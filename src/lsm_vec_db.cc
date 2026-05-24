#include "lsm_vec_db.h"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <limits>
#include <vector>

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace {
// 20260516_malloc_trim: file-local helper. Lives in the .cc so <malloc.h>
// doesn't leak into headers. No-op on non-glibc allocators (macOS / musl).
void trim_allocator_now() {
#ifdef __GLIBC__
    (void)::malloc_trim(0);
#endif
}
} // anonymous namespace

#include "distance.h"
#include "json.hpp"
#include "lsm_vec_index.h"
#include "logger.h"
#include "metadata.h"
#include "metadata_store.h"
#include "rocksdb/cache.h"
#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"

namespace lsm_vec
{
namespace {
constexpr char kDefaultVectorFileName[] = "vector.log";
constexpr char kDefaultLogFileName[] = "lsm_vec_db.log";
constexpr char kMetadataFileName[] = "lsm_vec_db.meta";
} // namespace

LSMVecDB::~LSMVecDB()
{
    // Tear down metadata resources in the right order if Close() was not called.
    metadata_store_.reset();
    if (metadata_cf_ && metadata_db_) {
        metadata_db_->DestroyColumnFamilyHandle(metadata_cf_);
        metadata_cf_ = nullptr;
    }
    metadata_db_.reset();
}

LSMVecDB::LSMVecDB(const std::string& db_path,
                   const LSMVecDBOptions& options,
                   std::unique_ptr<LSMVec> index,
                   std::unique_ptr<std::ostream> log_stream)
    : db_path_(db_path),
      options_(options),
      log_stream_(std::move(log_stream)),
      index_(std::move(index))
{
}

Status LSMVecDB::Open(const std::string& path,
                      const LSMVecDBOptions& opts,
                      std::unique_ptr<LSMVecDB>* db)
{
    if (!db) {
        return Status::InvalidArgument("db output must not be null");
    }
    if (opts.dim <= 0) {
        return Status::InvalidArgument("vector dimension must be positive");
    }
    if (opts.metric != DistanceMetric::kL2 &&
        opts.metric != DistanceMetric::kCosine) {
        return Status::NotSupported("unsupported distance metric");
    }

    initializeLogger(LogChoice::STDOUT, nullptr, LogSeverity::INFO);

    LSMVecDBOptions normalized_opts = opts;
    if (normalized_opts.vector_file_path.empty()) {
        normalized_opts.vector_file_path = path + "/" + kDefaultVectorFileName;
    }

    std::string log_path = opts.log_file_path.empty()
        ? path + "/" + kDefaultLogFileName
        : opts.log_file_path;
    auto log_stream = std::make_unique<std::ofstream>(log_path, std::ios::out);
    if (!static_cast<std::ofstream*>(log_stream.get())->is_open()) {
        return Status::IOError("failed to open log file");
    }

    auto index = std::make_unique<LSMVec>(path, normalized_opts, *log_stream);

    *db = std::unique_ptr<LSMVecDB>(
        new LSMVecDB(path, normalized_opts, std::move(index), std::move(log_stream)));

    if (!normalized_opts.reinit) {
        std::string metadata_path = path + "/" + kMetadataFileName;
        std::ifstream metadata_stream(metadata_path, std::ios::binary);
        if (metadata_stream.is_open()) {
            Status metadata_status = (*db)->index_->DeserializeMetadata(metadata_stream);
            if (!metadata_status.ok()) {
                return metadata_status;
            }
            // C5b: tombstoned set now lives inside LSMVec; no separate copy needed.
        }
    }

    // Open sibling rocksdb::DB at <path>/metadata for the metadata column family.
    {
        std::string meta_path = path + "/metadata";
        rocksdb::Options meta_opts;
        meta_opts.create_if_missing = true;
        meta_opts.create_missing_column_families = true;
        meta_opts.compression = rocksdb::kZSTD;
        // Cap total memtable memory across all CFs at 16 MB.
        meta_opts.db_write_buffer_size = 16 << 20;

        // One 16 MB LRU cache shared by default + metadata CFs. Without this,
        // each BlockBasedTableFactory would instantiate its own default cache
        // (32 MB each in current RocksDB). The default CF is never written to
        // so its contribution to the cache stays ~0.
        auto shared_block_cache = rocksdb::NewLRUCache(16 << 20);

        std::vector<rocksdb::ColumnFamilyDescriptor> cfds;

        rocksdb::BlockBasedTableOptions default_table_opts;
        default_table_opts.block_cache = shared_block_cache;
        rocksdb::ColumnFamilyOptions default_cfopts;
        default_cfopts.table_factory.reset(
            rocksdb::NewBlockBasedTableFactory(default_table_opts));
        cfds.emplace_back(rocksdb::kDefaultColumnFamilyName, default_cfopts);

        rocksdb::ColumnFamilyOptions cfopts;
        cfopts.compression = rocksdb::kZSTD;
        cfopts.write_buffer_size = 8 << 20;   // 8 MB per memtable
        cfopts.max_write_buffer_number = 2;   // up to 2 memtables: 2 * 8 MB = 16 MB
        rocksdb::BlockBasedTableOptions table_opts;
        table_opts.block_cache = shared_block_cache;
        table_opts.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
        cfopts.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opts));
        cfds.emplace_back("metadata", cfopts);

        std::vector<rocksdb::ColumnFamilyHandle*> handles;
        rocksdb::DB* meta_raw = nullptr;
        auto mst = rocksdb::DB::Open(meta_opts, meta_path, cfds, &handles, &meta_raw);
        if (!mst.ok()) {
            // Some RocksDB versions populate handles even on partial failure — clean up.
            for (auto* h : handles) {
                if (h && meta_raw) meta_raw->DestroyColumnFamilyHandle(h);
            }
            delete meta_raw;
            db->reset();  // API contract: *db is null on error.
            return mst;
        }

        (*db)->metadata_db_.reset(meta_raw);
        (*db)->metadata_cf_ = handles[1];
        (*db)->metadata_store_ =
            std::make_unique<MetadataStore>(meta_raw, handles[1]);
        // Default CF not used; destroy its handle now.
        meta_raw->DestroyColumnFamilyHandle(handles[0]);
    }

    return Status::OK();
}

Status LSMVecDB::ValidateVector(Span<float> vec) const
{
    if (vec.size() != static_cast<size_t>(options_.dim)) {
        return Status::InvalidArgument("vector dimension mismatch");
    }
    return Status::OK();
}

Status LSMVecDB::EnsureMetricSupported() const
{
    if (options_.metric != DistanceMetric::kL2 &&
        options_.metric != DistanceMetric::kCosine) {
        return Status::NotSupported("unsupported distance metric");
    }
    return Status::OK();
}

float LSMVecDB::ComputeDistance(Span<float> a, Span<float> b) const
{
    return distance::ComputeDistance(options_.metric,
                                     Span<const float>(a.data(), a.size()),
                                     Span<const float>(b.data(), b.size()));
}

Status LSMVecDB::Insert(node_id_t id, Span<float> vec)
{
    return Insert(id, vec, std::string_view{});
}

namespace {
// Hard cap on untrusted metadata input. Rejected before json::parse to
// prevent DoS via pathologically large payloads. Matches Milvus's
// JSON-field size limit. See docs/METADATA_FILTERING_FOLLOWUP.md (I1).
constexpr size_t kMaxMetadataJsonBytes = 64 * 1024;

// Parse and validate a metadata JSON string into a json object. Returns
// InvalidArgument on parse error, oversize input, or if the top-level
// value is not an object.
Status ParseMetadataObject(std::string_view json, nlohmann::json* out) {
    if (json.size() > kMaxMetadataJsonBytes) {
        return Status::InvalidArgument(
            "metadata JSON exceeds " + std::to_string(kMaxMetadataJsonBytes) + " bytes");
    }
    try {
        *out = nlohmann::json::parse(json);
    } catch (const std::exception& e) {
        return Status::InvalidArgument(std::string("metadata parse error: ") + e.what());
    }
    if (!out->is_object()) {
        return Status::InvalidArgument("metadata must be a JSON object");
    }
    return Status::OK();
}
}  // namespace

bool LSMVecDB::HasLiveVector(node_id_t id) const {
    return index_ && index_->is_alive(static_cast<real_id_t>(id));
}

Status LSMVecDB::Insert(node_id_t id, Span<float> vec, std::string_view metadata_json)
{
    const real_id_t r = static_cast<real_id_t>(id);
    if (r > kMaxRealId) {
        return Status::InvalidArgument("real_id must fit in 63 bits");
    }

    // Phase 3 (concurrent-writer-refactor-plan §5.2): acquire the
    // per-real-id transaction lock for the entire Insert/Upsert flow.
    // Two concurrent Insert(R, ...) calls now serialise on R's shard;
    // different real_ids hash into different shards and proceed in
    // parallel. recursive_mutex because UpdateInternal → Insert can
    // re-enter the same shard (re-insert path).
    std::lock_guard<std::recursive_mutex> txn_lk(real_id_lock(r));

    if (auto s = EnsureMetricSupported(); !s.ok()) return s;
    if (auto s = ValidateVector(vec); !s.ok()) return s;

    // Pre-parse metadata (fail-fast; no writes yet).
    const bool has_metadata = !metadata_json.empty() && metadata_json != "{}";
    if (has_metadata) {
        nlohmann::json parsed;
        auto pst = ParseMetadataObject(metadata_json, &parsed);
        if (!pst.ok()) return pst;
    }

    // A1: upsert on an alive real_id → route to Update.
    if (index_->is_alive(r)) {
        return UpdateInternal(id, vec, metadata_json);
    }

    // Q2: Insert after Delete → allocate a fresh update_id (matches Update path).
    // First-time Insert → use direct mapping.
    internal_id_t target_internal;
    const internal_id_t direct = static_cast<internal_id_t>(r);
    if (index_->is_tombstoned(direct)) {
        target_internal = index_->allocate_update_id();
        index_->record_update_mapping(r, target_internal);
    } else {
        target_internal = direct;
    }

    try {
        std::vector<float> data(vec.begin(), vec.end());
        index_->insertNode(target_internal, data);
    } catch (const std::exception& ex) {
        return Status::IOError(ex.what());
    }

    if (has_metadata && metadata_store_) {
        auto mst = metadata_store_->Put(target_internal, metadata_json);
        if (!mst.ok()) return mst;
    }
    return Status::OK();
}

Status LSMVecDB::GetPayload(node_id_t id, std::string* out_json)
{
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    const internal_id_t i = index_->resolve_internal(static_cast<real_id_t>(id));
    auto st = metadata_store_->Get(i, out_json);
    if (st.IsNotFound()) {
        *out_json = "{}";
        return Status::OK();
    }
    return st;
}

Status LSMVecDB::SetPayload(node_id_t id, std::string_view metadata_json)
{
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    const real_id_t r = static_cast<real_id_t>(id);
    if (!index_->is_alive(r)) return Status::NotFound("vector not found for id");

    nlohmann::json parsed;
    auto pst = ParseMetadataObject(metadata_json, &parsed);
    if (!pst.ok()) return pst;

    const internal_id_t i = index_->resolve_internal(r);
    return metadata_store_->Put(i, metadata_json);
}

Status LSMVecDB::UpdatePayload(node_id_t id, std::string_view partial_json)
{
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    const real_id_t r = static_cast<real_id_t>(id);
    if (!index_->is_alive(r)) return Status::NotFound("vector not found for id");

    nlohmann::json patch;
    auto pst = ParseMetadataObject(partial_json, &patch);
    if (!pst.ok()) return pst;

    const internal_id_t i = index_->resolve_internal(r);
    return metadata_store_->Update(i, patch);
}

Status LSMVecDB::DeletePayloadKeys(node_id_t id, Span<const std::string> keys)
{
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    const internal_id_t i = index_->resolve_internal(static_cast<real_id_t>(id));
    std::vector<std::string> key_vec(keys.data(), keys.data() + keys.size());
    return metadata_store_->DeleteKeys(i, key_vec);
}

Status LSMVecDB::Update(node_id_t id, Span<float> vec)
{
    return UpdateInternal(id, vec, std::string_view{});
}

Status LSMVecDB::UpdateInternal(node_id_t id, Span<float> vec,
                                std::string_view metadata_json)
{
    const real_id_t r = static_cast<real_id_t>(id);
    if (r > kMaxRealId) {
        return Status::InvalidArgument("real_id must fit in 63 bits");
    }

    // Phase 3 (§5.2): per-real-id transaction lock. Recursive — public
    // Update → UpdateInternal already holds it; Insert (when called
    // from this function via the re-insert path) re-enters the same
    // shard, harmless under recursive_mutex.
    std::lock_guard<std::recursive_mutex> txn_lk(real_id_lock(r));

    if (auto s = ValidateVector(vec); !s.ok()) return s;

    const bool has_metadata = !metadata_json.empty() && metadata_json != "{}";
    if (has_metadata) {
        nlohmann::json parsed;
        auto pst = ParseMetadataObject(metadata_json, &parsed);
        if (!pst.ok()) return pst;
    }

    // If r is not alive, route to Insert (first-time or after-Delete).
    // Insert's is_alive check will then fall through to the allocate-or-direct branch.
    if (!index_->is_alive(r)) {
        return Insert(id, vec, metadata_json);
    }

    // r is alive → allocate a fresh update internal_id, insert the new node,
    // tombstone the old, and migrate metadata.
    const internal_id_t old_internal = index_->resolve_internal(r);
    const internal_id_t new_internal = index_->allocate_update_id();

    try {
        std::vector<float> data(vec.begin(), vec.end());
        index_->insertNode(new_internal, data);
    } catch (const std::exception& ex) {
        return Status::IOError(ex.what());
    }

    // Record the new forward mapping (also erases stale reverse per A4).
    index_->record_update_mapping(r, new_internal);

    // Tombstone the previous version. Its vector and edges remain as routing
    // infrastructure; only its metadata row is deleted (see below).
    index_->tombstone(old_internal);

    // Metadata migration.
    if (metadata_store_) {
        if (has_metadata) {
            auto mst = metadata_store_->Put(new_internal, metadata_json);
            if (!mst.ok()) return mst;
        } else {
            // Preserve existing metadata across a vector-only Update.
            std::string old_meta;
            auto gst = metadata_store_->Get(old_internal, &old_meta);
            if (gst.ok()) {
                auto pst = metadata_store_->Put(new_internal, std::string_view{old_meta});
                if (!pst.ok()) return pst;
            }
        }
        auto dst = metadata_store_->Delete(old_internal);
        if (!dst.ok() && !dst.IsNotFound()) return dst;
    }
    return Status::OK();
}

Status LSMVecDB::Delete(node_id_t id)
{
    if (!index_) return Status::InvalidArgument("db not opened");
    const real_id_t r = static_cast<real_id_t>(id);

    // Phase 3 (§5.2): per-real-id transaction lock around the full
    // delete (resolve → tombstone → metadata delete → forget mapping).
    std::lock_guard<std::recursive_mutex> txn_lk(real_id_lock(r));

    const internal_id_t i = index_->resolve_internal(r);
    // Idempotent: if r never had a live vector, nothing to do (per design §5.3).
    if (!index_->vector_storage_ || !index_->vector_storage_->exists(i)) {
        return Status::OK();
    }

    // Tombstone the live internal (either direct real_id or current update_id).
    index_->tombstone(i);

    // A3: physical delete of the metadata row.
    if (metadata_store_) {
        auto mst = metadata_store_->Delete(i);
        if (!mst.ok() && !mst.IsNotFound()) return mst;
    }

    // A4: if r was update-allocated, drop the sparse-map entries so future
    // resolve_internal(r) returns identity.
    index_->forget_update_mapping(r);
    return Status::OK();
}

Status LSMVecDB::Get(node_id_t id, std::vector<float>* vec)
{
    if (!vec) return Status::InvalidArgument("output vector must not be null");
    if (!index_) return Status::InvalidArgument("db not opened");

    const real_id_t r = static_cast<real_id_t>(id);
    if (!index_->is_alive(r)) {
        return Status::NotFound("vector deleted or missing");
    }
    const internal_id_t i = index_->resolve_internal(r);

    try {
        auto timer = index_->stats.startTimer();
        index_->vector_storage_->readVectorFromDisk(i, *vec);
        index_->stats.accumulateTime(timer, index_->stats.vec_read_time);
        index_->stats.addCount(1, index_->stats.vec_read_count);
        if (timer.active) {
            DLOG(DEBUG) << "vector_read id=" << id << " time_s="
                        << (static_cast<double>(timer.duration_ns) / 1e9);
        }
    } catch (const std::exception& ex) {
        return Status::IOError(ex.what());
    }
    return Status::OK();
}

Status LSMVecDB::SearchKnn(Span<float> query,
                           const SearchOptions& options,
                           std::vector<SearchResult>* out)
{
    if (!out) {
        return Status::InvalidArgument("output results must not be null");
    }
    if (options.k <= 0) {
        return Status::InvalidArgument("k must be positive");
    }

    Status metric_status = EnsureMetricSupported();
    if (!metric_status.ok()) {
        return metric_status;
    }

    Status vec_status = ValidateVector(query);
    if (!vec_status.ok()) {
        return vec_status;
    }

    std::vector<float> query_vec(query.begin(), query.end());
    std::vector<SearchResult> results =
        index_->knnSearchK(query_vec, options.k, options.ef_search);
    if (results.empty()) {
        return Status::NotFound("index is empty");
    }

    out->clear();
    out->reserve(results.size());
    for (const auto& result : results) {
        // Defense in depth: LSMVec::knnSearchK already filters tombstones,
        // but redoing the check makes the translation safe if that changes.
        if (index_->is_tombstoned(result.id)) continue;
        SearchResult translated;
        translated.id       = index_->resolve_real(result.id);
        translated.distance = result.distance;
        out->push_back(translated);
        if (static_cast<int>(out->size()) >= options.k) break;
    }

    if (out->empty()) {
        return Status::NotFound("no available neighbors");
    }

    return Status::OK();
}

Status LSMVecDB::SearchKnn(Span<float> query,
                           const SearchOptions& options,
                           std::string_view filter_json,
                           std::vector<SearchResult>* out)
{
    if (!out) {
        return Status::InvalidArgument("output results must not be null");
    }
    if (options.k <= 0) {
        return Status::InvalidArgument("k must be positive");
    }

    Status metric_status = EnsureMetricSupported();
    if (!metric_status.ok()) {
        return metric_status;
    }

    Status vec_status = ValidateVector(query);
    if (!vec_status.ok()) {
        return vec_status;
    }

    // Fast path: empty or whitespace-only / "{}" filter routes to unfiltered overload.
    bool effectively_empty = true;
    size_t s = 0, e = filter_json.size();
    while (s < e && std::isspace(static_cast<unsigned char>(filter_json[s]))) ++s;
    while (e > s && std::isspace(static_cast<unsigned char>(filter_json[e - 1]))) --e;
    auto trimmed = filter_json.substr(s, e - s);
    if (!trimmed.empty() && trimmed != "{}") effectively_empty = false;

    if (effectively_empty) {
        return SearchKnn(query, options, out);
    }

    // Parse the filter predicate.
    metadata::Predicate pred;
    auto pst = metadata::ParsePredicate(filter_json, &pred);
    if (!pst.ok()) {
        return pst;
    }

    // Run filtered search.
    std::vector<float> qvec(query.begin(), query.end());
    std::vector<SearchResult> results = index_->knnSearchKFiltered(
        qvec,
        options.k,
        options.ef_search,
        &pred,
        options.max_scan_candidates,
        metadata_store_.get());

    out->clear();
    out->reserve(results.size());
    for (const auto& result : results) {
        if (index_->is_tombstoned(result.id)) continue;
        SearchResult translated;
        translated.id       = index_->resolve_real(result.id);
        translated.distance = result.distance;
        out->push_back(translated);
    }

    if (out->empty()) {
        return Status::NotFound("no available neighbors");
    }
    return Status::OK();
}

Status LSMVecDB::SearchKnn(Span<float> query,
                           std::vector<SearchResult>* out)
{
    SearchOptions opts;
    opts.k = options_.k;
    opts.ef_search = options_.ef_search;
    return SearchKnn(query, opts, out);
}

void LSMVecDB::printStatistics() const
{
    index_->printStatistics();
}

Status LSMVecDB::BulkBuild(Span<const float> vectors, int n,
                           const BulkBuildOptions& opts)
{
    if (!index_) return Status::InvalidArgument("db not opened");
    if (n <= 0) return Status::InvalidArgument("n must be > 0");
    if (options_.dim <= 0) {
        return Status::InvalidArgument("dim not set on db options");
    }
    if (vectors.size() !=
        static_cast<size_t>(n) * static_cast<size_t>(options_.dim)) {
        return Status::InvalidArgument(
            "vectors.size() must equal n * dim");
    }
    // Empty-DB precondition. BulkBuild is permanently initial-load
    // only — see docs/IN_MEMORY_BUILD_PLAN.md §0. We approximate
    // "empty" with: no entry point published, no in-memory upper-layer
    // nodes, no persisted dense slot. Caller that wants to rebuild
    // should Close() the DB, remove the data dir, and re-Open with
    // reinit=true (or pass an empty/new path).
    if (!index_->isEmpty()) {
        return Status::InvalidArgument(
            "BulkBuild requires an empty DB (entry_point already set). "
            "BulkBuild is initial-load-only; use Insert() for incremental "
            "updates on a non-empty DB.");
    }

    try {
        index_->bulkBuild(vectors.data(), n, opts);
    } catch (const std::exception& e) {
        return Status::IOError(std::string("BulkBuild failed: ") + e.what());
    } catch (...) {
        return Status::IOError("BulkBuild failed (unknown exception)");
    }

    // Match Insert's post-write side effects.
    flushVectorWrites();
    return Status::OK();
}

void LSMVecDB::flushVectorWrites()
{
    if (index_) {
        index_->vector_storage_->flushWrites();
    }
    // 20260516_malloc_trim: a bulk insert leaves the glibc heap fragmented
    // (RocksDB / Aster churn temporary buffers; our edge + page caches grow
    // and shrink). Trim now while we know we're at a write-batch boundary.
    trim_allocator_now();
}

void LSMVecDB::trimMemory()
{
    trim_allocator_now();
}

LSMVecDB::DeleteStats LSMVecDB::GetDeleteStats() const
{
    DeleteStats s{};
    if (!index_) return s;
    s.tombstones          = index_->tombstone_count();
    s.updated_real_ids    = index_->updated_real_id_count();
    s.total_inserts_ever  = index_->total_inserts_ever();
    s.tombstone_ratio     = index_->tombstone_ratio();
    s.bloom_capacity      = index_->bloom_capacity();
    s.bloom_fill_ratio    = index_->bloom_fill_ratio();
    s.bloom_rebuild_count = index_->bloom_rebuild_count();
    return s;
}

Status LSMVecDB::Close()
{
    if (!index_) {
        return Status::InvalidArgument("database not initialized");
    }

    // Tear down the metadata column family / DB FIRST, so file locks on
    // <path>/metadata are released even if the index-metadata file write
    // below fails (I3). These teardown steps do not fail; the destructor
    // performs the same sequence idempotently.
    if (metadata_store_) {
        metadata_store_.reset();
    }
    if (metadata_cf_ && metadata_db_) {
        metadata_db_->DestroyColumnFamilyHandle(metadata_cf_);
        metadata_cf_ = nullptr;
    }
    metadata_db_.reset();

    // Now persist the HNSW index metadata (may fail on disk-full etc.).
    std::string metadata_path = db_path_ + "/" + kMetadataFileName;
    std::ofstream metadata_stream(metadata_path, std::ios::binary | std::ios::trunc);
    if (!metadata_stream.is_open()) {
        return Status::IOError("failed to open metadata file for writing");
    }

    // C5b: LSMVec owns tombstoned_internal_ids_; no sync needed.
    Status metadata_status = index_->SerializeMetadata(metadata_stream);
    if (!metadata_status.ok()) {
        return metadata_status;
    }
    metadata_stream.flush();
    if (!metadata_stream.good()) {
        return Status::IOError("failed to flush metadata file");
    }
    if (log_stream_) {
        log_stream_->flush();
    }

    index_->close();
    return Status::OK();
}
} // namespace lsm_vec
