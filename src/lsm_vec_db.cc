#include "lsm_vec_db.h"

#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <vector>

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
            (*db)->deleted_ids_ = (*db)->index_->deletedIds();
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
    Status metric_status = EnsureMetricSupported();
    if (!metric_status.ok()) {
        return metric_status;
    }
    Status vec_status = ValidateVector(vec);
    if (!vec_status.ok()) {
        return vec_status;
    }

    try {
        std::vector<float> data(vec.begin(), vec.end());
        index_->insertNode(id, data);
        deleted_ids_.erase(id);
    } catch (const std::exception& ex) {
        return Status::IOError(ex.what());
    }

    return Status::OK();
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
    return index_
        && index_->vector_storage_
        && deleted_ids_.count(id) == 0
        && index_->vector_storage_->exists(id);
}

Status LSMVecDB::Insert(node_id_t id, Span<float> vec, std::string_view metadata_json)
{
    // 1. Validate vector up front (fail-fast; no writes yet).
    auto vst = ValidateVector(vec);
    if (!vst.ok()) return vst;

    // 2. Pre-parse metadata (fail-fast; no writes yet).
    const bool has_metadata = !metadata_json.empty() && metadata_json != "{}";
    if (has_metadata) {
        nlohmann::json parsed;
        auto pst = ParseMetadataObject(metadata_json, &parsed);
        if (!pst.ok()) return pst;
    }

    // 3. Insert vector via existing path.
    auto ist = Insert(id, vec);
    if (!ist.ok()) return ist;

    // 4. Persist metadata (if any). If this fails the vector remains inserted.
    if (has_metadata && metadata_store_) {
        auto mst = metadata_store_->Put(id, metadata_json);
        if (!mst.ok()) {
            return mst;
        }
    }
    return Status::OK();
}

Status LSMVecDB::GetPayload(node_id_t id, std::string* out_json)
{
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    auto st = metadata_store_->Get(id, out_json);
    if (st.IsNotFound()) {
        *out_json = "{}";
        return Status::OK();
    }
    return st;
}

Status LSMVecDB::SetPayload(node_id_t id, std::string_view metadata_json)
{
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    if (!HasLiveVector(id)) return Status::NotFound("vector not found for id");

    nlohmann::json parsed;
    auto pst = ParseMetadataObject(metadata_json, &parsed);
    if (!pst.ok()) return pst;

    return metadata_store_->Put(id, metadata_json);
}

Status LSMVecDB::UpdatePayload(node_id_t id, std::string_view partial_json)
{
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    if (!HasLiveVector(id)) return Status::NotFound("vector not found for id");

    nlohmann::json patch;
    auto pst = ParseMetadataObject(partial_json, &patch);
    if (!pst.ok()) return pst;

    return metadata_store_->Update(id, patch);
}

Status LSMVecDB::DeletePayloadKeys(node_id_t id, Span<const std::string> keys)
{
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    std::vector<std::string> key_vec(keys.data(), keys.data() + keys.size());
    return metadata_store_->DeleteKeys(id, key_vec);
}

Status LSMVecDB::Update(node_id_t id, Span<float> vec)
{
    Status vec_status = ValidateVector(vec);
    if (!vec_status.ok()) {
        return vec_status;
    }

    try {
        std::vector<float> data(vec.begin(), vec.end());
        auto timer = index_->stats.startTimer();
        index_->vector_storage_->storeVectorToDisk(id, data);
        index_->stats.accumulateTime(timer, index_->stats.vec_write_time);
        index_->stats.addCount(1, index_->stats.vec_write_count);
        if (timer.active) {
            DLOG(DEBUG) << "vector_write id=" << id << " time_s=" << timer.duration;
        }
        deleted_ids_.erase(id);
    } catch (const std::exception& ex) {
        return Status::IOError(ex.what());
    }

    return Status::OK();
}

Status LSMVecDB::Delete(node_id_t id)
{
    deleted_ids_.insert(id);
    Status delete_status = index_->deleteNode(id);
    if (!delete_status.ok()) {
        deleted_ids_.erase(id);
        return delete_status;
    }

    if (metadata_store_) {
        // Best-effort: ignore NotFound (id may have had no metadata).
        auto mst = metadata_store_->Delete(id);
        if (!mst.ok() && !mst.IsNotFound()) {
            return mst;
        }
    }
    return Status::OK();
}

Status LSMVecDB::Get(node_id_t id, std::vector<float>* vec)
{
    if (!vec) {
        return Status::InvalidArgument("output vector must not be null");
    }
    if (deleted_ids_.count(id) > 0) {
        return Status::NotFound("vector deleted");
    }

    try {
        auto timer = index_->stats.startTimer();
        index_->vector_storage_->readVectorFromDisk(id, *vec);
        index_->stats.accumulateTime(timer, index_->stats.vec_read_time);
        index_->stats.addCount(1, index_->stats.vec_read_count);
        if (timer.active) {
            DLOG(DEBUG) << "vector_read id=" << id << " time_s=" << timer.duration;
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
        if (deleted_ids_.count(result.id) > 0) {
            continue;
        }
        out->push_back(result);
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
        if (deleted_ids_.count(result.id) > 0) {
            continue;
        }
        out->push_back(result);
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

void LSMVecDB::flushVectorWrites()
{
    if (index_) {
        index_->vector_storage_->flushWrites();
    }
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

    index_->setDeletedIds(deleted_ids_);
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
