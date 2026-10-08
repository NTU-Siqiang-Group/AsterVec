#include "astervec_db.h"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_set>
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
#include "durable_io.h"
#include "json.hpp"
#include "astervec_index.h"
#include "logger.h"
#include "metadata.h"
#include "metadata_store.h"
#include "oplog.h"
#include "rocksdb/cache.h"
#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"

namespace astervec
{
namespace {
constexpr char kDefaultVectorFileName[] = "vector.log";
constexpr char kDefaultLogFileName[] = "lsm_vec_db.log";
constexpr char kMetadataFileName[] = "lsm_vec_db.meta";
constexpr char kOpLogDirName[] = "oplog";
// Present only while a new store is being created: written (with the chosen
// graph encoding) before the graph files exist, removed once the first
// metadata file is installed. Graph files with this marker and no metadata
// file are an interrupted create, not a damaged store.
constexpr char kCreatingMarkerName[] = "lsm_vec_db.creating";

// Nesting depth of WriteScope on this thread: public write calls call each
// other (Insert <-> UpdateInternal), and only the outermost takes the gate.
thread_local int t_write_scope_depth = 0;
} // namespace

class AsterVecDB::WriteScope {
public:
    explicit WriteScope(AsterVecDB& db)
        : db_(db), owner_(t_write_scope_depth == 0)
    {
        if (owner_) db_.write_gate_.lock_shared();
        ++t_write_scope_depth;
        db_.writes_since_checkpoint_.fetch_add(1, std::memory_order_relaxed);
    }
    ~WriteScope()
    {
        --t_write_scope_depth;
        if (!owner_) return;
        db_.write_gate_.unlock_shared();
        // checkpoint_interval_ops: checkpoint here, outside the shared gate
        // (Checkpoint takes it exclusively). One writer wins the flag.
        if (db_.checkpoint_due_.load(std::memory_order_relaxed) &&
            db_.checkpoint_due_.exchange(false)) {
            try {
                Status s = db_.Checkpoint();
                if (!s.ok()) {
                    LOG(ERR) << "automatic checkpoint failed: " << s.ToString();
                }
            } catch (const std::exception& e) {
                LOG(ERR) << "automatic checkpoint failed: " << e.what();
            }
        }
    }
    WriteScope(const WriteScope&) = delete;
    WriteScope& operator=(const WriteScope&) = delete;

private:
    AsterVecDB& db_;
    bool owner_;
};

AsterVecDB::~AsterVecDB()
{
    // Tear down metadata resources in the right order if Close() was not called.
    metadata_store_.reset();
}

AsterVecDB::AsterVecDB(const std::string& db_path,
                   const AsterVecDBOptions& options,
                   std::unique_ptr<AsterVec> index,
                   std::unique_ptr<std::ostream> log_stream)
    : db_path_(db_path),
      options_(options),
      log_stream_(std::move(log_stream)),
      index_(std::move(index))
{
}

Status AsterVecDB::Open(const std::string& path,
                      const AsterVecDBOptions& opts,
                      std::unique_ptr<AsterVecDB>* db)
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

    AsterVecDBOptions normalized_opts = opts;
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

    // Resolve the on-disk graph adjacency encoding BEFORE the index (and its
    // RocksGraph) is constructed. Aster treats the encoding as a constructor
    // parameter and never persists it, so the engine owns the choice: it is
    // fixed at create time, recorded in lsm_vec_db.meta (v3+), and replayed
    // on every open. Opening with the wrong value would silently mis-decode
    // every adjacency list.
    bool fresh_meta = false;
    bool write_create_marker = false;
    const std::string creating_marker = path + "/" + kCreatingMarkerName;
    {
        std::string metadata_path = path + "/" + kMetadataFileName;
        std::string metadata_bak = metadata_path + ".reinit_bak";
        if (normalized_opts.reinit) {
            // R1: two-phase, not delete. Aster's RocksGraph ctor exit(1)s on
            // an open failure (e.g. lock held by a live process), so a failed
            // reinit must leave the store's metadata recoverable: rename it
            // aside, and let the next Open restore the backup if the reinit
            // never completed. If a backup from a previous crashed attempt
            // already exists and the main file is gone, keep that backup.
            if (std::ifstream(metadata_path).good()) {
                std::remove(metadata_bak.c_str());
                if (std::rename(metadata_path.c_str(),
                                metadata_bak.c_str()) != 0) {
                    return Status::IOError(
                        "failed to stage metadata for reinit");
                }
            }
        } else if (!std::ifstream(metadata_path).good() &&
                   std::ifstream(metadata_bak).good()) {
            // Orphaned backup from a crashed/failed reinit: restore it.
            if (std::rename(metadata_bak.c_str(),
                            metadata_path.c_str()) != 0) {
                return Status::IOError(
                    "failed to restore metadata backup from a failed reinit");
            }
        }
        const std::string metadata_prev = metadata_path + ".prev";
        // A leftover temp file is an unfinished checkpoint; the previous
        // complete one is still in place.
        std::remove((metadata_path + ".tmp").c_str());
        if (normalized_opts.reinit) {
            std::remove(metadata_prev.c_str());
        }
        uint32_t enc = 0;
        Status pst = AsterVec::PeekGraphEncoding(metadata_path, &enc);
        if (!pst.ok() && !normalized_opts.reinit &&
            durable::file_exists(metadata_prev)) {
            Status prev_st = AsterVec::PeekGraphEncoding(metadata_prev, &enc);
            if (prev_st.ok()) pst = prev_st;
        }
        if (pst.ok()) {
            normalized_opts.graph_encoding_internal = enc;
        } else if (pst.IsNotFound()) {
            // A crash inside the Open that created this store: the graph
            // files exist but hold no user data (Open had not returned), and
            // the marker records the encoding they were created with.
            uint32_t marker_enc = 0;
            bool interrupted_create = false;
            if (!normalized_opts.reinit) {
                std::ifstream m(creating_marker);
                interrupted_create = static_cast<bool>(m >> marker_enc);
            }
            if (!normalized_opts.reinit && !interrupted_create &&
                std::ifstream(path + "/CURRENT").good()) {
                return Status::Corruption(
                    "graph data exists but lsm_vec_db.meta is missing; the "
                    "adjacency encoding cannot be determined — refusing to "
                    "guess. Rebuild the store (reinit) to recover.");
            }
            if (interrupted_create) {
                normalized_opts.graph_encoding_internal = marker_enc;
            } else {
                // True new DB: varint by default. ASTERVEC_GRAPH_VARINT=0 is
                // a bake-in kill-switch, honored at CREATE time only —
                // whatever is created gets recorded and replayed regardless
                // of env later.
                const char* kill = std::getenv("ASTERVEC_GRAPH_VARINT");
                bool use_varint = !(kill && kill[0] == '0');
                normalized_opts.graph_encoding_internal =
                    use_varint ? ENCODING_TYPE_VARINT : ENCODING_TYPE_NONE;
                write_create_marker = !normalized_opts.reinit;
            }
            fresh_meta = true;
        } else {
            return pst;  // corrupt metadata header
        }
    }

    if (write_create_marker) {
        std::error_code ec;
        std::filesystem::create_directories(path, ec);
        {
            std::ofstream m(creating_marker, std::ios::trunc);
            m << normalized_opts.graph_encoding_internal << "\n";
            if (!m) return Status::IOError("failed to write " + creating_marker);
        }
        if (!durable::fsync_file(creating_marker) || !durable::fsync_dir(path)) {
            return Status::IOError("failed to sync " + creating_marker);
        }
    }

    auto index = std::make_unique<AsterVec>(path, normalized_opts, *log_stream);

    *db = std::unique_ptr<AsterVecDB>(
        new AsterVecDB(path, normalized_opts, std::move(index), std::move(log_stream)));

    if (normalized_opts.reinit || fresh_meta) {
        // The graph and vector files were just recreated; payloads (reinit)
        // and logged operations of the old store must go with them.
        std::error_code ec;
        if (normalized_opts.reinit) {
            std::filesystem::remove_all(path + "/metadata", ec);
        }
        std::filesystem::remove_all(path + "/" + kOpLogDirName, ec);
    }
    if (!normalized_opts.reinit) {
        // Load the latest checkpoint; if it is damaged, fall back to the
        // previous one. The graph may be ahead of either (WAL-recovered
        // edges); traversal skips nodes the checkpoint does not know.
        const std::string metadata_path = path + "/" + kMetadataFileName;
        const std::string metadata_prev = metadata_path + ".prev";
        const bool have_main = durable::file_exists(metadata_path);
        Status ms = Status::OK();
        if (have_main) {
            std::ifstream in(metadata_path, std::ios::binary);
            ms = (*db)->index_->DeserializeMetadata(in);
        }
        if ((!have_main || !ms.ok()) && durable::file_exists(metadata_prev)) {
            std::ifstream in(metadata_prev, std::ios::binary);
            Status ps = (*db)->index_->DeserializeMetadata(in);
            if (!ps.ok()) {
                return have_main
                    ? Status::Corruption("both " + metadata_path + " (" +
                                         ms.ToString() + ") and " +
                                         metadata_prev + " (" + ps.ToString() +
                                         ") are unusable")
                    : ps;
            }
            if (have_main) {
                LOG(WARN) << metadata_path << " is unusable (" << ms.ToString()
                          << "); opened from " << metadata_prev;
                // Drop the damaged file so the next checkpoint does not
                // rotate it over the good previous copy.
                std::remove(metadata_path.c_str());
            }
        } else if (!ms.ok()) {
            return ms;
        }
        (*db)->main_checkpoint_lsn_ = (*db)->index_->checkpointLsn();
    }

    // Point-id mode: chosen when the store is created, then read back.
    {
        AsterVecDB& d = **db;
        if (fresh_meta) {
            d.index_->setIdMode(normalized_opts.id_mode == IdMode::kKeyed ? 1 : 0);
        } else {
            const IdMode stored =
                d.index_->idMode() == 1 ? IdMode::kKeyed : IdMode::kDense;
            if (normalized_opts.id_mode != IdMode::kAuto &&
                normalized_opts.id_mode != stored) {
                db->reset();
                return Status::InvalidArgument(
                    std::string("store was created with id_mode ") +
                    (stored == IdMode::kKeyed ? "keyed" : "dense"));
            }
        }
        d.keyed_ = d.index_->idMode() == 1;
        d.options_.id_mode = d.keyed_ ? IdMode::kKeyed : IdMode::kDense;
    }

    if (fresh_meta) {
        // Persist the encoding BEFORE any user write reaches the graph: a
        // crash before the first checkpoint must not leave it ambiguous. A
        // full (empty-index) serialize is used so the file is a valid meta.
        Status wst = (*db)->writeMetadataFile();
        if (!wst.ok()) {
            return wst;
        }
        // Reinit completed and the fresh meta is on disk: the staged backup
        // (if any) is now obsolete, and so is a create marker.
        std::remove((path + "/" + kMetadataFileName + ".reinit_bak").c_str());
        std::remove(creating_marker.c_str());
    }

    // Open the sibling metadata KV store (owned end-to-end by MetadataStore;
    // no RocksDB types cross this boundary — storage seam).
    {
        Status mst = MetadataStore::Open(path + "/metadata",
                                         &(*db)->metadata_store_,
                                         (*db)->keyed_);
        if (!mst.ok()) {
            db->reset();  // API contract: *db is null on error.
            return mst;
        }
        if ((*db)->keyed_) {
            node_id_t max_id = 0;
            bool any = false;
            Status ks = (*db)->metadata_store_->MaxMappedId(&max_id, &any);
            if (!ks.ok()) {
                db->reset();
                return ks;
            }
            (*db)->next_dense_id_.store(any ? max_id + 1 : 0,
                                        std::memory_order_relaxed);
        }
    }

    // Restore the live vector count persisted by a prior flush/Close (0 on a
    // fresh or wiped index, or after an unclean crash with no persisted file).
    // The count recorded in the checkpoint matches it exactly (also after a
    // fallback to .prev); the vector_count file serves pre-v4 stores.
    (*db)->loadCount();
    if ((*db)->index_->checkpointLiveCount() >= 0) {
        (*db)->live_count_.store((*db)->index_->checkpointLiveCount(),
                                 std::memory_order_relaxed);
    }

    // Re-apply operations logged after the checkpoint, then start the log.
    if (Status rs = (*db)->recoverFromOpLog(); !rs.ok()) {
        db->reset();
        return rs;
    }
    return Status::OK();
}

Status AsterVecDB::logOp(OpType op, node_id_t real_id,
                         node_id_t internal_id, node_id_t old_internal_id,
                         Span<float> vec, std::string_view payload,
                         std::string_view key, std::uint64_t* lsn,
                         const char** encoded)
{
    *lsn = 0;
    *encoded = nullptr;
    if (options_.checkpoint_interval_ops > 0 &&
        ops_since_checkpoint_.fetch_add(1, std::memory_order_relaxed) + 1 >=
            options_.checkpoint_interval_ops) {
        checkpoint_due_.store(true, std::memory_order_relaxed);
    }
    if (!oplog_) return Status::OK();

    OpRecord rec;
    rec.op = op;
    rec.real_id = real_id;
    rec.internal_id = internal_id;
    rec.aux = old_internal_id;
    rec.key = key;
    rec.payload = payload;
    if (!vec.empty()) {
        // Encode once: the log and the store get the same bytes.
        IVectorStorage* vs = index_->vector_storage_.get();
        thread_local std::vector<char> record;
        record.resize(vs->recordSize());
        vs->encodeRecord(vec.data(), record.data());
        rec.encoding = vs->encodingTag();
        rec.record = std::string_view(record.data(), record.size());
        *encoded = record.data();
    }
    return oplog_->Append(rec, lsn);
}

void AsterVecDB::logAbort(std::uint64_t lsn)
{
    if (!oplog_ || lsn == 0) return;
    OpRecord rec;
    rec.op = OpType::kAbort;
    rec.aux = lsn;
    std::uint64_t abort_lsn = 0;
    Status s = oplog_->Append(rec, &abort_lsn);
    if (!s.ok()) {
        LOG(ERR) << "cannot log abort of lsn " << lsn << ": " << s.ToString();
    }
}

Status AsterVecDB::recoverFromOpLog()
{
    const std::string dir = db_path_ + "/" + kOpLogDirName;
    const std::uint64_t from = index_->checkpointLsn() + 1;
    std::uint64_t next_lsn = from;
    recovery_stats_ = RecoveryStats{};
    recovery_stats_.checkpoint_lsn = from - 1;

    if (durable::file_exists(dir)) {
        const auto t0 = std::chrono::steady_clock::now();
        // Pass 1: operations that failed after they were logged.
        std::unordered_set<std::uint64_t> aborted;
        OpLog::ScanStats st;
        Status s = OpLog::Scan(dir, from, [&](const OpRecord& r) {
            if (r.op == OpType::kAbort) aborted.insert(r.aux);
            return Status::OK();
        }, &st);
        if (!s.ok()) return s;
        // Pass 2: apply in lsn order.
        std::vector<float> scratch(static_cast<size_t>(options_.dim));
        s = OpLog::Scan(dir, from, [&](const OpRecord& r) {
            return replayRecord(r, aborted.count(r.lsn) > 0, scratch);
        }, &st);
        if (!s.ok()) return s;
        if (st.records > 0) next_lsn = st.last_lsn + 1;

        recovery_stats_.replayed_ops = st.records;
        recovery_stats_.dropped_bytes = st.dropped_bytes;
        recovery_stats_.lsn_gap = st.gap;
        recovery_stats_.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        if (st.records > 0 || st.dropped_bytes > 0 || st.gap) {
            LOG(INFO) << "operation log: replayed " << st.records
                      << " ops after checkpoint lsn " << (from - 1) << " in "
                      << recovery_stats_.seconds << " s; dropped "
                      << st.dropped_bytes << " torn bytes"
                      << (st.gap ? "; stopped at a missing lsn" : "");
        }
    }

    if (options_.enable_oplog) {
        Status s = OpLog::Open(dir, next_lsn, options_.oplog_segment_bytes, &oplog_);
        if (!s.ok()) return s;
    }
    if (next_lsn != from) {
        // Make the replayed state the restart point so the next crash does
        // not replay it again.
        index_->setCheckpointLsn(next_lsn - 1);
        writes_since_checkpoint_.fetch_add(1, std::memory_order_relaxed);
        std::unique_lock<std::shared_mutex> gate(write_gate_);
        if (Status s = checkpointLocked(); !s.ok()) return s;
    }
    if (!options_.enable_oplog) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    return Status::OK();
}

Status AsterVecDB::replayRecord(const OpRecord& r, bool aborted,
                                std::vector<float>& scratch)
{
    const real_id_t real = static_cast<real_id_t>(r.real_id);
    const internal_id_t internal = static_cast<internal_id_t>(r.internal_id);
    switch (r.op) {
    case OpType::kAbort:
        return Status::OK();

    case OpType::kInsert:
    case OpType::kUpdate: {
        // A failed operation still consumed its update id (and its key its
        // dense id).
        if (is_update_id(internal)) index_->observe_update_id(internal);
        if (!r.key.empty() && metadata_store_ && metadata_store_->has_id_map()) {
            Status ks = metadata_store_->PutKey(r.key, real);
            if (!ks.ok()) return ks;
            observeDenseId(real);
        }
        if (aborted) return Status::OK();
        IVectorStorage* vs = index_->vector_storage_.get();
        if (r.encoding != vs->encodingTag() || r.record.size() != vs->recordSize()) {
            return Status::Corruption(
                "operation log record does not match the vector store encoding");
        }
        vs->decodeRecord(r.record.data(), scratch.data());
        const bool was_alive = index_->is_alive(real);
        if (r.op == OpType::kInsert && is_update_id(internal)) {
            index_->record_update_mapping(real, internal);
        }
        try {
            Status ist = index_->insertNode(internal, scratch, r.record.data());
            if (!ist.ok()) return ist;
        } catch (const std::exception& ex) {
            return Status::IOError(std::string("replay: ") + ex.what());
        }
        const internal_id_t old_internal = static_cast<internal_id_t>(r.aux);
        if (r.op == OpType::kUpdate) {
            index_->record_update_mapping(real, internal);
            index_->tombstone(old_internal);
        }
        if (!was_alive) live_count_.fetch_add(1, std::memory_order_relaxed);

        // Payloads have their own WAL and may already hold this write or a
        // later (unlogged) payload change: only fill a missing row.
        if (metadata_store_) {
            std::string existing;
            const bool have_new = metadata_store_->Get(internal, &existing).ok();
            if (!have_new && !r.payload.empty()) {
                Status ps = metadata_store_->Put(internal, r.payload);
                if (!ps.ok()) return ps;
            } else if (!have_new && r.op == OpType::kUpdate) {
                std::string old_meta;
                if (metadata_store_->Get(old_internal, &old_meta).ok()) {
                    Status ps = metadata_store_->Put(internal, std::string_view{old_meta});
                    if (!ps.ok()) return ps;
                }
            }
            if (r.op == OpType::kUpdate) {
                Status ds = metadata_store_->Delete(old_internal);
                if (!ds.ok() && !ds.IsNotFound()) return ds;
            }
        }
        break;
    }

    case OpType::kDelete: {
        if (aborted) return Status::OK();
        const bool was_alive = index_->is_alive(real);
        index_->tombstone(internal);
        if (metadata_store_) {
            Status ds = metadata_store_->Delete(internal);
            if (!ds.ok() && !ds.IsNotFound()) return ds;
        }
        index_->forget_update_mapping(real);
        if (was_alive) live_count_.fetch_sub(1, std::memory_order_relaxed);
        break;
    }

    default:
        return Status::Corruption("operation log: unknown op " +
                                  std::to_string(static_cast<int>(r.op)));
    }
    writes_since_checkpoint_.fetch_add(1, std::memory_order_relaxed);
    return Status::OK();
}

Status AsterVecDB::ValidateVector(Span<float> vec) const
{
    if (vec.size() != static_cast<size_t>(options_.dim)) {
        return Status::InvalidArgument("vector dimension mismatch");
    }
    return Status::OK();
}

Status AsterVecDB::EnsureMetricSupported() const
{
    if (options_.metric != DistanceMetric::kL2 &&
        options_.metric != DistanceMetric::kCosine) {
        return Status::NotSupported("unsupported distance metric");
    }
    return Status::OK();
}

float AsterVecDB::ComputeDistance(Span<float> a, Span<float> b) const
{
    return distance::ComputeDistance(options_.metric,
                                     Span<const float>(a.data(), a.size()),
                                     Span<const float>(b.data(), b.size()));
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

bool AsterVecDB::HasLiveVector(node_id_t id) const {
    return index_ && index_->is_alive(static_cast<real_id_t>(id));
}

Status AsterVecDB::validateInsertArgs(Span<float> vec,
                                      std::string_view metadata_json) const
{
    if (auto s = EnsureMetricSupported(); !s.ok()) return s;
    if (auto s = ValidateVector(vec); !s.ok()) return s;
    // Metadata present (non-empty string, incl. "{}") must be a JSON object
    // within the size cap. Absent (empty string) means "no metadata field".
    if (!metadata_json.empty()) {
        nlohmann::json parsed;
        if (auto s = ParseMetadataObject(metadata_json, &parsed); !s.ok()) return s;
    }
    return Status::OK();
}

Status AsterVecDB::insertDense(node_id_t id, Span<float> vec,
                               std::string_view metadata_json,
                               std::string_view key)
{
    WriteScope ws(*this);
    // Validate everything up front (metric, vector, metadata). Batch callers
    // run ValidateInsert for every item first, so a bad item writes nothing.
    if (auto s = validateInsertArgs(vec, metadata_json); !s.ok()) return s;
    const real_id_t r = static_cast<real_id_t>(id);

    // Phase 3 (concurrent-writer-refactor-plan §5.2): acquire the
    // per-real-id transaction lock for the entire Insert/Upsert flow.
    // Two concurrent Insert(R, ...) calls now serialise on R's shard;
    // different real_ids hash into different shards and proceed in
    // parallel. recursive_mutex because UpdateInternal → Insert can
    // re-enter the same shard (re-insert path).
    std::lock_guard<std::recursive_mutex> txn_lk(real_id_lock(r));

    // Metadata field present (incl. an explicit "{}") => write/replace the
    // payload ("{}" clears it). Absent (empty string) => preserve the existing.
    const bool has_metadata = !metadata_json.empty();

    // A1: upsert on an alive real_id → route to Update.
    if (index_->is_alive(r)) {
        return updateDense(id, vec, metadata_json, key);
    }

    // Q2: Insert after Delete → allocate a fresh update_id (matches Update path).
    // First-time Insert → use direct mapping.
    const internal_id_t direct = static_cast<internal_id_t>(r);
    const bool reinsert = index_->is_tombstoned(direct);
    const internal_id_t target_internal =
        reinsert ? index_->allocate_update_id() : direct;

    std::uint64_t lsn = 0;
    const char* encoded = nullptr;
    if (auto s = logOp(OpType::kInsert, r, target_internal, 0, vec,
                       metadata_json, key, &lsn, &encoded);
        !s.ok()) {
        return s;
    }
    if (reinsert) index_->record_update_mapping(r, target_internal);

    try {
        std::vector<float> data(vec.begin(), vec.end());
        {
            Status ist = index_->insertNode(target_internal, data, encoded);
            if (!ist.ok()) {
                logAbort(lsn);
                return ist;
            }
        }
    } catch (const std::exception& ex) {
        logAbort(lsn);
        return Status::IOError(ex.what());
    }
    // A new live vector now exists (this branch is reached only when r was not
    // alive — first insert or re-insert after delete). Upserts route to Update
    // above and don't reach here, so this never double-counts.
    live_count_.fetch_add(1, std::memory_order_relaxed);

    if (has_metadata && metadata_store_) {
        auto mst = metadata_store_->Put(target_internal, metadata_json);
        if (!mst.ok()) return mst;
    }
    return Status::OK();
}

Status AsterVecDB::getPayloadDense(node_id_t id, std::string* out_json)
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

Status AsterVecDB::setPayloadDense(node_id_t id, std::string_view metadata_json)
{
    WriteScope ws(*this);
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    const real_id_t r = static_cast<real_id_t>(id);
    if (!index_->is_alive(r)) return Status::NotFound("vector not found for id");

    nlohmann::json parsed;
    auto pst = ParseMetadataObject(metadata_json, &parsed);
    if (!pst.ok()) return pst;

    const internal_id_t i = index_->resolve_internal(r);
    return metadata_store_->Put(i, metadata_json);
}

Status AsterVecDB::SetPayloadBatch(
    const std::vector<std::pair<node_id_t, std::string>>& items)
{
    std::vector<std::pair<PointId, std::string>> pts;
    pts.reserve(items.size());
    for (const auto& kv : items) pts.emplace_back(PointId::Int(kv.first), kv.second);
    return SetPayloadBatch(pts);
}

Status AsterVecDB::SetPayloadBatch(
    const std::vector<std::pair<PointId, std::string>>& items)
{
    WriteScope ws(*this);
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    // Validate EVERY item (alive + metadata object within size cap) before any
    // write, then commit all rows in one atomic WriteBatch.
    std::vector<std::pair<node_id_t, std::string>> internal_items;
    internal_items.reserve(items.size());
    for (const auto& kv : items) {
        node_id_t dense = 0;
        std::string key;
        Status rs = resolveId(kv.first, /*create=*/false, &dense, &key);
        if (!rs.ok() && !rs.IsNotFound()) return rs;
        const real_id_t r = static_cast<real_id_t>(dense);
        if (rs.IsNotFound() || !index_->is_alive(r)) {
            return Status::NotFound(
                "vector not found for id " +
                (kv.first.is_int() ? std::to_string(kv.first.as_int()) : kv.first.as_str()));
        }
        nlohmann::json parsed;
        auto pst = ParseMetadataObject(kv.second, &parsed);
        if (!pst.ok()) return pst;
        const internal_id_t i = index_->resolve_internal(r);
        internal_items.emplace_back(static_cast<node_id_t>(i), kv.second);
    }
    return metadata_store_->PutBatch(internal_items);
}

Status AsterVecDB::updatePayloadDense(node_id_t id, std::string_view partial_json)
{
    WriteScope ws(*this);
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    const real_id_t r = static_cast<real_id_t>(id);
    if (!index_->is_alive(r)) return Status::NotFound("vector not found for id");

    nlohmann::json patch;
    auto pst = ParseMetadataObject(partial_json, &patch);
    if (!pst.ok()) return pst;

    const internal_id_t i = index_->resolve_internal(r);
    return metadata_store_->Update(i, patch);
}

Status AsterVecDB::deletePayloadKeysDense(node_id_t id, Span<const std::string> keys)
{
    WriteScope ws(*this);
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    const internal_id_t i = index_->resolve_internal(static_cast<real_id_t>(id));
    std::vector<std::string> key_vec(keys.data(), keys.data() + keys.size());
    return metadata_store_->DeleteKeys(i, key_vec);
}

Status AsterVecDB::updateDense(node_id_t id, Span<float> vec,
                               std::string_view metadata_json,
                               std::string_view key)
{
    WriteScope ws(*this);
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

    // Metadata field present (incl. an explicit "{}") => write/replace the
    // payload ("{}" clears it). Absent (empty string) => preserve the existing.
    const bool has_metadata = !metadata_json.empty();
    if (has_metadata) {
        nlohmann::json parsed;
        auto pst = ParseMetadataObject(metadata_json, &parsed);
        if (!pst.ok()) return pst;
    }

    // If r is not alive, route to Insert (first-time or after-Delete).
    // Insert's is_alive check will then fall through to the allocate-or-direct branch.
    if (!index_->is_alive(r)) {
        return insertDense(id, vec, metadata_json, key);
    }

    // r is alive → allocate a fresh update internal_id, insert the new node,
    // tombstone the old, and migrate metadata.
    const internal_id_t old_internal = index_->resolve_internal(r);
    const internal_id_t new_internal = index_->allocate_update_id();

    std::uint64_t lsn = 0;
    const char* encoded = nullptr;
    if (auto s = logOp(OpType::kUpdate, r, new_internal, old_internal, vec,
                       metadata_json, key, &lsn, &encoded);
        !s.ok()) {
        return s;
    }

    try {
        std::vector<float> data(vec.begin(), vec.end());
        {
            Status ist = index_->insertNode(new_internal, data, encoded);
            if (!ist.ok()) {
                logAbort(lsn);
                return ist;
            }
        }
    } catch (const std::exception& ex) {
        logAbort(lsn);
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

Status AsterVecDB::deleteDense(node_id_t id)
{
    WriteScope ws(*this);
    if (!index_) return Status::InvalidArgument("db not opened");
    const real_id_t r = static_cast<real_id_t>(id);

    // Phase 3 (§5.2): per-real-id transaction lock around the full
    // delete (resolve → tombstone → metadata delete → forget mapping).
    std::lock_guard<std::recursive_mutex> txn_lk(real_id_lock(r));

    // Capture liveness BEFORE tombstoning so the count only drops for a real
    // live->deleted transition (a repeat delete re-tombstones harmlessly but
    // must not decrement again).
    const bool was_alive = index_->is_alive(r);

    const internal_id_t i = index_->resolve_internal(r);
    // Idempotent: if r never had a live vector, nothing to do (per design §5.3).
    if (!index_->vector_storage_ || !index_->vector_storage_->exists(i)) {
        return Status::OK();
    }

    {
        std::uint64_t lsn = 0;
        const char* unused = nullptr;
        if (auto s = logOp(OpType::kDelete, r, i, 0, Span<float>(),
                           std::string_view{}, std::string_view{}, &lsn,
                           &unused);
            !s.ok()) {
            return s;
        }
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
    // Decrement only for a genuine live->deleted transition (was_alive captured
    // above); a repeat delete of an already-tombstoned id must not double-count.
    if (was_alive) live_count_.fetch_sub(1, std::memory_order_relaxed);
    return Status::OK();
}

Status AsterVecDB::getDense(node_id_t id, std::vector<float>* vec)
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

Status AsterVecDB::searchUnfilteredDense(Span<float> query,
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
    if (index_->graphCorrupted()) {
        return index_->graphCorruptionStatus();
    }

    // Nothing to return (empty index, every hit deleted) is an empty result,
    // not an error.
    out->clear();
    out->reserve(results.size());
    for (const auto& result : results) {
        // Defense in depth: AsterVec::knnSearchK already filters tombstones,
        // but redoing the check makes the translation safe if that changes.
        if (index_->is_tombstoned(result.id)) continue;
        SearchResult translated;
        translated.id       = index_->resolve_real(result.id);
        translated.distance = result.distance;
        out->push_back(translated);
        if (static_cast<int>(out->size()) >= options.k) break;
    }
    return Status::OK();
}

Status AsterVecDB::searchDense(Span<float> query,
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
        return searchUnfilteredDense(query, options, out);
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
    if (index_->graphCorrupted()) {
        return index_->graphCorruptionStatus();
    }

    out->clear();
    out->reserve(results.size());
    for (const auto& result : results) {
        if (index_->is_tombstoned(result.id)) continue;
        SearchResult translated;
        translated.id       = index_->resolve_real(result.id);
        translated.distance = result.distance;
        out->push_back(translated);
    }
    // No point matching the filter is an empty result, not an error.
    return Status::OK();
}

Status AsterVecDB::SearchKnn(Span<float> query,
                           std::vector<SearchResult>* out)
{
    SearchOptions opts;
    opts.k = options_.k;
    opts.ef_search = options_.ef_search;
    return SearchKnn(query, opts, out);
}

void AsterVecDB::printStatistics() const
{
    index_->printStatistics();
}

void AsterVecDB::resetStatistics()
{
    index_->resetStatistics();
}

Status AsterVecDB::BulkBuild(Span<const float> vectors, int n,
                           const BulkBuildOptions& opts)
{
    return bulkBuildImpl(vectors, n, opts, nullptr);
}

Status AsterVecDB::BulkBuild(Span<const float> vectors,
                           const std::vector<PointId>& ids,
                           const BulkBuildOptions& opts)
{
    return bulkBuildImpl(vectors, static_cast<int>(ids.size()), opts, &ids);
}

Status AsterVecDB::bulkBuildImpl(Span<const float> vectors, int n,
                               const BulkBuildOptions& opts,
                               const std::vector<PointId>* ids)
{
    if (!index_) return Status::InvalidArgument("db not opened");
    if (n <= 0) return Status::InvalidArgument("n must be > 0");
    if (ids && !keyed_) {
        return Status::InvalidArgument(
            "BulkBuild with caller ids needs a store created with id_mode keyed");
    }
    std::unique_lock<std::shared_mutex> gate(write_gate_);
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
    if (!index_->isEmpty() ||
        (keyed_ && next_dense_id_.load(std::memory_order_relaxed) != 0)) {
        return Status::InvalidArgument(
            "BulkBuild requires an empty DB (entry_point already set). "
            "BulkBuild is initial-load-only; use Insert() for incremental "
            "updates on a non-empty DB.");
    }
    // Keyed stores: vector i gets dense id i and the key ids[i] (default:
    // the integer i). Keys must be distinct.
    std::vector<std::pair<std::string, node_id_t>> keys;
    if (keyed_) {
        keys.reserve(static_cast<size_t>(n));
        std::unordered_set<std::string> seen;
        seen.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const PointId pid = ids ? (*ids)[static_cast<size_t>(i)]
                                    : PointId::Int(static_cast<uint64_t>(i));
            if (!pid.is_int() && pid.as_str().size() > PointId::kMaxStringBytes) {
                return Status::InvalidArgument("string id longer than 1024 bytes");
            }
            std::string k = pid.Encode();
            if (!seen.insert(k).second) {
                return Status::InvalidArgument("BulkBuild ids must be distinct");
            }
            keys.emplace_back(std::move(k), static_cast<node_id_t>(i));
        }
    }

    try {
        index_->bulkBuild(vectors.data(), n, opts);
    } catch (const std::exception& e) {
        return Status::IOError(std::string("BulkBuild failed: ") + e.what());
    } catch (...) {
        return Status::IOError("BulkBuild failed (unknown exception)");
    }

    // Bulk build loads n vectors at dense ids 0..n-1 into a fresh, empty index.
    live_count_.store(static_cast<std::int64_t>(n), std::memory_order_relaxed);
    if (keyed_) {
        if (Status ks = metadata_store_->PutKeys(keys); !ks.ok()) return ks;
        next_dense_id_.store(static_cast<node_id_t>(n), std::memory_order_relaxed);
    }

    writes_since_checkpoint_.fetch_add(1, std::memory_order_relaxed);
    Status cs = checkpointLocked();
    trim_allocator_now();
    return cs;
}

std::size_t AsterVecDB::VectorCount() const
{
    const std::int64_t v = live_count_.load(std::memory_order_relaxed);
    return v > 0 ? static_cast<std::size_t>(v) : 0;
}

void AsterVecDB::persistCount() const
{
    const std::string path = db_path_ + "/vector_count";
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f.is_open()) return;
        f << live_count_.load(std::memory_order_relaxed) << "\n";
        if (!f) return;
    }
    std::rename(tmp.c_str(), path.c_str());
}

void AsterVecDB::loadCount()
{
    std::ifstream f(db_path_ + "/vector_count");
    long long v = 0;
    if (f.is_open() && (f >> v) && v >= 0) {
        live_count_.store(static_cast<std::int64_t>(v), std::memory_order_relaxed);
    } else {
        live_count_.store(0, std::memory_order_relaxed);
    }
}

Status AsterVecDB::writeMetadataFile()
{
    const std::string meta = db_path_ + "/" + kMetadataFileName;
    const std::string tmp = meta + ".tmp";
    const std::string prev = meta + ".prev";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            return Status::IOError("failed to open " + tmp);
        }
        Status s = index_->SerializeMetadata(out);
        if (!s.ok()) return s;
        out.flush();
        if (!out) return Status::IOError("failed to write " + tmp);
    }
    if (!durable::fsync_file(tmp)) {
        return Status::IOError("failed to fsync " + tmp);
    }
    // Keep the current checkpoint as .prev via a hard link, then replace the
    // main file with one atomic rename: lsm_vec_db.meta always names a
    // complete checkpoint, old or new.
    if (durable::file_exists(meta)) {
        std::remove(prev.c_str());
        if (::link(meta.c_str(), prev.c_str()) != 0) {
            return Status::IOError("failed to keep previous checkpoint " + prev);
        }
    }
    if (std::rename(tmp.c_str(), meta.c_str()) != 0) {
        return Status::IOError("failed to install " + meta);
    }
    if (!durable::fsync_dir(db_path_)) {
        return Status::IOError("failed to fsync " + db_path_);
    }
    return Status::OK();
}

Status AsterVecDB::Checkpoint()
{
    if (!index_ || closed_) return Status::InvalidArgument("db not opened");
    std::unique_lock<std::shared_mutex> gate(write_gate_);
    return checkpointLocked();
}

Status AsterVecDB::checkpointLocked()
{
    if (writes_since_checkpoint_.load(std::memory_order_relaxed) == 0 &&
        durable::file_exists(db_path_ + "/" + kMetadataFileName)) {
        return Status::OK();
    }
    // The checkpoint references vector slots and graph edges, so both must
    // be on disk before it is installed.
    if (!index_->vector_storage_->syncData()) {
        return Status::IOError("failed to sync vector data");
    }
    if (Status s = index_->syncGraph(); !s.ok()) return s;
    if (metadata_store_) {
        if (Status s = metadata_store_->SyncWAL(); !s.ok()) return s;
    }
    // Writes are excluded, so every logged operation up to here is applied.
    if (oplog_) index_->setCheckpointLsn(oplog_->LastLsn());
    index_->setCheckpointLiveCount(live_count_.load(std::memory_order_relaxed));
    if (Status s = writeMetadataFile(); !s.ok()) return s;
    persistCount();
    writes_since_checkpoint_.store(0, std::memory_order_relaxed);
    ops_since_checkpoint_.store(0, std::memory_order_relaxed);
    // The log must still cover the previous checkpoint (Open falls back to
    // it if the new one is damaged).
    if (oplog_) {
        Status ts = oplog_->TruncateBelow(main_checkpoint_lsn_ + 1);
        if (!ts.ok()) {
            LOG(WARN) << "operation log truncation: " << ts.ToString();
        }
    }
    main_checkpoint_lsn_ = index_->checkpointLsn();
    return Status::OK();
}

void AsterVecDB::flushVectorWrites()
{
    if (index_ && !closed_) {
        Status s = Checkpoint();
        if (!s.ok()) {
            LOG(ERR) << "checkpoint failed: " << s.ToString();
        }
    }
    // 20260516_malloc_trim: a bulk insert leaves the glibc heap fragmented
    // (RocksDB / Aster churn temporary buffers; our edge + page caches grow
    // and shrink). Trim now while we know we're at a write-batch boundary.
    trim_allocator_now();
}

void AsterVecDB::trimMemory()
{
    trim_allocator_now();
}

AsterVecDB::DeleteStats AsterVecDB::GetDeleteStats() const
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

// ---------------------------------------------------------------------------
// Point ids: public calls resolve the caller's id, then run on dense ids.
// ---------------------------------------------------------------------------

Status AsterVecDB::resolveId(const PointId& id, bool create, node_id_t* dense,
                             std::string* key)
{
    key->clear();
    if (!keyed_) {
        if (!id.is_int()) {
            return Status::InvalidArgument(
                "string ids need a store created with id_mode keyed");
        }
        if (id.as_int() > kMaxRealId) {
            return Status::InvalidArgument("real_id must fit in 63 bits");
        }
        *dense = static_cast<node_id_t>(id.as_int());
        return Status::OK();
    }
    if (!id.is_int() && id.as_str().size() > PointId::kMaxStringBytes) {
        return Status::InvalidArgument("string id longer than 1024 bytes");
    }
    if (!metadata_store_) return Status::IOError("id map unavailable");
    *key = id.Encode();
    // One lookup-or-allocate per key at a time, so a new key gets exactly one
    // dense id even under concurrent writers.
    std::lock_guard<std::mutex> g(key_lock(*key));
    Status s = metadata_store_->LookupKey(*key, dense);
    if (s.ok() || !s.IsNotFound() || !create) return s;
    const node_id_t next = next_dense_id_.fetch_add(1, std::memory_order_relaxed);
    if (next > kMaxRealId) return Status::InvalidArgument("id space exhausted");
    if (Status ps = metadata_store_->PutKey(*key, next); !ps.ok()) return ps;
    *dense = next;
    return Status::OK();
}

void AsterVecDB::observeDenseId(node_id_t dense)
{
    node_id_t cur = next_dense_id_.load(std::memory_order_relaxed);
    while (cur <= dense &&
           !next_dense_id_.compare_exchange_weak(cur, dense + 1,
                                                 std::memory_order_relaxed)) {
    }
}

Status AsterVecDB::toIntIds(std::vector<SearchResult>* out) const
{
    if (!keyed_) return Status::OK();
    size_t kept = 0;
    for (size_t i = 0; i < out->size(); ++i) {
        std::string key;
        Status s = metadata_store_->ReverseLookup((*out)[i].id, &key);
        if (s.IsNotFound()) continue;
        if (!s.ok()) return s;
        PointId pid;
        if (!PointId::Decode(key, &pid)) return Status::Corruption("bad id map entry");
        if (!pid.is_int()) {
            return Status::InvalidArgument(
                "store has string ids; use SearchPoints to read them");
        }
        (*out)[kept] = (*out)[i];
        (*out)[kept].id = static_cast<node_id_t>(pid.as_int());
        ++kept;
    }
    out->resize(kept);
    return Status::OK();
}

Status AsterVecDB::toPointHits(const std::vector<SearchResult>& in,
                               std::vector<PointHit>* out) const
{
    out->clear();
    out->reserve(in.size());
    for (const auto& r : in) {
        if (!keyed_) {
            out->push_back({PointId::Int(r.id), r.distance});
            continue;
        }
        std::string key;
        Status s = metadata_store_->ReverseLookup(r.id, &key);
        if (s.IsNotFound()) continue;
        if (!s.ok()) return s;
        PointId pid;
        if (!PointId::Decode(key, &pid)) return Status::Corruption("bad id map entry");
        out->push_back({std::move(pid), r.distance});
    }
    return Status::OK();
}

Status AsterVecDB::ValidateInsert(const PointId& id, Span<float> vec,
                                  std::string_view metadata_json) const
{
    if (keyed_) {
        if (!id.is_int() && id.as_str().size() > PointId::kMaxStringBytes) {
            return Status::InvalidArgument("string id longer than 1024 bytes");
        }
    } else {
        if (!id.is_int()) {
            return Status::InvalidArgument(
                "string ids need a store created with id_mode keyed");
        }
        if (id.as_int() > kMaxRealId) {
            return Status::InvalidArgument("real_id must fit in 63 bits");
        }
    }
    return validateInsertArgs(vec, metadata_json);
}

Status AsterVecDB::ValidateInsert(node_id_t id, Span<float> vec,
                                  std::string_view metadata_json) const
{
    return ValidateInsert(PointId::Int(id), vec, metadata_json);
}

Status AsterVecDB::Insert(const PointId& id, Span<float> vec,
                          std::string_view metadata_json)
{
    WriteScope ws(*this);
    // Validate before resolving, so a bad request never allocates an id.
    if (auto s = ValidateInsert(id, vec, metadata_json); !s.ok()) return s;
    node_id_t dense = 0;
    std::string key;
    if (auto s = resolveId(id, /*create=*/true, &dense, &key); !s.ok()) return s;
    return insertDense(dense, vec, metadata_json, key);
}

Status AsterVecDB::Insert(node_id_t id, Span<float> vec)
{
    return Insert(PointId::Int(id), vec, std::string_view{});
}

Status AsterVecDB::Insert(node_id_t id, Span<float> vec,
                          std::string_view metadata_json)
{
    return Insert(PointId::Int(id), vec, metadata_json);
}

Status AsterVecDB::Update(const PointId& id, Span<float> vec)
{
    WriteScope ws(*this);
    if (auto s = ValidateInsert(id, vec, std::string_view{}); !s.ok()) return s;
    node_id_t dense = 0;
    std::string key;
    if (auto s = resolveId(id, /*create=*/true, &dense, &key); !s.ok()) return s;
    return updateDense(dense, vec, std::string_view{}, key);
}

Status AsterVecDB::Update(node_id_t id, Span<float> vec)
{
    return Update(PointId::Int(id), vec);
}

Status AsterVecDB::Delete(const PointId& id)
{
    WriteScope ws(*this);
    node_id_t dense = 0;
    std::string key;
    Status s = resolveId(id, /*create=*/false, &dense, &key);
    if (s.IsNotFound()) return Status::OK();  // never written: nothing to delete
    if (!s.ok()) return s;
    return deleteDense(dense);
}

Status AsterVecDB::Delete(node_id_t id)
{
    return Delete(PointId::Int(id));
}

Status AsterVecDB::Get(const PointId& id, std::vector<float>* vec)
{
    node_id_t dense = 0;
    std::string key;
    Status s = resolveId(id, /*create=*/false, &dense, &key);
    if (s.IsNotFound()) return Status::NotFound("vector deleted or missing");
    if (!s.ok()) return s;
    return getDense(dense, vec);
}

Status AsterVecDB::Get(node_id_t id, std::vector<float>* vec)
{
    return Get(PointId::Int(id), vec);
}

Status AsterVecDB::GetPayload(const PointId& id, std::string* out_json)
{
    node_id_t dense = 0;
    std::string key;
    Status s = resolveId(id, /*create=*/false, &dense, &key);
    if (s.IsNotFound()) {
        *out_json = "{}";
        return Status::OK();
    }
    if (!s.ok()) return s;
    return getPayloadDense(dense, out_json);
}

Status AsterVecDB::GetPayload(node_id_t id, std::string* out_json)
{
    return GetPayload(PointId::Int(id), out_json);
}

Status AsterVecDB::SetPayload(const PointId& id, std::string_view metadata_json)
{
    WriteScope ws(*this);
    node_id_t dense = 0;
    std::string key;
    Status s = resolveId(id, /*create=*/false, &dense, &key);
    if (s.IsNotFound()) return Status::NotFound("vector not found for id");
    if (!s.ok()) return s;
    return setPayloadDense(dense, metadata_json);
}

Status AsterVecDB::SetPayload(node_id_t id, std::string_view metadata_json)
{
    return SetPayload(PointId::Int(id), metadata_json);
}

Status AsterVecDB::UpdatePayload(const PointId& id, std::string_view partial_json)
{
    WriteScope ws(*this);
    node_id_t dense = 0;
    std::string key;
    Status s = resolveId(id, /*create=*/false, &dense, &key);
    if (s.IsNotFound()) return Status::NotFound("vector not found for id");
    if (!s.ok()) return s;
    return updatePayloadDense(dense, partial_json);
}

Status AsterVecDB::UpdatePayload(node_id_t id, std::string_view partial_json)
{
    return UpdatePayload(PointId::Int(id), partial_json);
}

Status AsterVecDB::DeletePayloadKeys(const PointId& id, Span<const std::string> keys)
{
    WriteScope ws(*this);
    node_id_t dense = 0;
    std::string key;
    Status s = resolveId(id, /*create=*/false, &dense, &key);
    if (s.IsNotFound()) return Status::NotFound("vector not found for id");
    if (!s.ok()) return s;
    return deletePayloadKeysDense(dense, keys);
}

Status AsterVecDB::DeletePayloadKeys(node_id_t id, Span<const std::string> keys)
{
    return DeletePayloadKeys(PointId::Int(id), keys);
}

Status AsterVecDB::SearchKnn(Span<float> query, const SearchOptions& options,
                             std::vector<SearchResult>* out)
{
    Status s = searchUnfilteredDense(query, options, out);
    if (!s.ok()) return s;
    return toIntIds(out);
}

Status AsterVecDB::SearchKnn(Span<float> query, const SearchOptions& options,
                             std::string_view filter_json,
                             std::vector<SearchResult>* out)
{
    Status s = searchDense(query, options, filter_json, out);
    if (!s.ok()) return s;
    return toIntIds(out);
}

Status AsterVecDB::SearchPoints(Span<float> query, const SearchOptions& options,
                                std::string_view filter_json,
                                std::vector<PointHit>* out)
{
    if (!out) return Status::InvalidArgument("output results must not be null");
    std::vector<SearchResult> dense;
    Status s = searchDense(query, options, filter_json, &dense);
    if (!s.ok()) return s;
    return toPointHits(dense, out);
}

Status AsterVecDB::Close()
{
    if (!index_) {
        return Status::InvalidArgument("database not initialized");
    }
    if (closed_) {
        return Status::OK();
    }

    // Vectors, graph WAL, payload WAL, then the metadata file (Checkpoint
    // orders these). Tear down regardless, so file locks are released even
    // if the checkpoint fails (I3); report the checkpoint error afterwards.
    Status cs = Checkpoint();
    oplog_.reset();
    metadata_store_.reset();
    if (log_stream_) {
        log_stream_->flush();
    }
    index_->close();
    closed_ = true;
    return cs;
}
} // namespace astervec
