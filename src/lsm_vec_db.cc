#include "lsm_vec_db.h"

#include <cmath>
#include <exception>
#include <fstream>
#include <limits>

#include "distance.h"
#include "lsm_vec_index.h"
#include "logger.h"

namespace lsm_vec
{
namespace {
constexpr char kDefaultVectorFileName[] = "vector.log";
constexpr char kDefaultLogFileName[] = "lsm_vec_db.log";
constexpr char kMetadataFileName[] = "lsm_vec_db.meta";
} // namespace

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

void LSMVecDB::printStatistics() const
{
    index_->printStatistics();
}

Status LSMVecDB::Close()
{
    if (!index_) {
        return Status::InvalidArgument("database not initialized");
    }

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

    return Status::OK();
}
} // namespace lsm_vec
