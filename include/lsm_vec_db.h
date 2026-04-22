#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "disk_vector.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {
class DB;
class ColumnFamilyHandle;
}  // namespace ROCKSDB_NAMESPACE

namespace lsm_vec
{
class MetadataStore;  // forward declaration
using Status = ROCKSDB_NAMESPACE::Status;

enum class DistanceMetric {
    kL2,
    kCosine,
};

template <typename T>
class Span {
public:
    Span() : data_(nullptr), size_(0) {}
    Span(T* data, size_t size) : data_(data), size_(size) {}
    Span(std::vector<std::remove_const_t<T>>& vec)
        : data_(vec.data()), size_(vec.size()) {}
    Span(const std::vector<std::remove_const_t<T>>& vec)
        : data_(const_cast<T*>(vec.data())), size_(vec.size()) {}

    T* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T& operator[](size_t idx) const { return data_[idx]; }
    T* begin() const { return data_; }
    T* end() const { return data_ + size_; }

private:
    T* data_;
    size_t size_;
};

struct LSMVecDBOptions {
    int dim = 0;
    DistanceMetric metric = DistanceMetric::kL2;
    int m = 8;
    int m_max = 16;
    int m_level = 1;
    float ef_construction = 32.0f;
    size_t vec_file_capacity = 100000;
    size_t paged_max_cached_pages = 8192;
    int vector_storage_type = 1; 
    uint64_t db_target_size = 107374182400ULL;
    int random_seed = 12345;
    bool enable_stats = false;
    bool enable_batch_read = true;
    bool reinit = false;
    int k = 1;
    int ef_search = 64;
    size_t edge_cache_size = 100000;
    std::string vector_file_path;
    std::string log_file_path;
};

struct SearchOptions {
    int k = 1;
    int ef_search = 64;
};

struct SearchResult {
    node_id_t id;
    float distance;
};

class LSMVecDB {
public:
    ~LSMVecDB();

    static Status Open(const std::string& path,
                       const LSMVecDBOptions& opts,
                       std::unique_ptr<LSMVecDB>* db);
    Status Close();

    Status Insert(node_id_t id, Span<float> vec);
    Status Insert(node_id_t id, Span<float> vec, std::string_view metadata_json);
    Status GetPayload(node_id_t id, std::string* out_json);
    Status SetPayload(node_id_t id, std::string_view metadata_json);
    Status UpdatePayload(node_id_t id, std::string_view partial_json);
    Status DeletePayloadKeys(node_id_t id, Span<const std::string> keys);
    Status Update(node_id_t id, Span<float> vec);
    Status Delete(node_id_t id);
    Status Get(node_id_t id, std::vector<float>* vec);
    Status SearchKnn(Span<float> query,
                     const SearchOptions& options,
                     std::vector<SearchResult>* out);

    Status SearchKnn(Span<float> query,
                     std::vector<SearchResult>* out);

    void printStatistics() const;
    void flushVectorWrites();

private:
    LSMVecDB(const std::string& db_path,
             const LSMVecDBOptions& options,
             std::unique_ptr<class LSMVec> index,
             std::unique_ptr<std::ostream> log_stream);

    Status ValidateVector(Span<float> vec) const;
    Status EnsureMetricSupported() const;
    float ComputeDistance(Span<float> a, Span<float> b) const;

    // True iff id has a live vector in the index (not deleted, slot assigned).
    // Centralizes the "id must have a vector" invariant used by payload CRUD.
    bool HasLiveVector(node_id_t id) const;

    std::string db_path_;
    LSMVecDBOptions options_;
    std::unique_ptr<std::ostream> log_stream_;
    std::unique_ptr<class LSMVec> index_;
    std::unordered_set<node_id_t> deleted_ids_;

    std::unique_ptr<ROCKSDB_NAMESPACE::DB>  metadata_db_;
    ROCKSDB_NAMESPACE::ColumnFamilyHandle*  metadata_cf_ = nullptr;
    std::unique_ptr<MetadataStore>          metadata_store_;
};
} // namespace lsm_vec
