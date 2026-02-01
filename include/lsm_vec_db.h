#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "disk_vector.h"
#include "rocksdb/status.h"

namespace lsm_vec
{
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
    float ef_construction = 64.0f;
    size_t vec_file_capacity = 100000;
    size_t paged_max_cached_pages = 256;
    int vector_storage_type = 0;
    uint64_t db_target_size = 107374182400ULL;
    int random_seed = 12345;
    bool enable_stats = false;
    bool reinit = false;
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
    static Status Open(const std::string& path,
                       const LSMVecDBOptions& opts,
                       std::unique_ptr<LSMVecDB>* db);
    Status Close();

    Status Insert(node_id_t id, Span<float> vec);
    Status Update(node_id_t id, Span<float> vec);
    Status Delete(node_id_t id);
    Status Get(node_id_t id, std::vector<float>* vec);
    Status SearchKnn(Span<float> query,
                     const SearchOptions& options,
                     std::vector<SearchResult>* out);

    void printStatistics() const;

private:
    LSMVecDB(const std::string& db_path,
             const LSMVecDBOptions& options,
             std::unique_ptr<class LSMVec> index,
             std::unique_ptr<std::ostream> log_stream);

    Status ValidateVector(Span<float> vec) const;
    Status EnsureMetricSupported() const;
    float ComputeDistance(Span<float> a, Span<float> b) const;

    std::string db_path_;
    LSMVecDBOptions options_;
    std::unique_ptr<std::ostream> log_stream_;
    std::unique_ptr<class LSMVec> index_;
    std::unordered_set<node_id_t> deleted_ids_;
};
} // namespace lsm_vec
