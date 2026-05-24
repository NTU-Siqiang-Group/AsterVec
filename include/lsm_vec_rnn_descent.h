#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "lsm_vec_db.h"  // DistanceMetric

namespace lsm_vec::rnn_descent {

struct Neighbor {
    int id;
    float distance;
    bool flag;  // true = "fresh"; subject to RNG-pruning comparison

    Neighbor() = default;
    Neighbor(int i, float d, bool f) : id(i), distance(d), flag(f) {}

    bool operator<(const Neighbor& o) const { return distance < o.distance; }
};

// Per-node bucket: guarded pool of candidate neighbors. mutex is
// non-movable, so explicit move ops are provided (move-only type).
struct Nhood {
    std::mutex lock;
    std::vector<Neighbor> pool;

    Nhood() = default;
    Nhood(const Nhood&) = delete;
    Nhood& operator=(const Nhood&) = delete;
    Nhood(Nhood&& o) noexcept : pool(std::move(o.pool)) {}
    Nhood& operator=(Nhood&& o) noexcept {
        pool = std::move(o.pool);
        return *this;
    }
};

// Distance computer over a flat contiguous float* blob of n * dim
// values, indexed by int in [0, n). Wraps lsm_vec::distance kernels
// (SIMD-accelerated). For cosine, norms are precomputed on construction.
class FlatStorageDistance {
public:
    FlatStorageDistance(const float* data, int n, int dim, DistanceMetric metric);

    // Squared L2 (or cosine distance, 1 - cos_sim) between two stored
    // vectors. Smaller is closer — caller must wrap inner-product
    // metrics with negation if needed.
    float operator()(int i, int j) const;

    int dim() const { return dim_; }
    int n() const { return n_; }
    const float* vector(int i) const { return data_ + static_cast<size_t>(i) * dim_; }

private:
    const float* data_;
    int n_;
    int dim_;
    DistanceMetric metric_;
    std::vector<float> norms_;  // populated only for kCosine
};

// Build a k-NN-like graph in memory using RNN-Descent (Ono & Matsui,
// CVPR 2024). Output is a CSR adjacency: `final_graph[offsets[u]..
// offsets[u+1])` are u's neighbor ids, sorted by ascending distance.
//
// Parallelism: build() spawns `num_threads` std::threads internally;
// each iteration of update_neighbors / add_reverse_edges is
// embarrassingly parallel over nodes (per-Nhood mutex).
class RNNDescentBuilder {
public:
    explicit RNNDescentBuilder(int d) : d(d) {}

    // Build the graph. After return, final_graph and offsets are
    // populated; the internal Nhood buffers are released.
    void build(FlatStorageDistance& dist, int n);

    // Inputs.
    int d;                       // vector dimension
    int S = 16;                  // initial random neighbor count
    int T1 = 4;                  // outer iterations
    int T2 = 15;                 // inner iterations per outer
    int R = 64;                  // pool cap = max output degree
    int num_threads = 0;         // 0 = single-threaded
    uint32_t random_seed = 2021;

    // Outputs (filled by build()).
    int ntotal = 0;
    std::vector<int> final_graph;
    std::vector<int> offsets;

private:
    std::vector<Nhood> graph_;

    void init_graph(FlatStorageDistance& dist);
    void update_neighbors(FlatStorageDistance& dist);
    void add_reverse_edges();
    void insert_nn(int id, int nn_id, float distance, bool flag);
};

}  // namespace lsm_vec::rnn_descent
