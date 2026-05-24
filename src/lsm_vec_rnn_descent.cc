#include "lsm_vec_rnn_descent.h"

#include <algorithm>
#include <atomic>
#include <random>
#include <thread>

#include "distance.h"

namespace lsm_vec::rnn_descent {

namespace {

// parallel_for: split [begin, end) into chunks of `chunk_size` and
// dispatch across `num_threads` std::threads using an atomic cursor.
// Joins all threads before returning, so reference-captures in `body`
// stay valid. Falls back to a serial loop for tiny ranges or
// single-thread mode.
template <typename Body>
void parallel_for(int begin, int end, int chunk_size, int num_threads,
                  Body body) {
    if (num_threads <= 1 || end - begin <= chunk_size) {
        for (int i = begin; i < end; ++i) body(i);
        return;
    }
    std::atomic<int> cursor{begin};
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            while (true) {
                int start = cursor.fetch_add(chunk_size,
                                             std::memory_order_relaxed);
                if (start >= end) break;
                int stop = std::min(start + chunk_size, end);
                for (int i = start; i < stop; ++i) body(i);
            }
        });
    }
    for (auto& th : threads) th.join();
}

// SplitMix64-style hash — produces well-distributed seeds from a
// monotonic counter. mt19937 seeded with consecutive ints has highly
// correlated initial state, which biases per-node random samples;
// pre-hashing decorrelates them.
uint32_t mix_seed(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return static_cast<uint32_t>(x);
}

// Pick `size` distinct ids in [0, N), placed at addr[0..size).
// Port of faiss::nndescent::gen_random — sort + de-duplicate + offset.
void gen_random(std::mt19937& rng, int* addr, int size, int N) {
    for (int i = 0; i < size; ++i) {
        addr[i] = static_cast<int>(rng() % (N - size));
    }
    std::sort(addr, addr + size);
    for (int i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    int off = static_cast<int>(rng() % N);
    for (int i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

}  // namespace

// ---------------------------------------------------------------------
// FlatStorageDistance
// ---------------------------------------------------------------------

FlatStorageDistance::FlatStorageDistance(const float* data, int n, int dim,
                                         DistanceMetric metric)
    : data_(data), n_(n), dim_(dim), metric_(metric) {
    if (metric_ == DistanceMetric::kCosine) {
        norms_.resize(static_cast<size_t>(n_));
        for (int i = 0; i < n_; ++i) {
            norms_[i] = lsm_vec::distance::L2Norm(
                data_ + static_cast<size_t>(i) * dim_, dim_);
        }
    }
}

float FlatStorageDistance::operator()(int i, int j) const {
    const float* a = data_ + static_cast<size_t>(i) * dim_;
    const float* b = data_ + static_cast<size_t>(j) * dim_;
    if (metric_ == DistanceMetric::kCosine) {
        return lsm_vec::distance::CosineDistance(a, b, dim_, norms_[i],
                                                 norms_[j]);
    }
    // RNND only uses these distances for ordering (sort, top-k,
    // RNG-prune compare). Squared L2 preserves the ordering and skips
    // the per-call sqrt — ~30-40% speed-up of the build's hottest loop.
    return lsm_vec::distance::L2SquaredDistance(a, b, dim_);
}

// ---------------------------------------------------------------------
// RNNDescentBuilder
// ---------------------------------------------------------------------

void RNNDescentBuilder::insert_nn(int id, int nn_id, float distance,
                                  bool flag) {
    auto& nhood = graph_[id];
    std::lock_guard<std::mutex> guard(nhood.lock);
    nhood.pool.emplace_back(nn_id, distance, flag);
}

void RNNDescentBuilder::init_graph(FlatStorageDistance& dist) {
    graph_.clear();
    // resize() default-constructs in place; Nhood is move-only but no
    // moves happen here because the storage is fresh.
    graph_ = std::vector<Nhood>{};
    graph_.reserve(static_cast<size_t>(ntotal));
    for (int i = 0; i < ntotal; ++i) {
        graph_.emplace_back();
    }

    const int chunk = 256;
    const int nthreads = num_threads > 0 ? num_threads : 1;
    const uint32_t seed_base = random_seed * 7741u;

    parallel_for(0, ntotal, chunk, nthreads, [&](int i) {
        // Deterministic per-node seed (hashed) — uncorrelated initial
        // state across nodes despite consecutive `i`s.
        std::mt19937 rng(mix_seed((static_cast<uint64_t>(seed_base) << 32) |
                                  static_cast<uint32_t>(i)));
        std::vector<int> tmp(S);
        gen_random(rng, tmp.data(), S, ntotal);

        auto& pool = graph_[i].pool;
        pool.reserve(static_cast<size_t>(2 * R));
        for (int j = 0; j < S; ++j) {
            int id = tmp[j];
            if (id == i) continue;
            float d = dist(i, id);
            pool.emplace_back(id, d, true);
        }
    });
}

void RNNDescentBuilder::update_neighbors(FlatStorageDistance& dist) {
    const int chunk = 256;
    const int nthreads = num_threads > 0 ? num_threads : 1;

    parallel_for(0, ntotal, chunk, nthreads, [&](int u) {
        // Reuse the per-thread scratch buffers across iterations.
        // Profiling on SIFT-100K c=1 attributed ~25% of update_neighbors
        // time to libsystem_malloc operator-new/free pairs because
        // these vectors were allocated and destroyed per call (6M
        // pairs total for SIFT-100K @ T1*T2=60 iters). thread_local
        // retains the capacity across calls.
        thread_local std::vector<Neighbor> old_pool;
        thread_local std::vector<Neighbor> new_pool;
        old_pool.clear();
        new_pool.clear();
        auto& nhood = graph_[u];
        {
            std::lock_guard<std::mutex> guard(nhood.lock);
            old_pool.swap(nhood.pool);
        }
        std::sort(old_pool.begin(), old_pool.end());
        old_pool.erase(std::unique(old_pool.begin(), old_pool.end(),
                                   [](const Neighbor& a, const Neighbor& b) {
                                       return a.id == b.id;
                                   }),
                       old_pool.end());

        // The RNN-Descent RNG-pruning core: for each candidate `nn`,
        // accept it only if no already-accepted neighbor `other` is
        // strictly closer to `nn` than `nn` is to u. If some `other`
        // beats it, redirect `nn` to that `other` (cross-pollination).
        for (auto& nn : old_pool) {
            bool ok = true;
            for (auto& other : new_pool) {
                if (!nn.flag && !other.flag) continue;
                if (nn.id == other.id) {
                    ok = false;
                    break;
                }
                float d = dist(nn.id, other.id);
                if (d < nn.distance) {
                    ok = false;
                    insert_nn(other.id, nn.id, d, true);
                    break;
                }
            }
            if (ok) new_pool.push_back(nn);
        }
        for (auto& nn : new_pool) nn.flag = false;

        {
            std::lock_guard<std::mutex> guard(nhood.lock);
            nhood.pool.insert(nhood.pool.end(), new_pool.begin(),
                              new_pool.end());
        }
    });
}

void RNNDescentBuilder::add_reverse_edges() {
    std::vector<std::vector<Neighbor>> reverse_pools(
        static_cast<size_t>(ntotal));

    const int chunk = 256;
    const int nthreads = num_threads > 0 ? num_threads : 1;

    // Pass 1: scatter reverse edges into per-target buckets. Reuse
    // graph_[nn.id].lock as the bucket lock — fewer mutexes, same
    // correctness.
    parallel_for(0, ntotal, chunk, nthreads, [&](int u) {
        for (auto& nn : graph_[u].pool) {
            std::lock_guard<std::mutex> guard(graph_[nn.id].lock);
            reverse_pools[nn.id].emplace_back(u, nn.distance, nn.flag);
        }
    });

    // Pass 2: per-node merge of forward + reverse, dedup, truncate to R.
    parallel_for(0, ntotal, chunk, nthreads, [&](int u) {
        auto& pool = graph_[u].pool;
        for (auto& nn : pool) nn.flag = true;
        auto& rpool = reverse_pools[u];
        rpool.insert(rpool.end(), pool.begin(), pool.end());
        pool.clear();
        std::sort(rpool.begin(), rpool.end());
        rpool.erase(std::unique(rpool.begin(), rpool.end(),
                                [](const Neighbor& a, const Neighbor& b) {
                                    return a.id == b.id;
                                }),
                    rpool.end());
        if (static_cast<int>(rpool.size()) > R) {
            rpool.resize(static_cast<size_t>(R));
        }
    });

    // Pass 3: push reverse_pools[u] entries back onto the forward
    // pools of their *original* sources. This is the cross-pollination
    // step that lets future update_neighbors iterations see new
    // candidates beyond the initial random S.
    parallel_for(0, ntotal, chunk, nthreads, [&](int u) {
        for (auto& nn : reverse_pools[u]) {
            std::lock_guard<std::mutex> guard(graph_[nn.id].lock);
            graph_[nn.id].pool.emplace_back(u, nn.distance, nn.flag);
        }
    });

    // Pass 4: final sort + truncate per node.
    parallel_for(0, ntotal, chunk, nthreads, [&](int u) {
        auto& pool = graph_[u].pool;
        std::sort(pool.begin(), pool.end());
        if (static_cast<int>(pool.size()) > R) {
            pool.resize(static_cast<size_t>(R));
        }
    });
}

void RNNDescentBuilder::build(FlatStorageDistance& dist, int n) {
    ntotal = n;
    if (ntotal <= 1) {
        // Degenerate cases: 0 or 1 vector → no edges possible.
        final_graph.clear();
        offsets.assign(static_cast<size_t>(std::max(0, ntotal)) + 1, 0);
        return;
    }
    if (S >= ntotal) S = ntotal - 1;

    init_graph(dist);

    for (int t1 = 0; t1 < T1; ++t1) {
        for (int t2 = 0; t2 < T2; ++t2) {
            update_neighbors(dist);
        }
        if (t1 != T1 - 1) {
            add_reverse_edges();
        }
    }

    const int chunk = 256;
    const int nthreads = num_threads > 0 ? num_threads : 1;

    // Final dedup + sort + truncate.
    parallel_for(0, ntotal, chunk, nthreads, [&](int u) {
        auto& pool = graph_[u].pool;
        std::sort(pool.begin(), pool.end());
        pool.erase(std::unique(pool.begin(), pool.end(),
                               [](const Neighbor& a, const Neighbor& b) {
                                   return a.id == b.id;
                               }),
                   pool.end());
        if (static_cast<int>(pool.size()) > R) {
            pool.resize(static_cast<size_t>(R));
        }
    });

    // Build CSR. Offsets is computed sequentially (cheap, O(n)).
    offsets.assign(static_cast<size_t>(ntotal) + 1, 0);
    for (int u = 0; u < ntotal; ++u) {
        offsets[u + 1] = offsets[u] + static_cast<int>(graph_[u].pool.size());
    }
    final_graph.assign(static_cast<size_t>(offsets.back()), -1);
    parallel_for(0, ntotal, chunk, nthreads, [&](int u) {
        const auto& pool = graph_[u].pool;
        int off = offsets[u];
        for (size_t i = 0; i < pool.size(); ++i) {
            final_graph[off + static_cast<int>(i)] = pool[i].id;
        }
    });

    // Reclaim the per-Nhood buffers — the caller only needs the CSR.
    std::vector<Nhood>().swap(graph_);
}

}  // namespace lsm_vec::rnn_descent
