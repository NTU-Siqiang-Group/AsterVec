// One-shot bulk build for AsterVec, used by pgastervec's in-memory
// CREATE INDEX path. Pipeline (see pgastervec/doc/in-memory-build-plan.md):
//
//   1. RNN-Descent (astervec_rnn_descent) builds a layer-0 KNN graph
//      from the in-memory float* blob, fully multi-threaded.
//   2. Per-node HNSW levels are drawn via randomLevel().
//   3. Vectors + layer-0 edges are bulk-written to AsterDB. No per-row
//      searchLayer / linkNeighbors machinery — RNND already produced
//      the adjacency.
//   4. Nodes with level >= 1 go through addPointUpperLayersOnly, which
//      is the existing insertNode upper-layer logic with the layer-0
//      branch removed.
//
// Phases B/D/E run in parallel; A/C/F are short enough to stay serial.

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "distance.h"
#include "astervec_index.h"
#include "astervec_rnn_descent.h"
#include "logger.h"
#include "rocksdb/graph.h"

namespace astervec {

namespace {

// Local copy of the RNND port's parallel_for. Inlined so we don't
// expose threading internals through the header.
template <typename Body>
void parallel_for(int begin, int end, int chunk_size, int num_threads,
                  Body body) {
    if (num_threads <= 1 || end - begin <= chunk_size) {
        for (int i = begin; i < end; ++i) body(i);
        return;
    }
    std::atomic<int> cursor{begin};
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(num_threads));
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

}  // namespace

// ---------------------------------------------------------------------
// Phase D — vectors + layer-0 edges to disk
// ---------------------------------------------------------------------
void AsterVec::bulkLoadLayer0(const float* vectors, int n,
                            const int* csr_offsets, const int* csr_neighbors,
                            const int* levels,
                            int num_threads) {
    const int dim = vector_dim_;
    const int nthreads = num_threads > 0 ? num_threads : 1;

    // Step 1: register vertices in AsterDB. Sequential — rocksdb's
    // WriteThread serialises this anyway, and the calls are cheap.
    for (int u = 0; u < n; ++u) {
        auto st = db_->AddVertex(static_cast<rocksdb::node_id_t>(u));
        if (!st.ok()) {
            throw std::runtime_error(
                "bulkLoadLayer0: AddVertex failed: " + st.ToString());
        }
    }

    // Step 2: pick a sectionKey per vertex. Guaranteed to be a node
    // at level >= section_layer_, mirroring streaming insertNode's
    // contract (sectionKey = neighbours[0].id at section_layer_).
    //
    // Two-stage strategy chosen for total cost:
    //
    //   Fast path: u's first RNND neighbour at level >= section_layer_.
    //              O(R) per node, hits most of the time. For M=8,
    //              section_layer_=2 → ~64% of nodes find a qualifying
    //              neighbour within R=64.
    //   Fallback:  HNSW greedy descent through the upper-layer graph
    //              (built in phase E above) down to section_layer_.
    //              O(log n × ef) per node, only ~36% of nodes hit
    //              this. Total fallback cost ~ 36K × ~50 ops ≈ 2M
    //              distance ops, negligible vs RNND.
    //
    // The greedy-descent fallback guarantees the returned id is a real
    // upper-layer node, so the section is a proper HNSW "centroid" —
    // page utilisation matches streaming.
    std::vector<node_id_t> section_keys(static_cast<size_t>(n));
    const int sl = section_layer_;
    parallel_for(0, n, 1024, nthreads, [&](int u) {
        if (levels[u] >= sl) {
            section_keys[u] = static_cast<node_id_t>(u);
            return;
        }
        int begin = csr_offsets[u];
        int end   = csr_offsets[u + 1];
        // Fast path: first RNND neighbour at level >= sl
        for (int i = begin; i < end; ++i) {
            int v = csr_neighbors[i];
            if (v >= 0 && levels[v] >= sl) {
                section_keys[u] = static_cast<node_id_t>(v);
                return;
            }
        }
        // Fallback: HNSW greedy descent. Phase E built the upper graph
        // above, so entry_point_ is valid.
        node_id_t ep = entry_point_.load(std::memory_order_acquire);
        int observed_max = max_layer_.load(std::memory_order_acquire);
        if (ep == k_invalid_node_id || observed_max < sl) {
            section_keys[u] = static_cast<node_id_t>(u);
            return;
        }
        std::vector<float> qv(vectors + static_cast<size_t>(u) * dim,
                              vectors + static_cast<size_t>(u + 1) * dim);
        for (int l = observed_max; l > sl; --l) {
            ep = greedySearchUpperLayer(qv, ep, l);
        }
        ep = greedySearchUpperLayer(qv, ep, sl);
        section_keys[u] = ep;
    });

    // Step 2b: per-section grouping. Different sections in parallel,
    // same section serial within a thread.
    {
        std::unordered_map<node_id_t, std::vector<int>> by_section;
        by_section.reserve(static_cast<size_t>(n));
        for (int u = 0; u < n; ++u) {
            by_section[section_keys[u]].push_back(u);
        }
        std::vector<node_id_t> sec_list;
        sec_list.reserve(by_section.size());
        for (const auto& kv : by_section) sec_list.push_back(kv.first);
        LOG(INFO) << "[bulkBuild] step 2b per-section "
                  << "sections=" << sec_list.size()
                  << " threads=" << nthreads;
        parallel_for(0, static_cast<int>(sec_list.size()), 1, nthreads,
                     [&](int si) {
            node_id_t sk = sec_list[si];
            const auto& us = by_section[sk];
            for (int u : us) {
                std::vector<float> vec(
                    vectors + static_cast<size_t>(u) * dim,
                    vectors + static_cast<size_t>(u + 1) * dim);
                storeVectorWithStats(static_cast<node_id_t>(u), vec, sk);
            }
        });
    }

    // Step 3a: build per-vertex outgoing adjacency in memory, starting
    // from RNND's top-m_max for each u.
    std::vector<std::vector<int>> adj(static_cast<size_t>(n));
    for (int u = 0; u < n; ++u) {
        int begin = csr_offsets[u];
        int end   = csr_offsets[u + 1];
        int deg   = end - begin;
        if (deg > m_max_) deg = m_max_;
        adj[u].reserve(static_cast<size_t>(2 * m_max_));
        for (int i = 0; i < deg; ++i) {
            int v = csr_neighbors[begin + i];
            if (v >= 0 && v != u) adj[u].push_back(v);
        }
    }

    // Step 3b: symmetrize. For every forward edge u→v, also add v→u.
    // Without this step the final graph would have asymmetric reachability
    // and search would miss some neighbours. Serial pass — each (u,v)
    // costs a linear find in adj[v] which is bounded by ~2*m_max_, so
    // total work is O(n * m_max_^2) ≈ 1-2 s at SIFT-1M scale.
    for (int u = 0; u < n; ++u) {
        // Snapshot u's neighbour list before appending to adj[u] itself
        // via the reverse direction (which never happens — adj[u]
        // appends only when v iterates and finds u absent). Capture
        // size now.
        const size_t base = adj[u].size();
        for (size_t i = 0; i < base; ++i) {
            int v = adj[u][i];
            auto& av = adj[v];
            bool found = false;
            for (int w : av) { if (w == u) { found = true; break; } }
            if (!found) av.push_back(u);
        }
    }

    // Step 3c: cap per-vertex degree. After symmetrization, "popular"
    // nodes (hubs of the original RNND graph) can have out-degree much
    // larger than m_max_ because many u's pointed at them. For each
    // such v we keep the m_max_ closest neighbours by distance. Without
    // this cap the on-disk layer-0 graph violates the HNSW M_max
    // invariant that subsequent INSERT/SEARCH paths assume, breaks
    // lightweight-resource budget (more edges per visit), and ages
    // poorly because streaming INSERT's per-neighbour shrink only
    // touches neighbours of the new node.
    parallel_for(0, n, 256, nthreads, [&](int v) {
        auto& a = adj[v];
        if (static_cast<int>(a.size()) <= m_max_) return;

        const float* vv = vectors + static_cast<size_t>(v) * dim;
        std::vector<std::pair<float, int>> scored;
        scored.reserve(a.size());
        for (int w : a) {
            const float* wv = vectors + static_cast<size_t>(w) * dim;
            float d = astervec::distance::ComputeDistance(
                db_options_.metric, vv, wv,
                static_cast<size_t>(dim));
            scored.emplace_back(d, w);
        }
        std::partial_sort(scored.begin(), scored.begin() + m_max_,
                          scored.end());
        a.clear();
        a.reserve(static_cast<size_t>(m_max_));
        for (int i = 0; i < m_max_; ++i) a.push_back(scored[i].second);
    });

    // Step 3d: emit edges. Symmetry is now in the adjacency lists, so
    // we only emit forward edges — every (v, u) pair appears as part of
    // v's own iteration when u is in adj[v].
    std::vector<std::pair<rocksdb::node_id_t, rocksdb::node_id_t>> all_edges;
    size_t total_edges = 0;
    for (int u = 0; u < n; ++u) total_edges += adj[u].size();
    all_edges.reserve(total_edges);
    for (int u = 0; u < n; ++u) {
        for (int v : adj[u]) {
            all_edges.emplace_back(static_cast<rocksdb::node_id_t>(u),
                                   static_cast<rocksdb::node_id_t>(v));
        }
    }
    std::vector<std::vector<int>>().swap(adj);  // free working memory

    // Step 4: dispatch in large chunks, single-threaded. Per-chunk cost
    // is dominated by rocksdb WriteThread serialisation; bigger chunks
    // → fewer round-trips. 64 K edges/chunk is comfortable for the WAL.
    //
    // Important: AddEdgeBatch under EDGE_UPDATE_EAGER is a per-vertex
    // read-modify-write (graph.cc::AddEdgeBatch → GetAllEdges + merge +
    // Put). If a single source vertex's outgoing edges span two chunks
    // they will incur an extra GetAllEdges round-trip but are still
    // *correct* serial-side because each chunk's WriteBatch commits
    // atomically. **Do not parallelise this loop without partitioning
    // chunks by source vertex** — two threads touching the same
    // out_by_vertex key would lose one update. Future bulk-load
    // optimisation should use AddVertexForBulkLoad / SST ingest rather
    // than parallel AddEdgeBatch.
    constexpr size_t kEdgeChunk = 64 * 1024;
    for (size_t off = 0; off < all_edges.size(); off += kEdgeChunk) {
        size_t end = std::min(off + kEdgeChunk, all_edges.size());
        std::vector<std::pair<rocksdb::node_id_t, rocksdb::node_id_t>> chunk(
            all_edges.begin() + static_cast<ptrdiff_t>(off),
            all_edges.begin() + static_cast<ptrdiff_t>(end));
        auto st = db_->AddEdgeBatch(chunk);
        if (!st.ok()) {
            throw std::runtime_error(
                "bulkLoadLayer0: AddEdgeBatch failed: " + st.ToString());
        }
    }
}

// ---------------------------------------------------------------------
// Phase E worker — upper-layer-only insertNode
// ---------------------------------------------------------------------
//
// Mirrors the layer >= 1 logic of AsterVec::insertNode (steps 4, 5, 8 in
// the source comments). Differences:
//   - presetLevel is given by the caller (drawn once up front), not via
//     randomLevel().
//   - storeVectorWithStats / linkNeighborsAsterDB / layer-0 shrink /
//     entry-point CAS are all skipped.
//   - Caller guarantees nodes_[nodeId] is pre-published with the
//     vector and empty neighbours map, and that entry_point_ and
//     max_layer_ are already set to the bulk-build's final values.
void AsterVec::addPointUpperLayersOnly(node_id_t nodeId,
                                     const float* vec_ptr,
                                     int presetLevel) {
    if (presetLevel < 1) return;  // nothing to do — caller may filter

    std::vector<float> vector(vec_ptr, vec_ptr + vector_dim_);

    node_id_t currentEntryPoint =
        entry_point_.load(std::memory_order_acquire);
    if (currentEntryPoint == k_invalid_node_id) {
        // First-node should be entry_point set by bulkBuild before
        // this loop. If we hit this it's a caller bug.
        return;
    }

    int observed_max_layer = max_layer_.load(std::memory_order_acquire);
    if (observed_max_layer < 0) observed_max_layer = 0;

    // Greedy descent from max_layer down to presetLevel + 1.
    for (int l = observed_max_layer; l > presetLevel; --l) {
        currentEntryPoint =
            greedySearchUpperLayer(vector, currentEntryPoint, l);
    }

    // searchLayer + selectNeighbors at each upper layer down to 1.
    thread_local SearchScratch upper_scratch;
    const int min_layer = std::min(observed_max_layer, presetLevel);
    std::unordered_map<int, std::vector<node_id_t>> selectedPerLayer;
    for (int l = min_layer; l >= 1; --l) {
        std::vector<SearchResult> candidates = searchLayer(
            vector, currentEntryPoint, ef_construction_, l, upper_scratch);
        std::vector<node_id_t> selected =
            selectNeighbors(vector, candidates, m_, l);
        if (!candidates.empty()) currentEntryPoint = candidates[0].id;
        selectedPerLayer[l] = std::move(selected);
    }

    if (min_layer < 1) return;

    // Forward + backward edges + shrink, all in the nodes_ map.
    std::shared_lock<std::shared_mutex> nodes_g(nodes_mu_);
    auto nodeIt = nodes_.find(nodeId);
    if (nodeIt == nodes_.end()) {
        LOG(ERR) << "bulkBuild: nodes_[" << nodeId
                 << "] missing in addPointUpperLayersOnly";
        return;
    }

    for (int l = min_layer; l >= 1; --l) {
        const std::vector<node_id_t>& selectedNeighbors = selectedPerLayer[l];

        // Forward edges.
        {
            std::lock_guard<std::mutex> g(node_shard(nodeId));
            nodeIt->second.neighbors[l] = selectedNeighbors;
        }

        // Backward edges + per-neighbour shrink (same convention as
        // insertNode step 5: m_max_ shrink threshold at every layer).
        for (node_id_t neighbor : selectedNeighbors) {
            auto neighborIt = nodes_.find(neighbor);
            if (neighborIt == nodes_.end()) {
                LOG(WARN) << "bulkBuild: missing neighbour " << neighbor
                          << " at layer " << l;
                continue;
            }
            std::lock_guard<std::mutex> g(node_shard(neighbor));
            neighborIt->second.neighbors[l].push_back(nodeId);
            auto& eConnRef = neighborIt->second.neighbors[l];
            if (eConnRef.size() > static_cast<size_t>(m_max_)) {
                std::vector<node_id_t> eConn = eConnRef;
                std::vector<node_id_t> eNewConn =
                    selectNeighbors(neighborIt->second.point, eConn, m_max_, l);
                eConnRef = std::move(eNewConn);
            }
        }
    }
}

// ---------------------------------------------------------------------
// Public entry: bulkBuild
// ---------------------------------------------------------------------
void AsterVec::bulkBuild(const float* vectors, int n,
                       const BulkBuildOptions& opts) {
    if (n <= 0) return;
    if (vector_dim_ <= 0) {
        throw std::runtime_error("bulkBuild: vector_dim_ is not set");
    }
    if (entry_point_.load(std::memory_order_acquire) != k_invalid_node_id) {
        throw std::runtime_error(
            "bulkBuild: index must be empty before bulk build");
    }

    const int dim = vector_dim_;
    const int nthreads = opts.num_threads > 0 ? opts.num_threads : 1;

    LOG(INFO) << "[bulkBuild] n=" << n << " dim=" << dim
              << " threads=" << nthreads << " START";

    // ----- Phase B: RNN-Descent layer-0 KNN graph -----
    rnn_descent::FlatStorageDistance fdist(vectors, n, dim, db_options_.metric);
    rnn_descent::RNNDescentBuilder rnnd(dim);
    rnnd.S            = opts.rnnd_S;
    rnnd.T1           = opts.rnnd_T1;
    rnnd.T2           = opts.rnnd_T2;
    rnnd.R            = opts.rnnd_R;
    rnnd.num_threads  = nthreads;
    LOG(INFO) << "[bulkBuild] RNND build start";
    rnnd.build(fdist, n);
    LOG(INFO) << "[bulkBuild] RNND build done";

    // ----- Phase C: per-node HNSW level draw -----
    // randomLevel() uses thread_local mt19937 (Phase 4a), safe to call
    // serially here. n random draws is microseconds even at n=10M.
    std::vector<int> levels(n);
    int max_level = 0;
    node_id_t entry_node = 0;
    for (int u = 0; u < n; ++u) {
        levels[u] = randomLevel();
        if (levels[u] > max_level ||
            (levels[u] == max_level &&
             static_cast<node_id_t>(u) < entry_node)) {
            max_level = levels[u];
            entry_node = static_cast<node_id_t>(u);
        }
    }

    // ----- Pre-publish nodes_ entries for all upper-layer nodes -----
    {
        std::unique_lock<std::shared_mutex> g(nodes_mu_);
        for (int u = 0; u < n; ++u) {
            if (levels[u] >= 1) {
                Node node;
                node.id = static_cast<node_id_t>(u);
                node.point.assign(vectors + static_cast<size_t>(u) * dim,
                                  vectors + static_cast<size_t>(u + 1) * dim);
                nodes_[static_cast<node_id_t>(u)] = std::move(node);
            }
        }
    }
    max_layer_.store(max_level, std::memory_order_release);
    entry_point_.store(entry_node, std::memory_order_release);

    // ----- Phase E (BEFORE phase D) -----
    if (max_level > 0) {
        std::vector<std::vector<node_id_t>> by_level(
            static_cast<size_t>(max_level) + 1);
        for (int u = 0; u < n; ++u) {
            if (levels[u] >= 1) {
                by_level[levels[u]].push_back(static_cast<node_id_t>(u));
            }
        }
        LOG(INFO) << "[bulkBuild] phase E start, max_level=" << max_level;
        for (int level = max_level; level >= 1; --level) {
            const auto& nodes_at_level = by_level[level];
            if (nodes_at_level.empty()) continue;
            LOG(INFO) << "[bulkBuild] phase E level=" << level
                      << " nodes=" << nodes_at_level.size();
            parallel_for(0, static_cast<int>(nodes_at_level.size()),
                         32, nthreads, [&](int idx) {
                node_id_t u = nodes_at_level[idx];
                if (u == entry_node) return;
                const float* vec_ptr =
                    vectors + static_cast<size_t>(u) * dim;
                addPointUpperLayersOnly(u, vec_ptr, levels[u]);
            });
        }
    }

    LOG(INFO) << "[bulkBuild] phase D bulk-load start";
    bulkLoadLayer0(vectors, n,
                   rnnd.offsets.data(), rnnd.final_graph.data(),
                   levels.data(),
                   nthreads);
    LOG(INFO) << "[bulkBuild] phase D bulk-load done";

    std::vector<int>().swap(rnnd.final_graph);
    std::vector<int>().swap(rnnd.offsets);
}

}  // namespace astervec
