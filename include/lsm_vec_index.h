#pragma once
#include <array>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <random>
#include <string>
#include "rocksdb/graph.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/statistics.h"
#include "rocksdb/status.h"
#include <iostream>
#include "bloom_filter.h"
#include "cgroup_monitor.h"
#include "disk_vector.h"
#include "id_types.h"
#include "lsm_vec_db.h"
#include "metadata.h"
#include "statistics.h"
#include "logger.h"

namespace lsm_vec
{
using namespace ROCKSDB_NAMESPACE;

// Thread-safety contract for LSMVec
// =================================
//
// LSMVec assumes the caller serializes read-vs-write coordination
// through an external reader-writer lock (e.g. a per-index
// std::shared_mutex held by the embedding application). State that
// must remain consistent between concurrent SHARED-lock holders
// (page cache, edge cache, stats counters, search scratch) IS
// synchronized internally. State that the external lock already
// protects (graph maps, tombstone set, update maps, entry_point_,
// max_layer_) is NOT internally synchronized.
//
// Lock holding requirements per public method:
//
//   SHARED:
//     - knnSearchK
//     - knnSearchKFiltered
//     - getNodeVector
//     - resolve_internal, resolve_real, is_alive, is_tombstoned
//     - tombstone_count, updated_real_id_count, ...
//     - printStatistics, printState, printIndexStatus
//
//   EXCLUSIVE:
//     - insertNode
//     - deleteNode
//     - tombstone, record_update_mapping, forget_update_mapping
//     - allocate_update_id
//     - setDeletedIds
//     - SerializeMetadata, DeserializeMetadata
//     - close
//
// LSMSearchIterator is special: its constructor (called via
// LSMVecDB::NewSearchIterator) and Next() method each acquire the
// SHARED lock for the duration of one call. The lock is NOT held
// across calls, so iterator-private state (visited_, candidates_,
// yield_queue_) may observe a graph that has changed between calls.
// is_tombstoned() is re-checked at yield time so deleted nodes are
// suppressed; concurrent inserts may or may not be visible depending
// on whether the iterator's frontier reaches them. This is acceptable
// for ANN semantics and avoids long-lived shared locks blocking
// writers when SQL cursors hold an iterator open.

    // Sharded thread-safe LRU cache for L0 edge lists.
    //
    // The previous single-mutex design was the dominant contention
    // point at c >= 4 concurrent searches because every level-0
    // searchLayer expansion calls `getEdgesCached` repeatedly.
    //
    // R-EdgeShard / port of 6e676c9 Phase 2 (split from the bundled
    // packed-atomic chunk array commit, which conflicts with our
    // narrow_idtopage perf commit and is deferred separately):
    //
    //   - kNumShards = 64. shard_id = id % kNumShards; each shard
    //     owns its own mutex, list, map. Different ids on different
    //     shards run lock-free against each other.
    //   - get(): drop-on-hit — does NOT splice. For HNSW search, hot
    //     entry-point neighbors get re-inserted on every miss anyway,
    //     so they stay near the front. Avoids cache-line bouncing on
    //     the list head.
    //   - put(): existing entry → splice-to-front. New entry, full
    //     shard → evict tail. Only put() mutates LRU order.
    //   - Per-shard capacity = total_capacity / 64. Eviction per-shard.
    //   - Counters (hits / misses / evictions / invalidations) are
    //     global atomics for observability parity.
    class EdgeLRUCache {
    public:
        static constexpr size_t kNumShards = 64;

        explicit EdgeLRUCache(size_t capacity)
            : capacity_(capacity),
              per_shard_capacity_(std::max<size_t>(1, capacity / kNumShards)) {}

        // Returns true and writes the cached neighbour list into *out
        // when present. Returns false on miss; *out is unchanged in
        // that case. Copy happens under the shard lock.
        bool get(node_id_t id, std::vector<node_id_t>* out) {
            Shard& s = shard_for(id);
            std::lock_guard<std::mutex> lock(s.mu);
            auto it = s.map.find(id);
            if (it == s.map.end()) {
                misses_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            hits_.fetch_add(1, std::memory_order_relaxed);
            // Drop-on-hit: no splice. Just copy and return.
            *out = it->second->second;
            return true;
        }

        void put(node_id_t id, std::vector<node_id_t> neighbors) {
            Shard& s = shard_for(id);
            std::lock_guard<std::mutex> lock(s.mu);
            auto it = s.map.find(id);
            if (it != s.map.end()) {
                it->second->second = std::move(neighbors);
                // Splice to front so this entry is treated as freshly
                // inserted (only place we mutate LRU order).
                s.list.splice(s.list.begin(), s.list, it->second);
                return;
            }
            if (s.map.size() >= per_shard_capacity_) {
                auto& back = s.list.back();
                s.map.erase(back.first);
                s.list.pop_back();
                evictions_.fetch_add(1, std::memory_order_relaxed);
            }
            s.list.emplace_front(id, std::move(neighbors));
            s.map[id] = s.list.begin();
        }

        void erase(node_id_t id) {
            Shard& s = shard_for(id);
            std::lock_guard<std::mutex> lock(s.mu);
            auto it = s.map.find(id);
            if (it == s.map.end()) return;
            s.list.erase(it->second);
            s.map.erase(it);
            invalidations_.fetch_add(1, std::memory_order_relaxed);
        }

        size_t hits() const { return hits_.load(std::memory_order_relaxed); }
        size_t misses() const { return misses_.load(std::memory_order_relaxed); }
        size_t evictions() const { return evictions_.load(std::memory_order_relaxed); }
        size_t invalidations() const { return invalidations_.load(std::memory_order_relaxed); }
        size_t capacity() const { return capacity_; }

    private:
        using Entry = std::pair<node_id_t, std::vector<node_id_t>>;
        struct Shard {
            mutable std::mutex                                  mu;
            std::list<Entry>                                    list;
            std::unordered_map<node_id_t,
                               std::list<Entry>::iterator>      map;
        };

        Shard& shard_for(node_id_t id) {
            // Hash-mix the id so consecutive monotonic ids spread
            // across shards instead of all landing in shard 0.
            std::uint64_t h = static_cast<std::uint64_t>(id);
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            return shards_[h % kNumShards];
        }

        size_t capacity_;
        size_t per_shard_capacity_;
        std::array<Shard, kNumShards> shards_;
        std::atomic<size_t> hits_{0};
        std::atomic<size_t> misses_{0};
        std::atomic<size_t> evictions_{0};
        std::atomic<size_t> invalidations_{0};
    };

    // Per-search scratch state. Previously these were members of LSMVec
    // and reused across calls; that prevents two concurrent searches on
    // the same index from running safely. The caller owns one instance
    // (typically as a thread_local in the worker thread) and passes it
    // into searchLayer.
    //
    // visited_version is uint32_t: at 1k searches/sec/thread, overflow
    // happens roughly every 50 days; visited_map is cleared and version
    // resets to 1 when that happens. With uint16_t the same wraparound
    // would fire every 65 seconds.
    struct SearchScratch {
        std::vector<float> batch_read_buf;
        std::unordered_map<node_id_t, uint32_t> visited_map;
        uint32_t visited_version = 0;
    };

    class LSMVec
    {
    public:
        // 20260516_compact_neighbors: per-Node layer→neighbors store.
        // The old type was std::unordered_map<int, std::vector<node_id_t>>;
        // for typical 1-2-layer upper nodes the hash-table headers + bucket
        // overhead (~64 bytes/empty + ~32 bytes/entry) dwarf the data. This
        // adapter presents the same API but stores entries contiguously,
        // dropping ~70 bytes/node on typical layouts. Linear scan is also
        // faster than std::unordered_map lookup at N ≤ ~4 entries.
        class LayerNeighbors {
        public:
            using value_type = std::pair<int, std::vector<node_id_t>>;
            using container  = std::vector<value_type>;

            std::vector<node_id_t>& operator[](int layer) {
                for (auto& e : data_) if (e.first == layer) return e.second;
                data_.emplace_back(layer, std::vector<node_id_t>{});
                return data_.back().second;
            }

            container::iterator       find(int layer) {
                for (auto it = data_.begin(); it != data_.end(); ++it)
                    if (it->first == layer) return it;
                return data_.end();
            }
            container::const_iterator find(int layer) const {
                for (auto it = data_.begin(); it != data_.end(); ++it)
                    if (it->first == layer) return it;
                return data_.end();
            }

            void emplace(int layer, std::vector<node_id_t> vec) {
                for (auto& e : data_) {
                    if (e.first == layer) { e.second = std::move(vec); return; }
                }
                data_.emplace_back(layer, std::move(vec));
            }

            container::iterator       begin()       { return data_.begin(); }
            container::const_iterator begin() const { return data_.begin(); }
            container::iterator       end()         { return data_.end(); }
            container::const_iterator end()   const { return data_.end(); }

            size_t size()     const { return data_.size(); }
            size_t capacity() const { return data_.capacity(); }
            bool   empty()    const { return data_.empty(); }

        private:
            container data_;
        };

        struct Node
        {
            node_id_t id;
            // float32 vector. Empty once the node has been quantized
            // (q8_data is non-empty in that case). Mutually exclusive with q8_data.
            std::vector<float> point;
            // 20260516_upper_layer_sq8: SQ8-quantized payload for upper-layer
            // nodes (layer ≥ 1). dim bytes; q_min/q_max are the dequant params.
            // Mutually exclusive with `point` per node.
            std::vector<uint8_t> q8_data;
            float q_min = 0.0f;
            float q_max = 0.0f;
            LayerNeighbors neighbors;  // Layer -> neighbors
        };

        mutable HNSWStats stats;

        bool use_heuristic_neighbor_selection_ = true;

        void setNeighborSelection(bool useHeuristic) {
            use_heuristic_neighbor_selection_ = useHeuristic;
        }

        LSMVec(const std::string& db_path,
               const LSMVecDBOptions& options,
               std::ostream &outFile);

        Status SerializeMetadata(std::ostream& out) const;
        Status DeserializeMetadata(std::istream& in);
        // Backward-compatible accessors: now backed by tombstoned_internal_ids_.
        // node_id_t and internal_id_t are both aliases for uint64_t, so callers
        // (LSMVecDB, persistence) compile unchanged.
        // Phase 3: returning a reference to a mutex-protected set is unsafe
        // for concurrent callers; persistence layer (the only caller) runs
        // under the engine's lifecycle exclusive gate so the reference is
        // stable for the call's duration.
        const std::unordered_set<node_id_t>& deletedIds() const {
            std::shared_lock<std::shared_mutex> g(tombstone_mu_);
            return tombstoned_internal_ids_;
        }
        void setDeletedIds(const std::unordered_set<node_id_t>& ids) {
            std::unique_lock<std::shared_mutex> g(tombstone_mu_);
            tombstoned_internal_ids_ = ids;
        }

        void insertNode(node_id_t nodeId, const std::vector<float> &vector);
        node_id_t knnSearch(const std::vector<float> &queryVector);
        Status deleteNode(node_id_t id);
        Status getNodeVector(node_id_t id, std::vector<float>* out);

        // ----- Bulk build (in-memory) -----
        // Build the full index from `n` contiguous vectors using
        // RNN-Descent for the layer-0 KNN graph and MIRAGE-style
        // top-down upper-layer insertion. Initial-load only:
        // caller must invoke on an empty index (entry_point_ ==
        // k_invalid_node_id). Throws std::runtime_error on failure.
        //
        // See docs/IN_MEMORY_BUILD_PLAN.md for the algorithm and
        // pipeline details.
        void bulkBuild(const float* vectors, int n,
                       const BulkBuildOptions& opts);

        // True iff the index has never had a node published (no
        // Insert / BulkBuild has succeeded). Used by LSMVecDB::BulkBuild
        // to enforce the initial-load-only precondition.
        bool isEmpty() const {
            return entry_point_.load(std::memory_order_acquire)
                   == k_invalid_node_id;
        }
        std::vector<SearchResult> knnSearchK(const std::vector<float>& query, int k, int ef_search);
        std::vector<SearchResult> knnSearchKFiltered(
            const std::vector<float>& query,
            int k,
            int ef_search,
            const metadata::Predicate* pred,
            int max_scan_candidates,
            const class MetadataStore* meta_store);
        std::unordered_set<node_id_t> highest_layer_nodes_;

        void printIndexStatus() const;
        void printStatistics() const;
        void printState() const;
        void close();

        std::string vector_file_path_;
        int vector_dim_ = 0;
        std::unique_ptr<IVectorStorage> vector_storage_;
        LSMVecDBOptions db_options_;

        // ----------------------------------------------------------------
        // Delete / update primitives (V1; see docs/DELETE_DESIGN.md §4-§5).
        // Public so LSMVecDB can drive the lifecycle and tests can verify
        // the resolvers in isolation. Read-only accessors below for stats.
        // ----------------------------------------------------------------
        internal_id_t resolve_internal(real_id_t r) const;
        real_id_t     resolve_real(internal_id_t i) const;
        bool          is_alive(real_id_t r) const;

        bool          is_tombstoned(internal_id_t i) const {
            std::shared_lock<std::shared_mutex> g(tombstone_mu_);
            return tombstoned_internal_ids_.count(i) > 0;
        }
        void          tombstone(internal_id_t i) {
            std::unique_lock<std::shared_mutex> g(tombstone_mu_);
            tombstoned_internal_ids_.insert(i);
        }

        // Atomic counter (concurrent-writer-refactor-plan §5.2).
        internal_id_t allocate_update_id() {
            return next_update_internal_id_.fetch_add(1, std::memory_order_relaxed);
        }
        void          record_update_mapping(real_id_t r, internal_id_t i);

        // A4: Drop both forward and reverse sparse-map entries for real_id.
        // Used by LSMVecDB::Delete so future resolve_internal(r) returns
        // identity. Bloom-filter bit stays set (FP falls through harmlessly).
        void          forget_update_mapping(real_id_t r);

        // Read-only accessors for stats / tests.
        std::size_t   tombstone_count()    const {
            std::shared_lock<std::shared_mutex> g(tombstone_mu_);
            return tombstoned_internal_ids_.size();
        }
        std::size_t   updated_real_id_count() const {
            std::shared_lock<std::shared_mutex> g(mapping_mu_);
            return updated_real_to_internal_.size();
        }
        std::size_t   updated_internal_to_real_size() const {
            std::shared_lock<std::shared_mutex> g(mapping_mu_);
            return updated_internal_to_real_.size();
        }
        std::size_t   bloom_capacity()     const {
            std::shared_lock<std::shared_mutex> g(mapping_mu_);
            return update_bloom_.capacity();
        }
        double        bloom_fill_ratio()   const {
            std::shared_lock<std::shared_mutex> g(mapping_mu_);
            return update_bloom_.fill_ratio();
        }
        internal_id_t next_update_internal_id() const {
            return next_update_internal_id_.load(std::memory_order_relaxed);
        }

        // C8 observability counters. `total_inserts_ever_` is in-memory-only
        // (resets on Open — a long-running service's ratio becomes accurate
        // only after the first Insert post-restart). Acceptable for V1.
        // R3.a / internal review: atomic relaxed because every concurrent
        // Insert increments it; plain std::size_t would be a TSan-flaggable
        // data race even though the counter is just observability.
        std::size_t   total_inserts_ever() const {
            return total_inserts_ever_.load(std::memory_order_relaxed);
        }
        std::size_t   bloom_rebuild_count() const { return bloom_rebuild_count_; }
        // tombstones / total_inserts_ever; returns 0.0 if no inserts seen yet.
        double        tombstone_ratio() const {
            std::shared_lock<std::shared_mutex> g(tombstone_mu_);
            std::size_t n = total_inserts_ever_.load(std::memory_order_relaxed);
            return n == 0
                 ? 0.0
                 : static_cast<double>(tombstoned_internal_ids_.size()) /
                   static_cast<double>(n);
        }

    private:
        // ----- Bulk build internals (see bulkBuild() above + plan) -----
        // Phase D: write all vectors + layer-0 edges to disk in parallel.
        // csr_offsets/csr_neighbors come from RNN-Descent in phase B;
        // levels[u] is the pre-assigned HNSW level of node u.
        void bulkLoadLayer0(const float* vectors, int n,
                            const int* csr_offsets,
                            const int* csr_neighbors,
                            const int* levels,
                            int num_threads);
        // Phase E: upper-layer-only insert. Factored from insertNode
        // with layer-0 work removed. Expects the node to be pre-published
        // into nodes_ with `point` set, and entry_point_/max_layer_
        // already pointing at the bulk-build's final entry node.
        void addPointUpperLayersOnly(node_id_t nodeId,
                                     const float* vec_ptr,
                                     int presetLevel);

        // 20260516_upper_layer_sq8 helpers.
        // quantizeNodePoint: destructive — writes n.q8_data/q_min/q_max from
        // n.point, then swaps n.point to empty.
        static void quantizeNodePoint(Node& n);
        // dequantizeNodePoint: writes a fresh float vector into `out`
        // (resized to q8_data.size()). Returns true if q8_data is non-empty;
        // false if the node has neither point nor q8_data.
        bool dequantizeNodePoint(const Node& n, std::vector<float>& out) const;
        // nodePointView: returns a Span<const float> over the node's vector.
        // If point is populated, returns a span over it directly. Else
        // materializes into `scratch` and returns a span over scratch.
        // Caller keeps scratch alive while the span is in use.
        Span<const float> nodePointView(const Node& n,
                                         std::vector<float>& scratch) const;

        int randomLevel();
        float computeDistance(Span<const float> vectorA,
                              Span<const float> vectorB) const;
        void storeVectorWithStats(node_id_t id,
                                  const std::vector<float>& vec,
                                  node_id_t sectionKey);
        void readVectorWithStats(node_id_t id,
                                 std::vector<float>& vec);
        std::vector<SearchResult> searchLayer(const std::vector<float> &queryVector,
                                              node_id_t entryPointId,
                                              int efSearch,
                                              int layer,
                                              SearchScratch& scratch);
        std::vector<SearchResult> searchLayer(const std::vector<float> &queryVector,
                                              node_id_t entryPointId,
                                              int efSearch,
                                              int layer,
                                              const metadata::Predicate* pred,
                                              int k,
                                              int max_scan_candidates,
                                              const class MetadataStore* meta_store,
                                              SearchScratch& scratch);
        node_id_t greedySearchUpperLayer(const std::vector<float>& query,
                                          node_id_t entryPoint, int layer);
        void linkNeighbors(node_id_t nodeId, const std::vector<node_id_t> &neighborIds, int layer);
        void linkNeighborsAsterDB(node_id_t nodeId, const std::vector<node_id_t> &neighborIds);
        std::vector<node_id_t> getEdgesCached(node_id_t id);

        std::vector<node_id_t> selectNeighbors(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        std::vector<node_id_t> selectNeighborsSimple(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        std::vector<node_id_t> selectNeighborsHeuristic1(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        std::vector<node_id_t> selectNeighborsHeuristic2(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        // Overloads accepting SearchResult (with precomputed distances)
        std::vector<node_id_t> selectNeighbors(
            const std::vector<float>& vector,
            const std::vector<SearchResult>& candidates,
            int maxNeighbors,
            int layer
        );

        std::vector<node_id_t> selectNeighborsHeuristic2(
            const std::vector<float>& vector,
            const std::vector<SearchResult>& candidates,
            int maxNeighbors,
            int layer
        );

        int m_;                      // Number of established connections
        int m_max_;                  // Maximum neighbors per layer
        int m_level_;                // Normalization factor for level generation
        float ef_construction_;      // Parameter for candidate selection

        std::unique_ptr<rocksdb::RocksGraph> db_;
        rocksdb::Options options_;
        
        std::ostream &out_file_;     // Output stream for logging

        std::unordered_map<node_id_t, Node> nodes_; // In-memory nodes for layers > 0

        // Phase 4d (§5.1.1 — relaxed primitive). std::shared_mutex
        // protecting nodes_ map mutations. Reads (find / at) take
        // shared; the single insert site (nodes_[id] = newNode) takes
        // exclusive. The plan's full §5.1.1 chunk-array primitive
        // doesn't fit naturally over the sparse direct-id +
        // high-bit-update-id namespace; the simpler shared_mutex is
        // correct, with the option to swap in a concurrent hash map
        // (TBB / folly::ConcurrentHashMap) if profiling on the eval
        // server shows contention.
        mutable std::shared_mutex nodes_mu_;

        // Phase 4c (concurrent-writer-refactor-plan §5.1.4 — relaxed
        // primitive). 256-shard mutex protecting Node.neighbors[layer]
        // mutations and reads. std::mutex (not shared_mutex) because
        // shared_mutex acquire on macOS / glibc is 5-10× more expensive;
        // same-node read serialisation is rare given the visited-set
        // dedup in search scratch. Both reads (greedy descent,
        // searchLayer) and writes (linkNeighbors, shrink) take the
        // shard mutex briefly. Write paths also take their own shard
        // when modifying their target node.
        //
        // The plan's full §5.1.4 design (atomic-slot fixed-array +
        // lock-free CAS-bounded append + prune-only spinlock) is
        // deferred until profiling on a stable benchmark host (not
        // the dev macOS where single-thread variance is too high to
        // measure ~10% overheads reliably). The current design has
        // the same 256-way scaling shape and is correctness-equivalent.
        static constexpr size_t kNumNeighborShards = 256;
        mutable std::array<std::mutex, kNumNeighborShards> neighbor_shards_;
        std::mutex& node_shard(node_id_t id) const {
            std::uint64_t h = static_cast<std::uint64_t>(id);
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            return neighbor_shards_[h % kNumNeighborShards];
        }

        // RNG removed — Phase 4a (§5.1.5) replaced shared mt19937 with
        // thread_local in LSMVec::randomLevel().
        //
        // Phase 4b (§5.1.3): independent atomics on entry_point_ and
        // max_layer_. Greedy descent at src/lsm_vec_index.cc is graceful
        // (breaks on missing node or missing layer), so transient
        // (NEW, OLD) / (OLD, NEW) observations are absorbed without a
        // paired-publish primitive. First-insert + promotion use CAS.
        // Search side guards against k_invalid_node_id explicitly.
        std::atomic<int>       max_layer_{-1};
        std::atomic<node_id_t> entry_point_{k_invalid_node_id};

        // V1 delete/update infrastructure — sparse maps over the divergent
        // subset of (real_id, internal_id) plus a Bloom shortcut for forward
        // resolve. Wiring into the operation layer happens in C5b.
        //
        // Phase 3 of concurrent-writer-refactor-plan §5.2: these are now
        // protected by mapping_mu_ (the maps + bloom) and tombstone_mu_
        // (the tombstone set). next_update_internal_id_ is atomic. Reads
        // (resolve_internal, resolve_real, is_alive, is_tombstoned) take
        // shared; writes (record_update_mapping, forget_update_mapping,
        // tombstone, setDeletedIds) take exclusive. The full sharded
        // version specced in §5.2 is deferred until profiling shows the
        // single mutex contended.
        std::unordered_map<real_id_t, internal_id_t> updated_real_to_internal_;
        std::unordered_map<internal_id_t, real_id_t> updated_internal_to_real_;
        std::unordered_set<internal_id_t>            tombstoned_internal_ids_;
        BloomFilter                                   update_bloom_{512, 0.01};
        std::atomic<internal_id_t>                    next_update_internal_id_{kFirstUpdateId};
        mutable std::shared_mutex                     mapping_mu_;
        mutable std::shared_mutex                     tombstone_mu_;

        // C8 transient observability counters (reset on Open).
        std::atomic<std::size_t>                      total_inserts_ever_{0};
        std::size_t                                   bloom_rebuild_count_ = 0;

        void rebuild_bloom_to(std::size_t new_capacity);
        // Variant for callers already holding mapping_mu_ exclusive
        // (e.g. record_update_mapping). Phase 3.
        void rebuild_bloom_to_locked(std::size_t new_capacity);

        int section_layer_ = 1;
        bool enable_batch_read_ = true;

        // Per-search scratch (visited map, batch read buffer, version
        // counter) used to live as members of LSMVec and was reused across
        // calls. That made the search path racy under concurrent callers.
        // Scratch now lives in SearchScratch (defined above the LSMVec
        // class) and is passed by reference into searchLayer; callers own
        // it (worker threads typically declare it as thread_local).

        // Edge LRU cache for L0 GetAllEdges
        std::unique_ptr<EdgeLRUCache> edge_cache_;

        // ----------------------------------------------------------------
        // Adaptive Direct I/O (see docs/local/..._ADAPTIVE_DIRECT_IO_*).
        // During an incremental build we start with buffered I/O for the
        // RocksGraph store. A background CgroupMemoryMonitor watches for
        // sustained page-cache thrash under the cgroup cap; on trip, the
        // insert driver performs a ONE-WAY escalation to RocksDB Direct I/O
        // (close + reopen the RocksGraph handle) and sheds the stale SST page
        // cache. The in-memory HNSW state (nodes_, entry_point_, edge_cache_)
        // survives the swap untouched.
        // ----------------------------------------------------------------
        std::string                          db_path_;          // for RocksGraph reopen
        std::unique_ptr<CgroupMemoryMonitor> mem_monitor_;
        std::atomic<bool>                    direct_io_active_{false}; // one-way latch
        std::mutex                           dio_switch_mu_;

        void     startAdaptiveDirectIO();
        void     maybeEscalateDirectIO();    // cheap check at insert boundaries
        void     performDirectIOSwitch();    // heavyweight close+reopen
        bool     directIOSupported();        // probe O_DIRECT on the data FS
        size_t   dropSSTPageCache();         // posix_fadvise(DONTNEED) over *.sst
    };
} // namespace ROCKSDB_NAMESPACE
