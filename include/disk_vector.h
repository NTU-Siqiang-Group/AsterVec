#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "id_types.h"

// Thread-safety contract for PagedVectorStorage
// =============================================
//
// Read-side methods (readVectorFromDisk, readVectorsBatch,
// readVectorsBatchFlat, exists) are safe to call concurrently from
// multiple threads. Internally, the page cache (pageCache_ /
// pageOrder_) is protected by a std::shared_mutex; cache lookups take
// shared locks, insertions and evictions take unique locks. Disk I/O
// happens outside the cache mutex.
//
// Write-side concurrency
// ----------------------
// Concurrent-writer-refactor-plan §5.3 (r3.2): the "external EXCLUSIVE
// lock" contract is replaced with internal sharded synchronisation so
// concurrent inserts on different sections can proceed in parallel.
//
//   * storeVectorToDisk acquires:
//       - sectionShards_[sectionIdx % 64] for section bookkeeping
//         (sectionFreeSlots_, sectionOpenPages_)
//       - sectionWriteBufsMu_ for the section write-buffer map
//         (Phase 6 / internal review: a single mutex covers map
//         structure + per-entry content; readers
//         overlayWriteBuf / tryReadFromWriteBuf{,Flat} take the same
//         mutex, with an atomic-counter fast-path for the empty case)
//       - pages_alloc_mu_ briefly when allocating a fresh page
//         (pages_.push_back + ensurePageAllocated + the dense-array
//         capacity grow path expandCapacity)
//       - section_index_mu_ briefly when resolving a sectionKey to a
//         sectionIdx for the first time
//       - update_loc_mu_ briefly when assigning an update-id location
//       - the actual record pwrite is performed UNLOCKED (different
//         (pageId, slot) tuples are non-overlapping byte ranges)
//   * expandCapacity / deserializeMetadata take pages_alloc_mu_
//     because they grow the dense location arrays.
//   * flushWrites runs under the engine's exclusive lifecycle gate
//     (FLUSH op in pglsmvec/csrc/lsmvec_worker_pool.cpp) and therefore
//     does not need to coordinate with concurrent storeVectorToDisk
//     callers.
//
// At Phase 1 (the engine's INSERT op still takes the lifecycle gate
// EXCLUSIVE), these internal locks are uncontended in practice — they
// are added now so that Phase 6 of the plan can drop the engine's
// exclusive without breaking storage thread-safety.

namespace lsm_vec
{
using node_id_t = std::uint64_t;
static constexpr node_id_t k_invalid_node_id =
    std::numeric_limits<node_id_t>::max();

class IVectorStorage {
public:
    struct PageCacheStats {
        std::size_t hits = 0;
        std::size_t misses = 0;
    };

    virtual ~IVectorStorage() = default;

    // Dimension of each vector (number of floats)
    virtual size_t getVectorDim() const = 0;

    // Maximum number of logical IDs supported
    virtual size_t getCapacity() const = 0;

    // Store vector for logical ID 'id' (basic API)
    virtual void storeVectorToDisk(node_id_t id,
                                   const std::vector<float>& vec) = 0;

    // Optional section-aware API. Default implementation ignores sectionKey
    // and forwards to the basic store.
    virtual void storeVectorToDisk(node_id_t id,
                                   const std::vector<float>& vec,
                                   node_id_t sectionKey)
    {
        (void)sectionKey;
        storeVectorToDisk(id, vec);
    }

    // Read vector for logical ID 'id'
    virtual void readVectorFromDisk(node_id_t id,
                                    std::vector<float>& vec) = 0;

    // Mark a vector as deleted for logical ID 'id'
    virtual void deleteVector(node_id_t id) = 0;

    // Check whether a vector exists (not deleted and assigned)
    virtual bool exists(node_id_t id) const = 0;

    // Optional prefetch API. Default is no-op.
    virtual void prefetchByIds(const std::vector<node_id_t>& /*ids*/) {}

    // Batch read API: read multiple vectors at once.
    // Default implementation reads one-by-one; PagedVectorStorage overrides
    // to group reads by page for fewer page lookups.
    virtual void readVectorsBatch(const std::vector<node_id_t>& ids,
                                  std::vector<std::vector<float>>& out) {
        out.resize(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            readVectorFromDisk(ids[i], out[i]);
        }
    }

    // Flat-buffer batch read: writes vectors contiguously into caller-owned buffer.
    // out must point to at least ids.size() * dim floats.
    virtual void readVectorsBatchFlat(const std::vector<node_id_t>& ids,
                                      float* out, size_t dim) {
        for (size_t i = 0; i < ids.size(); ++i) {
            std::vector<float> tmp(dim);
            readVectorFromDisk(ids[i], tmp);
            std::memcpy(out + i * dim, tmp.data(), dim * sizeof(float));
        }
    }

    // Flush buffered writes to disk. Default is no-op.
    virtual void flushWrites() {}

    // Optional page cache stats. Default returns zeros.
    virtual PageCacheStats getPageCacheStats() const { return {}; }
};

class BasicVectorStorage : public IVectorStorage {
private:
    std::string filePath_;
    std::string deleteFilePath_;
    size_t dim_;          // number of floats per vector
    size_t totalVectors_; // capacity (# of logical IDs)
    size_t fileSizeBytes_;

    std::fstream fileStream_;
    std::fstream deleteStream_;
    std::vector<uint8_t> deletedFlags_;

private:
    void ensureDeleteFileSize(size_t targetSize) {
        deleteStream_.clear();
        deleteStream_.seekp(0, std::ios::end);
        size_t currentSize = static_cast<size_t>(deleteStream_.tellp());
        if (currentSize < targetSize) {
            deleteStream_.seekp(static_cast<std::streamoff>(targetSize - 1),
                                std::ios::beg);
            char zero = 0;
            deleteStream_.write(&zero, 1);
            deleteStream_.flush();
        }
    }

    void reloadDeleteFlags(size_t targetSize) {
        ensureDeleteFileSize(targetSize);
        deletedFlags_.assign(targetSize, 0);
        deleteStream_.clear();
        deleteStream_.seekg(0, std::ios::beg);
        deleteStream_.read(reinterpret_cast<char*>(deletedFlags_.data()),
                           static_cast<std::streamsize>(deletedFlags_.size()));
    }

    void openDeleteFile() {
        deleteStream_.open(deleteFilePath_,
                           std::ios::in | std::ios::out | std::ios::binary);
        if (!deleteStream_.is_open()) {
            deleteStream_.open(deleteFilePath_,
                               std::ios::out | std::ios::binary | std::ios::trunc);
            if (!deleteStream_.is_open()) {
                throw std::runtime_error("Failed to create delete marker file.");
            }
            deleteStream_.close();
            deleteStream_.open(deleteFilePath_,
                               std::ios::in | std::ios::out | std::ios::binary);
            if (!deleteStream_.is_open()) {
                throw std::runtime_error("Failed to open delete marker file.");
            }
        }

        reloadDeleteFlags(totalVectors_);
    }

    void writeDeleteFlag(node_id_t id, bool deleted) {
        deletedFlags_[static_cast<size_t>(id)] = deleted ? 1 : 0;
        deleteStream_.clear();
        deleteStream_.seekp(static_cast<std::streamoff>(id), std::ios::beg);
        char value = deleted ? 1 : 0;
        deleteStream_.write(&value, 1);
        deleteStream_.flush();
        if (!deleteStream_.good()) {
            throw std::runtime_error("Failed to update delete marker file.");
        }
    }

public:
    BasicVectorStorage(const std::string& path,
                       size_t dim,
                       size_t numVectors)
        : filePath_(path),
          deleteFilePath_(path + ".deleted"),
          dim_(dim),
          totalVectors_(numVectors)
    {
        if (dim_ == 0) {
            throw std::runtime_error("Vector dimension must be > 0.");
        }

        fileSizeBytes_ = dim_ * totalVectors_ * sizeof(float);

        fileStream_.open(filePath_,
                         std::ios::in | std::ios::out | std::ios::binary);
        if (!fileStream_.is_open())
        {
            // Try to create the file
            fileStream_.open(filePath_,
                             std::ios::out | std::ios::binary | std::ios::trunc);
            if (!fileStream_.is_open())
            {
                throw std::runtime_error("Failed to create file.");
            }
            fileStream_.close();

            fileStream_.open(filePath_,
                             std::ios::in | std::ios::out | std::ios::binary);
            if (!fileStream_.is_open())
            {
                throw std::runtime_error("Failed to open file.");
            }
        }

        // Ensure file has enough size
        fileStream_.seekp(0, std::ios::end);
        size_t currentSize = static_cast<size_t>(fileStream_.tellp());
        if (currentSize < fileSizeBytes_)
        {
            fileStream_.seekp(static_cast<std::streamoff>(fileSizeBytes_ - 1),
                              std::ios::beg);
            char zero = 0;
            fileStream_.write(&zero, 1);
            fileStream_.flush();
        }

        openDeleteFile();
    }

    ~BasicVectorStorage() override
    {
        if (fileStream_.is_open())
        {
            fileStream_.close();
        }
        if (deleteStream_.is_open())
        {
            deleteStream_.close();
        }
    }

    // ---- IVectorStorage interface ----

    size_t getVectorDim() const override {
        return dim_;
    }

    size_t getCapacity() const override {
        return totalVectors_;
    }

    void storeVectorToDisk(node_id_t id,
                           const std::vector<float>& vector) override
    {
        if (static_cast<size_t>(id) >= totalVectors_)
        {
            throw std::out_of_range("Vector ID out of range.");
        }
        if (vector.size() != dim_)
        {
            throw std::runtime_error("Vector size mismatch.");
        }

        size_t offset = static_cast<size_t>(id) * dim_ * sizeof(float);
        fileStream_.seekp(static_cast<std::streamoff>(offset), std::ios::beg);

        fileStream_.write(reinterpret_cast<const char*>(vector.data()),
                          static_cast<std::streamsize>(dim_ * sizeof(float)));
        if (!fileStream_.good())
        {
            throw std::runtime_error("Failed to write vector to file.");
        }
        fileStream_.flush();

        if (deletedFlags_[static_cast<size_t>(id)] != 0) {
            writeDeleteFlag(id, false);
        }
    }

    void readVectorFromDisk(node_id_t id,
                            std::vector<float>& vector) override
    {
        if (static_cast<size_t>(id) >= totalVectors_)
        {
            throw std::out_of_range("Vector ID out of range.");
        }

        if (vector.size() != dim_)
        {
            vector.resize(dim_);
        }

        size_t offset = static_cast<size_t>(id) * dim_ * sizeof(float);
        fileStream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);

        fileStream_.read(reinterpret_cast<char*>(vector.data()),
                         static_cast<std::streamsize>(dim_ * sizeof(float)));
        if (!fileStream_.good())
        {
            throw std::runtime_error("Failed to read vector from file.");
        }
    }

    void deleteVector(node_id_t id) override
    {
        if (static_cast<size_t>(id) >= totalVectors_)
        {
            throw std::out_of_range("Vector ID out of range.");
        }
        if (deletedFlags_[static_cast<size_t>(id)] == 0) {
            writeDeleteFlag(id, true);
        }
    }

    bool exists(node_id_t id) const override
    {
        if (static_cast<size_t>(id) >= totalVectors_)
        {
            return false;
        }
        return deletedFlags_[static_cast<size_t>(id)] == 0;
    }
};

class PagedVectorStorage : public IVectorStorage {
public:
    static constexpr size_t kPageSize = 4096; // 4KB OS page

private:
    std::string filePath_;
    std::string deleteFilePath_;
    size_t dim_;            // vector dimension
    size_t recordSize_;     // SQ8: dim_ + 2 * sizeof(float)
    size_t totalVectors_;   // logical ID capacity
    // Phase 6 (internal review): the lock-free page_of/slot_of hot
    // path was reverted (it was UB under the C++ memory model). This
    // atomic is no longer load-bearing for correctness; it remains
    // only for the future packed-atomic chunk-array primitive (§5.4).
    mutable std::atomic<size_t> totalVectors_atomic_{0};
    size_t vectorsPerPage_; // how many vectors fit into one page (floor division)

    std::fstream fileStream_;
    std::fstream deleteStream_;
    std::vector<uint8_t> deletedFlags_;

    // logical ID -> page & slot mapping.
    // 20260516_narrow_idtopage: page index is int32_t (was int64_t). -1 means
    // unassigned. 2^31 pages × 4 KB = 8 TB capacity per backing file — well
    // beyond practical sizes. The on-disk format still uses int64_t per entry
    // for backwards compatibility; widen on serialize, narrow on deserialize.
    std::vector<int32_t>  idToPage_;
    std::vector<uint16_t> idToSlotInPage_; // slot in [0, vectorsPerPage_)

    struct PageMeta {
        int sectionIdx;      // internal section index, -1 for unused
        uint16_t usedSlots;  // number of occupied slots in this page
    };

    // Concurrent-writer-refactor-plan §5.3: std::deque (not vector) so
    // push_back on a different thread never invalidates `pages_[i]`
    // references held by other threads. Only the push_back itself
    // needs pages_alloc_mu_; mutations of pages_[pageId] in the open-
    // page path proceed under just the section-shard mutex (different
    // sections own disjoint page sets, so different shards never touch
    // the same pages_[pageId]).
    std::deque<PageMeta> pages_; // pageId -> meta

    // Section mappings: external sectionKey -> internal sectionIdx
    std::unordered_map<node_id_t,int> sectionKeyToIdx_;
    std::vector<node_id_t>            sectionIdxToKey_;

    // Per-section bookkeeping (pages with free slots, free slots
    // within each page) lives INSIDE each shard struct below. An
    // earlier design had two top-level
    // std::unordered_map<int, ...> shared across all shards, which
    // was unsafe: the per-shard mutex only serialises same-sectionIdx
    // access, but concurrent operator[] for different sectionIdx
    // races on the underlying bucket structure (rehash, new-key
    // insertion, chain head pointer update). That race caused engine
    // crashes and parallel-build deadlocks under high write
    // concurrency. Per-shard maps make the per-shard mutex sufficient
    // — no cross-shard structural sharing.

    // C6: sparse map for update-allocated internal_ids (bit 63 = 1).
    // Direct ids (bit 63 = 0) continue to use the dense idToPage_ /
    // idToSlotInPage_ arrays above. The on-disk data file is shared across
    // both ranges so allocateSlotForSection can co-locate a direct vector
    // and an update vector on the same 4 KB page (E3 — locality).
    // Update-id deletion is tracked by LSMVec::tombstoned_internal_ids_,
    // not here, so no per-update-id deleted_flag is stored.
    struct UpdateLoc {
        int64_t  page = -1;
        uint16_t slot = 0;
    };
    std::unordered_map<node_id_t, UpdateLoc> update_locations_;

    // -------------------------------------------------------------
    // Phase 2 internal synchronisation (concurrent-writer-refactor-plan §5.3).
    // See the file-header thread-safety contract above.
    // -------------------------------------------------------------
    static constexpr size_t kNumSectionShards = 64;

    // Section coalescing write buffer (one 4KB page per active section).
    // Moved up here so SectionShard can hold a map of these.
    struct SectionWriteBuf {
        std::vector<char> data;  // kPageSize bytes
        size_t pageId;
        uint16_t filledSlots;
    };

    /*
     * Per-shard state: each shard owns its own mutex AND its own
     * (openPages, freeSlots, writeBufs) maps. This is the structural
     * fix for the race noted above — `operator[]` mutations on the
     * maps now happen only under THIS shard's mutex, never racing
     * with other shards' threads.
     *
     * writeBufs holds the per-section 4KB coalescing write buffer
     * (formerly a single global map under sectionWriteBufsMu_). Move
     * follows commit 4a1e216 (openPages/freeSlots) — internal review
     * for build-time scaling: same-shard accesses still serialise
     * (and now share the lock with allocateSlotForSection, which is
     * already called sequentially with writeToSectionBuffer for the
     * same section), but cross-section writes for different shards
     * run in parallel.
     */
    struct SectionShard {
        mutable std::mutex                                              mu;
        std::unordered_map<int, std::vector<size_t>>                    openPages;
        std::unordered_map<int, std::vector<std::pair<size_t,uint16_t>>> freeSlots;
        std::unordered_map<int, SectionWriteBuf>                        writeBufs;  // sectionIdx -> buf
    };
    mutable std::array<SectionShard, kNumSectionShards> sectionShards_;
    // Phase 5 (concurrent-writer-refactor-plan §5.3 / §5.4): upgraded
    // from std::mutex to std::shared_mutex so reads of the dense
    // location arrays (page_of / slot_of) take shared, while writes
    // (assign_location / expandCapacity / pages_.push_back) take
    // exclusive. The Phase 5 publish-visibility protocol is the
    // ordering: vector bytes are pwritten BEFORE the location is
    // published into idToPage_/idToSlotInPage_ via assign_location.
    // Readers that see page_of(id) >= 0 are guaranteed the bytes are
    // already on disk.
    mutable std::shared_mutex                         pages_alloc_mu_;
    mutable std::mutex                                section_index_mu_;
    mutable std::mutex                                update_loc_mu_;

    SectionShard& section_shard_state(int sectionIdx) const {
        return sectionShards_[
            static_cast<size_t>(sectionIdx) % kNumSectionShards];
    }
    std::mutex& section_shard(int sectionIdx) const {
        return section_shard_state(sectionIdx).mu;
    }

    // ----- Location dispatch helpers (V1, see DELETE_DESIGN.md §0.3) -----
    // These transparently route direct ids to dense arrays and update ids to
    // the sparse hashmap, so the public methods don't have to branch.

    // Phase 6 (internal review): the previous "lock-free hot read path"
    // optimisation was incorrect under the C++ memory model — a plain
    // std::vector<int64_t>::operator[] read concurrent with a plain
    // operator[] write is a data race regardless of natural-aligned
    // hardware atomicity. Reverted to a shared_lock(pages_alloc_mu_)
    // for the dense-id branch. The proper long-term fix is the §5.4
    // packed-atomic chunk array primitive (deferred). The macOS
    // measurement showed the lock-free path was within noise anyway
    // (~4164 QPS in both modes).
    int64_t page_of(node_id_t id) const {
        if (is_direct_id(id)) {
            // A1.15 introduced a lock-free read via totalVectors_atomic_;
            // A1.16 (Codex review) reverted it because concurrent
            // assign_location overflow into the sparse path makes the
            // lock-free read unsafe in edge cases. totalVectors_atomic_
            // remains for storeVectorToDisk publish ordering elsewhere.
            // 20260516_narrow_idtopage: idToPage_ is int32_t; widen for API.
            std::shared_lock<std::shared_mutex> g(pages_alloc_mu_);
            if (static_cast<size_t>(id) >= idToPage_.size()) return -1;
            return static_cast<int64_t>(idToPage_[static_cast<size_t>(id)]);
        }
        // update-id branch: sparse map under update_loc_mu_.
        std::lock_guard<std::mutex> g(update_loc_mu_);
        auto it = update_locations_.find(id);
        return (it == update_locations_.end()) ? -1 : it->second.page;
    }

    uint16_t slot_of(node_id_t id) const {
        if (is_direct_id(id)) {
            std::shared_lock<std::shared_mutex> g(pages_alloc_mu_);
            if (static_cast<size_t>(id) >= idToSlotInPage_.size()) return 0;
            return idToSlotInPage_[static_cast<size_t>(id)];
        }
        // update-id branch: sparse map under update_loc_mu_.
        std::lock_guard<std::mutex> g(update_loc_mu_);
        auto it = update_locations_.find(id);
        return (it == update_locations_.end()) ? 0 : it->second.slot;
    }

    bool is_deleted_at(node_id_t id) const {
        if (is_direct_id(id)) {
            if (static_cast<size_t>(id) >= totalVectors_) return false;
            return deletedFlags_[static_cast<size_t>(id)] != 0;
        }
        return false;  // LSMVec owns tombstone state for update ids
    }

    // For write paths: ensure dense storage is large enough for direct ids.
    // No-op for update ids (the hashmap auto-grows on assignment).
    void ensure_dense_capacity(node_id_t id) {
        if (is_direct_id(id) && static_cast<size_t>(id) >= totalVectors_) {
            expandCapacity(static_cast<size_t>(id) + 1);
        }
    }

    void assign_location(node_id_t id, size_t page, uint16_t slot) {
        if (is_direct_id(id)) {
            // 20260516_narrow_idtopage: 2^31 page ceiling = 8 TB per backing
            // file. Hitting this means a multi-TB single-file workload — bail
            // out loudly rather than silently truncate the high bits.
            //
            // Phase 5 publish protocol (§5.4): vector bytes are already on
            // disk by the time storeVectorToDisk reaches here; this
            // unique_lock publish makes the (id → page, slot) mapping
            // visible atomically to readers that use shared-lock
            // page_of/slot_of. Readers observing page_of(id) >= 0 are
            // guaranteed the bytes are durable.
            if (page > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error(
                    "PagedVectorStorage: page index exceeds int32_t ceiling "
                    "(8 TB) — rebuild with int64_t idToPage_ or split the "
                    "storage file");
            }
            std::unique_lock<std::shared_mutex> g(pages_alloc_mu_);
            idToPage_[static_cast<size_t>(id)]       = static_cast<int32_t>(page);
            idToSlotInPage_[static_cast<size_t>(id)] = slot;
        } else {
            std::lock_guard<std::mutex> g(update_loc_mu_);
            UpdateLoc& loc = update_locations_[id];
            loc.page = static_cast<int64_t>(page);
            loc.slot = slot;
        }
    }

    void mark_deleted_at(node_id_t id, bool deleted) {
        if (is_direct_id(id) && static_cast<size_t>(id) < totalVectors_
            && (deletedFlags_[static_cast<size_t>(id)] != 0) != deleted) {
            writeDeleteFlag(id, deleted);
        }
        // update ids: no-op (LSMVec owns the canonical tombstone state)
    }

    // Page cache (FIFO) in units of full pages (4KB each).
    //
    // Concurrency: pageCache_ + pageOrder_ are protected by page_mu_,
    // a shared_mutex. Read paths take shared_lock for lookups; insertions
    // and evictions take unique_lock. Disk I/O (pread) is performed
    // outside any lock — concurrent readers may pread the same page on
    // miss, but only one wins the insertion (double-check after taking
    // unique_lock). Writers from the LSMVec write path are protected by
    // the external per-index RWLock, so they cannot race with readers
    // here.
    size_t maxCachedPages_;
    std::atomic<size_t> pageCacheHits_{0};
    std::atomic<size_t> pageCacheMisses_{0};

    struct PageBuf {
        std::vector<char> data; // always kPageSize bytes
    };

    std::unordered_map<size_t, PageBuf> pageCache_; // pageId -> data
    std::deque<size_t>                  pageOrder_; // FIFO order of pageIds
    mutable std::shared_mutex           page_mu_;

    int readFd_ = -1;  // file descriptor for pread-based I/O

    // Scratch buffers for readVectorsBatchFlat. These were class-level
    // fields reused across calls; that prevented concurrent reads on
    // the same instance from being safe. Now declared as thread_local
    // inside readVectorsBatchFlat (see implementation), one instance per
    // worker thread, capacity warmed across calls within a thread.
    struct MissEntry { size_t idx; size_t pageId; uint16_t slot; };

    // sq8TempBuf_ removed: directReadRecord and writeRecord allocate a
    // small stack-local buffer per call (recordSize_ ≈ dim+8 bytes, sub-µs
    // alloc).

    // Section write buffer: per-section 4KB buffer for coalesced writes via pwrite
    // SectionWriteBuf is defined above (next to SectionShard) so the
    // shard struct can own its writeBufs map. The buffers themselves
    // live in sectionShards_[*].writeBufs; this counter is a global
    // fast-path "is anything buffered anywhere?" hint that lets
    // readers (overlayWriteBuf / tryReadFromWriteBuf{,Flat}) skip the
    // shard mutex when no writer has buffered anything yet.
    std::atomic<size_t>                      sectionWriteBufsCount_{0};
    int writeFd_ = -1;  // file descriptor for pwrite-based writes

    // SQ8 quantize: float32[dim] → [min(4B), max(4B), uint8[dim]]
    static void quantize(const float* vec, size_t dim, char* record) {
        float minVal = vec[0], maxVal = vec[0];
        for (size_t i = 1; i < dim; ++i) {
            if (vec[i] < minVal) minVal = vec[i];
            if (vec[i] > maxVal) maxVal = vec[i];
        }
        float range = maxVal - minVal;
        if (range < 1e-10f) range = 1e-10f;
        std::memcpy(record, &minVal, sizeof(float));
        std::memcpy(record + sizeof(float), &maxVal, sizeof(float));
        float scale = 255.0f / range;
        uint8_t* qdata = reinterpret_cast<uint8_t*>(record + 2 * sizeof(float));
        for (size_t i = 0; i < dim; ++i) {
            float normalized = (vec[i] - minVal) * scale;
            int q = static_cast<int>(normalized + 0.5f);
            qdata[i] = static_cast<uint8_t>(std::min(255, std::max(0, q)));
        }
    }

    // SQ8 dequantize: [min(4B), max(4B), uint8[dim]] → float32[dim]
    static void dequantize(const char* record, float* vec, size_t dim) {
        float minVal, maxVal;
        std::memcpy(&minVal, record, sizeof(float));
        std::memcpy(&maxVal, record + sizeof(float), sizeof(float));
        float invScale = (maxVal - minVal) / 255.0f;
        const uint8_t* qdata = reinterpret_cast<const uint8_t*>(record + 2 * sizeof(float));
        for (size_t i = 0; i < dim; ++i) {
            vec[i] = static_cast<float>(qdata[i]) * invScale + minVal;
        }
    }

private:
    // Open or create the underlying file.
    void openFile() {
        fileStream_.open(filePath_, std::ios::in | std::ios::out | std::ios::binary);
        if (!fileStream_.is_open()) {
            // Try to create a new file
            fileStream_.open(filePath_, std::ios::out | std::ios::binary | std::ios::trunc);
            if (!fileStream_.is_open()) {
                throw std::runtime_error("Failed to create vector file.");
            }
            fileStream_.close();
            fileStream_.open(filePath_, std::ios::in | std::ios::out | std::ios::binary);
            if (!fileStream_.is_open()) {
                throw std::runtime_error("Failed to reopen vector file.");
            }
        }
        readFd_ = ::open(filePath_.c_str(), O_RDONLY);
        writeFd_ = ::open(filePath_.c_str(), O_WRONLY | O_CREAT, 0644);
    }

    void ensureDeleteFileSize(size_t targetSize) {
        deleteStream_.clear();
        deleteStream_.seekp(0, std::ios::end);
        size_t currentSize = static_cast<size_t>(deleteStream_.tellp());
        if (currentSize < targetSize) {
            deleteStream_.seekp(static_cast<std::streamoff>(targetSize - 1),
                                std::ios::beg);
            char zero = 0;
            deleteStream_.write(&zero, 1);
            deleteStream_.flush();
        }
    }

    void reloadDeleteFlags(size_t targetSize) {
        ensureDeleteFileSize(targetSize);
        deletedFlags_.assign(targetSize, 0);
        deleteStream_.clear();
        deleteStream_.seekg(0, std::ios::beg);
        deleteStream_.read(reinterpret_cast<char*>(deletedFlags_.data()),
                           static_cast<std::streamsize>(deletedFlags_.size()));
    }

    void openDeleteFile() {
        deleteStream_.open(deleteFilePath_,
                           std::ios::in | std::ios::out | std::ios::binary);
        if (!deleteStream_.is_open()) {
            deleteStream_.open(deleteFilePath_,
                               std::ios::out | std::ios::binary | std::ios::trunc);
            if (!deleteStream_.is_open()) {
                throw std::runtime_error("Failed to create delete marker file.");
            }
            deleteStream_.close();
            deleteStream_.open(deleteFilePath_,
                               std::ios::in | std::ios::out | std::ios::binary);
            if (!deleteStream_.is_open()) {
                throw std::runtime_error("Failed to open delete marker file.");
            }
        }

        reloadDeleteFlags(totalVectors_);
    }

    void writeDeleteFlag(node_id_t id, bool deleted) {
        deletedFlags_[static_cast<size_t>(id)] = deleted ? 1 : 0;
        deleteStream_.clear();
        deleteStream_.seekp(static_cast<std::streamoff>(id), std::ios::beg);
        char value = deleted ? 1 : 0;
        deleteStream_.write(&value, 1);
        deleteStream_.flush();
        if (!deleteStream_.good()) {
            throw std::runtime_error("Failed to update delete marker file.");
        }
    }

    // Grow the dense direct-id arrays + the on-disk delete-marker file.
    // Concurrent-writer-refactor-plan §5.3: takes pages_alloc_mu_ so that
    // a writer growing `pages_` and another writer growing the dense
    // location arrays don't both call non-thread-safe std::vector::resize
    // simultaneously, and so that a racing reader that briefly observes
    // a vector mid-grow is impossible (Phase 6 readers will acquire
    // pages_alloc_mu_ shared via a future shared_mutex; for Phase 1 the
    // lifecycle gate already excludes readers during writes).
    void expandCapacity(size_t minCapacity) {
        std::unique_lock<std::shared_mutex> g(pages_alloc_mu_);
        expandCapacityLocked(minCapacity);
    }

    // Pre-locked variant for callers that already hold pages_alloc_mu_
    // (e.g., allocateSlotForSection's "new page" path which grows
    // `pages_` and may also need to grow direct-id arrays).
    void expandCapacityLocked(size_t minCapacity) {
        if (minCapacity <= totalVectors_) {
            return;
        }

        size_t target = totalVectors_ == 0 ? minCapacity : totalVectors_;
        while (target < minCapacity) {
            target *= 2;
        }

        deleteStream_.clear();
        deleteStream_.seekp(0, std::ios::end);
        size_t currentSize = static_cast<size_t>(deleteStream_.tellp());
        if (currentSize < target) {
            deleteStream_.seekp(static_cast<std::streamoff>(target - 1), std::ios::beg);
            char zero = 0;
            deleteStream_.write(&zero, 1);
            deleteStream_.flush();
            if (!deleteStream_.good()) {
                throw std::runtime_error("Failed to expand delete marker file.");
            }
        }

        idToPage_.resize(target, -1);
        idToSlotInPage_.resize(target, 0);
        deletedFlags_.resize(target, 0);
        totalVectors_ = target;
        // Mirror to the atomic field for the future packed-atomic chunk
        // array; no longer load-bearing for correctness now that
        // page_of / slot_of take pages_alloc_mu_ shared again.
        totalVectors_atomic_.store(target, std::memory_order_release);
    }

    // Ensure that the file is large enough to contain pageId (0-based)
    void ensurePageAllocated(size_t pageId) {
        if (writeFd_ < 0) return;
        size_t requiredSize = (pageId + 1) * kPageSize;
        struct stat st;
        if (::fstat(writeFd_, &st) == 0 &&
            static_cast<size_t>(st.st_size) < requiredSize) {
            char zero = 0;
            ::pwrite(writeFd_, &zero, 1,
                     static_cast<off_t>(requiredSize - 1));
        }
    }

    // Flush one section write buffer to disk via pwrite (one 4KB aligned write).
    // Write-path; called under the external per-index exclusive lock, so
    // no readers are concurrent. The unique_lock here is defence-in-depth
    // for the page cache structure itself.
    void flushSectionWriteBuf(SectionWriteBuf& swb) {
        off_t offset = static_cast<off_t>(swb.pageId * kPageSize);
        if (writeFd_ >= 0) {
            ::pwrite(writeFd_, swb.data.data(), kPageSize, offset);
        }
        // Update read cache if page is cached
        std::unique_lock<std::shared_mutex> lk(page_mu_);
        auto cit = pageCache_.find(swb.pageId);
        if (cit != pageCache_.end()) {
            std::memcpy(cit->second.data.data(), swb.data.data(), kPageSize);
        }
    }

    // Write a vector into the section's write buffer.
    // When the page is full, flush the 4KB buffer to disk in one write.
    //
    // internal review (build scaling): writeBufs is per-shard, so
    // different sections (different shards) write in parallel. Same-
    // shard accesses still serialise on shard.mu, which also covers
    // openPages/freeSlots — but allocateSlotForSection and
    // writeToSectionBuffer are called sequentially for the same section
    // (in storeVectorToDisk), so combining the lock doesn't worsen
    // serialisation for same-section traffic.
    void writeToSectionBuffer(int sectionIdx, size_t pageId, uint16_t slot,
                              const std::vector<float>& vec) {
        // Lock-order rule:
        //   shard.mu is acquired BELOW; page_mu_ (via
        //   updateCacheAfterWrite) is acquired AFTER releasing
        //   shard.mu. Readers (readVectorsBatchFlat pass 1) take
        //   page_mu_ shared FIRST and never re-enter any shard.mu
        //   under that lock, so the two paths cannot deadlock.
        SectionShard& shard = section_shard_state(sectionIdx);
        {
            std::lock_guard<std::mutex> g(shard.mu);

            auto it = shard.writeBufs.find(sectionIdx);
            if (it == shard.writeBufs.end()) {
                SectionWriteBuf swb;
                swb.pageId = pageId;
                swb.data.resize(kPageSize, 0);
                it = shard.writeBufs.emplace(sectionIdx, std::move(swb)).first;
                sectionWriteBufsCount_.fetch_add(1, std::memory_order_relaxed);
            }

            auto& swb = it->second;
            // If the section moved to a new page, flush the old page first
            if (swb.pageId != pageId) {
                flushSectionWriteBuf(swb);
                swb.pageId = pageId;
                std::fill(swb.data.begin(), swb.data.end(), 0);
            }

            // Quantize vector into the buffer
            size_t offsetInPage = static_cast<size_t>(slot) * recordSize_;
            quantize(vec.data(), dim_, swb.data.data() + offsetInPage);

            // If page is full, flush to disk and free the buffer
            if (slot + 1 >= vectorsPerPage_) {
                flushSectionWriteBuf(swb);
                shard.writeBufs.erase(it);
                sectionWriteBufsCount_.fetch_sub(1, std::memory_order_relaxed);
            }
        }

        // Page-cache update OUTSIDE shard.mu to avoid the lock-order
        // inversion noted above. updateCacheAfterWrite takes page_mu_
        // exclusive on its own.
        updateCacheAfterWrite(pageId, slot, vec);
    }

    // Look up pages_[pageId].sectionIdx safely under pages_alloc_mu_.
    // Returns -1 if pageId is out of range (concurrent allocator may
    // have grown pages_ between caller and us, but we conservatively
    // miss in that case — the entry can't be in sectionWriteBufs_ yet).
    int section_idx_of_page_locked(size_t pageId) const {
        std::shared_lock<std::shared_mutex> g(pages_alloc_mu_);
        if (pageId >= pages_.size()) return -1;
        return pages_[pageId].sectionIdx;
    }

    // Overlay write buffer content onto a page buffer loaded from disk.
    // Uses pages_[pageId].sectionIdx for O(1) lookup instead of scanning all buffers.
    void overlayWriteBuf(size_t pageId, PageBuf& buf) const {
        if (sectionWriteBufsCount_.load(std::memory_order_relaxed) == 0) return;
        int sectionIdx = section_idx_of_page_locked(pageId);
        if (sectionIdx < 0) return;
        const SectionShard& shard = section_shard_state(sectionIdx);
        std::lock_guard<std::mutex> g(shard.mu);
        auto it = shard.writeBufs.find(sectionIdx);
        if (it != shard.writeBufs.end() && it->second.pageId == pageId) {
            std::memcpy(buf.data.data(), it->second.data.data(), kPageSize);
        }
    }

    // Try to read a vector from the section write buffer (for unflushed pages).
    bool tryReadFromWriteBuf(size_t pageId, uint16_t slot, std::vector<float>& vec) const {
        if (sectionWriteBufsCount_.load(std::memory_order_relaxed) == 0) return false;
        int sectionIdx = section_idx_of_page_locked(pageId);
        if (sectionIdx < 0) return false;
        const SectionShard& shard = section_shard_state(sectionIdx);
        std::lock_guard<std::mutex> g(shard.mu);
        auto it = shard.writeBufs.find(sectionIdx);
        if (it == shard.writeBufs.end() || it->second.pageId != pageId) return false;
        size_t offsetInPage = static_cast<size_t>(slot) * recordSize_;
        if (vec.size() != dim_) vec.resize(dim_);
        dequantize(it->second.data.data() + offsetInPage, vec.data(), dim_);
        return true;
    }

    // Flat version for batch reads: dequantize into a float* buffer.
    bool tryReadFromWriteBufFlat(size_t pageId, uint16_t slot, float* out, size_t dim) const {
        if (sectionWriteBufsCount_.load(std::memory_order_relaxed) == 0) return false;
        int sectionIdx = section_idx_of_page_locked(pageId);
        if (sectionIdx < 0) return false;
        const SectionShard& shard = section_shard_state(sectionIdx);
        std::lock_guard<std::mutex> g(shard.mu);
        auto it = shard.writeBufs.find(sectionIdx);
        if (it == shard.writeBufs.end() || it->second.pageId != pageId) return false;
        size_t offsetInPage = static_cast<size_t>(slot) * recordSize_;
        dequantize(it->second.data.data() + offsetInPage, out, dim);
        return true;
    }

    // Evict pages if cache size exceeds the limit (FIFO). Caller must
    // ensure the cache mutex is appropriately held; this is called from
    // setMaxCachedPages and from inside loadPageToCache (which already
    // holds the unique lock).
    void evictIfNeeded_locked() {
        if (maxCachedPages_ == 0) return;
        while (pageCache_.size() > maxCachedPages_) {
            size_t victim = pageOrder_.front();
            pageOrder_.pop_front();
            pageCache_.erase(victim);
        }
    }

    // Load a full 4KB page into the cache.
    //
    // Concurrency (A1.4 — port of f8110f0, merged with our page_buf_recycle):
    // disk I/O runs WITHOUT page_mu_ held, so concurrent misses on
    // different pages parallelize. Cache lookup, eviction, and insert
    // run under unique page_mu_. Double-checked locking handles the
    // rare race where another thread loaded the same page while we
    // were preading: we drop our buffer.
    //
    // 20260516_page_buf_recycle: when the cache is full, extract the
    // victim's node_handle (C++17) and reuse its 4 KB PageBuf
    // allocation instead of erase + new alloc. Saves the malloc/free
    // round-trip on every miss past warmup.
    //
    // Short-read uniformity: pread returning 0, -1, or partial count
    // is handled the same way — `got` is clamped to [0, kPageSize] and
    // the tail is memset to zero so callers see well-defined bytes.
    //
    // TODO(A1.8/A1.16): once per-section mutexes land, overlayWriteBuf
    // must take section_mu_[section] in shared mode BEFORE re-acquiring
    // page_mu_ to avoid the section→cache lock-order deadlock risk
    // documented in the design docs §0 / internal review. At this
    // step in the port the section state is still guarded by the
    // external per-index lock, so the simple ordering is safe.
    void loadPageToCache(size_t pageId) {
        if (maxCachedPages_ == 0) return;             // caching disabled

        // (1) Fast check under shared lock.
        {
            std::shared_lock<std::shared_mutex> lk(page_mu_);
            if (pageCache_.find(pageId) != pageCache_.end()) return;
        }

        // (2) pread WITHOUT cache lock — multiple thread misses parallelize.
        PageBuf buf;
        buf.data.resize(kPageSize, 0);

        size_t offset = pageId * kPageSize;
        auto fill_buf = [&](char* dst) {
            ssize_t n = 0;
            if (readFd_ >= 0) {
                n = ::pread(readFd_, dst, kPageSize, static_cast<off_t>(offset));
            } else {
                // Fallback fileStream_ is not thread-safe; in practice
                // readFd_ is always set. Guard with the page cache mutex
                // on the slow path so we don't crash if it isn't.
                std::unique_lock<std::shared_mutex> lk(page_mu_);
                fileStream_.clear();
                fileStream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
                fileStream_.read(dst, static_cast<std::streamsize>(kPageSize));
                n = static_cast<ssize_t>(fileStream_.gcount());
            }
            size_t got = (n > 0) ? static_cast<size_t>(n) : 0;
            if (got < kPageSize) std::memset(dst + got, 0, kPageSize - got);
        };
        fill_buf(buf.data.data());

        // A1.16: overlayWriteBuf now takes sectionWriteBufsMu_
        // internally, so it is safe under concurrent
        // writeToSectionBuffer (internal review). The earlier TODO
        // about lock-order between section_mu_ and page_mu_ is
        // resolved by pushing the section lock inside overlayWriteBuf.
        overlayWriteBuf(pageId, buf);

        // (3) Re-acquire unique; re-check, then recycle or insert.
        std::unique_lock<std::shared_mutex> lk(page_mu_);
        if (pageCache_.find(pageId) != pageCache_.end()) return;   // raced

        if (pageCache_.size() >= maxCachedPages_) {
            // 20260516_page_buf_recycle: extract victim node_handle to
            // reuse its 4 KB allocation. Move our fresh buf into it.
            size_t victim = pageOrder_.front();
            pageOrder_.pop_front();
            auto node = pageCache_.extract(victim);
            if (!node.empty()) {
                node.mapped() = std::move(buf);
                node.key() = pageId;
                pageCache_.insert(std::move(node));
                pageOrder_.push_back(pageId);
                return;
            }
            // Fallthrough if extract failed (shouldn't happen — pageOrder_
            // is the canonical source of victim ids).
        }
        pageCache_.emplace(pageId, std::move(buf));
        pageOrder_.push_back(pageId);
    }

    // Try to read a vector from cache by pageId & slot. Dequantizes SQ8 → float32.
    // Takes shared lock for the cache lookup; copy/dequantize happens
    // while holding the lock so the cached buffer cannot be evicted from
    // under us.
    bool tryReadFromCache(size_t pageId, uint16_t slot, std::vector<float>& vec) {
        if (maxCachedPages_ == 0) return false;
        std::shared_lock<std::shared_mutex> lk(page_mu_);
        auto it = pageCache_.find(pageId);
        if (it == pageCache_.end()) return false;

        const PageBuf& buf = it->second;
        size_t offsetInPage = static_cast<size_t>(slot) * recordSize_;
        if (offsetInPage + recordSize_ > buf.data.size()) {
            return false;
        }
        if (vec.size() != dim_) vec.resize(dim_);
        dequantize(buf.data.data() + offsetInPage, vec.data(), dim_);
        return true;
    }

    // Direct read of a single record into caller buffer (no cache interaction). Dequantizes.
    void directReadRecord(size_t pageId, uint16_t slot, float* out) {
        // Check write buffer first
        if (tryReadFromWriteBufFlat(pageId, slot, out, dim_)) return;
        std::vector<char> tmp(recordSize_);
        size_t offset = pageId * kPageSize + static_cast<size_t>(slot) * recordSize_;
        if (readFd_ >= 0) {
            ::pread(readFd_, tmp.data(), recordSize_, static_cast<off_t>(offset));
        } else {
            fileStream_.clear();
            fileStream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
            fileStream_.read(tmp.data(), static_cast<std::streamsize>(recordSize_));
        }
        dequantize(tmp.data(), out, dim_);
    }

    // Write a single record (one vector) at (pageId, slot) to disk. Quantizes to SQ8.
    // Write-path; called under the external per-index exclusive lock.
    void writeRecord(size_t pageId, uint16_t slot, const std::vector<float>& vec) {
        if (vec.size() != dim_) {
            throw std::runtime_error("Vector size mismatch in writeRecord.");
        }
        std::vector<char> tmp(recordSize_);
        quantize(vec.data(), dim_, tmp.data());
        size_t offset = pageId * kPageSize + static_cast<size_t>(slot) * recordSize_;
        fileStream_.clear();
        fileStream_.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
        fileStream_.write(tmp.data(), static_cast<std::streamsize>(recordSize_));
        if (!fileStream_.good()) {
            throw std::runtime_error("Failed to write vector to file.");
        }
        fileStream_.flush();
    }

    // If the page is cached, update the cached copy after a write (stores quantized SQ8).
    // Write-path; called under the external per-index exclusive lock.
    // Takes unique_lock as defence-in-depth against future relaxation of
    // the contract.
    void updateCacheAfterWrite(size_t pageId, uint16_t slot, const std::vector<float>& vec) {
        if (maxCachedPages_ == 0) return;
        std::unique_lock<std::shared_mutex> lk(page_mu_);
        auto it = pageCache_.find(pageId);
        if (it == pageCache_.end()) return;

        PageBuf& buf = it->second;
        size_t offsetInPage = static_cast<size_t>(slot) * recordSize_;
        if (offsetInPage + recordSize_ > buf.data.size()) {
            return;
        }
        quantize(vec.data(), dim_, buf.data.data() + offsetInPage);
    }

    // Allocate a free slot for a given sectionKey (creates section/pages on demand).
    // Two-phase allocation per concurrent-writer-refactor-plan §5.3:
    //   Phase A: section-local — under section_shard(sectionIdx).
    //            If a free slot or an open page is available, allocate
    //            from there.
    //   Phase B: page allocator — under pages_alloc_mu_. Grow `pages_`
    //            and the on-disk file. Then re-enter the section shard
    //            to publish the new page into sectionOpenPages_.
    //
    // Different sections (different shards) can run Phase A in parallel.
    // Same-section concurrent allocations serialise on the section shard
    // for ~µs of bookkeeping; the actual record write happens later,
    // unlocked. Phase B serialises ALL fresh-page allocations across
    // sections (`pages_.push_back` is the constraint), but new pages
    // are rare relative to slot allocations so this is rarely the
    // bottleneck.
    // R3.b / internal review: returns (pageId, slot, sectionIdx). The
    // sectionIdx is returned to avoid a second unlocked read of
    // sectionKeyToIdx_ at the caller (storeVectorToDisk) which would be
    // UB while concurrent writers may rehash the map.
    std::tuple<size_t,uint16_t,int> allocateSlotForSection(node_id_t sectionKey) {
        // Section-key → section-idx, under a tiny critical section
        // (read-mostly, writes only on first vector of a new section).
        int sectionIdx;
        {
            std::lock_guard<std::mutex> g(section_index_mu_);
            auto it = sectionKeyToIdx_.find(sectionKey);
            if (it == sectionKeyToIdx_.end()) {
                sectionIdx = static_cast<int>(sectionIdxToKey_.size());
                sectionKeyToIdx_[sectionKey] = sectionIdx;
                sectionIdxToKey_.push_back(sectionKey);
            } else {
                sectionIdx = it->second;
            }
        }

        // Phase A: try free-slot or open-page reuse under the section
        // shard mutex. The shard owns its own maps, so all map
        // mutations (operator[] inserts a new sectionIdx entry,
        // potential rehash) happen under this single mutex.
        {
            SectionShard& shard = section_shard_state(sectionIdx);
            std::lock_guard<std::mutex> g(shard.mu);

            auto& freeList = shard.freeSlots[sectionIdx];
            if (!freeList.empty()) {
                auto [pageId, slot] = freeList.back();
                freeList.pop_back();
                return {pageId, slot, sectionIdx};
            }

            auto& openList = shard.openPages[sectionIdx];
            if (!openList.empty()) {
                size_t pageId = openList.back();
                openList.pop_back();
                // pages_ is a std::deque so reads/mutations of
                // pages_[pageId] are stable while another shard's
                // writer push_backs a new page. The section shard
                // mutex serialises any access to pages_[pageId]
                // belonging to THIS section.
                PageMeta& meta = pages_[pageId];
                if (meta.usedSlots < vectorsPerPage_) {
                    uint16_t slot = meta.usedSlots++;
                    if (meta.usedSlots < vectorsPerPage_) {
                        openList.push_back(pageId);
                    }
                    return {pageId, slot, sectionIdx};
                }
                // Defensive fallback (should not happen): page was full,
                // fall through to fresh-page allocation.
            }
        }

        // Phase B: allocate a fresh page for this section under the
        // global pages allocator mutex.
        size_t newPageId;
        {
            std::unique_lock<std::shared_mutex> a(pages_alloc_mu_);
            newPageId = pages_.size();
            pages_.push_back(PageMeta{sectionIdx, /*usedSlots=*/1});
            ensurePageAllocated(newPageId);
        }

        // Publish the new page into the per-shard openPages map.
        // The section shard mutex protects the map's structure
        // (since it lives inside the shard struct).
        if (vectorsPerPage_ > 1) {
            SectionShard& shard = section_shard_state(sectionIdx);
            std::lock_guard<std::mutex> g(shard.mu);
            shard.openPages[sectionIdx].push_back(newPageId);
        }

        return {newPageId, /*slot=*/0, sectionIdx};
    }

public:
    // Constructor
    PagedVectorStorage(const std::string& path,
                  size_t dim,
                  size_t capacity = 1000000,
                  size_t maxCachedPages = 128)
        : filePath_(path),
          deleteFilePath_(path + ".deleted"),
          dim_(dim),
          recordSize_(dim + 2 * sizeof(float)),  // SQ8: dim bytes quantized + 8 bytes min/max
          totalVectors_(capacity),
          totalVectors_atomic_(capacity),
          vectorsPerPage_(0),
          maxCachedPages_(maxCachedPages)
    {
        if (dim_ == 0) {
            throw std::runtime_error("Vector dimension must be > 0.");
        }
        if (recordSize_ > kPageSize) {
            throw std::runtime_error("Vector too large to fit into one page.");
        }

        vectorsPerPage_ = kPageSize / recordSize_;
        if (vectorsPerPage_ == 0) {
            throw std::runtime_error("Invalid vectorsPerPage (0).");
        }

        openFile();
        openDeleteFile();

        idToPage_.assign(totalVectors_, -1);
        idToSlotInPage_.assign(totalVectors_, 0);
    }

    ~PagedVectorStorage() {
        flushWrites();
        if (writeFd_ >= 0) { ::close(writeFd_); writeFd_ = -1; }
        if (readFd_ >= 0) { ::close(readFd_); readFd_ = -1; }
        if (fileStream_.is_open()) {
            fileStream_.close();
        }
        if (deleteStream_.is_open()) {
            deleteStream_.close();
        }
    }

    size_t getVectorDim() const override { return dim_; }
    size_t getCapacity() const override { return totalVectors_; }
    size_t vectorsPerPage() const { return vectorsPerPage_; }

    void setMaxCachedPages(size_t m) {
        std::unique_lock<std::shared_mutex> lk(page_mu_);
        maxCachedPages_ = m;
        evictIfNeeded_locked();
    }

    // Flush all remaining section write buffers (partially filled pages).
    // Runs under the engine's exclusive lifecycle gate, but also takes
    // each shard.mu for explicit serialisation against any in-flight
    // writeToSectionBuffer (defence-in-depth).
    void flushWrites() override {
        for (auto& shard : sectionShards_) {
            std::lock_guard<std::mutex> g(shard.mu);
            for (auto& [sectionIdx, swb] : shard.writeBufs) {
                flushSectionWriteBuf(swb);
            }
            shard.writeBufs.clear();
        }
        sectionWriteBufsCount_.store(0, std::memory_order_relaxed);
    }

    // Store vector with a section hint: 'sectionKey' is derived from HNSW Level-1 entry.
    // We assign the vector to some page belonging to that section; pages are always
    // 4KB and vectors never cross page boundaries.
    void storeVectorToDisk(node_id_t id,
                           const std::vector<float>& vec,
                           node_id_t sectionKey) override
    {
        ensure_dense_capacity(id);   // grows direct-id arrays if needed; no-op for update ids
        if (vec.size() != dim_) {
            throw std::runtime_error("Vector size mismatch.");
        }

        int64_t curPage = page_of(id);

        if (curPage >= 0) {
            // Overwrite existing slot
            uint16_t curSlot = slot_of(id);
            writeRecord(static_cast<size_t>(curPage), curSlot, vec);
            updateCacheAfterWrite(static_cast<size_t>(curPage), curSlot, vec);
            mark_deleted_at(id, false);
            return;
        }

        // R3.b / internal review: allocateSlotForSection returns
        // (pageId, slot, sectionIdx). Avoids a second unlocked read of
        // sectionKeyToIdx_ that would race against concurrent inserts
        // (unordered_map concurrent read+write is UB; same map is
        // mutated inside the function with section_index_mu_ held).
        auto [pageId, slot, sectionIdx] = allocateSlotForSection(sectionKey);

        // Phase 6 publish protocol (§5.4, internal review): write bytes
        // BEFORE publishing the location. Readers that subsequently
        // observe page_of(id) >= 0 are guaranteed to find durable bytes
        // (either in the section write buffer or on disk).
        if (sectionIdx >= 0 && writeFd_ >= 0) {
            writeToSectionBuffer(sectionIdx, pageId, slot, vec);
        } else {
            writeRecord(pageId, slot, vec);
            updateCacheAfterWrite(pageId, slot, vec);
        }
        mark_deleted_at(id, false);

        // Publish LAST. Acts as a release fence for the writes above.
        assign_location(id, pageId, slot);
    }

    // Backward-compatible version: if you don't care about sections,
    // use ID as the "sectionKey".
    void storeVectorToDisk(node_id_t id, const std::vector<float>& vec) override {
        storeVectorToDisk(id, vec, id);
    }

    // Read a vector by its logical ID.
    void readVectorFromDisk(node_id_t id, std::vector<float>& vec) override
    {
        int64_t page = page_of(id);
        if (page < 0) {
            std::fprintf(stderr, "Problematic id: %lu\n", static_cast<unsigned long>(id));
            throw std::runtime_error("Vector slot not assigned for this ID.");
        }
        uint16_t slot = slot_of(id);
        size_t pageId = static_cast<size_t>(page);

        // 0) Try unflushed write buffer first
        if (tryReadFromWriteBuf(pageId, slot, vec)) {
            return;
        }

        // 1) Try page cache
        if (maxCachedPages_ > 0) {
            if (tryReadFromCache(pageId, slot, vec)) {
                pageCacheHits_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            pageCacheMisses_.fetch_add(1, std::memory_order_relaxed);
        }

        // 2) If not cached, optionally cache the page then try again
        if (maxCachedPages_ > 0) {
            loadPageToCache(pageId);
            if (tryReadFromCache(pageId, slot, vec)) {
                return;
            }
        }

        // 3) Fallback: direct disk read of this record + dequantize
        if (vec.size() != dim_) vec.resize(dim_);
        std::vector<char> tmp(recordSize_);
        size_t offset = pageId * kPageSize + static_cast<size_t>(slot) * recordSize_;
        fileStream_.clear();
        fileStream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        fileStream_.read(tmp.data(), static_cast<std::streamsize>(recordSize_));
        if (!fileStream_.good()) {
            throw std::runtime_error("Failed to read vector from file.");
        }
        dequantize(tmp.data(), vec.data(), dim_);
    }

    void deleteVector(node_id_t id) override
    {
        // C6 / C5a: Delete is a no-op at the storage layer. Tombstoning lives
        // in LSMVec::tombstoned_internal_ids_; the slot stays bound so
        // tombstoned nodes can still be read for routing (tombstone-as-router).
        // The pre-C5a slot-freelist (sectionFreeSlots_) and the deletedFlags_
        // bit-write are intentionally dropped: V1 has no slot reuse and the
        // canonical "is this id deleted?" state is owned by LSMVec.
        (void)id;
    }

    bool exists(node_id_t id) const override
    {
        if (is_deleted_at(id)) return false;
        return page_of(id) >= 0;
    }

    // Prefetch pages corresponding to given vector IDs.
    // This simply ensures the relevant pages are loaded into the page cache.
    void prefetchByIds(const std::vector<node_id_t>& ids) override {
        if (maxCachedPages_ == 0) return;

        std::unordered_map<size_t, bool> seenPage;
        for (node_id_t id : ids) {
            int64_t page = page_of(id);
            if (page < 0) continue;
            size_t pageId = static_cast<size_t>(page);
            if (seenPage.emplace(pageId, true).second) {
                loadPageToCache(pageId);
            }
        }
    }

    // Batch read: groups IDs by page, loads each page once, then reads all
    // vectors from that page.  Eliminates redundant page lookups for vectors
    // sharing the same page (128-dim = 8 vectors/page).
    void readVectorsBatch(const std::vector<node_id_t>& ids,
                          std::vector<std::vector<float>>& out) override {
        out.resize(ids.size());

        // Group indices by pageId
        std::unordered_map<size_t, std::vector<size_t>> pageToIndices;
        for (size_t i = 0; i < ids.size(); ++i) {
            int64_t page = page_of(ids[i]);
            if (page < 0) {
                throw std::runtime_error("Vector slot not assigned in batch read.");
            }
            pageToIndices[static_cast<size_t>(page)].push_back(i);
        }

        // For each page, load it once and read all vectors from it
        for (auto& [pageId, indices] : pageToIndices) {
            loadPageToCache(pageId);

            for (size_t idx : indices) {
                uint16_t slot = slot_of(ids[idx]);
                if (!tryReadFromCache(pageId, slot, out[idx])) {
                    // Fallback: direct read + dequantize
                    if (out[idx].size() != dim_) out[idx].resize(dim_);
                    directReadRecord(pageId, slot, out[idx].data());
                }
            }
        }
    }

    // Flat-buffer two-pass batch read: avoids per-call map construction
    // and per-vector heap allocations. The scratch vectors are
    // thread_local — capacity stays warm across calls within one worker
    // thread, but each thread has its own copy so two readers don't
    // collide on the buffer.
    void readVectorsBatchFlat(const std::vector<node_id_t>& ids,
                              float* out, size_t dim) override {
        thread_local std::vector<MissEntry> miss_scratch;
        thread_local std::vector<size_t>    unique_page_scratch;
        miss_scratch.clear();

        // Pass 1a: try write buffer first WITHOUT page_mu_, collect
        // remaining for the cache-hit pass below. Phase 6 lock order
        // (internal review follow-up): tryReadFromWriteBufFlat takes
        // sectionWriteBufsMu_; writeToSectionBuffer also takes
        // sectionWriteBufsMu_ then page_mu_. To avoid AB/BA, readers
        // never enter sectionWriteBufsMu_ while holding page_mu_.
        for (size_t i = 0; i < ids.size(); ++i) {
            node_id_t id = ids[i];
            int64_t page = page_of(id);
            if (page < 0) {
                throw std::runtime_error("Vector slot not assigned in batch flat read.");
            }
            size_t pageId = static_cast<size_t>(page);
            uint16_t slot = slot_of(id);

            if (tryReadFromWriteBufFlat(pageId, slot, out + i * dim, dim_)) {
                continue;
            }
            miss_scratch.push_back({i, pageId, slot});
        }

        if (miss_scratch.empty()) return;

        // Pass 1b: cache-hit pass under page_mu_ shared. Move misses
        // requiring disk read into still_missing.
        thread_local std::vector<MissEntry> still_missing;
        still_missing.clear();
        {
            std::shared_lock<std::shared_mutex> lk(page_mu_);
            for (const auto& miss : miss_scratch) {
                if (maxCachedPages_ > 0) {
                    auto cit = pageCache_.find(miss.pageId);
                    if (cit != pageCache_.end()) {
                        size_t offsetInPage = static_cast<size_t>(miss.slot) * recordSize_;
                        dequantize(cit->second.data.data() + offsetInPage,
                                   out + miss.idx * dim, dim_);
                        pageCacheHits_.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                }
                pageCacheMisses_.fetch_add(1, std::memory_order_relaxed);
                still_missing.push_back(miss);
            }
        }
        miss_scratch.swap(still_missing);

        if (miss_scratch.empty()) return; // fast path: all cached

        // Pass 2: sort misses by pageId, load unique pages, then copy
        std::sort(miss_scratch.begin(), miss_scratch.end(),
                  [](const MissEntry& a, const MissEntry& b) {
                      return a.pageId < b.pageId;
                  });

        // Deduplicate pages and load them. loadPageToCache handles its
        // own locking (double-check + unique_lock for insertion).
        unique_page_scratch.clear();
        for (size_t j = 0; j < miss_scratch.size(); ++j) {
            if (j == 0 || miss_scratch[j].pageId != miss_scratch[j - 1].pageId) {
                unique_page_scratch.push_back(miss_scratch[j].pageId);
            }
        }

        for (size_t pageId : unique_page_scratch) {
            loadPageToCache(pageId);
        }

        // Dequantize from now-cached pages (or direct read as fallback).
        // Take shared lock once for the whole batch.
        {
            std::shared_lock<std::shared_mutex> lk(page_mu_);
            for (const auto& miss : miss_scratch) {
                auto cit = pageCache_.find(miss.pageId);
                if (cit != pageCache_.end()) {
                    size_t offsetInPage = static_cast<size_t>(miss.slot) * recordSize_;
                    dequantize(cit->second.data.data() + offsetInPage,
                               out + miss.idx * dim, dim_);
                } else {
                    // Cache full, page was evicted — direct read.
                    // directReadRecord does its own I/O without the page
                    // cache lock; safe to call while holding shared lock
                    // (it doesn't touch pageCache_ state).
                    directReadRecord(miss.pageId, miss.slot,
                                     out + miss.idx * dim);
                }
            }
        }
    }

    PageCacheStats getPageCacheStats() const override {
        return PageCacheStats{
            pageCacheHits_.load(std::memory_order_relaxed),
            pageCacheMisses_.load(std::memory_order_relaxed)};
    }

    bool serializeMetadata(std::ostream& out) const {
        // v1: original (totalVectors, pages, sections, idToPage, idToSlotInPage)
        // v2: + update_locations_ sparse map (C7 robust delete)
        constexpr uint32_t kMetaVersion = 2;
        auto write = [&](const auto& value) -> bool {
            out.write(reinterpret_cast<const char*>(&value), sizeof(value));
            return static_cast<bool>(out);
        };

        uint64_t totalVectors = static_cast<uint64_t>(totalVectors_);
        uint64_t dim = static_cast<uint64_t>(dim_);
        uint64_t vectorsPerPage = static_cast<uint64_t>(vectorsPerPage_);
        uint64_t pagesSize = static_cast<uint64_t>(pages_.size());
        uint64_t sectionKeysSize = static_cast<uint64_t>(sectionIdxToKey_.size());

        if (!write(kMetaVersion) ||
            !write(totalVectors) ||
            !write(dim) ||
            !write(vectorsPerPage) ||
            !write(pagesSize) ||
            !write(sectionKeysSize)) {
            return false;
        }

        for (const auto& page : pages_) {
            int32_t sectionIdx = static_cast<int32_t>(page.sectionIdx);
            uint16_t usedSlots = page.usedSlots;
            if (!write(sectionIdx) || !write(usedSlots)) {
                return false;
            }
        }

        for (node_id_t key : sectionIdxToKey_) {
            if (!write(key)) {
                return false;
            }
        }

        uint64_t idToPageSize = static_cast<uint64_t>(idToPage_.size());
        uint64_t idToSlotSize = static_cast<uint64_t>(idToSlotInPage_.size());
        if (!write(idToPageSize) || !write(idToSlotSize)) {
            return false;
        }
        // 20260516_narrow_idtopage: on-disk format keeps int64_t per element
        // for forward compatibility with v1/v2 readers; in-memory is int32_t.
        for (int32_t page : idToPage_) {
            int64_t widened = static_cast<int64_t>(page);
            if (!write(widened)) {
                return false;
            }
        }
        for (uint16_t slot : idToSlotInPage_) {
            if (!write(slot)) {
                return false;
            }
        }

        // v2: update_locations_ sparse map.
        uint64_t updateLocCount = static_cast<uint64_t>(update_locations_.size());
        if (!write(updateLocCount)) return false;
        for (const auto& kv : update_locations_) {
            uint64_t id   = static_cast<uint64_t>(kv.first);
            int64_t  page = kv.second.page;
            uint16_t slot = kv.second.slot;
            if (!write(id) || !write(page) || !write(slot)) return false;
        }

        return static_cast<bool>(out);
    }

    bool deserializeMetadata(std::istream& in) {
        constexpr uint32_t kMaxKnownVersion = 2;
        auto read = [&](auto* value) -> bool {
            in.read(reinterpret_cast<char*>(value), sizeof(*value));
            return static_cast<bool>(in);
        };

        uint32_t version = 0;
        uint64_t totalVectors = 0;
        uint64_t dim = 0;
        uint64_t vectorsPerPage = 0;
        uint64_t pagesSize = 0;
        uint64_t sectionKeysSize = 0;
        if (!read(&version) ||
            !read(&totalVectors) ||
            !read(&dim) ||
            !read(&vectorsPerPage) ||
            !read(&pagesSize) ||
            !read(&sectionKeysSize)) {
            return false;
        }

        if (version < 1 || version > kMaxKnownVersion) {
            return false;
        }
        if (dim != dim_ || vectorsPerPage != vectorsPerPage_) {
            return false;
        }

        if (totalVectors > totalVectors_) {
            expandCapacity(static_cast<size_t>(totalVectors));
        }

        pages_.clear();
        // (deque doesn't take reserve(); growth is per-block, no relocation)
        for (uint64_t i = 0; i < pagesSize; ++i) {
            int32_t sectionIdx = 0;
            uint16_t usedSlots = 0;
            if (!read(&sectionIdx) || !read(&usedSlots)) {
                return false;
            }
            if (usedSlots > vectorsPerPage_) {
                return false;
            }
            pages_.push_back(PageMeta{sectionIdx, usedSlots});
        }

        sectionIdxToKey_.clear();
        sectionIdxToKey_.reserve(static_cast<size_t>(sectionKeysSize));
        for (uint64_t i = 0; i < sectionKeysSize; ++i) {
            node_id_t key = 0;
            if (!read(&key)) {
                return false;
            }
            sectionIdxToKey_.push_back(key);
        }

        uint64_t idToPageSize = 0;
        uint64_t idToSlotSize = 0;
        if (!read(&idToPageSize) || !read(&idToSlotSize)) {
            return false;
        }
        if (idToPageSize != totalVectors || idToSlotSize != totalVectors) {
            return false;
        }

        idToPage_.assign(static_cast<size_t>(idToPageSize), -1);
        idToSlotInPage_.assign(static_cast<size_t>(idToSlotSize), 0);
        for (uint64_t i = 0; i < idToPageSize; ++i) {
            int64_t page = 0;
            if (!read(&page)) {
                return false;
            }
            // 20260516_narrow_idtopage: on-disk is still int64_t; narrow with
            // overflow guard. Out-of-int32 page values would mean > 8 TB of
            // vector data per logical id — implausible, but refuse rather
            // than silently truncate.
            if (page > static_cast<int64_t>(std::numeric_limits<int32_t>::max())
                || page < -1) {
                return false;
            }
            idToPage_[static_cast<size_t>(i)] = static_cast<int32_t>(page);
        }
        for (uint64_t i = 0; i < idToSlotSize; ++i) {
            uint16_t slot = 0;
            if (!read(&slot)) {
                return false;
            }
            idToSlotInPage_[static_cast<size_t>(i)] = slot;
        }

        sectionKeyToIdx_.clear();
        for (size_t idx = 0; idx < sectionIdxToKey_.size(); ++idx) {
            sectionKeyToIdx_[sectionIdxToKey_[idx]] = static_cast<int>(idx);
        }

        for (auto& s : sectionShards_) {
            std::lock_guard<std::mutex> g(s.mu);
            s.openPages.clear();
            s.freeSlots.clear();
        }

        std::vector<std::vector<uint8_t>> usedSlots;
        usedSlots.resize(pages_.size());
        for (size_t pageId = 0; pageId < pages_.size(); ++pageId) {
            usedSlots[pageId].assign(pages_[pageId].usedSlots, 0);
        }

        for (size_t id = 0; id < idToPage_.size(); ++id) {
            int64_t page = idToPage_[id];
            if (page < 0) {
                continue;
            }
            size_t pageId = static_cast<size_t>(page);
            if (pageId >= pages_.size()) {
                return false;
            }
            uint16_t slot = idToSlotInPage_[id];
            if (slot >= pages_[pageId].usedSlots) {
                return false;
            }
            if (slot < usedSlots[pageId].size()) {
                usedSlots[pageId][slot] = 1;
            }
        }

        for (size_t pageId = 0; pageId < pages_.size(); ++pageId) {
            const PageMeta& meta = pages_[pageId];
            if (meta.sectionIdx < 0) {
                continue;
            }
            SectionShard& shard = section_shard_state(meta.sectionIdx);
            std::lock_guard<std::mutex> g(shard.mu);
            if (static_cast<size_t>(meta.usedSlots) < vectorsPerPage_) {
                shard.openPages[meta.sectionIdx].push_back(pageId);
            }
            auto& freeList = shard.freeSlots[meta.sectionIdx];
            for (size_t slot = 0; slot < meta.usedSlots; ++slot) {
                if (slot < usedSlots[pageId].size() && usedSlots[pageId][slot] == 0) {
                    freeList.emplace_back(pageId, static_cast<uint16_t>(slot));
                }
            }
        }

        // v2: read update_locations_ sparse map. v1 files have no such block.
        update_locations_.clear();
        if (version >= 2) {
            uint64_t updateLocCount = 0;
            if (!read(&updateLocCount)) return false;
            update_locations_.reserve(static_cast<size_t>(updateLocCount));
            for (uint64_t i = 0; i < updateLocCount; ++i) {
                uint64_t id = 0;
                int64_t  page = 0;
                uint16_t slot = 0;
                if (!read(&id) || !read(&page) || !read(&slot)) return false;
                UpdateLoc& loc = update_locations_[static_cast<node_id_t>(id)];
                loc.page = page;
                loc.slot = slot;
            }
        }

        reloadDeleteFlags(totalVectors_);
        {
            std::unique_lock<std::shared_mutex> lk(page_mu_);
            pageCache_.clear();
            pageOrder_.clear();
        }
        pageCacheHits_.store(0, std::memory_order_relaxed);
        pageCacheMisses_.store(0, std::memory_order_relaxed);

        return static_cast<bool>(in);
    }
};
}
