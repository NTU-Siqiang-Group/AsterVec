#pragma once

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

    std::vector<PageMeta> pages_; // pageId -> meta

    // Section mappings: external sectionKey -> internal sectionIdx
    std::unordered_map<node_id_t,int> sectionKeyToIdx_;
    std::vector<node_id_t>            sectionIdxToKey_;

    // For each sectionIdx, list of pages that still have free slots
    std::unordered_map<int,std::vector<size_t>> sectionOpenPages_;
    std::unordered_map<int,std::vector<std::pair<size_t,uint16_t>>> sectionFreeSlots_;

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

    // ----- Location dispatch helpers (V1, see DELETE_DESIGN.md §0.3) -----
    // These transparently route direct ids to dense arrays and update ids to
    // the sparse hashmap, so the public methods don't have to branch.

    int64_t page_of(node_id_t id) const {
        if (is_direct_id(id)) {
            if (static_cast<size_t>(id) >= totalVectors_) return -1;
            // 20260516_narrow_idtopage: idToPage_ is int32_t; widen for the API contract.
            return static_cast<int64_t>(idToPage_[static_cast<size_t>(id)]);
        }
        auto it = update_locations_.find(id);
        return (it == update_locations_.end()) ? -1 : it->second.page;
    }

    uint16_t slot_of(node_id_t id) const {
        if (is_direct_id(id)) {
            if (static_cast<size_t>(id) >= totalVectors_) return 0;
            return idToSlotInPage_[static_cast<size_t>(id)];
        }
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
            if (page > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error(
                    "PagedVectorStorage: page index exceeds int32_t ceiling "
                    "(8 TB) — rebuild with int64_t idToPage_ or split the "
                    "storage file");
            }
            idToPage_[static_cast<size_t>(id)]       = static_cast<int32_t>(page);
            idToSlotInPage_[static_cast<size_t>(id)] = slot;
        } else {
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

    // Page cache (FIFO) in units of full pages (4KB each)
    size_t maxCachedPages_;
    size_t pageCacheHits_ = 0;
    size_t pageCacheMisses_ = 0;

    struct PageBuf {
        std::vector<char> data; // always kPageSize bytes
    };

    std::unordered_map<size_t, PageBuf> pageCache_; // pageId -> data
    std::deque<size_t>                  pageOrder_; // FIFO order of pageIds

    int readFd_ = -1;  // file descriptor for pread-based I/O

    // Scratch buffers for readVectorsBatchFlat (avoid per-call allocations)
    struct MissEntry { size_t idx; size_t pageId; uint16_t slot; };
    std::vector<MissEntry> missScratch_;
    std::vector<size_t> uniquePageScratch_;

    // Reusable temp buffer for SQ8 record I/O
    mutable std::vector<char> sq8TempBuf_;

    // Section write buffer: per-section 4KB buffer for coalesced writes via pwrite
    struct SectionWriteBuf {
        std::vector<char> data;  // kPageSize bytes
        size_t pageId;
        uint16_t filledSlots;
    };
    std::unordered_map<int, SectionWriteBuf> sectionWriteBufs_; // sectionIdx -> buf
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

    void expandCapacity(size_t minCapacity) {
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
    void flushSectionWriteBuf(SectionWriteBuf& swb) {
        off_t offset = static_cast<off_t>(swb.pageId * kPageSize);
        if (writeFd_ >= 0) {
            ::pwrite(writeFd_, swb.data.data(), kPageSize, offset);
        }
        // Update read cache if page is cached
        auto cit = pageCache_.find(swb.pageId);
        if (cit != pageCache_.end()) {
            std::memcpy(cit->second.data.data(), swb.data.data(), kPageSize);
        }
    }

    // Write a vector into the section's write buffer.
    // When the page is full, flush the 4KB buffer to disk in one write.
    void writeToSectionBuffer(int sectionIdx, size_t pageId, uint16_t slot,
                              const std::vector<float>& vec) {
        auto it = sectionWriteBufs_.find(sectionIdx);
        if (it == sectionWriteBufs_.end()) {
            SectionWriteBuf swb;
            swb.pageId = pageId;
            swb.data.resize(kPageSize, 0);
            it = sectionWriteBufs_.emplace(sectionIdx, std::move(swb)).first;
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

        // Also update read cache if the page is cached (read consistency)
        updateCacheAfterWrite(pageId, slot, vec);

        // If page is full, flush to disk and free the buffer
        if (slot + 1 >= vectorsPerPage_) {
            flushSectionWriteBuf(swb);
            sectionWriteBufs_.erase(it);
        }
    }

    // Overlay write buffer content onto a page buffer loaded from disk.
    // Uses pages_[pageId].sectionIdx for O(1) lookup instead of scanning all buffers.
    void overlayWriteBuf(size_t pageId, PageBuf& buf) const {
        if (sectionWriteBufs_.empty() || pageId >= pages_.size()) return;
        int sectionIdx = pages_[pageId].sectionIdx;
        auto it = sectionWriteBufs_.find(sectionIdx);
        if (it != sectionWriteBufs_.end() && it->second.pageId == pageId) {
            std::memcpy(buf.data.data(), it->second.data.data(), kPageSize);
        }
    }

    // Try to read a vector from the section write buffer (for unflushed pages).
    bool tryReadFromWriteBuf(size_t pageId, uint16_t slot, std::vector<float>& vec) const {
        if (sectionWriteBufs_.empty()) return false;
        if (pageId >= pages_.size()) return false;
        int sectionIdx = pages_[pageId].sectionIdx;
        auto it = sectionWriteBufs_.find(sectionIdx);
        if (it == sectionWriteBufs_.end() || it->second.pageId != pageId) return false;
        size_t offsetInPage = static_cast<size_t>(slot) * recordSize_;
        if (vec.size() != dim_) vec.resize(dim_);
        dequantize(it->second.data.data() + offsetInPage, vec.data(), dim_);
        return true;
    }

    // Flat version for batch reads: dequantize into a float* buffer.
    bool tryReadFromWriteBufFlat(size_t pageId, uint16_t slot, float* out, size_t dim) const {
        if (sectionWriteBufs_.empty()) return false;
        if (pageId >= pages_.size()) return false;
        int sectionIdx = pages_[pageId].sectionIdx;
        auto it = sectionWriteBufs_.find(sectionIdx);
        if (it == sectionWriteBufs_.end() || it->second.pageId != pageId) return false;
        size_t offsetInPage = static_cast<size_t>(slot) * recordSize_;
        dequantize(it->second.data.data() + offsetInPage, out, dim);
        return true;
    }

    // Evict pages if cache size exceeds the limit (FIFO).
    void evictIfNeeded() {
        if (maxCachedPages_ == 0) return;
        while (pageCache_.size() > maxCachedPages_) {
            size_t victim = pageOrder_.front();
            pageOrder_.pop_front();
            pageCache_.erase(victim);
        }
    }

    // Load a full 4KB page into the cache.
    // 20260516_page_buf_recycle: on eviction, recycle the victim's 4 KB
    // PageBuf allocation via unordered_map::extract (C++17 node_handle)
    // instead of erase + new alloc. Saves the malloc/free round-trip on
    // every miss past warmup.
    void loadPageToCache(size_t pageId) {
        if (maxCachedPages_ == 0) return;             // caching disabled
        if (pageCache_.find(pageId) != pageCache_.end()) return; // already cached

        size_t offset = pageId * kPageSize;

        // Helper: read a full page into `dst`. Zero-fill any short tail
        // (near EOF, or pread error) so callers (tryReadFromCache,
        // dequantize, etc.) always see well-defined bytes. Uniform handling
        // of n < 0, n == 0, and 0 < n < kPageSize.
        auto fill_buf = [&](char* dst) {
            ssize_t n = 0;
            if (readFd_ >= 0) {
                n = ::pread(readFd_, dst, kPageSize, static_cast<off_t>(offset));
            } else {
                fileStream_.clear();
                fileStream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
                fileStream_.read(dst, static_cast<std::streamsize>(kPageSize));
                n = static_cast<ssize_t>(fileStream_.gcount());
            }
            size_t got = (n > 0) ? static_cast<size_t>(n) : 0;
            if (got < kPageSize) {
                std::memset(dst + got, 0, kPageSize - got);
            }
        };

        if (pageCache_.size() >= maxCachedPages_) {
            // Recycle: extract the victim's node_handle so its 4 KB
            // allocation survives. Refill, re-key, re-insert.
            size_t victim = pageOrder_.front();
            pageOrder_.pop_front();
            auto node = pageCache_.extract(victim);
            if (!node.empty()) {
                PageBuf& buf = node.mapped();
                if (buf.data.size() != kPageSize) buf.data.resize(kPageSize);
                fill_buf(buf.data.data());
                overlayWriteBuf(pageId, buf);
                node.key() = pageId;
                pageCache_.insert(std::move(node));
                pageOrder_.push_back(pageId);
                return;
            }
            // Fallthrough if extract failed (shouldn't happen — pageOrder_
            // is the canonical source of victim ids).
        }

        PageBuf buf;
        buf.data.resize(kPageSize, 0);
        fill_buf(buf.data.data());
        overlayWriteBuf(pageId, buf);
        pageCache_.emplace(pageId, std::move(buf));
        pageOrder_.push_back(pageId);
    }

    // Try to read a vector from cache by pageId & slot. Dequantizes SQ8 → float32.
    bool tryReadFromCache(size_t pageId, uint16_t slot, std::vector<float>& vec) {
        if (maxCachedPages_ == 0) return false;
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
        sq8TempBuf_.resize(recordSize_);
        size_t offset = pageId * kPageSize + static_cast<size_t>(slot) * recordSize_;
        if (readFd_ >= 0) {
            ::pread(readFd_, sq8TempBuf_.data(), recordSize_, static_cast<off_t>(offset));
        } else {
            fileStream_.clear();
            fileStream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
            fileStream_.read(sq8TempBuf_.data(),
                             static_cast<std::streamsize>(recordSize_));
        }
        dequantize(sq8TempBuf_.data(), out, dim_);
    }

    // Write a single record (one vector) at (pageId, slot) to disk. Quantizes to SQ8.
    void writeRecord(size_t pageId, uint16_t slot, const std::vector<float>& vec) {
        if (vec.size() != dim_) {
            throw std::runtime_error("Vector size mismatch in writeRecord.");
        }
        sq8TempBuf_.resize(recordSize_);
        quantize(vec.data(), dim_, sq8TempBuf_.data());
        size_t offset = pageId * kPageSize + static_cast<size_t>(slot) * recordSize_;
        fileStream_.clear();
        fileStream_.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
        fileStream_.write(sq8TempBuf_.data(),
                          static_cast<std::streamsize>(recordSize_));
        if (!fileStream_.good()) {
            throw std::runtime_error("Failed to write vector to file.");
        }
        fileStream_.flush();
    }

    // If the page is cached, update the cached copy after a write (stores quantized SQ8).
    void updateCacheAfterWrite(size_t pageId, uint16_t slot, const std::vector<float>& vec) {
        if (maxCachedPages_ == 0) return;
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
    std::pair<size_t,uint16_t> allocateSlotForSection(node_id_t sectionKey) {
        // Map external sectionKey to internal sectionIdx
        int sectionIdx;
        auto it = sectionKeyToIdx_.find(sectionKey);
        if (it == sectionKeyToIdx_.end()) {
            sectionIdx = static_cast<int>(sectionIdxToKey_.size());
            sectionKeyToIdx_[sectionKey] = sectionIdx;
            sectionIdxToKey_.push_back(sectionKey);
        } else {
            sectionIdx = it->second;
        }

        auto& freeList = sectionFreeSlots_[sectionIdx];
        if (!freeList.empty()) {
            auto [pageId, slot] = freeList.back();
            freeList.pop_back();
            return {pageId, slot};
        }

        auto& openList = sectionOpenPages_[sectionIdx];

        // If there is a not-full page in this section, use it
        if (!openList.empty()) {
            size_t pageId = openList.back();
            openList.pop_back();
            PageMeta& meta = pages_[pageId];
            if (meta.usedSlots >= vectorsPerPage_) {
                // Should not happen, but guard anyway
                return allocateSlotForSection(sectionKey);
            }
            uint16_t slot = meta.usedSlots++;
            if (meta.usedSlots < vectorsPerPage_) {
                openList.push_back(pageId);
            }
            return {pageId, slot};
        }

        // Otherwise, allocate a new page for this section
        size_t pageId = pages_.size();
        pages_.push_back(PageMeta{sectionIdx, 0});
        ensurePageAllocated(pageId);

        PageMeta& meta = pages_.back();
        meta.usedSlots = 1;
        uint16_t slot = 0;

        if (meta.usedSlots < vectorsPerPage_) {
            openList.push_back(pageId);
        }

        return {pageId, slot};
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
        sq8TempBuf_.resize(recordSize_);
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
        maxCachedPages_ = m;
        evictIfNeeded();
    }

    // Flush all remaining section write buffers (partially filled pages).
    void flushWrites() override {
        for (auto& [sectionIdx, swb] : sectionWriteBufs_) {
            flushSectionWriteBuf(swb);
        }
        sectionWriteBufs_.clear();
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

        auto [pageId, slot] = allocateSlotForSection(sectionKey);
        assign_location(id, pageId, slot);

        // Look up the internal sectionIdx for write buffer
        auto secIt = sectionKeyToIdx_.find(sectionKey);
        int sectionIdx = (secIt != sectionKeyToIdx_.end()) ? secIt->second : -1;

        if (sectionIdx >= 0 && writeFd_ >= 0) {
            writeToSectionBuffer(sectionIdx, pageId, slot, vec);
        } else {
            writeRecord(pageId, slot, vec);
            updateCacheAfterWrite(pageId, slot, vec);
        }
        mark_deleted_at(id, false);
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
                ++pageCacheHits_;
                return;
            }
            ++pageCacheMisses_;
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
        sq8TempBuf_.resize(recordSize_);
        size_t offset = pageId * kPageSize + static_cast<size_t>(slot) * recordSize_;
        fileStream_.clear();
        fileStream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        fileStream_.read(sq8TempBuf_.data(),
                         static_cast<std::streamsize>(recordSize_));
        if (!fileStream_.good()) {
            throw std::runtime_error("Failed to read vector from file.");
        }
        dequantize(sq8TempBuf_.data(), vec.data(), dim_);
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
    // and per-vector heap allocations.
    void readVectorsBatchFlat(const std::vector<node_id_t>& ids,
                              float* out, size_t dim) override {
        missScratch_.clear();

        // Pass 1: serve cache hits directly, collect misses
        for (size_t i = 0; i < ids.size(); ++i) {
            node_id_t id = ids[i];
            int64_t page = page_of(id);
            if (page < 0) {
                throw std::runtime_error("Vector slot not assigned in batch flat read.");
            }
            size_t pageId = static_cast<size_t>(page);
            uint16_t slot = slot_of(id);

            // Try write buffer first (unflushed data)
            if (tryReadFromWriteBufFlat(pageId, slot, out + i * dim, dim_)) {
                continue;
            }
            // Try cache hit — dequantize SQ8 → float
            if (maxCachedPages_ > 0) {
                auto cit = pageCache_.find(pageId);
                if (cit != pageCache_.end()) {
                    size_t offsetInPage = static_cast<size_t>(slot) * recordSize_;
                    dequantize(cit->second.data.data() + offsetInPage, out + i * dim, dim_);
                    ++pageCacheHits_;
                    continue;
                }
            }
            ++pageCacheMisses_;
            missScratch_.push_back({i, pageId, slot});
        }

        if (missScratch_.empty()) return; // fast path: all cached

        // Pass 2: sort misses by pageId, load unique pages, then copy
        std::sort(missScratch_.begin(), missScratch_.end(),
                  [](const MissEntry& a, const MissEntry& b) {
                      return a.pageId < b.pageId;
                  });

        // Deduplicate pages and load them
        uniquePageScratch_.clear();
        for (size_t j = 0; j < missScratch_.size(); ++j) {
            if (j == 0 || missScratch_[j].pageId != missScratch_[j - 1].pageId) {
                uniquePageScratch_.push_back(missScratch_[j].pageId);
            }
        }

        for (size_t pageId : uniquePageScratch_) {
            loadPageToCache(pageId);
        }

        // Dequantize from now-cached pages (or direct read as fallback)
        for (const auto& miss : missScratch_) {
            auto cit = pageCache_.find(miss.pageId);
            if (cit != pageCache_.end()) {
                size_t offsetInPage = static_cast<size_t>(miss.slot) * recordSize_;
                dequantize(cit->second.data.data() + offsetInPage,
                           out + miss.idx * dim, dim_);
            } else {
                // Cache full, page was evicted — direct read
                directReadRecord(miss.pageId, miss.slot, out + miss.idx * dim);
            }
        }
    }

    PageCacheStats getPageCacheStats() const override {
        return PageCacheStats{pageCacheHits_, pageCacheMisses_};
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
        pages_.reserve(static_cast<size_t>(pagesSize));
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

        sectionOpenPages_.clear();
        sectionFreeSlots_.clear();

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
            if (static_cast<size_t>(meta.usedSlots) < vectorsPerPage_) {
                sectionOpenPages_[meta.sectionIdx].push_back(pageId);
            }
            auto& freeList = sectionFreeSlots_[meta.sectionIdx];
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
        pageCache_.clear();
        pageOrder_.clear();
        pageCacheHits_ = 0;
        pageCacheMisses_ = 0;

        return static_cast<bool>(in);
    }
};
}
