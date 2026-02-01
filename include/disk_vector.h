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
    size_t recordSize_;     // dim_ * sizeof(float)
    size_t totalVectors_;   // logical ID capacity
    size_t vectorsPerPage_; // how many vectors fit into one page (floor division)

    std::fstream fileStream_;
    std::fstream deleteStream_;
    std::vector<uint8_t> deletedFlags_;

    // logical ID -> page & slot mapping
    std::vector<int64_t>  idToPage_;       // -1 means unassigned
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

    // Page cache (FIFO) in units of full pages (4KB each)
    size_t maxCachedPages_;
    size_t pageCacheHits_ = 0;
    size_t pageCacheMisses_ = 0;

    struct PageBuf {
        std::vector<char> data; // always kPageSize bytes
    };

    std::unordered_map<size_t, PageBuf> pageCache_; // pageId -> data
    std::deque<size_t>                  pageOrder_; // FIFO order of pageIds

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
        size_t requiredSize = (pageId + 1) * kPageSize;
        fileStream_.seekp(0, std::ios::end);
        size_t currentSize = static_cast<size_t>(fileStream_.tellp());
        if (currentSize < requiredSize) {
            fileStream_.seekp(static_cast<std::streamoff>(requiredSize - 1), std::ios::beg);
            char zero = 0;
            fileStream_.write(&zero, 1);
            fileStream_.flush();
        }
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
    void loadPageToCache(size_t pageId) {
        if (maxCachedPages_ == 0) return;             // caching disabled
        if (pageCache_.find(pageId) != pageCache_.end()) return; // already cached

        // FIFO eviction
        if (pageCache_.size() >= maxCachedPages_) {
            size_t victim = pageOrder_.front();
            pageOrder_.pop_front();
            pageCache_.erase(victim);
        }

        PageBuf buf;
        buf.data.resize(kPageSize, 0);

        size_t offset = pageId * kPageSize;
        fileStream_.clear();
        fileStream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        fileStream_.read(buf.data.data(), static_cast<std::streamsize>(kPageSize));
        // If the read is short (near EOF), remaining bytes stay zero.

        pageCache_.emplace(pageId, std::move(buf));
        pageOrder_.push_back(pageId);
    }

    // Try to read a vector from cache by pageId & slot.
    bool tryReadFromCache(size_t pageId, uint16_t slot, std::vector<float>& vec) {
        if (maxCachedPages_ == 0) return false;
        auto it = pageCache_.find(pageId);
        if (it == pageCache_.end()) return false;

        const PageBuf& buf = it->second;
        size_t offsetInPage = static_cast<size_t>(slot) * recordSize_;
        if (offsetInPage + recordSize_ > buf.data.size()) {
            return false; // corrupted / inconsistent
        }
        if (vec.size() != dim_) vec.resize(dim_);
        std::memcpy(vec.data(), buf.data.data() + offsetInPage, recordSize_);
        return true;
    }

    // Write a single record (one vector) at (pageId, slot) to disk.
    void writeRecord(size_t pageId, uint16_t slot, const std::vector<float>& vec) {
        if (vec.size() != dim_) {
            throw std::runtime_error("Vector size mismatch in writeRecord.");
        }
        size_t offset = pageId * kPageSize + static_cast<size_t>(slot) * recordSize_;
        fileStream_.clear();
        fileStream_.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
        fileStream_.write(reinterpret_cast<const char*>(vec.data()),
                          static_cast<std::streamsize>(recordSize_));
        if (!fileStream_.good()) {
            throw std::runtime_error("Failed to write vector to file.");
        }
        fileStream_.flush();
    }

    // If the page is cached, update the cached copy after a write.
    void updateCacheAfterWrite(size_t pageId, uint16_t slot, const std::vector<float>& vec) {
        if (maxCachedPages_ == 0) return;
        auto it = pageCache_.find(pageId);
        if (it == pageCache_.end()) return;

        PageBuf& buf = it->second;
        size_t offsetInPage = static_cast<size_t>(slot) * recordSize_;
        if (offsetInPage + recordSize_ > buf.data.size()) {
            return;
        }
        std::memcpy(buf.data.data() + offsetInPage,
                    reinterpret_cast<const char*>(vec.data()),
                    recordSize_);
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
          recordSize_(dim * sizeof(float)),
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
    }

    ~PagedVectorStorage() {
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

    // Store vector with a section hint: 'sectionKey' is derived from HNSW Level-1 entry.
    // We assign the vector to some page belonging to that section; pages are always
    // 4KB and vectors never cross page boundaries.
    void storeVectorToDisk(node_id_t id,
                           const std::vector<float>& vec,
                           node_id_t sectionKey) override
    {
        if (static_cast<size_t>(id) >= totalVectors_) {
            expandCapacity(static_cast<size_t>(id) + 1);
        }
        if (vec.size() != dim_) {
            throw std::runtime_error("Vector size mismatch.");
        }

        int64_t curPage = idToPage_[static_cast<size_t>(id)];
        uint16_t curSlot = idToSlotInPage_[static_cast<size_t>(id)];

        if (curPage >= 0) {
            // Overwrite existing slot
            writeRecord(static_cast<size_t>(curPage), curSlot, vec);
            updateCacheAfterWrite(static_cast<size_t>(curPage), curSlot, vec);
            if (deletedFlags_[static_cast<size_t>(id)] != 0) {
                writeDeleteFlag(id, false);
            }
            return;
        }

        auto [pageId, slot] = allocateSlotForSection(sectionKey);
        idToPage_[static_cast<size_t>(id)]       = static_cast<int64_t>(pageId);
        idToSlotInPage_[static_cast<size_t>(id)] = slot;

        writeRecord(pageId, slot, vec);
        updateCacheAfterWrite(pageId, slot, vec);
        if (deletedFlags_[static_cast<size_t>(id)] != 0) {
            writeDeleteFlag(id, false);
        }
    }

    // Backward-compatible version: if you don't care about sections,
    // use ID as the "sectionKey".
    void storeVectorToDisk(node_id_t id, const std::vector<float>& vec) override {
        storeVectorToDisk(id, vec, id);
    }

    // Read a vector by its logical ID.
    void readVectorFromDisk(node_id_t id, std::vector<float>& vec) override
    {
        if (static_cast<size_t>(id) >= totalVectors_) {
            throw std::out_of_range("Vector ID out of range.");
        }

        int64_t page = idToPage_[static_cast<size_t>(id)];
        uint16_t slot = idToSlotInPage_[static_cast<size_t>(id)];

        if (page < 0) {
            std::fprintf(stderr, "Problematic id: %lu\n", static_cast<unsigned long>(id));
            throw std::runtime_error("Vector slot not assigned for this ID.");
        }

        size_t pageId = static_cast<size_t>(page);

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

        // 3) Fallback: direct disk read of this record
        if (vec.size() != dim_) vec.resize(dim_);
        size_t offset = pageId * kPageSize + static_cast<size_t>(slot) * recordSize_;
        fileStream_.clear();
        fileStream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        fileStream_.read(reinterpret_cast<char*>(vec.data()),
                         static_cast<std::streamsize>(recordSize_));
        if (!fileStream_.good()) {
            throw std::runtime_error("Failed to read vector from file.");
        }
    }

    void deleteVector(node_id_t id) override
    {
        if (static_cast<size_t>(id) >= totalVectors_) {
            throw std::out_of_range("Vector ID out of range.");
        }
        if (deletedFlags_[static_cast<size_t>(id)] != 0) {
            return;
        }

        int64_t page = idToPage_[static_cast<size_t>(id)];
        uint16_t slot = idToSlotInPage_[static_cast<size_t>(id)];
        if (page >= 0) {
            size_t pageIndex = static_cast<size_t>(page);
            if (pageIndex < pages_.size()) {
                int sectionIdx = pages_[pageIndex].sectionIdx;
                sectionFreeSlots_[sectionIdx].emplace_back(pageIndex, slot);
            }
            idToPage_[static_cast<size_t>(id)] = -1;
            idToSlotInPage_[static_cast<size_t>(id)] = 0;
        }

        writeDeleteFlag(id, true);
    }

    bool exists(node_id_t id) const override
    {
        if (static_cast<size_t>(id) >= totalVectors_) {
            return false;
        }
        if (deletedFlags_[static_cast<size_t>(id)] != 0) {
            return false;
        }
        return idToPage_[static_cast<size_t>(id)] >= 0;
    }

    // Prefetch pages corresponding to given vector IDs.
    // This simply ensures the relevant pages are loaded into the page cache.
    void prefetchByIds(const std::vector<node_id_t>& ids) override {
        if (maxCachedPages_ == 0) return;

        std::unordered_map<size_t, bool> seenPage;
        for (node_id_t id : ids) {
            if (static_cast<size_t>(id) >= totalVectors_) continue;
            int64_t page = idToPage_[static_cast<size_t>(id)];
            if (page < 0) continue;
            size_t pageId = static_cast<size_t>(page);
            if (seenPage.emplace(pageId, true).second) {
                loadPageToCache(pageId);
            }
        }
    }

    PageCacheStats getPageCacheStats() const override {
        return PageCacheStats{pageCacheHits_, pageCacheMisses_};
    }

    bool serializeMetadata(std::ostream& out) const {
        constexpr uint32_t kMetaVersion = 1;
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
        for (int64_t page : idToPage_) {
            if (!write(page)) {
                return false;
            }
        }
        for (uint16_t slot : idToSlotInPage_) {
            if (!write(slot)) {
                return false;
            }
        }

        return static_cast<bool>(out);
    }

    bool deserializeMetadata(std::istream& in) {
        constexpr uint32_t kMetaVersion = 1;
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

        if (version != kMetaVersion) {
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
            idToPage_[static_cast<size_t>(i)] = page;
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

        reloadDeleteFlags(totalVectors_);
        pageCache_.clear();
        pageOrder_.clear();
        pageCacheHits_ = 0;
        pageCacheMisses_ = 0;

        return static_cast<bool>(in);
    }
};
}
