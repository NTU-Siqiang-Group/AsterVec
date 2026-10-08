#pragma once

// Operation log: every acknowledged insert, update and delete since the last
// checkpoint, so they survive a process crash (replayed on Open).
//
// Files: <dir>/<first lsn, 20 digits>.log segments, rotated at a size limit.
// A new segment is started on every Open, so a torn tail left by a crash is
// never followed by new records in the same file.
//
// Record (little-endian, native layout):
//   u32 body_len          bytes after this 8-byte prefix
//   u32 crc32             over the body after the lsn, then the lsn (so the
//                         bulk is checksummed outside the append lock)
//   u64 lsn
//   u8  op                OpType
//   u8  encoding          IVectorStorage::kEncoding* of `record`
//   u64 real_id
//   u64 internal_id
//   u64 aux               update: old internal id; abort: aborted lsn
//   u32 record_len, bytes record      vector in the store's encoding
//   u32 key_len, bytes key            caller's point id (keyed stores; else empty)
//   u32 payload_len, bytes payload    JSON; empty = no payload given
//
// No fsync per append: records reach the OS on every append, which is what
// survives a process crash. Power loss keeps only what the last checkpoint
// made durable.

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include "rocksdb/status.h"

namespace astervec {

using Status = ROCKSDB_NAMESPACE::Status;

enum class OpType : uint8_t {
    kInsert = 1,
    kUpdate = 2,
    kDelete = 3,
    kAbort = 4,  // the operation at `aux` failed after it was logged
};

struct OpRecord {
    uint64_t lsn = 0;  // assigned by Append
    OpType op = OpType::kInsert;
    uint8_t encoding = 0;
    uint64_t real_id = 0;
    uint64_t internal_id = 0;
    uint64_t aux = 0;
    std::string_view record;
    std::string_view key;  // PointId::Encode() in keyed stores, else empty
    std::string_view payload;
};

class OpLog {
public:
    static constexpr uint64_t kDefaultSegmentBytes = 64ull << 20;

    // Opens `dir` (created if missing) for appending; the first record gets
    // `next_lsn`, which must be one past the last record Scan accepted (or
    // the checkpoint lsn + 1). Segments starting at or after it are stale
    // and are removed; stale records inside an older segment always sit
    // behind the bad record that ended the scan, so Scan never reaches them.
    static Status Open(const std::string& dir, uint64_t next_lsn,
                       uint64_t segment_bytes, std::unique_ptr<OpLog>* out);
    ~OpLog();
    OpLog(const OpLog&) = delete;
    OpLog& operator=(const OpLog&) = delete;

    // Appends one record (rec.lsn is ignored) and returns its lsn. On an I/O
    // error nothing is appended and the log stays usable if it could be
    // rolled back, otherwise every later append fails.
    Status Append(const OpRecord& rec, uint64_t* lsn);

    // The last lsn handed out (next_lsn - 1 right after Open).
    uint64_t LastLsn() const;

    // Removes whole segments whose records all have lsn < `lsn`. The segment
    // being appended to is never removed.
    Status TruncateBelow(uint64_t lsn);

    struct ScanStats {
        uint64_t records = 0;        // records passed to the callback
        uint64_t last_lsn = 0;       // lsn of the last one (0 if none)
        uint64_t dropped_bytes = 0;  // torn or corrupt bytes skipped
        bool gap = false;            // stopped at a missing lsn
    };
    // Calls `fn` for every record with lsn >= from_lsn, in lsn order. LSNs
    // must be contiguous starting at from_lsn: a torn or corrupt record ends
    // its segment, and the scan stops for good at the first missing lsn.
    // A non-OK status from `fn` stops the scan and is returned.
    static Status Scan(const std::string& dir, uint64_t from_lsn,
                       const std::function<Status(const OpRecord&)>& fn,
                       ScanStats* stats);

    // Total bytes of all segments in `dir` (0 if it does not exist).
    static uint64_t DiskBytes(const std::string& dir);

private:
    OpLog(std::string dir, uint64_t next_lsn, uint64_t segment_bytes);
    Status openSegmentLocked();

    const std::string dir_;
    const uint64_t segment_bytes_;
    mutable std::mutex mu_;
    uint64_t next_lsn_;
    int fd_ = -1;
    uint64_t offset_ = 0;  // bytes in the current segment
    bool broken_ = false;
};

}  // namespace astervec
