#include "oplog.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

#include "crc32.h"
#include "durable_io.h"

namespace astervec {
namespace {

namespace fs = std::filesystem;

constexpr size_t kPrefixBytes = 8;   // body_len + crc
constexpr size_t kFixedBody = 8 + 1 + 1 + 8 + 8 + 8 + 4 + 4;
// Records larger than this are treated as corrupt (a garbage length must
// not make the scan allocate gigabytes).
constexpr uint32_t kMaxBodyBytes = 64u << 20;

struct Segment {
    uint64_t first_lsn;
    std::string path;
};

// Segments in `dir`, ordered by first lsn.
std::vector<Segment> listSegments(const std::string& dir)
{
    std::vector<Segment> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name.size() != 24 || name.compare(20, 4, ".log") != 0) continue;
        if (!std::all_of(name.begin(), name.begin() + 20,
                         [](char c) { return c >= '0' && c <= '9'; }))
            continue;
        out.push_back({std::stoull(name.substr(0, 20)), e.path().string()});
    }
    std::sort(out.begin(), out.end(),
              [](const Segment& a, const Segment& b) {
                  return a.first_lsn < b.first_lsn;
              });
    return out;
}

std::string segmentPath(const std::string& dir, uint64_t first_lsn)
{
    char name[32];
    std::snprintf(name, sizeof(name), "%020llu.log",
                  static_cast<unsigned long long>(first_lsn));
    return dir + "/" + name;
}

template <typename T>
void put(char*& p, T v)
{
    std::memcpy(p, &v, sizeof(T));
    p += sizeof(T);
}

template <typename T>
T get(const char*& p)
{
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

bool writeAll(int fd, const char* data, size_t n)
{
    while (n > 0) {
        ssize_t w = ::write(fd, data, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        data += w;
        n -= static_cast<size_t>(w);
    }
    return true;
}

}  // namespace

OpLog::OpLog(std::string dir, uint64_t next_lsn, uint64_t segment_bytes)
    : dir_(std::move(dir)),
      segment_bytes_(segment_bytes == 0 ? kDefaultSegmentBytes : segment_bytes),
      next_lsn_(next_lsn)
{
}

OpLog::~OpLog()
{
    if (fd_ >= 0) ::close(fd_);
}

Status OpLog::Open(const std::string& dir, uint64_t next_lsn,
                   uint64_t segment_bytes, std::unique_ptr<OpLog>* out)
{
    if (next_lsn == 0) return Status::InvalidArgument("lsn starts at 1");
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return Status::IOError("cannot create " + dir + ": " + ec.message());
    for (const auto& seg : listSegments(dir)) {
        if (seg.first_lsn >= next_lsn) {
            fs::remove(seg.path, ec);
            if (ec) return Status::IOError("cannot remove stale " + seg.path);
        }
    }
    std::unique_ptr<OpLog> log(new OpLog(dir, next_lsn, segment_bytes));
    {
        std::lock_guard<std::mutex> g(log->mu_);
        Status s = log->openSegmentLocked();
        if (!s.ok()) return s;
    }
    *out = std::move(log);
    return Status::OK();
}

Status OpLog::openSegmentLocked()
{
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    const std::string path = segmentPath(dir_, next_lsn_);
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
    if (fd_ < 0) {
        return Status::IOError("cannot open " + path + ": " + std::strerror(errno));
    }
    offset_ = 0;
    // The new name must survive a crash, or replay would not find the file.
    if (!durable::fsync_dir(dir_)) {
        return Status::IOError("cannot sync " + dir_);
    }
    return Status::OK();
}

Status OpLog::Append(const OpRecord& rec, uint64_t* lsn_out)
{
    const size_t body = kFixedBody + rec.record.size() + rec.payload.size();
    if (body > kMaxBodyBytes) return Status::InvalidArgument("log record too large");
    thread_local std::vector<char> buf;
    buf.resize(kPrefixBytes + body);

    char* p = buf.data();
    put<uint32_t>(p, static_cast<uint32_t>(body));
    put<uint32_t>(p, 0);  // crc, filled below
    put<uint64_t>(p, 0);  // lsn, filled below
    char* const after_lsn = p;
    put<uint8_t>(p, static_cast<uint8_t>(rec.op));
    put<uint8_t>(p, rec.encoding);
    put<uint64_t>(p, rec.real_id);
    put<uint64_t>(p, rec.internal_id);
    put<uint64_t>(p, rec.aux);
    put<uint32_t>(p, static_cast<uint32_t>(rec.record.size()));
    if (!rec.record.empty()) {
        std::memcpy(p, rec.record.data(), rec.record.size());
        p += rec.record.size();
    }
    put<uint32_t>(p, static_cast<uint32_t>(rec.payload.size()));
    if (!rec.payload.empty()) {
        std::memcpy(p, rec.payload.data(), rec.payload.size());
        p += rec.payload.size();
    }
    const uint32_t body_crc = crc32::Update(
        crc32::kInit, after_lsn, static_cast<size_t>(p - after_lsn));

    std::lock_guard<std::mutex> g(mu_);
    if (broken_) return Status::IOError("operation log unusable after an earlier error");
    if (offset_ >= segment_bytes_) {
        Status s = openSegmentLocked();
        if (!s.ok()) {
            broken_ = true;
            return s;
        }
    }
    const uint64_t lsn = next_lsn_;
    std::memcpy(buf.data() + kPrefixBytes, &lsn, sizeof(lsn));
    const uint32_t crc = crc32::Finish(crc32::Update(body_crc, &lsn, sizeof(lsn)));
    std::memcpy(buf.data() + 4, &crc, sizeof(crc));

    if (!writeAll(fd_, buf.data(), buf.size())) {
        const std::string err = std::strerror(errno);
        // Cut a partial record off so later records are not stranded
        // behind it; if even that fails, stop appending.
        if (::ftruncate(fd_, static_cast<off_t>(offset_)) != 0) broken_ = true;
        return Status::IOError("operation log write failed: " + err);
    }
    offset_ += buf.size();
    ++next_lsn_;
    *lsn_out = lsn;
    return Status::OK();
}

uint64_t OpLog::LastLsn() const
{
    std::lock_guard<std::mutex> g(mu_);
    return next_lsn_ - 1;
}

Status OpLog::TruncateBelow(uint64_t lsn)
{
    std::lock_guard<std::mutex> g(mu_);
    const auto segs = listSegments(dir_);
    // Segment i holds lsns [first_i, first_{i+1}); the last one is current.
    for (size_t i = 0; i + 1 < segs.size(); ++i) {
        if (segs[i + 1].first_lsn > lsn) break;
        std::error_code ec;
        fs::remove(segs[i].path, ec);
        if (ec) return Status::IOError("cannot remove " + segs[i].path);
    }
    return Status::OK();
}

Status OpLog::Scan(const std::string& dir, uint64_t from_lsn,
                   const std::function<Status(const OpRecord&)>& fn,
                   ScanStats* stats)
{
    ScanStats local;
    ScanStats& st = stats ? *stats : local;
    st = ScanStats{};
    const auto segs = listSegments(dir);
    uint64_t expected = from_lsn;
    std::vector<char> body;

    for (size_t i = 0; i < segs.size(); ++i) {
        // Skip segments that end before from_lsn.
        if (i + 1 < segs.size() && segs[i + 1].first_lsn <= from_lsn) continue;

        std::ifstream in(segs[i].path, std::ios::binary);
        if (!in.is_open()) return Status::IOError("cannot open " + segs[i].path);
        in.seekg(0, std::ios::end);
        const uint64_t file_size = static_cast<uint64_t>(in.tellg());
        in.seekg(0, std::ios::beg);
        uint64_t pos = 0;

        while (pos < file_size) {
            char prefix[kPrefixBytes];
            if (file_size - pos < kPrefixBytes ||
                !in.read(prefix, kPrefixBytes)) {
                break;
            }
            uint32_t body_len, crc;
            std::memcpy(&body_len, prefix, 4);
            std::memcpy(&crc, prefix + 4, 4);
            if (body_len < kFixedBody || body_len > kMaxBodyBytes ||
                file_size - pos - kPrefixBytes < body_len) {
                break;
            }
            body.resize(body_len);
            if (!in.read(body.data(), body_len)) break;
            const uint32_t actual = crc32::Finish(crc32::Update(
                crc32::Update(crc32::kInit, body.data() + 8, body_len - 8),
                body.data(), 8));
            if (actual != crc) break;

            const char* p = body.data();
            OpRecord rec;
            rec.lsn = get<uint64_t>(p);
            rec.op = static_cast<OpType>(get<uint8_t>(p));
            rec.encoding = get<uint8_t>(p);
            rec.real_id = get<uint64_t>(p);
            rec.internal_id = get<uint64_t>(p);
            rec.aux = get<uint64_t>(p);
            const char* const end = body.data() + body_len;
            const uint32_t rec_len = get<uint32_t>(p);
            if (rec_len > static_cast<size_t>(end - p) - 4) break;
            rec.record = std::string_view(p, rec_len);
            p += rec_len;
            const uint32_t pay_len = get<uint32_t>(p);
            if (pay_len != static_cast<size_t>(end - p)) break;
            rec.payload = std::string_view(p, pay_len);
            pos += kPrefixBytes + body_len;

            if (rec.lsn < from_lsn) continue;
            if (rec.lsn != expected) {
                st.gap = true;
                return Status::OK();
            }
            Status s = fn(rec);
            if (!s.ok()) return s;
            ++st.records;
            st.last_lsn = rec.lsn;
            ++expected;
        }
        // Torn or corrupt tail: skip the rest of this segment. The next
        // segment must continue the lsn sequence, which the check above
        // enforces.
        st.dropped_bytes += file_size - pos;
    }
    return Status::OK();
}

uint64_t OpLog::DiskBytes(const std::string& dir)
{
    uint64_t total = 0;
    std::error_code ec;
    for (const auto& seg : listSegments(dir)) {
        auto n = fs::file_size(seg.path, ec);
        if (!ec) total += n;
    }
    return total;
}

}  // namespace astervec
