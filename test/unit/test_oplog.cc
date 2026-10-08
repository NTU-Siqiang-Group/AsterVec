// Operation log file format: append/scan round trip, segment rotation,
// torn tails, corruption, lsn gaps and truncation.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "doctest.h"
#include "oplog.h"

using astervec::OpLog;
using astervec::OpRecord;
using astervec::OpType;
using astervec::Status;

namespace {

namespace fs = std::filesystem;

std::string make_dir() {
    char tmpl[] = "/tmp/astervec_oplog_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    return std::string(dir) + "/oplog";
}

std::vector<std::string> segments(const std::string& dir) {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(dir)) out.push_back(e.path().string());
    std::sort(out.begin(), out.end());
    return out;
}

struct Seen {
    uint64_t lsn;
    OpType op;
    uint64_t real_id;
    uint64_t internal_id;
    uint64_t aux;
    uint8_t encoding;
    std::string record;
    std::string key;
    std::string payload;
};

std::vector<Seen> scan_all(const std::string& dir, uint64_t from,
                           OpLog::ScanStats* st = nullptr) {
    std::vector<Seen> out;
    Status s = OpLog::Scan(dir, from, [&](const OpRecord& r) {
        out.push_back({r.lsn, r.op, r.real_id, r.internal_id, r.aux, r.encoding,
                       std::string(r.record), std::string(r.key),
                       std::string(r.payload)});
        return Status::OK();
    }, st);
    REQUIRE(s.ok());
    return out;
}

void append_n(OpLog& log, int n, int base = 0) {
    for (int i = 0; i < n; ++i) {
        std::string rec(40, static_cast<char>('a' + (i % 26)));
        std::string pay = "{\"i\":" + std::to_string(base + i) + "}";
        OpRecord r;
        r.op = OpType::kInsert;
        r.encoding = 1;
        r.real_id = base + i;
        r.internal_id = base + i;
        r.record = rec;
        r.payload = pay;
        uint64_t lsn = 0;
        REQUIRE(log.Append(r, &lsn).ok());
    }
}

}  // namespace

TEST_CASE("oplog: append and scan round trip") {
    const auto dir = make_dir();
    std::unique_ptr<OpLog> log;
    REQUIRE(OpLog::Open(dir, 1, 0, &log).ok());

    OpRecord ins;
    ins.op = OpType::kInsert;
    ins.encoding = 1;
    ins.real_id = 7;
    ins.internal_id = 7;
    ins.record = std::string_view("\x01\x02\x03\x00\x05", 5);
    ins.payload = R"({"a":1})";
    ins.key = std::string_view("\x02doc-1", 6);
    uint64_t lsn1 = 0, lsn2 = 0, lsn3 = 0;
    REQUIRE(log->Append(ins, &lsn1).ok());

    OpRecord upd;
    upd.op = OpType::kUpdate;
    upd.encoding = 1;
    upd.real_id = 7;
    upd.internal_id = (1ULL << 63) + 3;
    upd.aux = 7;
    upd.record = "abc";
    REQUIRE(log->Append(upd, &lsn2).ok());

    OpRecord del;
    del.op = OpType::kDelete;
    del.real_id = 7;
    del.internal_id = (1ULL << 63) + 3;
    REQUIRE(log->Append(del, &lsn3).ok());

    CHECK(lsn1 == 1);
    CHECK(lsn2 == 2);
    CHECK(lsn3 == 3);
    CHECK(log->LastLsn() == 3);

    OpLog::ScanStats st;
    auto seen = scan_all(dir, 1, &st);
    REQUIRE(seen.size() == 3);
    CHECK(seen[0].op == OpType::kInsert);
    CHECK(seen[0].record == std::string("\x01\x02\x03\x00\x05", 5));
    CHECK(seen[0].payload == R"({"a":1})");
    CHECK(seen[0].key == std::string("\x02doc-1", 6));
    CHECK(seen[1].key.empty());
    CHECK(seen[1].op == OpType::kUpdate);
    CHECK(seen[1].internal_id == (1ULL << 63) + 3);
    CHECK(seen[1].aux == 7);
    CHECK(seen[1].payload.empty());
    CHECK(seen[2].op == OpType::kDelete);
    CHECK(seen[2].record.empty());
    CHECK(st.records == 3);
    CHECK(st.last_lsn == 3);
    CHECK(st.dropped_bytes == 0);
    CHECK(!st.gap);

    // Starting later skips earlier records.
    auto tail = scan_all(dir, 3);
    REQUIRE(tail.size() == 1);
    CHECK(tail[0].lsn == 3);
}

TEST_CASE("oplog: segments rotate and scan continues across them") {
    const auto dir = make_dir();
    std::unique_ptr<OpLog> log;
    REQUIRE(OpLog::Open(dir, 1, 1024, &log).ok());
    append_n(*log, 100);
    CHECK(segments(dir).size() > 5);
    auto seen = scan_all(dir, 1);
    REQUIRE(seen.size() == 100);
    for (size_t i = 0; i < seen.size(); ++i) CHECK(seen[i].lsn == i + 1);
    auto tail = scan_all(dir, 57);
    REQUIRE(tail.size() == 44);
    CHECK(tail.front().lsn == 57);
}

TEST_CASE("oplog: a torn tail is dropped and reopening continues the sequence") {
    const auto dir = make_dir();
    {
        std::unique_ptr<OpLog> log;
        REQUIRE(OpLog::Open(dir, 1, 0, &log).ok());
        append_n(*log, 10);
    }
    // Cut the last record in half.
    const auto seg = segments(dir).back();
    const auto size = fs::file_size(seg);
    fs::resize_file(seg, size - 20);

    OpLog::ScanStats st;
    auto seen = scan_all(dir, 1, &st);
    CHECK(seen.size() == 9);
    CHECK(st.dropped_bytes > 0);
    CHECK(!st.gap);

    // Reopen after the last good record: new records go to a new segment.
    {
        std::unique_ptr<OpLog> log;
        REQUIRE(OpLog::Open(dir, 10, 0, &log).ok());
        append_n(*log, 5, 100);
        CHECK(log->LastLsn() == 14);
    }
    auto all = scan_all(dir, 1, &st);
    REQUIRE(all.size() == 14);
    for (size_t i = 0; i < all.size(); ++i) CHECK(all[i].lsn == i + 1);
    CHECK(all[9].real_id == 100);
}

TEST_CASE("oplog: a corrupt record stops the segment; a gap stops the scan") {
    const auto dir = make_dir();
    {
        std::unique_ptr<OpLog> log;
        REQUIRE(OpLog::Open(dir, 1, 0, &log).ok());
        append_n(*log, 10);
    }
    const auto seg = segments(dir).back();
    // Flip a byte inside record 5's vector bytes.
    {
        const auto size = fs::file_size(seg);
        const auto rec_bytes = size / 10;
        std::fstream f(seg, std::ios::binary | std::ios::in | std::ios::out);
        const std::streamoff off = static_cast<std::streamoff>(rec_bytes * 4 + 50);
        f.seekg(off);
        char c = 0;
        f.read(&c, 1);
        c = static_cast<char>(c ^ 0x40);
        f.seekp(off);
        f.write(&c, 1);
    }
    OpLog::ScanStats st;
    auto seen = scan_all(dir, 1, &st);
    CHECK(seen.size() == 4);
    CHECK(st.dropped_bytes > 0);

    // A later segment that does not continue at lsn 5 is a gap: nothing
    // after the damage is applied.
    {
        std::unique_ptr<OpLog> log;
        REQUIRE(OpLog::Open(dir, 11, 0, &log).ok());
        append_n(*log, 3);
    }
    seen = scan_all(dir, 1, &st);
    CHECK(seen.size() == 4);
    CHECK(st.gap);
}

TEST_CASE("oplog: reopen removes segments beyond the accepted lsn") {
    const auto dir = make_dir();
    {
        std::unique_ptr<OpLog> log;
        REQUIRE(OpLog::Open(dir, 1, 512, &log).ok());
        append_n(*log, 40);
    }
    const auto before = segments(dir);
    REQUIRE(before.size() > 3);
    // Replay stopped where the third segment starts (e.g. a gap).
    const uint64_t next = std::stoull(fs::path(before[2]).filename().string().substr(0, 20));
    {
        std::unique_ptr<OpLog> log;
        REQUIRE(OpLog::Open(dir, next, 512, &log).ok());
        append_n(*log, 2, 500);
    }
    CHECK(segments(dir).size() == 3);
    auto seen = scan_all(dir, 1);
    REQUIRE(seen.size() == next + 1);
    CHECK(seen[next - 1].real_id == 500);
    CHECK(seen[next].lsn == next + 1);
}

TEST_CASE("oplog: truncation keeps every segment that holds a needed lsn") {
    const auto dir = make_dir();
    std::unique_ptr<OpLog> log;
    REQUIRE(OpLog::Open(dir, 1, 1024, &log).ok());
    append_n(*log, 100);
    const auto before = segments(dir).size();
    REQUIRE(log->TruncateBelow(60).ok());
    const auto after = segments(dir).size();
    CHECK(after < before);
    CHECK(after >= 1);
    auto seen = scan_all(dir, 60);
    REQUIRE(seen.size() == 41);
    CHECK(seen.front().lsn == 60);

    // Truncating past the end keeps the current segment.
    REQUIRE(log->TruncateBelow(1000).ok());
    CHECK(segments(dir).size() == 1);
    uint64_t lsn = 0;
    OpRecord r;
    r.op = OpType::kDelete;
    REQUIRE(log->Append(r, &lsn).ok());
    CHECK(lsn == 101);
}
