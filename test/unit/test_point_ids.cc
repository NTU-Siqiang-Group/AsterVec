// Point ids: keyed stores accept any 64-bit integer or string id and map it
// to a dense internal id; dense stores keep integer ids as internal ids.

#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "doctest.h"
#include "astervec_db.h"

using astervec::AsterVecDB;
using astervec::AsterVecDBOptions;
using astervec::IdMode;
using astervec::PointHit;
using astervec::PointId;
using astervec::SearchOptions;
using astervec::SearchResult;
using astervec::Span;
using astervec::Status;

namespace {

constexpr int kDim = 16;

std::string make_path() {
    char tmpl[] = "/tmp/astervec_ids_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    return dir;
}

AsterVecDBOptions opts_for(const std::string& path, IdMode mode, bool oplog = false) {
    AsterVecDBOptions o;
    o.dim = kDim;
    o.vec_file_capacity = 64;  // small on purpose: keyed ids stay dense
    o.vector_file_path = path + "/vecs.bin";
    o.random_seed = 7;
    o.id_mode = mode;
    o.enable_oplog = oplog;
    return o;
}

std::unique_ptr<AsterVecDB> open_db(const std::string& path, const AsterVecDBOptions& o) {
    std::unique_ptr<AsterVecDB> db;
    Status s = AsterVecDB::Open(path, o, &db);
    INFO(s.ToString());
    REQUIRE(s.ok());
    return db;
}

std::vector<float> vec(uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(kDim);
    for (auto& x : v) x = nd(rng);
    return v;
}

std::string uuid_like(int i) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%08x-1234-4abc-8def-%012x", i * 2654435761u, i);
    return buf;
}

// The point found at rank 0 when searching for `v`.
PointId top1(AsterVecDB* db, const std::vector<float>& v) {
    SearchOptions so;
    so.k = 3;
    so.ef_search = 64;
    std::vector<PointHit> hits;
    auto q = v;
    REQUIRE(db->SearchPoints(Span<float>(q), so, "", &hits).ok());
    REQUIRE(!hits.empty());
    return hits[0].id;
}

void run_and_crash(const std::function<bool()>& body) {
    pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) _exit(body() ? 0 : 3);
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
}

}  // namespace

TEST_CASE("point ids: every kind of id round-trips in a keyed store") {
    const auto path = make_path();
    const std::vector<PointId> ids = {
        PointId::Int(0),
        PointId::Int(42),
        PointId::Int(std::numeric_limits<uint64_t>::max()),
        PointId::Int(1ULL << 63),
        PointId::Str("doc-17#chunk-3"),
        PointId::Str(uuid_like(1)),
        PointId::Str(""),
    };
    {
        auto db = open_db(path, opts_for(path, IdMode::kKeyed));
        CHECK(db->GetIdMode() == IdMode::kKeyed);
        for (size_t i = 0; i < ids.size(); ++i) {
            auto v = vec(static_cast<uint32_t>(i));
            REQUIRE(db->Insert(ids[i], Span<float>(v),
                               "{\"i\":" + std::to_string(i) + "}").ok());
        }
        // 50 more so search has a graph to walk.
        for (int i = 0; i < 50; ++i) {
            auto v = vec(1000 + i);
            REQUIRE(db->Insert(PointId::Str(uuid_like(100 + i)), Span<float>(v)).ok());
        }
        CHECK(db->VectorCount() == ids.size() + 50);
        REQUIRE(db->Close().ok());
    }
    auto db = open_db(path, opts_for(path, IdMode::kAuto));
    CHECK(db->GetIdMode() == IdMode::kKeyed);
    for (size_t i = 0; i < ids.size(); ++i) {
        CAPTURE(i);
        std::vector<float> got;
        REQUIRE(db->Get(ids[i], &got).ok());
        CHECK(got.size() == static_cast<size_t>(kDim));
        std::string payload;
        REQUIRE(db->GetPayload(ids[i], &payload).ok());
        CHECK(payload == "{\"i\":" + std::to_string(i) + "}");
        CHECK(top1(db.get(), vec(static_cast<uint32_t>(i))) == ids[i]);
    }

    // Update, payload calls and delete by string id.
    const PointId doc = PointId::Str("doc-17#chunk-3");
    auto nv = vec(777);
    REQUIRE(db->Update(doc, Span<float>(nv)).ok());
    CHECK(top1(db.get(), nv) == doc);
    REQUIRE(db->UpdatePayload(doc, R"({"extra":true})").ok());
    std::string payload;
    REQUIRE(db->GetPayload(doc, &payload).ok());
    CHECK(payload.find("extra") != std::string::npos);
    REQUIRE(db->Delete(doc).ok());
    std::vector<float> got;
    CHECK(db->Get(doc, &got).IsNotFound());
    CHECK(db->Delete(PointId::Str("never-written")).ok());
    CHECK(db->Get(PointId::Str("never-written"), &got).IsNotFound());
    CHECK(db->SetPayload(PointId::Str("never-written"), "{}").IsNotFound());
    REQUIRE(db->Close().ok());
}

TEST_CASE("point ids: the integer 7 and the string \"7\" are different points") {
    const auto path = make_path();
    auto db = open_db(path, opts_for(path, IdMode::kKeyed));
    auto a = vec(1), b = vec(2);
    REQUIRE(db->Insert(PointId::Int(7), Span<float>(a), R"({"kind":"int"})").ok());
    REQUIRE(db->Insert(PointId::Str("7"), Span<float>(b), R"({"kind":"str"})").ok());
    CHECK(db->VectorCount() == 2);
    std::string p;
    REQUIRE(db->GetPayload(PointId::Int(7), &p).ok());
    CHECK(p == R"({"kind":"int"})");
    REQUIRE(db->GetPayload(PointId::Str("7"), &p).ok());
    CHECK(p == R"({"kind":"str"})");
    REQUIRE(db->Close().ok());
}

TEST_CASE("point ids: delete and re-insert, upsert, and dense internal ids") {
    const auto path = make_path();
    auto db = open_db(path, opts_for(path, IdMode::kKeyed));
    // Huge integer ids stay dense inside: capacity 64 is plenty for 200.
    for (int i = 0; i < 200; ++i) {
        auto v = vec(i);
        const uint64_t big = 0xFEDCBA9876543210ULL ^ static_cast<uint64_t>(i);
        REQUIRE(db->Insert(PointId::Int(big), Span<float>(v)).ok());
    }
    const PointId k = PointId::Int(0xFEDCBA9876543210ULL);
    REQUIRE(db->Delete(k).ok());
    CHECK(db->VectorCount() == 199);
    auto v = vec(5000);
    REQUIRE(db->Insert(k, Span<float>(v)).ok());
    CHECK(db->VectorCount() == 200);
    CHECK(top1(db.get(), v) == k);
    // Upsert: inserting an existing id replaces it.
    auto w = vec(6000);
    REQUIRE(db->Insert(k, Span<float>(w)).ok());
    CHECK(db->VectorCount() == 200);
    CHECK(top1(db.get(), w) == k);

    // SearchKnn reports integer keys in a keyed store.
    SearchOptions so;
    so.k = 1;
    so.ef_search = 64;
    std::vector<SearchResult> res;
    REQUIRE(db->SearchKnn(Span<float>(w), so, &res).ok());
    REQUIRE(res.size() == 1);
    CHECK(res[0].id == 0xFEDCBA9876543210ULL);
    REQUIRE(db->Close().ok());
}

TEST_CASE("point ids: modes are fixed when the store is created") {
    // Dense store: string ids are refused, plain integers work as before.
    const auto dense_path = make_path();
    {
        auto db = open_db(dense_path, opts_for(dense_path, IdMode::kAuto));
        CHECK(db->GetIdMode() == IdMode::kDense);
        auto v = vec(1);
        CHECK(db->Insert(PointId::Str("a"), Span<float>(v)).IsInvalidArgument());
        CHECK(db->Insert(PointId::Int(1ULL << 63), Span<float>(v)).IsInvalidArgument());
        REQUIRE(db->Insert(3, Span<float>(v)).ok());
        REQUIRE(db->Close().ok());
    }
    {
        std::unique_ptr<AsterVecDB> db;
        CHECK(AsterVecDB::Open(dense_path, opts_for(dense_path, IdMode::kKeyed), &db)
                  .IsInvalidArgument());
        CHECK(db == nullptr);
    }
    {
        auto db = open_db(dense_path, opts_for(dense_path, IdMode::kDense));
        std::vector<float> got;
        CHECK(db->Get(3, &got).ok());
        REQUIRE(db->Close().ok());
    }

    // Keyed store: reopening asks for keyed or auto, never dense.
    const auto keyed_path = make_path();
    {
        auto db = open_db(keyed_path, opts_for(keyed_path, IdMode::kKeyed));
        auto v = vec(2);
        REQUIRE(db->Insert(PointId::Str("x"), Span<float>(v)).ok());
        REQUIRE(db->Close().ok());
    }
    {
        std::unique_ptr<AsterVecDB> db;
        CHECK(AsterVecDB::Open(keyed_path, opts_for(keyed_path, IdMode::kDense), &db)
                  .IsInvalidArgument());
    }
    auto db = open_db(keyed_path, opts_for(keyed_path, IdMode::kAuto));
    CHECK(db->GetIdMode() == IdMode::kKeyed);
    // SearchKnn cannot report a string id.
    auto q = vec(2);
    SearchOptions so;
    so.k = 1;
    std::vector<SearchResult> res;
    CHECK(db->SearchKnn(Span<float>(q), so, &res).IsInvalidArgument());
    REQUIRE(db->Close().ok());
}

TEST_CASE("point ids: bulk build with caller ids") {
    const auto path = make_path();
    auto db = open_db(path, opts_for(path, IdMode::kKeyed));
    const int n = 300;
    std::vector<float> flat;
    std::vector<PointId> ids;
    for (int i = 0; i < n; ++i) {
        auto v = vec(i);
        flat.insert(flat.end(), v.begin(), v.end());
        ids.push_back(PointId::Str(uuid_like(i)));
    }
    astervec::BulkBuildOptions bo;
    bo.num_threads = 2;
    std::vector<PointId> dup = ids;
    dup[5] = dup[4];
    CHECK(db->BulkBuild(Span<const float>(flat), dup, bo).IsInvalidArgument());
    REQUIRE(db->BulkBuild(Span<const float>(flat), ids, bo).ok());
    CHECK(db->VectorCount() == static_cast<size_t>(n));
    int found = 0;
    for (int i = 0; i < n; i += 10) found += top1(db.get(), vec(i)) == ids[i];
    CHECK(found >= 28);  // of 30
    // New keys after a bulk build get fresh dense ids.
    auto v = vec(9999);
    REQUIRE(db->Insert(PointId::Str("after-bulk"), Span<float>(v)).ok());
    CHECK(top1(db.get(), v) == PointId::Str("after-bulk"));
    std::vector<float> got;
    REQUIRE(db->Get(ids[0], &got).ok());
    REQUIRE(db->Close().ok());

    // Dense stores refuse caller ids.
    const auto dense_path = make_path();
    auto ddb = open_db(dense_path, opts_for(dense_path, IdMode::kDense));
    CHECK(ddb->BulkBuild(Span<const float>(flat), ids, bo).IsInvalidArgument());
    REQUIRE(ddb->Close().ok());
}

TEST_CASE("point ids: concurrent writers on overlapping keys get one id per key") {
    const auto path = make_path();
    auto db = open_db(path, opts_for(path, IdMode::kKeyed));
    std::atomic<bool> fail{false};
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t) {
        ts.emplace_back([&, t] {
            for (int round = 0; round < 3; ++round) {
                for (int i = 0; i < 100; ++i) {
                    const int k = (i * 7 + t * 13 + round) % 100;
                    auto v = vec(k);
                    if (!db->Insert(PointId::Str("key-" + std::to_string(k)),
                                    Span<float>(v)).ok()) {
                        fail = true;
                    }
                }
            }
        });
    }
    for (auto& t : ts) t.join();
    REQUIRE(!fail.load());
    CHECK(db->VectorCount() == 100);
    REQUIRE(db->Close().ok());
    db = open_db(path, opts_for(path, IdMode::kAuto));
    int found = 0;
    for (int k = 0; k < 100; ++k) {
        std::vector<float> got;
        REQUIRE(db->Get(PointId::Str("key-" + std::to_string(k)), &got).ok());
        found += top1(db.get(), vec(k)) == PointId::Str("key-" + std::to_string(k));
    }
    CHECK(found >= 95);
    REQUIRE(db->Close().ok());
}

TEST_CASE("point ids: a crash keeps keyed ids consistent, with and without the log") {
    for (bool oplog : {true, false}) {
        CAPTURE(oplog);
        const auto path = make_path();
        const auto o = opts_for(path, IdMode::kKeyed, oplog);
        {
            auto db = open_db(path, o);
            for (int i = 0; i < 100; ++i) {
                auto v = vec(i);
                REQUIRE(db->Insert(PointId::Str(uuid_like(i)), Span<float>(v)).ok());
            }
            REQUIRE(db->Close().ok());
        }
        run_and_crash([&] {
            std::unique_ptr<AsterVecDB> db;
            if (!AsterVecDB::Open(path, o, &db).ok()) return false;
            for (int i = 100; i < 200; ++i) {
                auto v = vec(i);
                if (!db->Insert(PointId::Str(uuid_like(i)), Span<float>(v)).ok()) return false;
            }
            for (int i = 0; i < 10; ++i) {
                if (!db->Delete(PointId::Str(uuid_like(i))).ok()) return false;
            }
            return true;
        });
        auto db = open_db(path, o);
        std::vector<float> got;
        if (oplog) {
            CHECK(db->VectorCount() == 190);
            CHECK(db->Get(PointId::Str(uuid_like(5)), &got).IsNotFound());
            REQUIRE(db->Get(PointId::Str(uuid_like(150)), &got).ok());
            CHECK(top1(db.get(), vec(150)) == PointId::Str(uuid_like(150)));
        } else {
            // Back to the checkpoint: the first 100, nothing after.
            CHECK(db->VectorCount() == 100);
            REQUIRE(db->Get(PointId::Str(uuid_like(5)), &got).ok());
            CHECK(db->Get(PointId::Str(uuid_like(150)), &got).IsNotFound());
        }
        // Keys keep their dense ids: re-inserting lost keys and adding new
        // ones never lands on another key's vector.
        for (int i = 100; i < 220; ++i) {
            auto v = vec(i);
            REQUIRE(db->Insert(PointId::Str(uuid_like(i)), Span<float>(v)).ok());
        }
        int found = 0;
        for (int i = 10; i < 220; i += 7) {
            REQUIRE(db->Get(PointId::Str(uuid_like(i)), &got).ok());
            found += top1(db.get(), vec(i)) == PointId::Str(uuid_like(i));
        }
        CHECK(found >= 28);  // of 30
        REQUIRE(db->Close().ok());
    }
}
