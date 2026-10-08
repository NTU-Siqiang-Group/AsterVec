// Operation log replay: with enable_oplog, every acknowledged insert, update
// and delete survives a process crash. A child process writes and exits
// without Close(); the parent reopens and compares against a reference store
// that ran the same operations without crashing.

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "doctest.h"
#include "astervec_db.h"
#include "astervec_index.h"
#include "graph_store.h"

using astervec::AsterVecDB;
using astervec::AsterVecDBOptions;
using astervec::SearchOptions;
using astervec::SearchResult;
using astervec::Span;
using astervec::Status;

namespace {

namespace fs = std::filesystem;
constexpr int kDim = 16;
constexpr int kIds = 520;  // every id the scenarios touch is below this

std::string make_path() {
    char tmpl[] = "/tmp/astervec_replay_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    return dir;
}

AsterVecDBOptions options_for(const std::string& path, bool oplog) {
    AsterVecDBOptions opts;
    opts.dim = kDim;
    opts.vec_file_capacity = 4096;
    opts.vector_file_path = path + "/vecs.bin";
    opts.random_seed = 7;
    opts.enable_oplog = oplog;
    return opts;
}

std::vector<std::vector<float>> make_vectors(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::vector<float>> out(n, std::vector<float>(kDim));
    for (auto& v : out)
        for (auto& x : v) x = nd(rng);
    return out;
}

const auto& vecs() {
    static const auto v = make_vectors(700, 1234);
    return v;
}

std::string payload(const char* key, int i) {
    return std::string("{\"") + key + "\":" + std::to_string(i) + "}";
}

// Runs `body` in a child process that then exits without any cleanup.
void run_and_crash(const std::function<bool()>& body) {
    pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        int rc = body() ? 0 : 3;
        _exit(rc);
    }
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
}

#define TRY(expr) do { if (!(expr).ok()) return false; } while (0)

bool phase1(AsterVecDB* db) {
    for (int i = 0; i < 200; ++i) {
        auto v = vecs()[i];
        TRY(db->Insert(i, Span<float>(v), payload("v", i)));
    }
    return true;
}

// Inserts, upserts with and without payloads, deletes, re-inserts after
// delete (fresh update ids) and repeated updates of one id.
bool phase2(AsterVecDB* db) {
    for (int i = 200; i < 400; ++i) {
        auto v = vecs()[i];
        TRY(db->Insert(i, Span<float>(v), i % 2 ? std::string() : payload("v", i)));
    }
    for (int i = 0; i < 30; ++i) {
        auto v = vecs()[400 + i];
        if (i % 3 == 0) {
            TRY(db->Insert(i, Span<float>(v), payload("u", i)));
        } else {
            TRY(db->Update(i, Span<float>(v)));
        }
    }
    for (int i = 30; i < 50; ++i) TRY(db->Delete(i));
    for (int i = 40; i < 45; ++i) {
        auto v = vecs()[450 + i];
        TRY(db->Insert(i, Span<float>(v), payload("r", i)));
    }
    for (int i = 200; i < 210; ++i) TRY(db->Delete(i));
    for (int k = 0; k < 3; ++k) {
        auto v = vecs()[500 + k];
        TRY(db->Update(210, Span<float>(v)));
    }
    return true;
}

bool phase3(AsterVecDB* db) {
    for (int i = 400; i < 450; ++i) {
        auto v = vecs()[i + 200];
        TRY(db->Insert(i, Span<float>(v), payload("p3", i)));
    }
    for (int i = 50; i < 60; ++i) TRY(db->Delete(i));
    return true;
}

// Every id's vector, payload and liveness, plus the count, must match.
void check_same_state(AsterVecDB* got, AsterVecDB* want) {
    CHECK(got->VectorCount() == want->VectorCount());
    int mismatches = 0;
    for (int id = 0; id < kIds; ++id) {
        std::vector<float> a, b;
        Status sa = got->Get(id, &a);
        Status sb = want->Get(id, &b);
        if (sa.ok() != sb.ok() || (sa.ok() && a != b)) {
            if (mismatches < 5) {
                MESSAGE("id " << id << ": got " << sa.ToString() << ", want "
                              << sb.ToString() << (sa.ok() ? " (vectors differ)" : ""));
            }
            ++mismatches;
            continue;
        }
        if (!sa.ok()) continue;
        std::string pa, pb;
        REQUIRE(got->GetPayload(id, &pa).ok());
        REQUIRE(want->GetPayload(id, &pb).ok());
        if (pa != pb) {
            if (mismatches < 5) {
                MESSAGE("id " << id << ": payload " << pa << ", want " << pb);
            }
            ++mismatches;
        }
    }
    CHECK(mismatches == 0);
}

// Live vectors are found by an exact-match search.
void check_searchable(AsterVecDB* db) {
    SearchOptions so;
    so.k = 5;
    so.ef_search = 64;
    int live = 0, found = 0;
    for (int id = 0; id < kIds; ++id) {
        std::vector<float> v;
        if (!db->Get(id, &v).ok()) continue;
        ++live;
        std::vector<SearchResult> out;
        REQUIRE(db->SearchKnn(Span<float>(v), so, &out).ok());
        if (!out.empty() && out[0].id == static_cast<uint64_t>(id)) ++found;
    }
    REQUIRE(live > 0);
    CHECK(found >= live * 95 / 100);
}

std::unique_ptr<AsterVecDB> open_db(const std::string& path,
                                    const AsterVecDBOptions& opts) {
    std::unique_ptr<AsterVecDB> db;
    Status s = AsterVecDB::Open(path, opts, &db);
    INFO(s.ToString());
    REQUIRE(s.ok());
    return db;
}

// Reference store: the same operations, closed cleanly.
std::unique_ptr<AsterVecDB> reference(const std::vector<bool (*)(AsterVecDB*)>& phases) {
    const auto path = make_path();
    auto db = open_db(path, options_for(path, false));
    for (auto* p : phases) REQUIRE(p(db.get()));
    return db;
}

std::vector<std::string> segments(const std::string& dir) {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(dir)) out.push_back(e.path().string());
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

TEST_CASE("oplog: every acknowledged write survives a crash") {
    const auto path = make_path();
    const auto opts = options_for(path, true);
    {
        auto db = open_db(path, opts);
        REQUIRE(phase1(db.get()));
        REQUIRE(db->Close().ok());
    }
    run_and_crash([&] {
        std::unique_ptr<AsterVecDB> db;
        if (!AsterVecDB::Open(path, opts, &db).ok()) return false;
        return phase2(db.get());
    });

    auto db = open_db(path, opts);
    const auto rs = db->GetRecoveryStats();
    CHECK(rs.replayed_ops == 200 + 30 + 20 + 5 + 10 + 3);
    CHECK(rs.dropped_bytes == 0);
    CHECK(!rs.lsn_gap);
    auto want = reference({phase1, phase2});
    check_same_state(db.get(), want.get());
    check_searchable(db.get());

    // A second crash after recovery: both tails survive.
    REQUIRE(db->Close().ok());
    run_and_crash([&] {
        std::unique_ptr<AsterVecDB> d;
        if (!AsterVecDB::Open(path, opts, &d).ok()) return false;
        return phase3(d.get());
    });
    db = open_db(path, opts);
    CHECK(db->GetRecoveryStats().replayed_ops == 60);
    auto want3 = reference({phase1, phase2, phase3});
    check_same_state(db.get(), want3.get());
    check_searchable(db.get());

    // Recovery checkpointed: reopening again replays nothing.
    REQUIRE(db->Close().ok());
    db = open_db(path, opts);
    CHECK(db->GetRecoveryStats().replayed_ops == 0);
    check_same_state(db.get(), want3.get());
    REQUIRE(db->Close().ok());
}

TEST_CASE("oplog: a log left behind is replayed even when reopened with the log off") {
    const auto path = make_path();
    {
        auto db = open_db(path, options_for(path, true));
        REQUIRE(phase1(db.get()));
        REQUIRE(db->Close().ok());
    }
    run_and_crash([&] {
        std::unique_ptr<AsterVecDB> db;
        if (!AsterVecDB::Open(path, options_for(path, true), &db).ok()) return false;
        return phase2(db.get());
    });
    auto db = open_db(path, options_for(path, false));
    CHECK(db->GetRecoveryStats().replayed_ops > 0);
    auto want = reference({phase1, phase2});
    check_same_state(db.get(), want.get());
    CHECK(!fs::exists(path + "/oplog"));
    REQUIRE(db->Close().ok());
}

TEST_CASE("oplog: off by default, and then a crash keeps only the checkpoint") {
    const auto path = make_path();
    const auto opts = options_for(path, false);
    CHECK(!AsterVecDBOptions{}.enable_oplog);
    {
        auto db = open_db(path, opts);
        REQUIRE(phase1(db.get()));
        REQUIRE(db->Close().ok());
    }
    run_and_crash([&] {
        std::unique_ptr<AsterVecDB> db;
        if (!AsterVecDB::Open(path, opts, &db).ok()) return false;
        return phase2(db.get());
    });
    CHECK(!fs::exists(path + "/oplog"));
    auto db = open_db(path, opts);
    CHECK(db->GetRecoveryStats().replayed_ops == 0);
    CHECK(db->VectorCount() == 200);
    std::vector<float> v;
    CHECK(db->Get(300, &v).IsNotFound());
    REQUIRE(db->Close().ok());
}

TEST_CASE("oplog: a torn last record is dropped, everything before it replays") {
    const auto path = make_path();
    const auto opts = options_for(path, true);
    run_and_crash([&] {
        std::unique_ptr<AsterVecDB> db;
        if (!AsterVecDB::Open(path, opts, &db).ok()) return false;
        for (int i = 0; i < 300; ++i) {
            auto v = vecs()[i];
            if (!db->Insert(i, Span<float>(v)).ok()) return false;
        }
        return true;
    });
    const auto seg = segments(path + "/oplog").back();
    fs::resize_file(seg, fs::file_size(seg) - 7);

    auto db = open_db(path, opts);
    const auto rs = db->GetRecoveryStats();
    CHECK(rs.replayed_ops == 299);
    CHECK(rs.dropped_bytes > 0);
    CHECK(db->VectorCount() == 299);
    std::vector<float> v;
    CHECK(db->Get(298, &v).ok());
    CHECK(db->Get(299, &v).IsNotFound());
    check_searchable(db.get());

    // The store keeps working and the log keeps a clean sequence.
    auto v299 = vecs()[299];
    REQUIRE(db->Insert(299, Span<float>(v299)).ok());
    REQUIRE(db->Close().ok());
    db = open_db(path, opts);
    CHECK(db->VectorCount() == 300);
    REQUIRE(db->Close().ok());
}

TEST_CASE("oplog: falling back to the previous checkpoint replays from its lsn") {
    const auto path = make_path();
    auto opts = options_for(path, true);
    opts.oplog_segment_bytes = 4096;  // many segments, so truncation runs
    {
        auto db = open_db(path, opts);
        for (int i = 0; i < 100; ++i) {
            auto v = vecs()[i];
            REQUIRE(db->Insert(i, Span<float>(v)).ok());
        }
        REQUIRE(db->Checkpoint().ok());
        for (int i = 100; i < 200; ++i) {
            auto v = vecs()[i];
            REQUIRE(db->Insert(i, Span<float>(v)).ok());
        }
        REQUIRE(db->Close().ok());
    }
    run_and_crash([&] {
        std::unique_ptr<AsterVecDB> db;
        if (!AsterVecDB::Open(path, opts, &db).ok()) return false;
        for (int i = 200; i < 250; ++i) {
            auto v = vecs()[i];
            if (!db->Insert(i, Span<float>(v)).ok()) return false;
        }
        return true;
    });
    // Damage the newest checkpoint.
    {
        const std::string meta = path + "/lsm_vec_db.meta";
        std::fstream f(meta, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(f.is_open());
        f.seekg(0, std::ios::end);
        const auto size = static_cast<std::streamoff>(f.tellg());
        f.seekg(size / 2);
        char c = 0;
        f.read(&c, 1);
        c = static_cast<char>(c ^ 0x5a);
        f.seekp(size / 2);
        f.write(&c, 1);
    }
    auto db = open_db(path, opts);
    const auto rs = db->GetRecoveryStats();
    CHECK(rs.checkpoint_lsn == 100);
    CHECK(rs.replayed_ops == 150);
    CHECK(db->VectorCount() == 250);
    check_searchable(db.get());
    REQUIRE(db->Close().ok());
}

namespace {
// Every graph write fails, so an insert fails after it was logged.
struct FailingGraphStore final : astervec::IGraphStore {
    Status AddVertex(uint64_t) override { return Status::IOError("injected"); }
    Status AddEdgeBatch(const std::vector<std::pair<uint64_t, uint64_t>>&) override {
        return Status::IOError("injected");
    }
    Status DeleteEdgeBatch(const std::vector<std::pair<uint64_t, uint64_t>>&) override {
        return Status::IOError("injected");
    }
    Status GetOutEdges(uint64_t, std::vector<uint64_t>* out) override {
        out->clear();
        return Status::IOError("injected");
    }
};
}  // namespace

TEST_CASE("oplog: a write that failed after logging is not replayed") {
    const auto path = make_path();
    auto opts = options_for(path, true);
    opts.edge_cache_size = 0;
    run_and_crash([&] {
        std::unique_ptr<AsterVecDB> db;
        if (!AsterVecDB::Open(path, opts, &db).ok()) return false;
        for (int i = 0; i < 100; ++i) {
            auto v = vecs()[i];
            if (!db->Insert(i, Span<float>(v)).ok()) return false;
        }
        db->TESTONLY_index()->TESTONLY_swapGraphStore(
            std::make_unique<FailingGraphStore>());
        auto v = vecs()[500];
        return !db->Insert(500, Span<float>(v)).ok();  // must fail
    });
    auto db = open_db(path, opts);
    CHECK(db->GetRecoveryStats().replayed_ops == 102);  // 101 ops + 1 abort
    CHECK(db->VectorCount() == 100);
    std::vector<float> v;
    CHECK(db->Get(500, &v).IsNotFound());
    CHECK(db->Get(99, &v).ok());
    REQUIRE(db->Close().ok());
}

TEST_CASE("checkpoint_interval_ops bounds what a crash has to replay or loses") {
    for (bool oplog : {true, false}) {
        CAPTURE(oplog);
        const auto path = make_path();
        auto opts = options_for(path, oplog);
        opts.checkpoint_interval_ops = 50;
        run_and_crash([&] {
            std::unique_ptr<AsterVecDB> db;
            if (!AsterVecDB::Open(path, opts, &db).ok()) return false;
            for (int i = 0; i < 175; ++i) {
                auto v = vecs()[i];
                if (!db->Insert(i, Span<float>(v)).ok()) return false;
            }
            return true;
        });
        auto db = open_db(path, opts);
        if (oplog) {
            CHECK(db->GetRecoveryStats().checkpoint_lsn == 150);
            CHECK(db->GetRecoveryStats().replayed_ops == 25);
            CHECK(db->VectorCount() == 175);
        } else {
            CHECK(db->VectorCount() == 150);
        }
        check_searchable(db.get());
        REQUIRE(db->Close().ok());
    }
}

TEST_CASE("oplog: concurrent writers with automatic checkpoints") {
    const auto path = make_path();
    auto opts = options_for(path, true);
    opts.checkpoint_interval_ops = 100;
    opts.oplog_segment_bytes = 8192;
    {
        auto db = open_db(path, opts);
        std::atomic<bool> fail{false};
        std::atomic<bool> stop{false};
        std::vector<std::thread> writers;
        for (int t = 0; t < 4; ++t) {
            writers.emplace_back([&, t] {
                for (int i = t; i < 400; i += 4) {
                    auto v = vecs()[i];
                    if (!db->Insert(i, Span<float>(v), payload("t", t)).ok()) fail = true;
                    if (i % 10 == 0 && !db->Delete(i).ok()) fail = true;
                }
            });
        }
        std::thread searcher([&] {
            SearchOptions so;
            so.k = 5;
            so.ef_search = 32;
            while (!stop.load()) {
                auto q = vecs()[650];
                std::vector<SearchResult> out;
                auto s = db->SearchKnn(Span<float>(q), so, &out);
                if (!s.ok() && !s.IsNotFound()) fail = true;
            }
        });
        for (auto& w : writers) w.join();
        stop = true;
        searcher.join();
        REQUIRE(!fail.load());
        CHECK(db->VectorCount() == 360);
        REQUIRE(db->Close().ok());
    }
    auto db = open_db(path, opts);
    CHECK(db->GetRecoveryStats().replayed_ops == 0);
    CHECK(db->VectorCount() == 360);
    check_searchable(db.get());
    REQUIRE(db->Close().ok());
}
