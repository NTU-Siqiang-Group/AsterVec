// Crash recovery: the index must stay searchable after an unclean shutdown.
// A child process writes and exits without Close() (a real crash: no
// destructors, no metadata write); the parent reopens and searches.

#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "doctest.h"
#include "astervec_db.h"
#include "astervec_index.h"
#include "durable_io.h"

using astervec::AsterVecDB;
using astervec::AsterVecDBOptions;
using astervec::SearchOptions;
using astervec::SearchResult;
using astervec::Span;

namespace {

constexpr int kDim = 16;

std::string make_path() {
    char tmpl[] = "/tmp/astervec_crash_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    return dir;
}

AsterVecDBOptions options_for(const std::string& path) {
    AsterVecDBOptions opts;
    opts.dim              = kDim;
    opts.vec_file_capacity = 4096;
    opts.vector_file_path = path + "/vecs.bin";
    opts.random_seed      = 7;
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

// Runs `body` in a child process that then exits without any cleanup.
template <typename F>
void run_and_crash(F body) {
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

}  // namespace

TEST_CASE("crash: index stays searchable after writes that were never checkpointed") {
    const auto path = make_path();
    const auto opts = options_for(path);
    const auto vecs = make_vectors(600, 11);

    // 300 vectors, then a clean close (a checkpoint).
    {
        std::unique_ptr<AsterVecDB> db;
        REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
        for (int i = 0; i < 300; ++i) {
            auto v = vecs[i];
            REQUIRE(db->Insert(i, Span<float>(v)).ok());
        }
        REQUIRE(db->Close().ok());
    }

    // 300 more inserts and a few updates and deletes, then a crash.
    run_and_crash([&] {
        std::unique_ptr<AsterVecDB> db;
        if (!AsterVecDB::Open(path, opts, &db).ok()) return false;
        for (int i = 300; i < 600; ++i) {
            auto v = vecs[i];
            if (!db->Insert(i, Span<float>(v)).ok()) return false;
        }
        for (int i = 0; i < 20; ++i) {
            auto v = vecs[599 - i];
            if (!db->Update(i, Span<float>(v)).ok()) return false;
        }
        return true;
    });

    std::unique_ptr<AsterVecDB> db;
    REQUIRE(AsterVecDB::Open(path, opts, &db).ok());

    SearchOptions so;
    so.k = 10;
    so.ef_search = 64;

    // Every search succeeds and only returns checkpointed ids.
    const auto queries = make_vectors(100, 99);
    for (const auto& q : queries) {
        auto qq = q;
        std::vector<SearchResult> out;
        REQUIRE(db->SearchKnn(Span<float>(qq), so, &out).ok());
        CHECK(!out.empty());
        for (const auto& r : out) CHECK(r.id < 300);
    }

    // Checkpointed vectors are still found exactly.
    int found = 0;
    for (int i = 20; i < 300; i += 7) {
        auto qq = vecs[i];
        std::vector<SearchResult> out;
        REQUIRE(db->SearchKnn(Span<float>(qq), so, &out).ok());
        if (!out.empty() && out[0].id == static_cast<uint64_t>(i)) ++found;
    }
    CHECK(found >= 38);  // of 40

    // The graph really did hold edges to the lost nodes.
    CHECK(db->index_for_test()->danglingEdgesSkipped() > 0);

    // The store keeps working: the lost ids can be written again.
    for (int i = 300; i < 600; ++i) {
        auto v = vecs[i];
        REQUIRE(db->Insert(i, Span<float>(v)).ok());
    }
    int refound = 0;
    for (int i = 300; i < 600; i += 15) {
        auto qq = vecs[i];
        std::vector<SearchResult> out;
        REQUIRE(db->SearchKnn(Span<float>(qq), so, &out).ok());
        if (!out.empty() && out[0].id == static_cast<uint64_t>(i)) ++refound;
    }
    CHECK(refound >= 18);  // of 20
    REQUIRE(db->Close().ok());
}

TEST_CASE("crash: everything up to the last Checkpoint() survives") {
    const auto path = make_path();
    const auto opts = options_for(path);
    const auto vecs = make_vectors(500, 21);

    run_and_crash([&] {
        std::unique_ptr<AsterVecDB> db;
        if (!AsterVecDB::Open(path, opts, &db).ok()) return false;
        for (int i = 0; i < 300; ++i) {
            auto v = vecs[i];
            if (!db->Insert(i, Span<float>(v), R"({"k":1})").ok()) return false;
        }
        if (!db->Checkpoint().ok()) return false;
        for (int i = 300; i < 500; ++i) {
            auto v = vecs[i];
            if (!db->Insert(i, Span<float>(v)).ok()) return false;
        }
        return true;
    });

    std::unique_ptr<AsterVecDB> db;
    REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
    SearchOptions so;
    so.k = 5;
    so.ef_search = 64;
    int found = 0;
    for (int i = 0; i < 300; ++i) {
        auto qq = vecs[i];
        std::vector<SearchResult> out;
        REQUIRE(db->SearchKnn(Span<float>(qq), so, &out).ok());
        for (const auto& r : out) CHECK(r.id < 300);
        if (!out.empty() && out[0].id == static_cast<uint64_t>(i)) ++found;
    }
    CHECK(found >= 290);
    std::string payload;
    REQUIRE(db->GetPayload(5, &payload).ok());
    CHECK(payload == R"({"k":1})");
    CHECK(db->VectorCount() == 300);
    REQUIRE(db->Close().ok());
}

TEST_CASE("checkpoint: a damaged latest checkpoint falls back to the previous one") {
    const auto path = make_path();
    const auto opts = options_for(path);
    const auto vecs = make_vectors(300, 31);
    const std::string meta = path + "/lsm_vec_db.meta";

    {
        std::unique_ptr<AsterVecDB> db;
        REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
        for (int i = 0; i < 200; ++i) {
            auto v = vecs[i];
            REQUIRE(db->Insert(i, Span<float>(v)).ok());
        }
        REQUIRE(db->Checkpoint().ok());
        for (int i = 200; i < 300; ++i) {
            auto v = vecs[i];
            REQUIRE(db->Insert(i, Span<float>(v)).ok());
        }
        REQUIRE(db->Close().ok());
    }
    REQUIRE(astervec::durable::file_exists(meta + ".prev"));

    // Flip one byte in the middle of the newest checkpoint.
    {
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

    std::unique_ptr<AsterVecDB> db;
    REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
    SearchOptions so;
    so.k = 5;
    so.ef_search = 64;
    int found = 0;
    for (int i = 0; i < 200; i += 4) {
        auto qq = vecs[i];
        std::vector<SearchResult> out;
        REQUIRE(db->SearchKnn(Span<float>(qq), so, &out).ok());
        for (const auto& r : out) CHECK(r.id < 200);
        if (!out.empty() && out[0].id == static_cast<uint64_t>(i)) ++found;
    }
    CHECK(found >= 48);  // of 50
    REQUIRE(db->Close().ok());
}

TEST_CASE("checkpoint: a half-written checkpoint file is ignored") {
    const auto path = make_path();
    const auto opts = options_for(path);
    const auto vecs = make_vectors(100, 41);
    const std::string meta = path + "/lsm_vec_db.meta";
    {
        std::unique_ptr<AsterVecDB> db;
        REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
        for (int i = 0; i < 100; ++i) {
            auto v = vecs[i];
            REQUIRE(db->Insert(i, Span<float>(v)).ok());
        }
        REQUIRE(db->Close().ok());
    }
    // What a crash in the middle of writing the next checkpoint leaves behind.
    {
        std::ofstream tmp(meta + ".tmp", std::ios::binary);
        tmp << "LSMVMETA-truncated";
    }
    std::unique_ptr<AsterVecDB> db;
    REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
    CHECK(!astervec::durable::file_exists(meta + ".tmp"));
    CHECK(db->VectorCount() == 100);
    REQUIRE(db->Close().ok());
}

TEST_CASE("checkpoint: reinit also clears the old payloads") {
    const auto path = make_path();
    auto opts = options_for(path);
    std::vector<float> v(kDim, 0.5f);
    {
        std::unique_ptr<AsterVecDB> db;
        REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
        REQUIRE(db->Insert(3, Span<float>(v), R"({"old":true})").ok());
        REQUIRE(db->Close().ok());
    }
    opts.reinit = true;
    std::unique_ptr<AsterVecDB> db;
    REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
    REQUIRE(db->Insert(3, Span<float>(v)).ok());
    std::string payload;
    REQUIRE(db->GetPayload(3, &payload).ok());
    CHECK(payload == "{}");
    REQUIRE(db->Close().ok());
}

TEST_CASE("checkpoint: a crash while creating a new store does not brick it") {
    const auto path = make_path();
    const auto opts = options_for(path);
    const std::string meta = path + "/lsm_vec_db.meta";
    {
        std::unique_ptr<AsterVecDB> db;
        REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
        REQUIRE(db->Close().ok());
    }
    CHECK(!astervec::durable::file_exists(path + "/lsm_vec_db.creating"));

    // What a kill inside the creating Open leaves: graph files, the marker
    // (varint = 2), no metadata file yet.
    std::remove(meta.c_str());
    std::remove((meta + ".prev").c_str());
    {
        std::ofstream m(path + "/lsm_vec_db.creating");
        m << 2 << "\n";
    }
    std::unique_ptr<AsterVecDB> db;
    REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
    CHECK(!astervec::durable::file_exists(path + "/lsm_vec_db.creating"));
    auto v = make_vectors(1, 5)[0];
    REQUIRE(db->Insert(1, Span<float>(v)).ok());
    REQUIRE(db->Close().ok());

    // Without the marker, missing metadata next to graph files stays an error.
    std::remove(meta.c_str());
    std::remove((meta + ".prev").c_str());
    db.reset();
    CHECK(AsterVecDB::Open(path, opts, &db).IsCorruption());
}
