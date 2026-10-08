// Batch2 S6: graph adjacency encoding persistence (meta v3) tests.
//
// The differential deliberately compares ONE index with itself across the
// close/reopen persistence boundary (encode -> disk -> decode on the real
// path) instead of comparing two independently built indexes: randomLevel()
// uses thread_local RNG state, so two same-seed builds in one process do
// not produce identical graphs (review 2026-09-08, pt.4).

#include "doctest.h"
#include "astervec_db.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

std::string NewTempDir(const char* prefix) {
    std::string tmpl = std::string("/tmp/") + prefix + "_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* dir = mkdtemp(buf.data());
    REQUIRE(dir != nullptr);
    return std::string(dir);
}

astervec::AsterVecDBOptions BaseOpts(const std::string& path, int dim) {
    astervec::AsterVecDBOptions opts;
    opts.dim = dim;
    opts.m = 8;
    opts.m_max = 24;
    opts.ef_construction = 64;
    opts.vec_file_capacity = 4096;
    opts.vector_file_path = path + "/vecs.bin";
    return opts;
}

std::vector<std::vector<float>> MakeVectors(int n, int dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<std::vector<float>> vecs(n, std::vector<float>(dim));
    for (auto& v : vecs)
        for (auto& x : v) x = u(rng);
    return vecs;
}

std::vector<std::vector<astervec::node_id_t>> SearchAll(
    astervec::AsterVecDB* db,
    const std::vector<std::vector<float>>& queries, int k) {
    std::vector<std::vector<astervec::node_id_t>> out;
    for (const auto& q : queries) {
        astervec::SearchOptions so;
        so.k = k;
        so.ef_search = 64;
        std::vector<astervec::SearchResult> res;
        REQUIRE(db->SearchKnn(astervec::Span<float>(
                                  const_cast<float*>(q.data()), q.size()),
                              so, &res)
                    .ok());
        std::vector<astervec::node_id_t> ids;
        for (const auto& r : res) ids.push_back(r.id);
        out.push_back(std::move(ids));
    }
    return out;
}

uint32_t ReadMetaField32(const std::string& meta_path, size_t offset) {
    std::ifstream in(meta_path, std::ios::binary);
    REQUIRE(in.is_open());
    in.seekg(static_cast<std::streamoff>(offset));
    uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    REQUIRE(static_cast<bool>(in));
    return v;
}

// Meta layout: magic(8) version(4) entry(8) maxLayer(4) nodes(8) deleted(8)
constexpr size_t kVersionOff = 8;
constexpr size_t kEncodingOff = 40;

}  // namespace

TEST_CASE("encoding: varint survives close/reopen with identical results") {
    std::string dir = NewTempDir("enc_varint");
    std::string meta = dir + "/lsm_vec_db.meta";
    const int dim = 16, n = 400, q = 25;
    auto vecs = MakeVectors(n, dim, 0xA1);
    auto queries = MakeVectors(q, dim, 0xB2);

    unsetenv("ASTERVEC_GRAPH_VARINT");
    std::unique_ptr<astervec::AsterVecDB> db;
    auto opts = BaseOpts(dir, dim);
    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());

    // Create-time persistence: meta exists with v4 + VARINT before Close.
    CHECK(ReadMetaField32(meta, kVersionOff) == 4);
    CHECK(ReadMetaField32(meta, kEncodingOff) == 2);  // ENCODING_TYPE_VARINT

    for (int i = 0; i < n; i++) {
        REQUIRE(db->Insert(static_cast<astervec::node_id_t>(i),
                           astervec::Span<float>(vecs[i].data(), dim))
                    .ok());
    }
    // Churn: updates + deletes so update-ids (bit 63) reach the graph too.
    for (int i = 0; i < 40; i++) {
        REQUIRE(db->Update(static_cast<astervec::node_id_t>(i),
                           astervec::Span<float>(vecs[(i + 7) % n].data(), dim))
                    .ok());
    }
    for (int i = 300; i < 320; i++) {
        REQUIRE(db->Delete(static_cast<astervec::node_id_t>(i)).ok());
    }

    auto before = SearchAll(db.get(), queries, 10);
    REQUIRE(db->Close().ok());
    db.reset();

    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
    auto after = SearchAll(db.get(), queries, 10);
    CHECK(before == after);  // decode(encode(graph)) must be exact
    REQUIRE(db->Close().ok());
}

TEST_CASE("encoding: kill-switch creates NONE and it sticks without env") {
    std::string dir = NewTempDir("enc_none");
    std::string meta = dir + "/lsm_vec_db.meta";
    const int dim = 16, n = 200;
    auto vecs = MakeVectors(n, dim, 0xC3);
    auto queries = MakeVectors(15, dim, 0xD4);

    setenv("ASTERVEC_GRAPH_VARINT", "0", 1);
    std::unique_ptr<astervec::AsterVecDB> db;
    auto opts = BaseOpts(dir, dim);
    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
    unsetenv("ASTERVEC_GRAPH_VARINT");  // must not matter after create

    CHECK(ReadMetaField32(meta, kEncodingOff) == 0);  // ENCODING_TYPE_NONE

    for (int i = 0; i < n; i++) {
        REQUIRE(db->Insert(static_cast<astervec::node_id_t>(i),
                           astervec::Span<float>(vecs[i].data(), dim))
                    .ok());
    }
    auto before = SearchAll(db.get(), queries, 10);
    REQUIRE(db->Close().ok());
    db.reset();

    // Reopen WITHOUT the env var: recorded encoding must be replayed.
    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
    CHECK(ReadMetaField32(meta, kEncodingOff) == 0);
    auto after = SearchAll(db.get(), queries, 10);
    CHECK(before == after);
    REQUIRE(db->Close().ok());
}

TEST_CASE("encoding: legal enum flip is caught by the header CRC") {
    std::string dir = NewTempDir("enc_flip");
    std::string meta = dir + "/lsm_vec_db.meta";
    const int dim = 16;
    auto vecs = MakeVectors(50, dim, 0xE5);

    unsetenv("ASTERVEC_GRAPH_VARINT");
    std::unique_ptr<astervec::AsterVecDB> db;
    auto opts = BaseOpts(dir, dim);
    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
    for (int i = 0; i < 50; i++) {
        REQUIRE(db->Insert(static_cast<astervec::node_id_t>(i),
                           astervec::Span<float>(vecs[i].data(), dim))
                    .ok());
    }
    REQUIRE(db->Close().ok());
    db.reset();

    // No previous checkpoint to fall back to (that path is covered in
    // test_crash_recovery.cc), so the damaged header must be refused.
    std::remove((meta + ".prev").c_str());
    // 2 (VARINT) -> 0 (NONE): both legal enum values; only the CRC can tell.
    {
        std::fstream f(meta, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(f.is_open());
        f.seekp(kEncodingOff);
        uint32_t zero = 0;
        f.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    }
    auto st = astervec::AsterVecDB::Open(dir, opts, &db);
    CHECK(st.IsCorruption());
    CHECK(db == nullptr);
}

TEST_CASE("encoding: graph present but meta missing is refused") {
    std::string dir = NewTempDir("enc_absent");
    std::string meta = dir + "/lsm_vec_db.meta";
    const int dim = 16;
    auto vecs = MakeVectors(50, dim, 0xF6);

    unsetenv("ASTERVEC_GRAPH_VARINT");
    std::unique_ptr<astervec::AsterVecDB> db;
    auto opts = BaseOpts(dir, dim);
    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
    for (int i = 0; i < 50; i++) {
        REQUIRE(db->Insert(static_cast<astervec::node_id_t>(i),
                           astervec::Span<float>(vecs[i].data(), dim))
                    .ok());
    }
    REQUIRE(db->Close().ok());
    db.reset();

    REQUIRE(std::remove(meta.c_str()) == 0);
    std::remove((meta + ".prev").c_str());
    auto st = astervec::AsterVecDB::Open(dir, opts, &db);
    CHECK(st.IsCorruption());

    // reinit is the documented way out.
    auto ropts = opts;
    ropts.reinit = true;
    REQUIRE(astervec::AsterVecDB::Open(dir, ropts, &db).ok());
    CHECK(ReadMetaField32(meta, kEncodingOff) == 2);
    REQUIRE(db->Close().ok());
}

TEST_CASE("encoding: truncated meta header is rejected") {
    std::string dir = NewTempDir("enc_trunc");
    std::string meta = dir + "/lsm_vec_db.meta";
    const int dim = 16;

    unsetenv("ASTERVEC_GRAPH_VARINT");
    std::unique_ptr<astervec::AsterVecDB> db;
    auto opts = BaseOpts(dir, dim);
    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
    REQUIRE(db->Close().ok());
    db.reset();

    // Truncate to 20 bytes (mid-header).
    {
        std::ifstream in(meta, std::ios::binary);
        std::vector<char> head(20);
        in.read(head.data(), 20);
        REQUIRE(static_cast<bool>(in));
        std::ofstream out(meta, std::ios::binary | std::ios::trunc);
        out.write(head.data(), 20);
    }
    auto st = astervec::AsterVecDB::Open(dir, opts, &db);
    CHECK(st.IsCorruption());
}

// ---------- Batch2 R-review additions (2026-09-08) ----------

#include <thread>
#include "astervec_index.h"
#include "graph_store.h"

TEST_CASE("encoding: crash before first Close still replays the encoding") {
    // R5.2: create -> insert -> destroy WITHOUT Close -> reopen. The
    // create-time meta stub must carry the encoding; the index is empty
    // (today's no-WAL semantics) but consistent and correctly decoded.
    std::string dir = NewTempDir("enc_crash");
    std::string meta = dir + "/lsm_vec_db.meta";
    const int dim = 16;
    auto vecs = MakeVectors(50, dim, 0x11);

    unsetenv("ASTERVEC_GRAPH_VARINT");
    auto opts = BaseOpts(dir, dim);
    {
        std::unique_ptr<astervec::AsterVecDB> db;
        REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
        for (int i = 0; i < 50; i++) {
            REQUIRE(db->Insert(static_cast<astervec::node_id_t>(i),
                               astervec::Span<float>(vecs[i].data(), dim))
                        .ok());
        }
        db.reset();  // simulated crash: no Close()
    }
    CHECK(ReadMetaField32(meta, kEncodingOff) == 2);  // stub survived
    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
    // Index state is empty (no-WAL semantics: upper layers lived only in
    // memory); the pre-existing contract returns NotFound on empty indexes.
    auto q = MakeVectors(1, dim, 0x12);
    astervec::SearchOptions so;
    so.k = 5;
    so.ef_search = 64;
    std::vector<astervec::SearchResult> res;
    auto sst = db->SearchKnn(
        astervec::Span<float>(q[0].data(), q[0].size()), so, &res);
    CHECK(sst.IsNotFound());
    REQUIRE(db->Close().ok());
}

TEST_CASE("encoding: PeekGraphEncoding is safe under concurrent first use") {
    // R4: exercises the CRC-table init from many threads (TSAN gate).
    std::string dir = NewTempDir("enc_conc");
    std::string meta = dir + "/lsm_vec_db.meta";
    unsetenv("ASTERVEC_GRAPH_VARINT");
    {
        std::unique_ptr<astervec::AsterVecDB> db;
        auto opts = BaseOpts(dir, 16);
        REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
        REQUIRE(db->Close().ok());
    }
    std::vector<std::thread> ts;
    std::atomic<int> failures{0};
    for (int i = 0; i < 8; i++) {
        ts.emplace_back([&] {
            for (int j = 0; j < 100; j++) {
                uint32_t enc = 99;
                auto st = astervec::AsterVec::PeekGraphEncoding(meta, &enc);
                if (!st.ok() || enc != 2) failures.fetch_add(1);
            }
        });
    }
    for (auto& t : ts) t.join();
    CHECK(failures.load() == 0);
}

TEST_CASE("encoding: corrupted meta BODY is rejected by the payload CRC") {
    // R3: flip one byte in a node vector (well past the fixed header).
    std::string dir = NewTempDir("enc_body");
    std::string meta = dir + "/lsm_vec_db.meta";
    const int dim = 16;
    auto vecs = MakeVectors(80, dim, 0x21);
    unsetenv("ASTERVEC_GRAPH_VARINT");
    auto opts = BaseOpts(dir, dim);
    {
        std::unique_ptr<astervec::AsterVecDB> db;
        REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
        for (int i = 0; i < 80; i++) {
            REQUIRE(db->Insert(static_cast<astervec::node_id_t>(i),
                               astervec::Span<float>(vecs[i].data(), dim))
                        .ok());
        }
        REQUIRE(db->Close().ok());
    }
    {
        std::fstream f(meta, std::ios::binary | std::ios::in | std::ios::out);
        f.seekg(0, std::ios::end);
        auto size = static_cast<size_t>(f.tellg());
        REQUIRE(size > 120);
        f.seekp(static_cast<std::streamoff>(size / 2));  // deep in the body
        char b = 0x5A;
        f.write(&b, 1);
    }
    // Without a previous checkpoint to fall back to, the damage is fatal.
    std::remove((meta + ".prev").c_str());
    std::unique_ptr<astervec::AsterVecDB> db;
    CHECK(astervec::AsterVecDB::Open(dir, opts, &db).IsCorruption());
}

TEST_CASE("encoding: v3->v2 version relabel fails cleanly, no throw") {
    // R3: relabeling bypasses the v3 CRC path entirely; the bounded legacy
    // parse must return Corruption instead of dying in an allocation.
    std::string dir = NewTempDir("enc_relabel");
    std::string meta = dir + "/lsm_vec_db.meta";
    const int dim = 16;
    auto vecs = MakeVectors(60, dim, 0x31);
    unsetenv("ASTERVEC_GRAPH_VARINT");
    auto opts = BaseOpts(dir, dim);
    {
        std::unique_ptr<astervec::AsterVecDB> db;
        REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
        for (int i = 0; i < 60; i++) {
            REQUIRE(db->Insert(static_cast<astervec::node_id_t>(i),
                               astervec::Span<float>(vecs[i].data(), dim))
                        .ok());
        }
        REQUIRE(db->Close().ok());
    }
    {
        std::fstream f(meta, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(kVersionOff);
        uint32_t two = 2;
        f.write(reinterpret_cast<const char*>(&two), sizeof(two));
    }
    std::unique_ptr<astervec::AsterVecDB> db;
    auto st = astervec::AsterVecDB::Open(dir, opts, &db);
    CHECK((st.IsCorruption() || st.IsIOError() || st.IsInvalidArgument()));
}

namespace {
// R2: fault-injection backend — every graph operation fails.
struct FailingGraphStore final : astervec::IGraphStore {
    Status AddVertex(uint64_t) override {
        return Status::Corruption("injected");
    }
    Status AddEdgeBatch(
        const std::vector<std::pair<uint64_t, uint64_t>>&) override {
        return Status::Corruption("injected");
    }
    Status DeleteEdgeBatch(
        const std::vector<std::pair<uint64_t, uint64_t>>&) override {
        return Status::Corruption("injected");
    }
    Status GetOutEdges(uint64_t, std::vector<uint64_t>* out) override {
        out->clear();
        return Status::Corruption("injected");
    }
};
}  // namespace

TEST_CASE("R2: graph corruption fails Search and Insert at the public API") {
    std::string dir = NewTempDir("enc_inject");
    const int dim = 16;
    auto vecs = MakeVectors(120, dim, 0x41);
    unsetenv("ASTERVEC_GRAPH_VARINT");
    auto opts = BaseOpts(dir, dim);
    opts.edge_cache_size = 0;  // force every hop to the (failing) store

    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
    for (int i = 0; i < 120; i++) {
        REQUIRE(db->Insert(static_cast<astervec::node_id_t>(i),
                           astervec::Span<float>(vecs[i].data(), dim))
                    .ok());
    }

    db->TESTONLY_index()->TESTONLY_swapGraphStore(
        std::make_unique<FailingGraphStore>());

    astervec::SearchOptions so;
    so.k = 10;
    so.ef_search = 64;
    std::vector<astervec::SearchResult> res;
    auto sst = db->SearchKnn(
        astervec::Span<float>(vecs[0].data(), dim), so, &res);
    CHECK(sst.IsCorruption());  // NOT "OK with truncated results"

    // Sticky: subsequent calls fail too.
    auto sst2 = db->SearchKnn(
        astervec::Span<float>(vecs[1].data(), dim), so, &res);
    CHECK(sst2.IsCorruption());

    // Write path propagates directly.
    auto ist = db->Insert(500, astervec::Span<float>(vecs[2].data(), dim));
    CHECK(!ist.ok());
}

TEST_CASE("R5: NONE vs VARINT cross-process differential is id-exact") {
    const char* driver = ENCODING_DIFF_DRIVER;
    std::string base = NewTempDir("enc_diff");
    std::string cmd_v = std::string("ASTERVEC_GRAPH_VARINT=1 \"") + driver +
                        "\" " + base + "/v " + base + "/v.out > " + base +
                        "/v.log 2>&1";
    std::string cmd_n = std::string("ASTERVEC_GRAPH_VARINT=0 \"") + driver +
                        "\" " + base + "/n " + base + "/n.out > " + base +
                        "/n.log 2>&1";
    REQUIRE(system((std::string("mkdir -p ") + base + "/v " + base + "/n").c_str()) == 0);
    REQUIRE(system(cmd_v.c_str()) == 0);
    REQUIRE(system(cmd_n.c_str()) == 0);
    std::ifstream fv(base + "/v.out"), fn(base + "/n.out");
    REQUIRE(fv.is_open());
    REQUIRE(fn.is_open());
    std::string sv((std::istreambuf_iterator<char>(fv)),
                   std::istreambuf_iterator<char>());
    std::string sn((std::istreambuf_iterator<char>(fn)),
                   std::istreambuf_iterator<char>());
    REQUIRE(!sv.empty());
    CHECK(sv == sn);  // byte-identical result ids across encodings
}
