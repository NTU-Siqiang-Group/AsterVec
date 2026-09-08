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

    // Create-time persistence: meta exists with v3 + VARINT before Close.
    CHECK(ReadMetaField32(meta, kVersionOff) == 3);
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
