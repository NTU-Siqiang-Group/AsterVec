// Batch2 S6: back-compat against a REAL pre-batch store.
//
// fixtures/prebatch_db was built by the released v0.2.2 binary (pip wheel)
// via BulkBuild — so its meta is v2 (no encoding field), its adjacency is
// raw-encoded, and its property CFs contain the per-vertex records that
// pre-#44 AddVertex wrote. fixtures/prebatch_expected.txt holds the search
// results the v0.2.2 binary itself returned for prebatch_queries.txt.
//
// This proves, on a real artifact rather than by reasoning:
//  - v2 meta resolves to raw encoding and the store opens (S4 fallback);
//  - populated-but-no-longer-written property CFs are harmless (S3);
//  - search results are exactly what the old binary produced;
//  - Close upgrades the meta to v3 with encoding 0, and it reopens.

#include "doctest.h"
#include "astervec_db.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::string FixtureRoot() {
    // __FILE__ = <repo>/test/unit/test_prebatch_backcompat.cc
    return std::filesystem::path(__FILE__).parent_path() / "fixtures";
}

std::string CopyFixtureToTemp() {
    std::string tmpl = "/tmp/prebatch_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* dir = mkdtemp(buf.data());
    REQUIRE(dir != nullptr);
    std::filesystem::copy(FixtureRoot() + "/prebatch_db", dir,
                          std::filesystem::copy_options::recursive |
                              std::filesystem::copy_options::overwrite_existing);
    return std::string(dir);
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

}  // namespace

TEST_CASE("backcompat: v0.2.2 bulk-built store opens, matches, upgrades") {
    const int dim = 16;
    std::string dir = CopyFixtureToTemp();
    std::string meta = dir + "/lsm_vec_db.meta";
    REQUIRE(ReadMetaField32(meta, 8) == 2);  // pre-batch meta version

    astervec::AsterVecDBOptions opts;
    opts.dim = dim;
    opts.m = 8;
    opts.m_max = 24;
    opts.ef_construction = 64;
    opts.vec_file_capacity = 4096;
    opts.vector_file_path = dir + "/vecs.bin";

    unsetenv("ASTERVEC_GRAPH_VARINT");
    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());

    // Replay the recorded queries; results must match the old binary's.
    std::ifstream qf(FixtureRoot() + "/prebatch_queries.txt");
    std::ifstream ef(FixtureRoot() + "/prebatch_expected.txt");
    REQUIRE(qf.is_open());
    REQUIRE(ef.is_open());
    for (int qi = 0; qi < 10; qi++) {
        std::vector<float> q(dim);
        for (int d = 0; d < dim; d++) {
            double x;
            REQUIRE(static_cast<bool>(qf >> x));
            q[d] = static_cast<float>(x);
        }
        astervec::SearchOptions so;
        so.k = 10;
        so.ef_search = 64;
        std::vector<astervec::SearchResult> res;
        REQUIRE(db->SearchKnn(astervec::Span<float>(q.data(), q.size()), so,
                              &res)
                    .ok());
        REQUIRE(res.size() == 10);
        for (int k = 0; k < 10; k++) {
            uint64_t expect = 0;
            REQUIRE(static_cast<bool>(ef >> expect));
            CHECK(static_cast<uint64_t>(res[k].id) == expect);
        }
    }

    // The store stays writable, and Close upgrades the meta to v4/raw.
    std::vector<float> nv(dim, 0.25f);
    REQUIRE(db->Insert(2000, astervec::Span<float>(nv.data(), nv.size())).ok());
    REQUIRE(db->Close().ok());
    db.reset();
    CHECK(ReadMetaField32(meta, 8) == 4);    // upgraded
    CHECK(ReadMetaField32(meta, 40) == 0);   // still raw, forever

    REQUIRE(astervec::AsterVecDB::Open(dir, opts, &db).ok());
    REQUIRE(db->Close().ok());
}
