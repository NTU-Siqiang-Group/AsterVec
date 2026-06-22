#include "doctest.h"
#include "astervec_db.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

TEST_CASE("Unfiltered SearchKnn results unchanged after metadata feature added") {
    char tmpl[] = "/tmp/astervec_reg_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    std::string path = dir;

    astervec::AsterVecDBOptions opts;
    opts.dim = 16;
    opts.m = 8;
    opts.m_max = 16;
    opts.ef_construction = 32;
    opts.vec_file_capacity = 5000;
    opts.vector_file_path = path + "/vecs.bin";
    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1, 1);
    for (uint64_t i = 0; i < 200; ++i) {
        std::vector<float> v(16);
        for (auto& x : v) x = dist(rng);
        REQUIRE(db->Insert(i, astervec::Span<float>(v)).ok());  // no metadata
    }
    db->flushVectorWrites();

    std::vector<float> q(16);
    for (auto& x : q) x = dist(rng);

    astervec::SearchOptions so;
    so.k = 10;
    so.ef_search = 64;

    std::vector<astervec::SearchResult> out;
    REQUIRE(db->SearchKnn(astervec::Span<float>(q), so, &out).ok());
    CHECK(out.size() == 10);

    db->Close();
    std::filesystem::remove_all(path);
}
