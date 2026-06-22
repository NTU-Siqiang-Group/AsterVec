#include "doctest.h"
#include "astervec_db.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

static std::string NewTempDir() {
    char tmpl[] = "/tmp/astervec_fs_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    return std::string(dir);
}

static std::unique_ptr<astervec::AsterVecDB> OpenFresh(const std::string& path, int dim = 8) {
    astervec::AsterVecDBOptions opts;
    opts.dim = dim;
    opts.m = 8;
    opts.m_max = 16;
    opts.ef_construction = 32;
    opts.vec_file_capacity = 5000;
    opts.vector_file_path = path + "/vecs.bin";
    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());
    return db;
}

static std::vector<float> RandVec(int dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = d(rng);
    return v;
}

TEST_CASE("Filter: returns only matching ids (100 vectors, ~10% selectivity)") {
    std::string path = NewTempDir();
    auto db = OpenFresh(path);
    std::mt19937 rng(42);

    for (uint64_t i = 0; i < 100; ++i) {
        auto v = RandVec(8, rng);
        std::string md = (i % 10 == 0)
            ? R"({"tenant":"acme"})"
            : R"({"tenant":"other"})";
        REQUIRE(db->Insert(i, astervec::Span<float>(v), md).ok());
    }
    db->flushVectorWrites();

    auto q = RandVec(8, rng);
    astervec::SearchOptions opts;
    opts.k = 5;
    opts.ef_search = 64;

    std::vector<astervec::SearchResult> out;
    REQUIRE(db->SearchKnn(astervec::Span<float>(q), opts,
                          R"({"tenant":"acme"})", &out).ok());

    CHECK(out.size() == 5);
    for (const auto& r : out) {
        CHECK(r.id % 10 == 0);        // only matching ids
    }
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Filter: returns < k when matches are fewer than k") {
    std::string path = NewTempDir();
    auto db = OpenFresh(path);
    std::mt19937 rng(42);

    for (uint64_t i = 0; i < 100; ++i) {
        auto v = RandVec(8, rng);
        std::string md = (i < 3)
            ? R"({"tenant":"acme"})"
            : R"({"tenant":"other"})";
        REQUIRE(db->Insert(i, astervec::Span<float>(v), md).ok());
    }
    db->flushVectorWrites();

    auto q = RandVec(8, rng);
    astervec::SearchOptions opts;
    opts.k = 10;
    opts.ef_search = 64;

    std::vector<astervec::SearchResult> out;
    REQUIRE(db->SearchKnn(astervec::Span<float>(q), opts,
                          R"({"tenant":"acme"})", &out).ok());

    CHECK(out.size() <= 3);
    for (const auto& r : out) CHECK(r.id < 3);
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Filter: no metadata on vector -> does not match any filter") {
    std::string path = NewTempDir();
    auto db = OpenFresh(path);
    std::mt19937 rng(42);

    for (uint64_t i = 0; i < 50; ++i) {
        auto v = RandVec(8, rng);
        if (i < 10) {
            REQUIRE(db->Insert(i, astervec::Span<float>(v), R"({"has_meta":true})").ok());
        } else {
            REQUIRE(db->Insert(i, astervec::Span<float>(v)).ok());  // no metadata
        }
    }
    db->flushVectorWrites();

    auto q = RandVec(8, rng);
    astervec::SearchOptions opts;
    opts.k = 20;
    opts.ef_search = 64;

    std::vector<astervec::SearchResult> out;
    REQUIRE(db->SearchKnn(astervec::Span<float>(q), opts,
                          R"({"has_meta":true})", &out).ok());

    for (const auto& r : out) CHECK(r.id < 10);
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Empty filter = unfiltered search") {
    std::string path = NewTempDir();
    auto db = OpenFresh(path);
    std::mt19937 rng(42);

    for (uint64_t i = 0; i < 50; ++i) {
        auto v = RandVec(8, rng);
        REQUIRE(db->Insert(i, astervec::Span<float>(v), R"({"tag":"x"})").ok());
    }
    db->flushVectorWrites();

    auto q = RandVec(8, rng);
    astervec::SearchOptions opts;
    opts.k = 10;
    opts.ef_search = 64;

    std::vector<astervec::SearchResult> out1, out2;
    REQUIRE(db->SearchKnn(astervec::Span<float>(q), opts, "", &out1).ok());
    REQUIRE(db->SearchKnn(astervec::Span<float>(q), opts, &out2).ok());
    CHECK(out1.size() == out2.size());
    for (size_t i = 0; i < out1.size(); ++i) {
        CHECK(out1[i].id == out2[i].id);
        CHECK(out1[i].distance == doctest::Approx(out2[i].distance));
    }
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Filter: invalid JSON is InvalidArgument") {
    std::string path = NewTempDir();
    auto db = OpenFresh(path);
    auto q = std::vector<float>(8, 0.5f);
    std::vector<astervec::SearchResult> out;
    astervec::SearchOptions opts;
    auto st = db->SearchKnn(astervec::Span<float>(q), opts, R"({not json)", &out);
    CHECK(st.IsInvalidArgument());
    db->Close();
    std::filesystem::remove_all(path);
}
