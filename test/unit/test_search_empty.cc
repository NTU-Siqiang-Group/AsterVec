// A search with nothing to return is an empty result, not an error.

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "doctest.h"
#include "astervec_db.h"

using astervec::AsterVecDB;
using astervec::AsterVecDBOptions;
using astervec::SearchOptions;
using astervec::SearchResult;
using astervec::Span;

namespace {

constexpr int kDim = 8;

std::unique_ptr<AsterVecDB> open_fresh() {
    char tmpl[] = "/tmp/astervec_empty_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    AsterVecDBOptions opts;
    opts.dim = kDim;
    opts.vec_file_capacity = 256;
    opts.vector_file_path = std::string(dir) + "/vecs.bin";
    std::unique_ptr<AsterVecDB> db;
    REQUIRE(AsterVecDB::Open(dir, opts, &db).ok());
    return db;
}

std::vector<float> vec(int i) {
    std::vector<float> v(kDim);
    for (int d = 0; d < kDim; ++d) v[d] = static_cast<float>(i * 3 + d);
    return v;
}

}  // namespace

TEST_CASE("search: an empty index returns an empty result") {
    auto db = open_fresh();
    auto q = vec(1);
    SearchOptions so;
    so.k = 5;
    std::vector<SearchResult> out{{1, 0.f}};
    REQUIRE(db->SearchKnn(Span<float>(q), so, &out).ok());
    CHECK(out.empty());
    out = {{1, 0.f}};
    REQUIRE(db->SearchKnn(Span<float>(q), so, R"({"tag":"a"})", &out).ok());
    CHECK(out.empty());
}

TEST_CASE("search: every point deleted returns an empty result") {
    auto db = open_fresh();
    for (int i = 0; i < 20; ++i) {
        auto v = vec(i);
        REQUIRE(db->Insert(i, Span<float>(v)).ok());
    }
    for (int i = 0; i < 20; ++i) REQUIRE(db->Delete(i).ok());
    auto q = vec(3);
    SearchOptions so;
    so.k = 5;
    std::vector<SearchResult> out;
    REQUIRE(db->SearchKnn(Span<float>(q), so, &out).ok());
    CHECK(out.empty());
}

TEST_CASE("search: a filter that matches nothing returns an empty result") {
    auto db = open_fresh();
    for (int i = 0; i < 20; ++i) {
        auto v = vec(i);
        REQUIRE(db->Insert(i, Span<float>(v), R"({"tag":"a"})").ok());
    }
    auto q = vec(3);
    SearchOptions so;
    so.k = 5;
    std::vector<SearchResult> out;
    REQUIRE(db->SearchKnn(Span<float>(q), so, R"({"tag":"zzz"})", &out).ok());
    CHECK(out.empty());

    // A matching filter still finds points, and a malformed one is an error.
    REQUIRE(db->SearchKnn(Span<float>(q), so, R"({"tag":"a"})", &out).ok());
    CHECK(!out.empty());
    CHECK(!db->SearchKnn(Span<float>(q), so, "{not json", &out).ok());
}
