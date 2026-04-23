// Red tests for the delete redesign — see docs/DELETE_DESIGN.md and
// docs/DELETE_IMPL_PLAN.md §C4. These intentionally fail (or are flaky)
// against pre-pivot code; they pass after C5a + C5b complete.
//
// Coverage:
//   bug2: search after delete does not crash on dangling edges
//   bug3: Insert on a previously Deleted real_id succeeds (re-insert)
//   NS1:  Update replaces the vector behind a real_id
//   NS2:  Repeated Update yields the latest vector
//   NS3:  Insert on a still-alive real_id is treated as upsert (A1)
//   NS4:  Deleting an early-inserted id (likely entry_point) does not break search
//   NS6:  Delete cascades to metadata (regression guard from feat/metadata-filtering)

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "doctest.h"
#include "lsm_vec_db.h"

using lsm_vec::LSMVecDB;
using lsm_vec::LSMVecDBOptions;
using lsm_vec::SearchOptions;
using lsm_vec::SearchResult;
using lsm_vec::Span;
using lsm_vec::node_id_t;

namespace {

std::string make_temp_path() {
    char tmpl[] = "/tmp/lsmvec_C4_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    return dir;
}

void cleanup_path(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

std::unique_ptr<LSMVecDB> open_fresh(const std::string& path, int dim,
                                     uint64_t cap, uint64_t random_seed = 1) {
    LSMVecDBOptions opts;
    opts.dim                  = dim;
    opts.vec_file_capacity    = cap;
    opts.vector_file_path     = path + "/vecs.bin";
    opts.random_seed          = random_seed;
    std::unique_ptr<LSMVecDB> db;
    REQUIRE(LSMVecDB::Open(path, opts, &db).ok());
    return db;
}

void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

}  // namespace

TEST_CASE("bug2: search after delete does not crash on dangling edges") {
    auto path = make_temp_path();
    auto db   = open_fresh(path, /*dim=*/8, /*cap=*/2000);

    std::vector<float> v(8);
    std::mt19937 rng(42);
    for (uint64_t id = 0; id < 1000; ++id) {
        fill_random(v, rng);
        REQUIRE(db->Insert(id, Span<float>(v)).ok());
    }

    std::set<node_id_t> deleted;
    for (uint64_t k = 0; k < 200; ++k) {
        uint64_t id = (k * 7) % 1000;
        if (deleted.insert(id).second) {
            REQUIRE(db->Delete(id).ok());
        }
    }

    SearchOptions so; so.k = 10; so.ef_search = 64;
    std::vector<SearchResult> out;
    for (int q = 0; q < 50; ++q) {
        fill_random(v, rng);
        auto st = db->SearchKnn(Span<float>(v), so, &out);
        REQUIRE(st.ok());                  // no crash, no IOError
        CHECK(!out.empty());
        for (const auto& r : out) {
            CHECK(deleted.count(r.id) == 0);  // no deleted id returned
        }
    }

    db->Close();
    cleanup_path(path);
}

TEST_CASE("bug3: re-insert after delete succeeds and returns the new vector") {
    auto path = make_temp_path();
    auto db   = open_fresh(path, /*dim=*/4, /*cap=*/100);

    std::vector<float> v1{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> v2{9.0f, 8.0f, 7.0f, 6.0f};

    REQUIRE(db->Insert(42, Span<float>(v1)).ok());
    REQUIRE(db->Delete(42).ok());
    REQUIRE(db->Insert(42, Span<float>(v2)).ok());

    std::vector<float> got;
    REQUIRE(db->Get(42, &got).ok());
    CHECK(got == v2);

    db->Close();
    cleanup_path(path);
}

TEST_CASE("NS1: Update replaces the vector behind real_id") {
    auto path = make_temp_path();
    auto db   = open_fresh(path, 4, 100);

    std::vector<float> v1{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> v2{9.0f, 8.0f, 7.0f, 6.0f};
    REQUIRE(db->Insert(7, Span<float>(v1)).ok());
    REQUIRE(db->Update(7, Span<float>(v2)).ok());

    std::vector<float> got;
    REQUIRE(db->Get(7, &got).ok());
    CHECK(got == v2);

    db->Close();
    cleanup_path(path);
}

TEST_CASE("NS2: repeated Update on the same real_id yields the latest vector") {
    auto path = make_temp_path();
    auto db   = open_fresh(path, 4, 100);

    for (int i = 1; i <= 5; ++i) {
        std::vector<float> v(4, static_cast<float>(i));
        if (i == 1) {
            REQUIRE(db->Insert(7, Span<float>(v)).ok());
        } else {
            REQUIRE(db->Update(7, Span<float>(v)).ok());
        }
    }
    std::vector<float> got;
    REQUIRE(db->Get(7, &got).ok());
    CHECK(got == std::vector<float>(4, 5.0f));

    db->Close();
    cleanup_path(path);
}

TEST_CASE("NS3: Insert on an alive real_id is treated as upsert (A1)") {
    auto path = make_temp_path();
    auto db   = open_fresh(path, 4, 100);

    std::vector<float> v1{1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> v2{2.0f, 2.0f, 2.0f, 2.0f};

    REQUIRE(db->Insert(9, Span<float>(v1)).ok());
    auto st = db->Insert(9, Span<float>(v2));
    REQUIRE(st.ok());                          // not an AlreadyExists error

    std::vector<float> got;
    REQUIRE(db->Get(9, &got).ok());
    CHECK(got == v2);

    db->Close();
    cleanup_path(path);
}

TEST_CASE("NS4: deleting an early-inserted id (likely entry_point) does not break search") {
    auto path = make_temp_path();
    auto db   = open_fresh(path, /*dim=*/8, /*cap=*/600, /*seed=*/1);

    std::vector<float> v(8);
    std::mt19937 rng(7);
    for (uint64_t id = 0; id < 500; ++id) {
        fill_random(v, rng);
        REQUIRE(db->Insert(id, Span<float>(v)).ok());
    }

    // id=0 is the very first node and therefore the initial entry_point.
    // It may have been promoted out by later high-layer inserts, but with
    // 500 nodes it still likely occupies an upper layer.
    REQUIRE(db->Delete(0).ok());

    SearchOptions so; so.k = 10; so.ef_search = 64;
    std::vector<SearchResult> out;
    for (int q = 0; q < 20; ++q) {
        fill_random(v, rng);
        auto st = db->SearchKnn(Span<float>(v), so, &out);
        REQUIRE(st.ok());
        CHECK(!out.empty());
        for (const auto& r : out) {
            CHECK(r.id != 0);                  // deleted id never returned
        }
    }

    db->Close();
    cleanup_path(path);
}

TEST_CASE("NS6: Delete cascades to metadata (regression guard from metadata-filtering)") {
    auto path = make_temp_path();
    auto db   = open_fresh(path, 4, 100);

    std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
    REQUIRE(db->Insert(5, Span<float>(v), R"({"tag":"x"})").ok());

    // Before Delete: payload reflects what was inserted.
    std::string got;
    REQUIRE(db->GetPayload(5, &got).ok());
    CHECK(got == R"({"tag":"x"})");

    REQUIRE(db->Delete(5).ok());

    // After Delete: GetPayload masks NotFound as OK + "{}" (set by the
    // metadata-filtering era for $exists semantics). A "{}" here proves
    // the row was removed, not that it was never there — we wrote it above.
    REQUIRE(db->GetPayload(5, &got).ok());
    CHECK(got == "{}");

    db->Close();
    cleanup_path(path);
}
