// C9: scale / stress / round-trip tests for the robust-delete redesign.
//
// Stress 1: 2000 random vectors, delete 20%, recall@10 vs brute-force ground
//           truth on the alive set must stay above 90%.
// Stress 2: 1000 reals × 5 updates each — verify final state (counters + Get).
// Stress 3: continuation of 2 — Close+reopen, verify state survives, latest
//           vector still readable.
// Stress 4: bloom_fill_ratio after stress 2 must stay below the rebuild
//           threshold (i.e., the auto-rebuild kept up).

#include <algorithm>
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

namespace {

std::string make_path(const char* tag) {
    std::string tmpl = std::string("/tmp/lsmvec_C9_") + tag + "_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* dir = mkdtemp(buf.data());
    REQUIRE(dir != nullptr);
    return dir;
}

std::unique_ptr<LSMVecDB> open_db(const std::string& path, int dim, uint64_t cap,
                                   uint64_t seed = 1) {
    LSMVecDBOptions opts;
    opts.dim                  = dim;
    opts.vec_file_capacity    = cap;
    opts.vector_file_path     = path + "/vecs.bin";
    opts.random_seed          = seed;
    std::unique_ptr<LSMVecDB> db;
    REQUIRE(LSMVecDB::Open(path, opts, &db).ok());
    return db;
}

void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

float l2_dist_sq(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// Brute-force top-k over a list of (id, vector) excluding `excluded`.
std::vector<uint64_t> brute_force_top_k(
    const std::vector<float>& query,
    const std::vector<std::pair<uint64_t, std::vector<float>>>& corpus,
    const std::set<uint64_t>& excluded,
    int k) {
    std::vector<std::pair<float, uint64_t>> scored;
    scored.reserve(corpus.size());
    for (const auto& [id, vec] : corpus) {
        if (excluded.count(id)) continue;
        scored.emplace_back(l2_dist_sq(query, vec), id);
    }
    std::partial_sort(scored.begin(),
                      scored.begin() + std::min<size_t>(k, scored.size()),
                      scored.end());
    std::vector<uint64_t> result;
    result.reserve(k);
    for (int i = 0; i < k && i < static_cast<int>(scored.size()); ++i) {
        result.push_back(scored[i].second);
    }
    return result;
}

}  // namespace

TEST_CASE("Stress 1: delete 20% of 2000 vectors, recall@10 vs brute-force GT >= 90%") {
    constexpr int kDim       = 8;
    constexpr int kN         = 2000;
    constexpr int kQueries   = 50;
    constexpr int kK         = 10;
    constexpr int kEfSearch  = 64;
    constexpr double kRecallTarget = 0.90;

    auto path = make_path("s1");
    auto db = open_db(path, kDim, /*cap=*/static_cast<uint64_t>(kN));

    // Build ground-truth corpus alongside index inserts.
    std::vector<std::pair<uint64_t, std::vector<float>>> corpus;
    corpus.reserve(kN);
    std::mt19937 rng(12345);
    for (uint64_t id = 0; id < kN; ++id) {
        std::vector<float> v(kDim);
        fill_random(v, rng);
        corpus.emplace_back(id, v);
        REQUIRE(db->Insert(id, Span<float>(v)).ok());
    }

    // Delete 20% (every 5th id).
    std::set<uint64_t> deleted;
    for (uint64_t id = 0; id < kN; id += 5) {
        REQUIRE(db->Delete(id).ok());
        deleted.insert(id);
    }
    REQUIRE(db->GetDeleteStats().tombstones == deleted.size());

    // Run queries; compare with brute-force GT (alive set).
    SearchOptions so; so.k = kK; so.ef_search = kEfSearch;
    int total_overlap = 0;
    int total_possible = 0;
    for (int q = 0; q < kQueries; ++q) {
        std::vector<float> qvec(kDim);
        fill_random(qvec, rng);
        std::vector<SearchResult> got;
        REQUIRE(db->SearchKnn(Span<float>(qvec), so, &got).ok());

        // No deleted id may appear.
        for (const auto& r : got) CHECK(deleted.count(r.id) == 0);

        auto truth = brute_force_top_k(qvec, corpus, deleted, kK);

        std::set<uint64_t> truth_set(truth.begin(), truth.end());
        for (const auto& r : got) {
            if (truth_set.count(r.id)) ++total_overlap;
        }
        total_possible += static_cast<int>(truth.size());
    }
    const double recall =
        static_cast<double>(total_overlap) / static_cast<double>(total_possible);
    MESSAGE("Stress 1 recall@10 (alive-aware GT): " << recall);
    CHECK(recall >= kRecallTarget);

    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Stress 2: 1000 reals × 5 updates — counters + final-vector correctness") {
    constexpr int kDim    = 4;
    constexpr int kN      = 1000;
    constexpr int kRounds = 5;

    auto path = make_path("s2");
    auto db = open_db(path, kDim, /*cap=*/static_cast<uint64_t>(kN));

    // Round 0: fresh inserts.
    for (uint64_t id = 0; id < kN; ++id) {
        std::vector<float> v(kDim, static_cast<float>(id));
        REQUIRE(db->Insert(id, Span<float>(v)).ok());
    }
    // Rounds 1..kRounds-1: Update each real_id with a distinguishable value.
    for (int round = 1; round < kRounds; ++round) {
        for (uint64_t id = 0; id < kN; ++id) {
            std::vector<float> v(kDim, static_cast<float>(id) +
                                       static_cast<float>(round) * 10000.0f);
            REQUIRE(db->Update(id, Span<float>(v)).ok());
        }
    }

    auto s = db->GetDeleteStats();
    // Per real_id: 1 Insert + (kRounds-1) Updates = kRounds inserts at HNSW
    // level. (kRounds-1) tombstones per real (the previous version each time).
    CHECK(s.total_inserts_ever == static_cast<size_t>(kN * kRounds));
    CHECK(s.tombstones         == static_cast<size_t>(kN * (kRounds - 1)));
    CHECK(s.updated_real_ids   == static_cast<size_t>(kN));

    // Verify the LATEST vector survives for every real_id.
    const float expected_round = static_cast<float>(kRounds - 1);
    for (uint64_t id = 0; id < kN; ++id) {
        std::vector<float> got;
        REQUIRE(db->Get(id, &got).ok());
        std::vector<float> expected(kDim, static_cast<float>(id) +
                                          expected_round * 10000.0f);
        CHECK(got == expected);
    }

    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Stress 3: round-trip after heavy update — Close+reopen preserves all 1000 latest values") {
    constexpr int kDim    = 4;
    constexpr int kN      = 1000;
    constexpr int kRounds = 5;

    auto path = make_path("s3");

    // Phase 1: build, update, close.
    {
        auto db = open_db(path, kDim, /*cap=*/static_cast<uint64_t>(kN));
        for (uint64_t id = 0; id < kN; ++id) {
            std::vector<float> v(kDim, static_cast<float>(id));
            REQUIRE(db->Insert(id, Span<float>(v)).ok());
        }
        for (int round = 1; round < kRounds; ++round) {
            for (uint64_t id = 0; id < kN; ++id) {
                std::vector<float> v(kDim, static_cast<float>(id) +
                                           static_cast<float>(round) * 10000.0f);
                REQUIRE(db->Update(id, Span<float>(v)).ok());
            }
        }
        REQUIRE(db->Close().ok());
    }

    // Phase 2: reopen, verify counters + every Get returns the latest value.
    {
        auto db = open_db(path, kDim, /*cap=*/static_cast<uint64_t>(kN));
        auto s = db->GetDeleteStats();
        CHECK(s.tombstones       == static_cast<size_t>(kN * (kRounds - 1)));
        CHECK(s.updated_real_ids == static_cast<size_t>(kN));
        // total_inserts_ever_ is transient (resets on Open) — by design.
        CHECK(s.total_inserts_ever == 0);

        const float expected_round = static_cast<float>(kRounds - 1);
        for (uint64_t id = 0; id < kN; ++id) {
            std::vector<float> got;
            REQUIRE(db->Get(id, &got).ok());
            std::vector<float> expected(kDim, static_cast<float>(id) +
                                              expected_round * 10000.0f);
            CHECK(got == expected);
        }
        REQUIRE(db->Close().ok());
    }

    std::filesystem::remove_all(path);
}

TEST_CASE("Stress 4: bloom_fill_ratio stays below rebuild threshold after heavy update") {
    constexpr int kDim    = 4;
    constexpr int kN      = 1000;
    constexpr int kRounds = 5;

    auto path = make_path("s4");
    auto db = open_db(path, kDim, /*cap=*/static_cast<uint64_t>(kN));

    for (uint64_t id = 0; id < kN; ++id) {
        std::vector<float> v(kDim, static_cast<float>(id));
        REQUIRE(db->Insert(id, Span<float>(v)).ok());
    }
    for (int round = 1; round < kRounds; ++round) {
        for (uint64_t id = 0; id < kN; ++id) {
            std::vector<float> v(kDim, static_cast<float>(id) +
                                       static_cast<float>(round) * 10000.0f);
            REQUIRE(db->Update(id, Span<float>(v)).ok());
        }
    }

    auto s = db->GetDeleteStats();
    MESSAGE("Stress 4 bloom: capacity=" << s.bloom_capacity
            << " fill_ratio=" << s.bloom_fill_ratio
            << " rebuild_count=" << s.bloom_rebuild_count);
    // The auto-rebuild must have kept the filter under the 0.7 threshold.
    CHECK(s.bloom_fill_ratio < 0.7);
    CHECK(s.bloom_rebuild_count >= 1);

    db->Close();
    std::filesystem::remove_all(path);
}
