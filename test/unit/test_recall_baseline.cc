// Recall@10 baseline measurement.
//
// Concurrent-writer-refactor-plan §6 Phase 0 deliverable. This test
// records the SERIAL recall@10 number on a fixed synthetic dataset.
// Phase 4 onward will compare recall under concurrent inserts against
// this baseline; the budget is < 1.0 % absolute drift at N=16
// concurrent writers (concurrent-writer-refactor-plan.md §1.2).
//
// The test always passes; its purpose is to PRINT the baseline number
// so it can be captured (manually or via doctest --reporters) into the
// local-only BENCHMARK.md as a reference for later phases.
//
// Tagged [phase0-baseline] so it doesn't interleave with the rest of
// the unit-test run timing-wise.

#include "doctest.h"
#include "lsm_vec_db.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
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

std::unique_ptr<lsm_vec::LSMVecDB>
OpenFresh(const std::string& path, int dim, std::size_t capacity) {
    lsm_vec::LSMVecDBOptions opts;
    opts.dim = dim;
    opts.m = 8;
    opts.m_max = 24;
    opts.ef_construction = 32;
    opts.vec_file_capacity = capacity;
    opts.vector_file_path = path + "/vecs.bin";
    std::unique_ptr<lsm_vec::LSMVecDB> db;
    REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());
    return db;
}

std::vector<float> RandVec(int dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = d(rng);
    return v;
}

float L2Sq(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// Brute-force top-k ground truth.
std::vector<std::uint64_t>
BruteForceTopK(const std::vector<std::vector<float>>& base,
               const std::vector<float>& q, int k) {
    std::vector<std::pair<float, std::uint64_t>> dists;
    dists.reserve(base.size());
    for (std::size_t i = 0; i < base.size(); ++i) {
        dists.emplace_back(L2Sq(base[i], q), static_cast<std::uint64_t>(i));
    }
    int kk = std::min<int>(k, static_cast<int>(dists.size()));
    std::partial_sort(dists.begin(), dists.begin() + kk, dists.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });
    dists.resize(kk);
    std::vector<std::uint64_t> ids;
    ids.reserve(kk);
    for (auto& p : dists) ids.push_back(p.second);
    return ids;
}

}  // namespace

TEST_CASE("Phase-0 baseline: serial recall@10 on synthetic 128-dim, 2000 vectors"
          * doctest::test_suite("phase0-baseline")) {
    constexpr int kDim         = 128;
    constexpr int kNumVectors  = 2000;
    constexpr int kNumQueries  = 200;
    constexpr int kK           = 10;
    constexpr int kEfSearch    = 64;
    constexpr std::uint64_t kSeed = 0xB45EUL;  // "BASE"

    std::string path = NewTempDir("lsmvec_recall_baseline");
    auto db = OpenFresh(path, kDim, kNumVectors * 2);

    // Reproducible base set.
    std::mt19937 rng(kSeed);
    std::vector<std::vector<float>> base;
    base.reserve(kNumVectors);
    for (int i = 0; i < kNumVectors; ++i) {
        base.push_back(RandVec(kDim, rng));
        REQUIRE(db->Insert(static_cast<std::uint64_t>(i),
                           lsm_vec::Span<float>(base[i]))
                    .ok());
    }
    db->flushVectorWrites();

    // Independent query distribution (different seed lane).
    std::mt19937 qrng(kSeed ^ 0xCAFEUL);
    std::vector<std::vector<float>> queries;
    queries.reserve(kNumQueries);
    for (int q = 0; q < kNumQueries; ++q) {
        queries.push_back(RandVec(kDim, qrng));
    }

    lsm_vec::SearchOptions opts;
    opts.k = kK;
    opts.ef_search = kEfSearch;

    long matched = 0;
    long total   = 0;
    for (int q = 0; q < kNumQueries; ++q) {
        auto gt = BruteForceTopK(base, queries[q], kK);
        std::unordered_set<std::uint64_t> gt_set(gt.begin(), gt.end());

        std::vector<lsm_vec::SearchResult> out;
        REQUIRE(db->SearchKnn(lsm_vec::Span<float>(queries[q]), opts, &out)
                    .ok());
        for (auto& r : out) {
            if (gt_set.count(r.id)) ++matched;
        }
        total += static_cast<long>(gt.size());
    }

    const double recall = total > 0 ? static_cast<double>(matched) / total
                                    : 0.0;

    // Print the baseline. `doctest` echoes MESSAGE / std::cout so it's
    // captured by `make unit_test` output, which the operator pastes
    // into BENCHMARK.md.
    std::cout << "\n[phase0-baseline] dim=" << kDim
              << " n=" << kNumVectors
              << " queries=" << kNumQueries
              << " k=" << kK
              << " ef_search=" << kEfSearch
              << " seed=0x" << std::hex << kSeed << std::dec
              << " recall@10=" << std::fixed << std::setprecision(4)
              << recall << "\n";

    // The baseline test always passes — it's recording, not asserting.
    // We only sanity-bound recall to "obviously not broken" territory
    // so a regression in the search path itself is caught.
    CHECK(recall > 0.50);

    std::filesystem::remove_all(path);
}
