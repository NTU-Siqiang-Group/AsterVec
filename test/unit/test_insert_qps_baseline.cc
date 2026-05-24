// Single-thread INSERT QPS baseline.
//
// Concurrent-writer-refactor-plan §6 Phase 0 deliverable. Records
// the 1× INSERT throughput baseline that Phase 6 acceptance compares
// the N-worker scaling against ("INSERT QPS at engine_worker_threads=N
// reaches ≥ 50 % of N × single-thread INSERT").
//
// The number is workload-dependent (dim, m, ef_construction, host).
// Test always passes; the printed line is the operator's input to
// BENCHMARK.md.

#include "doctest.h"
#include "lsm_vec_db.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
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

}  // namespace

TEST_CASE("Phase-0 baseline: single-thread INSERT QPS"
          * doctest::test_suite("phase0-baseline")) {
    constexpr int kDim         = 128;
    constexpr int kNumVectors  = 5000;
    constexpr std::uint64_t kSeed = 0x9B5UL;  // "QPS-ish"

    std::string path = NewTempDir("lsmvec_qps_baseline");
    auto db = OpenFresh(path, kDim, kNumVectors * 2);

    std::mt19937 rng(kSeed);

    // Pre-generate vectors so RNG cost isn't on the timing path.
    std::vector<std::vector<float>> base;
    base.reserve(kNumVectors);
    for (int i = 0; i < kNumVectors; ++i) {
        base.push_back(RandVec(kDim, rng));
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kNumVectors; ++i) {
        REQUIRE(db->Insert(static_cast<std::uint64_t>(i),
                           lsm_vec::Span<float>(base[i]))
                    .ok());
    }
    db->flushVectorWrites();
    auto t1 = std::chrono::steady_clock::now();

    const double seconds =
        std::chrono::duration<double>(t1 - t0).count();
    const double qps = static_cast<double>(kNumVectors) / seconds;

    std::cout << "\n[phase0-baseline] dim=" << kDim
              << " n=" << kNumVectors
              << " seed=0x" << std::hex << kSeed << std::dec
              << " serial_insert_seconds=" << std::fixed
              << std::setprecision(3) << seconds
              << " serial_insert_qps=" << std::setprecision(1) << qps
              << "\n";

    // Sanity floor — anything below ~50 QPS at this scale on a modern
    // host indicates the test environment is broken (e.g., disk
    // saturation, debug build with no optimisation).
    CHECK(qps > 50.0);

    std::filesystem::remove_all(path);
}
