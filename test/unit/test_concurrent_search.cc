// Concurrent search regression test.
//
// Verifies that N reader threads invoking SearchKnn against the same
// AsterVecDB instance produce results identical to a single-threaded
// baseline. Step 8 of the thread-safety refactor (see
// pgastervec/doc/thread-safety-refactor-plan.md).
//
// Sanitizer notes:
//   - TSAN on macOS x86_64h: TSAN's own allocator crashes inside
//     std::string deallocation paths during dyld init. This is a
//     known TSAN-on-macOS issue interacting with the prebuilt
//     RocksDB static archive; not a real race.
//   - ASAN on macOS: triggers a bad-free inside RocksDB's static
//     initializer for write_stall_stats; same pre-built-archive
//     issue.
// On Linux CI (where Aiven's actual deployment runs) both sanitizers
// should run cleanly against this test. The plan-of-record is to
// gate this test under -fsanitize=thread in Linux CI.
#include "doctest.h"
#include "astervec_db.h"
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <thread>
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

std::unique_ptr<astervec::AsterVecDB>
OpenFresh(const std::string& path, int dim) {
    astervec::AsterVecDBOptions opts;
    opts.dim = dim;
    opts.m = 8;
    opts.m_max = 24;
    opts.ef_construction = 32;
    opts.vec_file_capacity = 20000;
    opts.vector_file_path = path + "/vecs.bin";
    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());
    return db;
}

std::vector<float> RandVec(int dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = d(rng);
    return v;
}

}  // namespace

TEST_CASE("Concurrent search: 8 threads × 200 queries match single-thread baseline") {
    constexpr int kDim = 16;
    constexpr int kNumVectors = 2000;
    constexpr int kNumQueries = 200;
    constexpr int kNumThreads = 8;
    constexpr int kK = 10;

    std::string path = NewTempDir("astervec_concsearch");
    auto db = OpenFresh(path, kDim);

    // Insert deterministic dataset.
    std::mt19937 rng(0xC0FFEE);
    for (int i = 0; i < kNumVectors; ++i) {
        auto v = RandVec(kDim, rng);
        REQUIRE(db->Insert(static_cast<uint64_t>(i),
                           astervec::Span<float>(v))
                    .ok());
    }
    db->flushVectorWrites();

    // Generate queries deterministically and record single-thread baseline.
    std::vector<std::vector<float>> queries;
    queries.reserve(kNumQueries);
    for (int q = 0; q < kNumQueries; ++q) {
        queries.push_back(RandVec(kDim, rng));
    }

    astervec::SearchOptions opts;
    opts.k = kK;
    opts.ef_search = 64;

    std::vector<std::vector<astervec::SearchResult>> baseline(kNumQueries);
    for (int q = 0; q < kNumQueries; ++q) {
        REQUIRE(db->SearchKnn(astervec::Span<float>(queries[q]),
                              opts, &baseline[q])
                    .ok());
    }

    // Each thread runs all queries; results must match the baseline byte-
    // for-byte (same id sequence, same distance values).
    std::atomic<int> mismatch_count{0};
    std::atomic<int> error_count{0};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int q = 0; q < kNumQueries; ++q) {
                std::vector<astervec::SearchResult> out;
                auto st = db->SearchKnn(astervec::Span<float>(queries[q]),
                                        opts, &out);
                if (!st.ok()) {
                    error_count.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (out.size() != baseline[q].size()) {
                    mismatch_count.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                for (size_t i = 0; i < out.size(); ++i) {
                    if (out[i].id != baseline[q][i].id ||
                        out[i].distance != baseline[q][i].distance) {
                        mismatch_count.fetch_add(1,
                                                  std::memory_order_relaxed);
                        break;
                    }
                }
            }
            (void)t;
        });
    }
    for (auto& th : threads) th.join();

    CHECK(error_count.load() == 0);
    CHECK(mismatch_count.load() == 0);

    std::filesystem::remove_all(path);
}

TEST_CASE("Concurrent search: many small queries do not crash or leak state") {
    // Stress test for the page cache and edge cache locks. Queries are
    // intentionally varied so different threads hit different cache
    // entries, exercising the concurrent insert/evict paths.
    constexpr int kDim = 8;
    constexpr int kNumVectors = 500;
    constexpr int kNumThreads = 8;
    constexpr int kQueriesPerThread = 500;

    std::string path = NewTempDir("astervec_concstress");
    auto db = OpenFresh(path, kDim);

    std::mt19937 rng(0xF00D);
    for (int i = 0; i < kNumVectors; ++i) {
        auto v = RandVec(kDim, rng);
        REQUIRE(db->Insert(static_cast<uint64_t>(i),
                           astervec::Span<float>(v))
                    .ok());
    }
    db->flushVectorWrites();

    astervec::SearchOptions opts;
    opts.k = 5;
    opts.ef_search = 32;

    std::atomic<int> error_count{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 trng(static_cast<unsigned>(0xABCD ^ t));
            for (int q = 0; q < kQueriesPerThread; ++q) {
                auto query = RandVec(kDim, trng);
                std::vector<astervec::SearchResult> out;
                auto st = db->SearchKnn(astervec::Span<float>(query),
                                        opts, &out);
                if (!st.ok() || out.empty()) {
                    error_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    CHECK(error_count.load() == 0);

    std::filesystem::remove_all(path);
}
