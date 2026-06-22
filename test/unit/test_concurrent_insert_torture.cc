// Concurrent-INSERT torture test.
//
// Concurrent-writer-refactor-plan §6 Phase 0 deliverable: this test
// exercises N writer threads inserting into ONE AsterVecDB instance
// concurrently, with M reader threads searching alongside. Today it
// will (correctly) race / crash under TSan because the AsterVec graph
// is not yet thread-safe for concurrent writers — that's the point.
// This test is the regression-boundary fixture: the
// concurrent-writer-refactor-plan §6 Phase 4 acceptance gate is "this
// test passes under TSan".
//
// **SKIPPED BY DEFAULT** via `doctest::skip()` so a normal
//   `make unit_test`
// run does not crash. To execute:
//   make unit_test DOCTEST_TAGS="--no-skip --test-suite=concurrent-writer-torture"
// Or under TSan:
//   make unit_test_tsan DOCTEST_TAGS="--no-skip --test-suite=concurrent-writer-torture"
//
// The plan specifies dim=128 (SIFT-128 dim), N writer threads, 100K
// vectors per thread. To keep the test runtime reasonable we use a
// smaller per-thread count here and let CI / large runs override
// via the `ASTERVEC_TORTURE_PER_THREAD` env var.

#include "doctest.h"
#include "astervec_db.h"
#include "astervec_index.h"   // needed for index_for_test()->is_alive()

#include <atomic>
#include <cstdint>
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
OpenFresh(const std::string& path, int dim, std::size_t capacity) {
    astervec::AsterVecDBOptions opts;
    opts.dim = dim;
    opts.m = 8;
    opts.m_max = 24;
    opts.ef_construction = 32;
    opts.vec_file_capacity = capacity;
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

int env_int(const char* name, int defv) {
    if (const char* v = std::getenv(name)) {
        char* end = nullptr;
        long n = std::strtol(v, &end, 10);
        if (end != v && n > 0 && n < (1LL << 28)) return static_cast<int>(n);
    }
    return defv;
}

}  // namespace

TEST_CASE("Concurrent INSERT torture: N writer threads on one AsterVecDB"
          * doctest::test_suite("concurrent-writer-torture")
          * doctest::skip()) {
    constexpr int kDim = 128;
    const int kWriterThreads  = env_int("ASTERVEC_TORTURE_WRITERS", 8);
    const int kPerThread      = env_int("ASTERVEC_TORTURE_PER_THREAD", 4000);
    const int kReaderThreads  = env_int("ASTERVEC_TORTURE_READERS", 2);
    const int kReaderQueries  = env_int("ASTERVEC_TORTURE_READER_QUERIES", 200);

    const std::size_t kTotalIds =
        static_cast<std::size_t>(kWriterThreads) * kPerThread;

    std::string path = NewTempDir("astervec_insert_torture");
    auto db = OpenFresh(path, kDim, kTotalIds + 16);

    std::atomic<bool> stop_readers{false};
    std::atomic<int>  insert_failures{0};
    std::atomic<int>  search_failures{0};

    // Disjoint id ranges per writer so Insert(real_id, vec) doesn't
    // collide on the same key. The per-real-id transaction lock
    // (Phase 3, concurrent-writer-refactor-plan.md §5.2) is what
    // covers the same-key case; this test isolates the graph-level
    // concurrency only.
    auto writer_main = [&](int t) {
        std::mt19937 rng(0xA110 + t);
        const std::uint64_t base = static_cast<std::uint64_t>(t) * kPerThread;
        for (int i = 0; i < kPerThread; ++i) {
            auto v = RandVec(kDim, rng);
            auto st = db->Insert(base + static_cast<std::uint64_t>(i),
                                 astervec::Span<float>(v));
            if (!st.ok()) {
                insert_failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    auto reader_main = [&](int t) {
        std::mt19937 rng(0xBABE + t);
        astervec::SearchOptions opts;
        opts.k = 10;
        opts.ef_search = 64;
        for (int i = 0; i < kReaderQueries && !stop_readers.load(); ++i) {
            auto q = RandVec(kDim, rng);
            std::vector<astervec::SearchResult> out;
            auto st = db->SearchKnn(astervec::Span<float>(q), opts, &out);
            // NotFound is expected during the brief window before any
            // insert publishes entry_point_; only count IOError /
            // InvalidArgument / etc as real failures.
            if (!st.ok() && !st.IsNotFound()) {
                search_failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> writers;
    writers.reserve(kWriterThreads);
    for (int t = 0; t < kWriterThreads; ++t) {
        writers.emplace_back(writer_main, t);
    }
    std::vector<std::thread> readers;
    readers.reserve(kReaderThreads);
    for (int t = 0; t < kReaderThreads; ++t) {
        readers.emplace_back(reader_main, t);
    }

    for (auto& th : writers) th.join();
    stop_readers.store(true);
    for (auto& th : readers) th.join();

    CHECK(insert_failures.load() == 0);
    CHECK(search_failures.load() == 0);

    // Post-condition: every id we inserted must be alive and findable
    // via a getNodeVector / SearchKnn round-trip. This is the strongest
    // check that no insert was lost to a race.
    db->flushVectorWrites();
    int missing = 0;
    // Use the test-only accessor instead of HasLiveVector — the latter
    // is private (we don't port 9f14958 which would expose it publicly
    // for pgastervec).
    auto* index = db->index_for_test();
    for (std::uint64_t id = 0; id < kTotalIds; ++id) {
        if (!index->is_alive(id)) ++missing;
    }
    CHECK(missing == 0);

    std::filesystem::remove_all(path);
}
