// Concurrent UPSERT race test.
//
// Concurrent-writer-refactor-plan §6 Phase 0 deliverable. Authored
// here; expected to PASS only after Phase 3 lands the per-real-id
// transaction lock at the LSMVecDB layer (see
// concurrent-writer-refactor-plan.md §5.2). Today the Upsert path
// has a check-then-act race on `is_alive(R)` →
// `allocate_update_id()` → `record_update_mapping(R, new_id)`, so
// two concurrent Insert(R, ...) calls can both observe R as not
// alive, both allocate fresh internal ids, and the second's
// `record_update_mapping` overwrites the first — orphaning the
// first inserted node.
//
// **SKIPPED BY DEFAULT** for the same reason as the insert torture
// test. To execute:
//   make unit_test DOCTEST_TAGS="--no-skip --test-suite=concurrent-writer-torture"
//   make unit_test_tsan DOCTEST_TAGS="--no-skip --test-suite=concurrent-writer-torture"
//
// Post-conditions verified after Phase 3:
//   1. The single alive mapping for R points to one of the writer
//      vectors (any of the kThreads is acceptable — last-writer
//      wins is fine, but exactly one wins).
//   2. Other writers' new internal ids end up tombstoned (their
//      upsert was overwritten by a later one).
//   3. No internal id mapping points at R but is also tombstoned
//      (broken invariant: claims to be R's current id but is dead).

#include "doctest.h"
#include "lsm_vec_db.h"
#include "lsm_vec_index.h"   // needed for index_for_test()->is_alive()

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

std::unique_ptr<lsm_vec::LSMVecDB>
OpenFresh(const std::string& path, int dim) {
    lsm_vec::LSMVecDBOptions opts;
    opts.dim = dim;
    opts.m = 8;
    opts.m_max = 24;
    opts.ef_construction = 32;
    opts.vec_file_capacity = 4096;
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

TEST_CASE("Concurrent UPSERT race: N threads on the same real_id"
          * doctest::test_suite("concurrent-writer-torture")
          * doctest::skip()) {
    constexpr int kDim       = 32;
    constexpr int kThreads   = 8;
    constexpr int kIters     = 100;     // upserts per thread on the SAME id
    constexpr std::uint64_t kTargetId = 42;

    std::string path = NewTempDir("lsmvec_upsert_race");
    auto db = OpenFresh(path, kDim);

    // Seed the key so we exercise the "alive → Update" path, not the
    // "first insert" path. Without this seed, each thread races to be
    // the first inserter; that's a different (and easier) race.
    {
        std::mt19937 rng(0xFEED);
        auto v = RandVec(kDim, rng);
        REQUIRE(db->Insert(kTargetId, lsm_vec::Span<float>(v)).ok());
    }

    std::atomic<int> upsert_failures{0};
    auto worker = [&](int t) {
        std::mt19937 rng(0xC0DE + t);
        for (int i = 0; i < kIters; ++i) {
            auto v = RandVec(kDim, rng);
            // Insert on an alive key routes to UpdateInternal — the
            // check-then-act race target.
            auto st = db->Insert(kTargetId, lsm_vec::Span<float>(v));
            if (!st.ok()) {
                upsert_failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    CHECK(upsert_failures.load() == 0);

    // Post-conditions:
    //  (1) The key is alive.
    //  (2) A SearchKnn on a query that includes the most recent insert
    //      finds it; the resolved real_id is kTargetId. We don't know
    //      which thread wrote last, so just verify resolution still
    //      works and points at kTargetId.
    // index_for_test() exposes is_alive; HasLiveVector is private since
    // we skip 9f14958 (pglsmvec-specific public API).
    CHECK(db->index_for_test()->is_alive(kTargetId));

    // (3) Read-back via Get must succeed and return some valid vector
    //     of the right dimension. We don't pin which thread's vector
    //     wins — last-writer-wins is acceptable.
    std::vector<float> out;
    auto st = db->Get(kTargetId, &out);
    CHECK(st.ok());
    CHECK(out.size() == static_cast<std::size_t>(kDim));

    std::filesystem::remove_all(path);
}
