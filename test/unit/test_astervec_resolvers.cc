#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

#include "doctest.h"
#include "id_types.h"
#include "astervec_db.h"
#include "astervec_index.h"

using astervec::internal_id_t;
using astervec::kFirstUpdateId;
using astervec::kMaxRealId;
using astervec::AsterVec;
using astervec::AsterVecDB;
using astervec::AsterVecDBOptions;
using astervec::real_id_t;

namespace {
// Minimal disposable AsterVecDB for resolver tests. Constructs a fresh
// temp-dir DB; cleans up on destruction. Exposes the inner AsterVec.
struct TempEnv {
    std::string                path;
    std::unique_ptr<AsterVecDB>  db;

    explicit TempEnv(int dim = 4, uint64_t cap = 100) {
        char tmpl[] = "/tmp/astervec_C3_XXXXXX";
        char* dir = mkdtemp(tmpl);
        REQUIRE(dir != nullptr);
        path = dir;

        AsterVecDBOptions opts;
        opts.dim                  = dim;
        opts.vec_file_capacity    = cap;
        opts.vector_file_path     = path + "/vecs.bin";
        REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
    }

    ~TempEnv() {
        if (db) db->Close();
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    AsterVec& index() { return *db->index_for_test(); }
};
}  // namespace

TEST_CASE("resolve_internal on a fresh DB is the identity function") {
    TempEnv env;
    auto& v = env.index();
    CHECK(v.resolve_internal(0)            == 0);
    CHECK(v.resolve_internal(42)           == 42);
    CHECK(v.resolve_internal(kMaxRealId)   == kMaxRealId);
    CHECK(v.tombstone_count()              == 0);
    CHECK(v.updated_real_id_count()        == 0);
    CHECK(v.next_update_internal_id()      == kFirstUpdateId);
}

TEST_CASE("resolve_real on a bit-63=0 internal_id is the identity") {
    TempEnv env;
    auto& v = env.index();
    CHECK(v.resolve_real(0)            == 0);
    CHECK(v.resolve_real(42)           == 42);
    CHECK(v.resolve_real(kMaxRealId)   == kMaxRealId);
}

TEST_CASE("record_update_mapping wires both directions") {
    TempEnv env;
    auto& v = env.index();
    v.record_update_mapping(42, kFirstUpdateId);
    CHECK(v.resolve_internal(42)             == kFirstUpdateId);
    CHECK(v.resolve_real(kFirstUpdateId)     == 42);
    CHECK(v.updated_real_id_count()          == 1);
    CHECK(v.updated_internal_to_real_size()  == 1);
}

TEST_CASE("double update on same real_id erases the stale reverse entry (A4)") {
    TempEnv env;
    auto& v = env.index();
    v.record_update_mapping(42, kFirstUpdateId);
    v.record_update_mapping(42, kFirstUpdateId + 1);

    // Forward map points to the latest internal_id only.
    CHECK(v.resolve_internal(42) == kFirstUpdateId + 1);
    // Reverse map drops the previous-version entry — only one live entry remains.
    CHECK(v.updated_internal_to_real_size() == 1);
    CHECK(v.resolve_real(kFirstUpdateId + 1) == 42);
}

TEST_CASE("resolve_internal returns identity for real_ids outside the map (Bloom FP-safe)") {
    TempEnv env;
    auto& v = env.index();
    // Populate enough mappings that the Bloom filter starts producing some FPs.
    for (real_id_t r = 0; r < 5000; ++r) {
        v.record_update_mapping(r, kFirstUpdateId + r);
    }
    // Query for real_ids NOT in the map. Bloom may say "might_contain" (FP)
    // but the hashmap miss must fall through to the identity mapping.
    for (real_id_t r = 100'000; r < 100'100; ++r) {
        CHECK(v.resolve_internal(r) == r);
    }
}

TEST_CASE("Bloom filter auto-rebuilds when fill_ratio crosses 0.7 (Q3)") {
    TempEnv env;
    auto& v = env.index();
    const std::size_t initial_capacity = v.bloom_capacity();
    REQUIRE(initial_capacity > 0);

    // Insert enough entries to cross the 70% fill threshold.
    // Initial capacity is 512, so 360 entries (~70%) trigger.
    for (real_id_t r = 0; r < 400; ++r) {
        v.record_update_mapping(r, kFirstUpdateId + r);
    }

    // Capacity must have grown to at least 2x.
    CHECK(v.bloom_capacity() >= initial_capacity * 2);

    // After rebuild, every previously-recorded mapping still resolves.
    for (real_id_t r = 0; r < 400; ++r) {
        CHECK(v.resolve_internal(r) == kFirstUpdateId + r);
    }
}
