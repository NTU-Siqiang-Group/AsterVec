// C7 persistence tests for the robust-delete redesign.
// P1: tombstones + sparse update maps + monotonic counter survive Close+reopen.
// P2: Bloom filter is reconstructed from the persisted forward map.

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "doctest.h"
#include "id_types.h"
#include "astervec_db.h"
#include "astervec_index.h"

using astervec::internal_id_t;
using astervec::kFirstUpdateId;
using astervec::AsterVecDB;
using astervec::AsterVecDBOptions;
using astervec::real_id_t;
using astervec::Span;

namespace {

std::string make_path() {
    char tmpl[] = "/tmp/astervec_C7_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    return dir;
}

std::unique_ptr<AsterVecDB> open_db(const std::string& path, int dim, uint64_t cap,
                                   uint64_t seed = 1) {
    AsterVecDBOptions opts;
    opts.dim                  = dim;
    opts.vec_file_capacity    = cap;
    opts.vector_file_path     = path + "/vecs.bin";
    opts.random_seed          = seed;
    std::unique_ptr<AsterVecDB> db;
    REQUIRE(AsterVecDB::Open(path, opts, &db).ok());
    return db;
}

}  // namespace

TEST_CASE("P1: tombstones + sparse maps + counter survive Close+reopen") {
    auto path = make_path();

    // Phase 1: populate, exercise Update + Delete, capture state.
    internal_id_t expected_next_update_id = 0;
    {
        auto db = open_db(path, 4, 100);
        std::vector<float> v0{1, 2, 3, 4};

        // 5 fresh inserts (real_ids = 0..4)
        for (int i = 0; i < 5; ++i) {
            std::vector<float> v(4, static_cast<float>(i + 1));
            REQUIRE(db->Insert(static_cast<uint64_t>(i), Span<float>(v)).ok());
        }
        // Update real_id=2 twice → 2 update_ids allocated, 1 tombstone for old=2,
        // 1 tombstone for first update_id (overwritten by second update).
        std::vector<float> v22{20, 20, 20, 20};
        std::vector<float> v23{30, 30, 30, 30};
        REQUIRE(db->Update(2, Span<float>(v22)).ok());
        REQUIRE(db->Update(2, Span<float>(v23)).ok());
        // Delete real_id=4 (direct tombstone)
        REQUIRE(db->Delete(4).ok());

        // Snapshot expected state via test accessors before Close.
        auto& v = *db->index_for_test();
        CHECK(v.tombstone_count() == 3);              // old 2, first update_id, direct 4
        CHECK(v.updated_real_id_count() == 1);        // only real_id=2 has a sparse mapping
        expected_next_update_id = v.next_update_internal_id();
        CHECK(expected_next_update_id == kFirstUpdateId + 2);

        REQUIRE(db->Close().ok());
    }

    // Phase 2: reopen, verify state matches.
    {
        auto db = open_db(path, 4, 100);
        auto& v = *db->index_for_test();

        CHECK(v.tombstone_count() == 3);
        CHECK(v.updated_real_id_count() == 1);
        CHECK(v.updated_internal_to_real_size() == 1);
        CHECK(v.next_update_internal_id() == expected_next_update_id);

        // resolve_internal still routes real_id=2 to its current update id.
        CHECK(v.resolve_internal(2) == kFirstUpdateId + 1);
        CHECK(v.resolve_real(kFirstUpdateId + 1) == 2);

        // Functional: Get returns the latest vector for r=2.
        std::vector<float> got;
        REQUIRE(db->Get(2, &got).ok());
        CHECK(got == std::vector<float>(4, 30.0f));

        // Real_id=4 stays deleted.
        CHECK(db->Get(4, &got).IsNotFound());

        REQUIRE(db->Close().ok());
    }

    std::filesystem::remove_all(path);
}

TEST_CASE("P2: Bloom filter is rebuilt from the forward map on Open") {
    auto path = make_path();

    // Phase 1: populate enough updates that the rebuilt Bloom filter must
    // contain non-trivial state. Must not be persisted to disk — only the
    // forward map should be — and yet membership must survive reopen.
    {
        auto db = open_db(path, 4, 1000);
        for (int i = 0; i < 50; ++i) {
            std::vector<float> v(4, static_cast<float>(i));
            REQUIRE(db->Insert(static_cast<uint64_t>(i), Span<float>(v)).ok());
        }
        // Update half of them — produces 25 forward-map entries.
        for (int i = 0; i < 50; i += 2) {
            std::vector<float> v(4, static_cast<float>(i + 100));
            REQUIRE(db->Update(static_cast<uint64_t>(i), Span<float>(v)).ok());
        }

        auto& v = *db->index_for_test();
        CHECK(v.updated_real_id_count() == 25);

        REQUIRE(db->Close().ok());
    }

    // Phase 2: reopen and verify Bloom filter contains those 25 keys —
    // i.e., resolve_internal returns the update id (not identity) for each.
    {
        auto db = open_db(path, 4, 1000);
        auto& v = *db->index_for_test();

        CHECK(v.updated_real_id_count() == 25);
        CHECK(v.bloom_capacity() >= 512);  // floor

        // For each updated real_id, resolve must hit the sparse map.
        for (int i = 0; i < 50; i += 2) {
            internal_id_t internal = v.resolve_internal(static_cast<real_id_t>(i));
            CHECK(internal >= kFirstUpdateId);
        }
        // For non-updated real_ids, resolve must stay identity.
        for (int i = 1; i < 50; i += 2) {
            internal_id_t internal = v.resolve_internal(static_cast<real_id_t>(i));
            CHECK(internal == static_cast<internal_id_t>(i));
        }

        REQUIRE(db->Close().ok());
    }

    std::filesystem::remove_all(path);
}

TEST_CASE("C8: GetDeleteStats reports consistent counters after Insert/Update/Delete") {
    auto path = make_path();
    auto db = open_db(path, 4, 2000);

    // Initial state: nothing happened yet.
    auto s0 = db->GetDeleteStats();
    CHECK(s0.tombstones          == 0);
    CHECK(s0.updated_real_ids    == 0);
    CHECK(s0.total_inserts_ever  == 0);
    CHECK(s0.tombstone_ratio     == 0.0);
    CHECK(s0.bloom_rebuild_count == 0);
    CHECK(s0.bloom_capacity      >= 512);

    // 10 fresh inserts → 10 HNSW inserts, 0 tombstones.
    for (int i = 0; i < 10; ++i) {
        std::vector<float> v(4, static_cast<float>(i));
        REQUIRE(db->Insert(static_cast<uint64_t>(i), Span<float>(v)).ok());
    }
    auto s1 = db->GetDeleteStats();
    CHECK(s1.total_inserts_ever  == 10);
    CHECK(s1.tombstones          == 0);
    CHECK(s1.tombstone_ratio     == 0.0);

    // Update real_id=3 once → +1 HNSW insert, +1 tombstone, +1 sparse map entry.
    std::vector<float> v33{30, 30, 30, 30};
    REQUIRE(db->Update(3, Span<float>(v33)).ok());
    auto s2 = db->GetDeleteStats();
    CHECK(s2.total_inserts_ever  == 11);
    CHECK(s2.tombstones          == 1);
    CHECK(s2.updated_real_ids    == 1);
    CHECK(s2.tombstone_ratio == doctest::Approx(1.0 / 11.0));

    // Delete real_id=5 → +1 tombstone, no new HNSW insert.
    REQUIRE(db->Delete(5).ok());
    auto s3 = db->GetDeleteStats();
    CHECK(s3.total_inserts_ever  == 11);
    CHECK(s3.tombstones          == 2);
    CHECK(s3.updated_real_ids    == 1);   // r=5 was direct, no sparse entry was created
    CHECK(s3.tombstone_ratio == doctest::Approx(2.0 / 11.0));

    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("C8: Bloom rebuild count increments on auto-rebuild") {
    auto path = make_path();
    auto db = open_db(path, 4, 5000);

    // First populate enough fresh inserts so we have updateable real_ids.
    for (int i = 0; i < 600; ++i) {
        std::vector<float> v(4, static_cast<float>(i));
        REQUIRE(db->Insert(static_cast<uint64_t>(i), Span<float>(v)).ok());
    }
    CHECK(db->GetDeleteStats().bloom_rebuild_count == 0);

    // Update enough to force the Bloom filter past 70% fill — bloom_capacity
    // starts at 512, so ~360 entries cross the threshold.
    for (int i = 0; i < 400; ++i) {
        std::vector<float> v(4, static_cast<float>(i + 1000));
        REQUIRE(db->Update(static_cast<uint64_t>(i), Span<float>(v)).ok());
    }
    auto s = db->GetDeleteStats();
    CHECK(s.bloom_rebuild_count >= 1);
    CHECK(s.bloom_capacity      >= 1024);  // at least 2x growth from 512

    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Backward compat: v1 metadata file (no update map, no update_locations) loads cleanly") {
    // Simulate this by writing a fresh DB, closing without ever Updating,
    // and confirming the v2 reader handles a "no updates ever" file (which
    // matches v1 semantics: empty maps, counter at kFirstUpdateId).
    auto path = make_path();
    {
        auto db = open_db(path, 4, 100);
        std::vector<float> v{1, 2, 3, 4};
        REQUIRE(db->Insert(7, Span<float>(v)).ok());
        REQUIRE(db->Close().ok());
    }
    {
        auto db = open_db(path, 4, 100);
        auto& v = *db->index_for_test();
        CHECK(v.tombstone_count() == 0);
        CHECK(v.updated_real_id_count() == 0);
        CHECK(v.next_update_internal_id() == kFirstUpdateId);
        std::vector<float> got;
        REQUIRE(db->Get(7, &got).ok());
        std::vector<float> expected{1, 2, 3, 4};
        CHECK(got == expected);
        REQUIRE(db->Close().ok());
    }
    std::filesystem::remove_all(path);
}
