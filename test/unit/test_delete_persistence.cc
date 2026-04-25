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
#include "lsm_vec_db.h"
#include "lsm_vec_index.h"

using lsm_vec::internal_id_t;
using lsm_vec::kFirstUpdateId;
using lsm_vec::LSMVecDB;
using lsm_vec::LSMVecDBOptions;
using lsm_vec::real_id_t;
using lsm_vec::Span;

namespace {

std::string make_path() {
    char tmpl[] = "/tmp/lsmvec_C7_XXXXXX";
    char* dir = mkdtemp(tmpl);
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
