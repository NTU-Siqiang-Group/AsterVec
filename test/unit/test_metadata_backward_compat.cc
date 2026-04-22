#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

#include "doctest.h"
#include "lsm_vec_db.h"

TEST_CASE("Opening a DB that was created before metadata feature works") {
    char tmpl[] = "/tmp/lsmvec_bc_XXXXXX";
    std::string path = mkdtemp(tmpl);

    // First open: creates metadata CF. Insert a few vectors without metadata.
    {
        lsm_vec::LSMVecDBOptions opts;
        opts.dim = 4;
        opts.vec_file_capacity = 100;
        opts.vector_file_path = path + "/vecs.bin";
        std::unique_ptr<lsm_vec::LSMVecDB> db;
        REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());
        std::vector<float> v{1, 2, 3, 4};
        for (uint64_t i = 0; i < 10; ++i) REQUIRE(db->Insert(i, lsm_vec::Span<float>(v)).ok());
        db->Close();
    }

    // Simulate an "old" DB by removing the metadata/ subdirectory.
    std::filesystem::remove_all(path + "/metadata");
    REQUIRE_FALSE(std::filesystem::exists(path + "/metadata"));

    // Second open: metadata CF must be re-created; vectors still accessible.
    {
        lsm_vec::LSMVecDBOptions opts;
        opts.dim = 4;
        opts.vec_file_capacity = 100;
        opts.vector_file_path = path + "/vecs.bin";
        std::unique_ptr<lsm_vec::LSMVecDB> db;
        REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());
        CHECK(std::filesystem::exists(path + "/metadata"));

        std::vector<float> out;
        REQUIRE(db->Get(0, &out).ok());   // vectors still work
        db->Close();
    }

    std::filesystem::remove_all(path);
}
