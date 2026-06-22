#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "doctest.h"
#include "json.hpp"
#include "astervec_db.h"

TEST_CASE("Opening a DB that was created before metadata feature works") {
    char tmpl[] = "/tmp/astervec_bc_XXXXXX";
    std::string path = mkdtemp(tmpl);

    // First open: creates metadata CF. Insert a few vectors without metadata.
    {
        astervec::AsterVecDBOptions opts;
        opts.dim = 4;
        opts.vec_file_capacity = 100;
        opts.vector_file_path = path + "/vecs.bin";
        std::unique_ptr<astervec::AsterVecDB> db;
        REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());
        std::vector<float> v{1, 2, 3, 4};
        for (uint64_t i = 0; i < 10; ++i) REQUIRE(db->Insert(i, astervec::Span<float>(v)).ok());
        db->Close();
    }

    // Simulate an "old" DB by removing the metadata/ subdirectory.
    std::filesystem::remove_all(path + "/metadata");
    REQUIRE_FALSE(std::filesystem::exists(path + "/metadata"));

    // Second open: metadata CF must be re-created; vectors still accessible.
    {
        astervec::AsterVecDBOptions opts;
        opts.dim = 4;
        opts.vec_file_capacity = 100;
        opts.vector_file_path = path + "/vecs.bin";
        std::unique_ptr<astervec::AsterVecDB> db;
        REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());
        CHECK(std::filesystem::exists(path + "/metadata"));

        std::vector<float> out;
        REQUIRE(db->Get(0, &out).ok());   // vectors still work
        db->Close();
    }

    std::filesystem::remove_all(path);
}

TEST_CASE("Metadata survives clean close+reopen round-trip (I2)") {
    // Insert vectors with metadata, Close(), reopen the SAME path,
    // and verify metadata round-trips through RocksDB's WAL / compaction.
    // Also exercises the full lifecycle: payload CRUD after reopen.
    char tmpl[] = "/tmp/astervec_rtt_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    std::string path = dir;

    // Phase 1: populate.
    {
        astervec::AsterVecDBOptions opts;
        opts.dim = 4;
        opts.vec_file_capacity = 100;
        opts.vector_file_path = path + "/vecs.bin";
        std::unique_ptr<astervec::AsterVecDB> db;
        REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());

        std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
        REQUIRE(db->Insert(1, astervec::Span<float>(v),
                           R"({"tenant":"acme","n":1})").ok());
        REQUIRE(db->Insert(2, astervec::Span<float>(v),
                           R"({"tenant":"acme","n":2})").ok());
        REQUIRE(db->Insert(3, astervec::Span<float>(v)).ok());  // no metadata

        // Mutate via the payload APIs to make sure RMW updates also persist.
        REQUIRE(db->UpdatePayload(2, R"({"n":22,"new":true})").ok());
        REQUIRE(db->SetPayload(3, R"({"created_at":1700000000})").ok());

        REQUIRE(db->Close().ok());
    }

    // Phase 2: reopen same path, verify every payload mutation survived.
    {
        astervec::AsterVecDBOptions opts;
        opts.dim = 4;
        opts.vec_file_capacity = 100;
        opts.vector_file_path = path + "/vecs.bin";
        std::unique_ptr<astervec::AsterVecDB> db;
        REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());

        std::string md;
        REQUIRE(db->GetPayload(1, &md).ok());
        CHECK(md == R"({"tenant":"acme","n":1})");

        REQUIRE(db->GetPayload(2, &md).ok());
        // Merge-patch on {"tenant":"acme","n":2} with {"n":22,"new":true}.
        // Key order is implementation-defined; compare as JSON.
        CHECK(nlohmann::json::parse(md) ==
              nlohmann::json::parse(R"({"tenant":"acme","n":22,"new":true})"));

        REQUIRE(db->GetPayload(3, &md).ok());
        CHECK(md == R"({"created_at":1700000000})");

        REQUIRE(db->Close().ok());
    }

    std::filesystem::remove_all(path);
}
