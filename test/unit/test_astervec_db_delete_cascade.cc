#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

#include "doctest.h"
#include "astervec_db.h"
#include "metadata_store.h"

// NOTE: Full round-trip (Insert -> Delete -> Get returns NotFound) is deferred
// to M3 when GetPayload is available. Here we only assert that Delete does
// not fail on a previously-inserted id with metadata.

TEST_CASE("Delete on id with metadata does not error") {
    char tmpl[] = "/tmp/astervec_delcascade_XXXXXX";
    std::string path = mkdtemp(tmpl);

    astervec::AsterVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = path + "/vecs.bin";

    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());

    std::vector<float> v{1, 2, 3, 4};
    REQUIRE(db->Insert(7, astervec::Span<float>(v), R"({"tag":"x"})").ok());
    REQUIRE(db->Delete(7).ok());

    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Delete cascades: GetPayload after Delete returns empty") {
    char tmpl[] = "/tmp/astervec_delrt_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    std::string path = dir;

    astervec::AsterVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = path + "/vecs.bin";
    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());

    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(7, astervec::Span<float>(v), R"({"tag":"x"})").ok());

    std::string got;
    REQUIRE(db->GetPayload(7, &got).ok());
    CHECK(got == R"({"tag":"x"})");

    REQUIRE(db->Delete(7).ok());

    REQUIRE(db->GetPayload(7, &got).ok());
    CHECK(got == "{}");

    db->Close();
    std::filesystem::remove_all(path);
}
