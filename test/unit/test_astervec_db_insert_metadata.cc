#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "doctest.h"
#include "astervec_db.h"

TEST_CASE("Insert with metadata persists JSON") {
    char tmpl[] = "/tmp/astervec_insertmd_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    std::string path = dir;

    astervec::AsterVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = path + "/vecs.bin";

    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());

    std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
    REQUIRE(db->Insert(7, astervec::Span<float>(v), R"({"tenant":"acme"})").ok());

    // Full GetPayload verification happens in M3.1; here we just assert Insert succeeded
    // and the binary didn't crash.
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Insert with invalid JSON metadata rejected") {
    char tmpl[] = "/tmp/astervec_insertbadmd_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    std::string path = dir;

    astervec::AsterVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = path + "/vecs.bin";

    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());

    std::vector<float> v{1, 2, 3, 4};
    auto st = db->Insert(7, astervec::Span<float>(v), R"({not json)");
    CHECK(st.IsInvalidArgument());

    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Insert: metadata JSON exceeds 64 KB rejected (I1)") {
    char tmpl[] = "/tmp/astervec_bigmd_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    std::string path = dir;

    astervec::AsterVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = path + "/vecs.bin";

    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());

    // Build a syntactically-valid metadata object whose serialized size
    // is comfortably over 64 KB: {"blob":"aaa...aaa"}.
    std::string big_value(70 * 1024, 'a');
    std::string oversize = std::string(R"({"blob":")") + big_value + R"("})";
    REQUIRE(oversize.size() > 64 * 1024);

    std::vector<float> v{1, 2, 3, 4};
    auto st = db->Insert(7, astervec::Span<float>(v), oversize);
    CHECK(st.IsInvalidArgument());

    db->Close();
    std::filesystem::remove_all(path);
}
