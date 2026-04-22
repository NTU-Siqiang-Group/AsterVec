#include "doctest.h"
#include "lsm_vec_db.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"

static std::string NewTempPath() {
    char tmpl[] = "/tmp/lsmvec_crud_XXXXXX";
    char* dir = mkdtemp(tmpl);
    REQUIRE(dir != nullptr);
    return std::string(dir);
}

static std::unique_ptr<lsm_vec::LSMVecDB> OpenFresh(std::string* out_path) {
    *out_path = NewTempPath();
    lsm_vec::LSMVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = *out_path + "/vecs.bin";
    std::unique_ptr<lsm_vec::LSMVecDB> db;
    REQUIRE(lsm_vec::LSMVecDB::Open(*out_path, opts, &db).ok());
    return db;
}

TEST_CASE("GetPayload: id without metadata returns empty object") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v)).ok());   // no metadata

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(got == "{}");
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("GetPayload: id with metadata returns stored JSON") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v), R"({"k":"v"})").ok());

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(got == R"({"k":"v"})");
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("SetPayload: overwrites existing metadata") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v), R"({"k":"v1"})").ok());

    REQUIRE(db->SetPayload(1, R"({"k":"v2","new":1})").ok());

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(got == R"({"k":"v2","new":1})");
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("SetPayload: NotFound on id without vector") {
    std::string path;
    auto db = OpenFresh(&path);
    auto st = db->SetPayload(99, R"({"k":"v"})");
    CHECK(st.IsNotFound());
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("SetPayload: invalid JSON rejected") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v)).ok());
    auto st = db->SetPayload(1, R"({not json)");
    CHECK(st.IsInvalidArgument());
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("UpdatePayload: merge-patch adds fields") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v), R"({"a":1})").ok());

    REQUIRE(db->UpdatePayload(1, R"({"b":2})").ok());

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(nlohmann::json::parse(got) ==
          nlohmann::json::parse(R"({"a":1,"b":2})"));
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("UpdatePayload: null deletes field") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v), R"({"a":1,"b":2})").ok());

    REQUIRE(db->UpdatePayload(1, R"({"a":null})").ok());

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(nlohmann::json::parse(got) == nlohmann::json::parse(R"({"b":2})"));
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("UpdatePayload: NotFound on id without vector") {
    std::string path;
    auto db = OpenFresh(&path);
    auto st = db->UpdatePayload(99, R"({"a":1})");
    CHECK(st.IsNotFound());
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("DeletePayloadKeys: removes keys, idempotent for missing") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v), R"({"a":1,"b":2,"c":3})").ok());

    std::vector<std::string> keys{"b", "nonexistent"};
    REQUIRE(db->DeletePayloadKeys(1, lsm_vec::Span<const std::string>(keys)).ok());

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(nlohmann::json::parse(got) == nlohmann::json::parse(R"({"a":1,"c":3})"));
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("DeletePayloadKeys: NotFound on id with no metadata") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v)).ok());  // no metadata

    std::vector<std::string> keys{"a"};
    auto st = db->DeletePayloadKeys(1, lsm_vec::Span<const std::string>(keys));
    CHECK(st.IsNotFound());
    db->Close();
    std::filesystem::remove_all(path);
}
