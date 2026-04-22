#include "doctest.h"
#include "metadata_store.h"
#include "metadata_store_fixture.h"

using namespace lsm_vec;
using namespace lsm_vec::test;

TEST_CASE("MetadataStore: Put then Get round-trip") {
    TempDB t;
    MetadataStore store(t.db, t.cf_metadata);

    auto put_st = store.Put(42, R"({"tenant":"acme","count":7})");
    REQUIRE(put_st.ok());

    std::string bytes;
    auto get_st = store.Get(42, &bytes);
    REQUIRE(get_st.ok());
    CHECK(bytes == R"({"tenant":"acme","count":7})");
}

TEST_CASE("MetadataStore: Get of non-existent id is NotFound") {
    TempDB t;
    MetadataStore store(t.db, t.cf_metadata);

    std::string bytes;
    auto st = store.Get(99, &bytes);
    CHECK(st.IsNotFound());
}

TEST_CASE("MetadataStore: Put with JSON object overload") {
    TempDB t;
    MetadataStore store(t.db, t.cf_metadata);

    MetadataStore::Json j = {{"x", 1}, {"y", "foo"}};
    REQUIRE(store.Put(1, j).ok());

    MetadataStore::Json out;
    REQUIRE(store.Get(1, &out).ok());
    CHECK(out == j);
}

TEST_CASE("MetadataStore: Put overwrites") {
    TempDB t;
    MetadataStore store(t.db, t.cf_metadata);
    REQUIRE(store.Put(7, R"({"v":1})").ok());
    REQUIRE(store.Put(7, R"({"v":2})").ok());
    std::string bytes;
    REQUIRE(store.Get(7, &bytes).ok());
    CHECK(bytes == R"({"v":2})");
}

TEST_CASE("MetadataStore: Delete is idempotent") {
    TempDB t;
    MetadataStore store(t.db, t.cf_metadata);
    REQUIRE(store.Put(5, R"({})").ok());
    REQUIRE(store.Delete(5).ok());
    REQUIRE(store.Delete(5).ok());           // double delete OK
    std::string bytes;
    CHECK(store.Get(5, &bytes).IsNotFound());
}
