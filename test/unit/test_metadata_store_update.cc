#include "doctest.h"
#include "metadata_store.h"
#include "metadata_store_fixture.h"
using namespace astervec;
using namespace astervec::test;

TEST_CASE("Update: merge-patch adds new fields") {
    TempDB t;
    MetadataStore s(t.db, t.cf_metadata);
    REQUIRE(s.Put(1, R"({"a":1})").ok());

    REQUIRE(s.Update(1, MetadataStore::Json{{"b", 2}}).ok());

    MetadataStore::Json out;
    REQUIRE(s.Get(1, &out).ok());
    CHECK(out == MetadataStore::Json::parse(R"({"a":1,"b":2})"));
}

TEST_CASE("Update: merge-patch overwrites scalar fields") {
    TempDB t;
    MetadataStore s(t.db, t.cf_metadata);
    REQUIRE(s.Put(1, R"({"a":1})").ok());

    REQUIRE(s.Update(1, MetadataStore::Json{{"a", 99}}).ok());

    MetadataStore::Json out;
    REQUIRE(s.Get(1, &out).ok());
    CHECK(out == MetadataStore::Json::parse(R"({"a":99})"));
}

TEST_CASE("Update: null value deletes field (RFC 7396)") {
    TempDB t;
    MetadataStore s(t.db, t.cf_metadata);
    REQUIRE(s.Put(1, R"({"a":1,"b":2})").ok());

    REQUIRE(s.Update(1, MetadataStore::Json::parse(R"({"a":null})")).ok());

    MetadataStore::Json out;
    REQUIRE(s.Get(1, &out).ok());
    CHECK(out == MetadataStore::Json::parse(R"({"b":2})"));
}

TEST_CASE("Update: nested merge") {
    TempDB t;
    MetadataStore s(t.db, t.cf_metadata);
    REQUIRE(s.Put(1, R"({"meta":{"x":1,"y":2}})").ok());

    REQUIRE(s.Update(1, MetadataStore::Json::parse(R"({"meta":{"y":99,"z":3}})")).ok());

    MetadataStore::Json out;
    REQUIRE(s.Get(1, &out).ok());
    CHECK(out == MetadataStore::Json::parse(R"({"meta":{"x":1,"y":99,"z":3}})"));
}

TEST_CASE("Update on non-existent id creates row") {
    TempDB t;
    MetadataStore s(t.db, t.cf_metadata);

    REQUIRE(s.Update(42, MetadataStore::Json{{"hello", "world"}}).ok());

    MetadataStore::Json out;
    REQUIRE(s.Get(42, &out).ok());
    CHECK(out == MetadataStore::Json{{"hello", "world"}});
}

TEST_CASE("DeleteKeys: removes listed keys") {
    TempDB t;
    MetadataStore s(t.db, t.cf_metadata);
    REQUIRE(s.Put(1, R"({"a":1,"b":2,"c":3})").ok());

    REQUIRE(s.DeleteKeys(1, {"b", "c"}).ok());

    MetadataStore::Json out;
    REQUIRE(s.Get(1, &out).ok());
    CHECK(out == MetadataStore::Json::parse(R"({"a":1})"));
}

TEST_CASE("DeleteKeys: missing keys are ignored") {
    TempDB t;
    MetadataStore s(t.db, t.cf_metadata);
    REQUIRE(s.Put(1, R"({"a":1})").ok());

    REQUIRE(s.DeleteKeys(1, {"nonexistent", "a"}).ok());

    MetadataStore::Json out;
    REQUIRE(s.Get(1, &out).ok());
    CHECK(out == MetadataStore::Json::object());
}

TEST_CASE("DeleteKeys on non-existent id is NotFound") {
    TempDB t;
    MetadataStore s(t.db, t.cf_metadata);
    auto st = s.DeleteKeys(99, {"a"});
    CHECK(st.IsNotFound());
}
