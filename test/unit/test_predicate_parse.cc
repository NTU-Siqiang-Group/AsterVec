#include "doctest.h"
#include "metadata.h"
using namespace lsm_vec::metadata;

static Predicate parseOrFail(std::string_view s) {
    Predicate p;
    auto st = ParsePredicate(s, &p);
    REQUIRE(st.ok());
    return p;
}

TEST_CASE("Parse: empty filter = match all") {
    auto p = parseOrFail("");
    CHECK(p.matches(Json::object()));
    CHECK(p.matches(Json{{"anything", 1}}));

    auto p2 = parseOrFail("{}");
    CHECK(p2.matches(Json::object()));
}

TEST_CASE("Parse: implicit eq with scalar value") {
    auto p = parseOrFail(R"({"tenant_id":"acme"})");
    CHECK(p.matches(Json{{"tenant_id","acme"}}));
    CHECK_FALSE(p.matches(Json{{"tenant_id","other"}}));
}

TEST_CASE("Parse: explicit operators") {
    auto p = parseOrFail(R"({"age":{"$gt":18,"$lt":65}})");
    CHECK(p.matches(Json{{"age", 30}}));
    CHECK_FALSE(p.matches(Json{{"age", 10}}));
    CHECK_FALSE(p.matches(Json{{"age", 70}}));
}

TEST_CASE("Parse: $in, $nin") {
    auto p = parseOrFail(R"({"tag":{"$in":["py","ml"]}})");
    CHECK(p.matches(Json{{"tag","py"}}));
    CHECK_FALSE(p.matches(Json{{"tag","rust"}}));
}

TEST_CASE("Parse: $contains_any, $contains_all") {
    auto pany = parseOrFail(R"({"tags":{"$contains_any":["py","ml"]}})");
    CHECK(pany.matches(Json::parse(R"({"tags":["py","rust"]})")));

    auto pall = parseOrFail(R"({"tags":{"$contains_all":["py","ml"]}})");
    CHECK(pall.matches(Json::parse(R"({"tags":["py","ml","rust"]})")));
    CHECK_FALSE(pall.matches(Json::parse(R"({"tags":["py"]})")));
}

TEST_CASE("Parse: $exists") {
    auto p = parseOrFail(R"({"optional":{"$exists":true}})");
    CHECK(p.matches(Json{{"optional",1}}));
    CHECK_FALSE(p.matches(Json::object()));
}

TEST_CASE("Parse: $and / $or") {
    auto p = parseOrFail(R"({"$or":[{"a":1},{"b":2}]})");
    CHECK(p.matches(Json{{"a",1}}));
    CHECK(p.matches(Json{{"b",2}}));
    CHECK_FALSE(p.matches(Json{{"a",0},{"b",0}}));
}

TEST_CASE("Parse: dot-path key") {
    auto p = parseOrFail(R"({"author.name":"alice"})");
    CHECK(p.matches(Json::parse(R"({"author":{"name":"alice"}})")));
}

TEST_CASE("Parse: top-level multi-key = implicit AND") {
    auto p = parseOrFail(R"({"a":1,"b":2})");
    CHECK(p.matches(Json{{"a",1},{"b",2}}));
    CHECK_FALSE(p.matches(Json{{"a",1}}));
}

TEST_CASE("Parse: implicit eq with object value (deep equality)") {
    auto p = parseOrFail(R"({"author":{"name":"alice","age":30}})");
    CHECK(p.matches(Json::parse(R"({"author":{"name":"alice","age":30}})")));
    CHECK_FALSE(p.matches(Json::parse(R"({"author":{"name":"alice","age":31}})")));
    CHECK_FALSE(p.matches(Json::parse(R"({"author":{"name":"bob","age":30}})")));
    CHECK_FALSE(p.matches(Json::object()));
}

TEST_CASE("Parse: implicit eq with array value") {
    auto p = parseOrFail(R"({"tags":["py","ml"]})");
    CHECK(p.matches(Json::parse(R"({"tags":["py","ml"]})")));
    CHECK_FALSE(p.matches(Json::parse(R"({"tags":["py"]})")));           // shorter
    CHECK_FALSE(p.matches(Json::parse(R"({"tags":["ml","py"]})")));      // reordered — array eq is order-sensitive
}

TEST_CASE("Parse: empty object value is implicit eq with empty object") {
    // {} as a value is not an operator-object (no keys) → should be implicit eq with empty object.
    auto p = parseOrFail(R"({"meta":{}})");
    CHECK(p.matches(Json::parse(R"({"meta":{}})")));
    CHECK_FALSE(p.matches(Json::parse(R"({"meta":{"x":1}})")));
}
