#include "doctest.h"
#include "metadata.h"
using namespace lsm_vec::metadata;

TEST_CASE("Dot-path: nested object") {
    Predicate p;
    p.kind = Predicate::Kind::Eq;
    p.path = {"author", "name"};
    p.value = std::string("alice");

    CHECK(p.matches(Json::parse(R"({"author":{"name":"alice"}})")));
    CHECK_FALSE(p.matches(Json::parse(R"({"author":{"name":"bob"}})")));
    CHECK_FALSE(p.matches(Json::parse(R"({"author":{}})")));
    CHECK_FALSE(p.matches(Json::object()));
}

TEST_CASE("Dot-path: intermediate segment is not object -> missing") {
    Predicate p;
    p.kind = Predicate::Kind::Eq;
    p.path = {"author", "name"};
    p.value = std::string("alice");

    CHECK_FALSE(p.matches(Json::parse(R"({"author":"alice"})")));      // "author" is string, can't descend
    CHECK_FALSE(p.matches(Json::parse(R"({"author":["alice"]})")));    // array, not object
}

TEST_CASE("Dot-path: deep nesting") {
    Predicate p;
    p.kind = Predicate::Kind::Eq;
    p.path = {"a", "b", "c", "d"};
    p.value = 42;

    CHECK(p.matches(Json::parse(R"({"a":{"b":{"c":{"d":42}}}})")));
    CHECK_FALSE(p.matches(Json::parse(R"({"a":{"b":{"c":{"d":43}}}})")));
}

TEST_CASE("Dot-path: literal key that looks like an array index") {
    // Path "tags.0" looks like "tags" array element 0, but Level-2 does NOT do array indexing.
    // It looks up the literal key "0" in the object "tags".
    Predicate p;
    p.kind = Predicate::Kind::Eq;
    p.path = {"tags", "0"};
    p.value = std::string("py");

    CHECK_FALSE(p.matches(Json::parse(R"({"tags":["py","ml"]})")));  // array, not object
    CHECK(p.matches(Json::parse(R"({"tags":{"0":"py"}})")));          // literal "0" key
}
