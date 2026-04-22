#include "doctest.h"
#include "metadata.h"
using namespace lsm_vec::metadata;

static void expectError(std::string_view s) {
    Predicate p;
    auto st = ParsePredicate(s, &p);
    CHECK(st.IsInvalidArgument());
}

TEST_CASE("Parse errors: malformed JSON") {
    expectError("{");
    expectError(R"({"a":)");
}

TEST_CASE("Parse errors: non-object filter root") {
    expectError(R"([1,2,3])");
    expectError(R"("string")");
}

TEST_CASE("Parse errors: unknown operator") {
    expectError(R"({"x":{"$foo":1}})");
}

TEST_CASE("Parse errors: $in / $nin / $contains_any / $contains_all require array") {
    expectError(R"({"x":{"$in":"not an array"}})");
    expectError(R"({"x":{"$nin":42}})");
    expectError(R"({"x":{"$contains_any":"nope"}})");
    expectError(R"({"x":{"$contains_all":{}}})");
}

TEST_CASE("Parse errors: $and / $or must be array of objects") {
    expectError(R"({"$and":"not-an-array"})");
    expectError(R"({"$or":[1,2,3]})");
    expectError(R"({"$and":[{"a":1},"not-an-object"]})");
}

TEST_CASE("Parse errors: $exists must be bool") {
    expectError(R"({"x":{"$exists":"yes"}})");
    expectError(R"({"x":{"$exists":1}})");
}

TEST_CASE("Parse errors: mixed operator and field keys in value object") {
    expectError(R"({"x":{"$gt":5,"y":10}})");
    expectError(R"({"x":{"a":1,"$eq":2}})");
}
