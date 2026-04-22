#include "doctest.h"
#include "metadata.h"
#include <string>
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

TEST_CASE("Parse errors: filter JSON exceeds 8 KB (I1)") {
    // Build a syntactically-valid but oversize filter: {"k":"aaaa...aaaa"}.
    // Total size > 8 KB → must be rejected before Json::parse touches it.
    std::string big_value(9000, 'a');
    std::string oversize = std::string(R"({"k":")") + big_value + R"("})";
    CHECK(oversize.size() > 8 * 1024);
    expectError(oversize);
}

TEST_CASE("Parse errors: predicate nesting exceeds 32 levels (I1/M5)") {
    // Build {"$and":[{"$and":[...40 levels deep...]}]} programmatically.
    // The parser's depth cap (kMaxPredicateDepth = 32) must reject this
    // before the AST is fully constructed.
    std::string deep = R"({"x":1})";
    for (int i = 0; i < 40; ++i) {
        deep = std::string(R"({"$and":[)") + deep + "]}";
    }
    // Sanity: our test payload is still under the 8 KB byte cap, so it will
    // reach the depth check rather than being rejected for size.
    CHECK(deep.size() < 8 * 1024);
    expectError(deep);
}
