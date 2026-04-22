#include "doctest.h"
#include "metadata.h"
using namespace lsm_vec::metadata;

static Predicate leaf(Predicate::Kind k, FieldPath path, Json v) {
    Predicate p;
    p.kind = k;
    p.path = std::move(path);
    p.value = std::move(v);
    return p;
}

TEST_CASE("$ne: field missing returns true (Mongo semantics)") {
    auto p = leaf(Predicate::Kind::Ne, {"x"}, 5);
    CHECK(p.matches(Json::object()));                // missing → true
    CHECK_FALSE(p.matches(Json{{"x", 5}}));           // equal → false
    CHECK(p.matches(Json{{"x", 6}}));                 // different → true
    CHECK(p.matches(Json{{"x", "5"}}));               // type mismatch → true (not equal)
}

TEST_CASE("$gt / $gte / $lt / $lte numeric") {
    CHECK(leaf(Predicate::Kind::Gt, {"x"}, 5).matches(Json{{"x", 6}}));
    CHECK_FALSE(leaf(Predicate::Kind::Gt, {"x"}, 5).matches(Json{{"x", 5}}));
    CHECK(leaf(Predicate::Kind::Gte, {"x"}, 5).matches(Json{{"x", 5}}));
    CHECK(leaf(Predicate::Kind::Lt, {"x"}, 5).matches(Json{{"x", 4}}));
    CHECK(leaf(Predicate::Kind::Lte, {"x"}, 5).matches(Json{{"x", 5}}));

    // Missing → false
    CHECK_FALSE(leaf(Predicate::Kind::Gt, {"x"}, 5).matches(Json::object()));
    // Type mismatch (string vs int) → false
    CHECK_FALSE(leaf(Predicate::Kind::Gt, {"x"}, 5).matches(Json{{"x", "foo"}}));
}

TEST_CASE("$gt / $lt also work on strings") {
    CHECK(leaf(Predicate::Kind::Gt, {"x"}, std::string("b")).matches(Json{{"x", "c"}}));
    CHECK_FALSE(leaf(Predicate::Kind::Gt, {"x"}, std::string("b")).matches(Json{{"x", "a"}}));
}
