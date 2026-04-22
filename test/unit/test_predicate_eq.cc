#include "doctest.h"
#include "metadata.h"

using namespace lsm_vec::metadata;

TEST_CASE("$eq: matches equal scalar") {
    Predicate p;
    p.kind  = Predicate::Kind::Eq;
    p.path  = {"name"};
    p.value = "alice";

    Json doc  = {{"name", "alice"}};
    Json doc2 = {{"name", "bob"}};
    Json doc3 = {};

    CHECK(p.matches(doc));
    CHECK_FALSE(p.matches(doc2));
    CHECK_FALSE(p.matches(doc3));
}

TEST_CASE("$eq: scalar types (int, double, bool, null)") {
    Predicate p;
    p.kind = Predicate::Kind::Eq;
    p.path = {"x"};

    p.value = 42;
    CHECK(p.matches(Json{{"x", 42}}));
    CHECK_FALSE(p.matches(Json{{"x", 43}}));
    CHECK_FALSE(p.matches(Json{{"x", "42"}}));  // type mismatch

    p.value = 3.14;
    CHECK(p.matches(Json{{"x", 3.14}}));

    p.value = true;
    CHECK(p.matches(Json{{"x", true}}));
    CHECK_FALSE(p.matches(Json{{"x", false}}));

    p.value = nullptr;
    CHECK(p.matches(Json{{"x", nullptr}}));
    CHECK_FALSE(p.matches(Json{{"x", 0}}));  // null vs 0 → unequal
}
