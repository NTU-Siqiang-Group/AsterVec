#include "doctest.h"
#include "metadata.h"
using namespace lsm_vec::metadata;

static Predicate eqLeaf(FieldPath path, Json v) {
    Predicate p;
    p.kind = Predicate::Kind::Eq;
    p.path = std::move(path);
    p.value = std::move(v);
    return p;
}

TEST_CASE("$and: conjunction; empty = true") {
    Predicate p;
    p.kind = Predicate::Kind::And;
    p.children.push_back(eqLeaf({"a"}, 1));
    p.children.push_back(eqLeaf({"b"}, 2));

    CHECK(p.matches(Json{{"a",1},{"b",2}}));
    CHECK_FALSE(p.matches(Json{{"a",1},{"b",3}}));
    CHECK_FALSE(p.matches(Json{{"a",0},{"b",2}}));

    // Empty AND matches everything (used for empty filter)
    Predicate empty;
    empty.kind = Predicate::Kind::And;
    CHECK(empty.matches(Json::object()));
    CHECK(empty.matches(Json{{"a",1}}));
}

TEST_CASE("$or: disjunction; empty = false") {
    Predicate p;
    p.kind = Predicate::Kind::Or;
    p.children.push_back(eqLeaf({"a"}, 1));
    p.children.push_back(eqLeaf({"b"}, 2));

    CHECK(p.matches(Json{{"a",1}}));
    CHECK(p.matches(Json{{"b",2}}));
    CHECK(p.matches(Json{{"a",1},{"b",2}}));
    CHECK_FALSE(p.matches(Json{{"a",0},{"b",0}}));

    Predicate empty;
    empty.kind = Predicate::Kind::Or;
    CHECK_FALSE(empty.matches(Json::object()));
}

TEST_CASE("Nested $and/$or") {
    Predicate inner;
    inner.kind = Predicate::Kind::Or;
    inner.children.push_back(eqLeaf({"a"}, 1));
    inner.children.push_back(eqLeaf({"a"}, 2));

    Predicate outer;
    outer.kind = Predicate::Kind::And;
    outer.children.push_back(std::move(inner));
    outer.children.push_back(eqLeaf({"b"}, 10));

    CHECK(outer.matches(Json{{"a",1},{"b",10}}));
    CHECK(outer.matches(Json{{"a",2},{"b",10}}));
    CHECK_FALSE(outer.matches(Json{{"a",3},{"b",10}}));
    CHECK_FALSE(outer.matches(Json{{"a",1},{"b",99}}));
}
