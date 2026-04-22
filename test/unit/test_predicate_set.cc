#include "doctest.h"
#include "metadata.h"
using namespace lsm_vec::metadata;

static Predicate inLeaf(Predicate::Kind k, FieldPath path, std::vector<Json> vals) {
    Predicate p;
    p.kind = k;
    p.path = std::move(path);
    p.values = std::move(vals);
    return p;
}

TEST_CASE("$in: element membership") {
    auto p = inLeaf(Predicate::Kind::In, {"x"}, {1, 2, 3});
    CHECK(p.matches(Json{{"x", 2}}));
    CHECK_FALSE(p.matches(Json{{"x", 4}}));
    CHECK_FALSE(p.matches(Json::object()));  // missing → false
    CHECK_FALSE(p.matches(Json{{"x", "2"}}));  // type mismatch
}

TEST_CASE("$nin: negated membership; missing → true") {
    auto p = inLeaf(Predicate::Kind::Nin, {"x"}, {1, 2, 3});
    CHECK(p.matches(Json{{"x", 4}}));
    CHECK_FALSE(p.matches(Json{{"x", 2}}));
    CHECK(p.matches(Json::object()));        // missing → true
    CHECK(p.matches(Json{{"x", "2"}}));      // type mismatch → true
}

TEST_CASE("$in with string values") {
    auto p = inLeaf(Predicate::Kind::In, {"tag"}, {std::string("py"), std::string("ml")});
    CHECK(p.matches(Json{{"tag", "py"}}));
    CHECK_FALSE(p.matches(Json{{"tag", "rust"}}));
}
