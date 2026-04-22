#include "doctest.h"
#include "metadata.h"
using namespace lsm_vec::metadata;

TEST_CASE("$exists: true/false semantics") {
    Predicate p;
    p.kind = Predicate::Kind::Exists;
    p.path = {"maybe"};

    p.exists_expected = true;
    CHECK(p.matches(Json{{"maybe", 1}}));
    CHECK(p.matches(Json{{"maybe", nullptr}}));  // null value still counts as "exists"
    CHECK_FALSE(p.matches(Json::object()));

    p.exists_expected = false;
    CHECK_FALSE(p.matches(Json{{"maybe", 1}}));
    CHECK(p.matches(Json::object()));
}

TEST_CASE("$contains_any") {
    Predicate p;
    p.kind = Predicate::Kind::ContainsAny;
    p.path = {"tags"};
    p.values = {std::string("py"), std::string("ml")};

    CHECK(p.matches(Json{{"tags", {"py", "rust"}}}));
    CHECK(p.matches(Json{{"tags", {"ml"}}}));
    CHECK_FALSE(p.matches(Json{{"tags", {"rust", "go"}}}));
    CHECK_FALSE(p.matches(Json::object()));                 // missing → false
    CHECK_FALSE(p.matches(Json{{"tags", "py"}}));           // not array → false
}

TEST_CASE("$contains_all") {
    Predicate p;
    p.kind = Predicate::Kind::ContainsAll;
    p.path = {"tags"};
    p.values = {std::string("py"), std::string("ml")};

    CHECK(p.matches(Json{{"tags", {"py", "ml", "rust"}}}));
    CHECK_FALSE(p.matches(Json{{"tags", {"py"}}}));         // missing "ml"
    CHECK_FALSE(p.matches(Json::object()));
}
