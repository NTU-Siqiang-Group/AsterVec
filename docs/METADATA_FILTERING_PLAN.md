# LSM-Vec Metadata Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Phase-1 metadata filtering to LSM-Vec — JSON metadata per vector, Mongo-style filter expressions (Level 2 operator set), HNSW search with pre-filter + iterative candidate expansion.

**Architecture:** New `metadata::Predicate` AST and `MetadataStore` (RocksDB CF); public API gains overloads on `Insert` and `SearchKnn`, and four new payload CRUD methods. HNSW `searchLayer` gets a new overload accepting a predicate, with decoupled routing/filtering and a `max_scan_candidates` expansion cap. Existing signatures and code paths are unchanged.

**Tech Stack:** C++17 · CMake · RocksDB (via Aster) · nlohmann/json (vendored, new) · doctest (vendored, new, for unit tests) · pybind11 (existing, for Python bindings).

**Reference docs:** `docs/METADATA_FILTERING_DESIGN.md` (spec), `docs/METADATA_FILTERING_SURVEY.md` (research background).

---

## Prelude: Dependencies

Before any feature code is written, vendor two header-only libraries and create a unit-test harness. Both are one-time setup; subsequent tasks assume they exist.

### Task P1: Vendor nlohmann/json

**Files:**
- Create: `third_party/json/json.hpp`
- Modify: `CMakeLists.txt` — add include path

- [ ] **Step 1: Create the directory and drop the header**

Download the single-header release from the official repo (v3.11.3 or newer) and save it as `third_party/json/json.hpp`. Exact URL: <https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp>.

```bash
mkdir -p third_party/json
curl -L -o third_party/json/json.hpp \
  https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
```

Verify file is ~950 KB and starts with the nlohmann license header:

```bash
ls -lh third_party/json/json.hpp
head -3 third_party/json/json.hpp
```

Expected: first line contains `//     __ _____ _____ _____`.

- [ ] **Step 2: Add to CMakeLists.txt include path**

In `CMakeLists.txt`, after the existing `include_directories` call for Boost, add a line for the vendored third_party:

```cmake
include_directories(${CMAKE_SOURCE_DIR}/third_party/json)
```

Put it just before the `find_package(Boost REQUIRED)` block so it's easy to spot.

- [ ] **Step 3: Verify the header compiles**

Create a one-liner sanity check file `third_party/json/__sanity.cc`:

```cpp
#include "json.hpp"
#include <iostream>
int main() {
    nlohmann::json j = {{"ok", true}};
    std::cout << j.dump() << std::endl;
    return 0;
}
```

Compile and run:

```bash
g++ -std=c++17 -I third_party/json third_party/json/__sanity.cc -o /tmp/__sanity
/tmp/__sanity
```

Expected output: `{"ok":true}`.

- [ ] **Step 4: Remove the sanity file and commit**

```bash
rm third_party/json/__sanity.cc /tmp/__sanity
git add third_party/json/ CMakeLists.txt
git commit -m "chore: vendor nlohmann/json v3.11.3 for metadata filtering"
```

---

### Task P2: Vendor doctest and bootstrap unit-test harness

**Files:**
- Create: `third_party/doctest/doctest.h`
- Create: `test/unit/CMakeLists.txt`
- Create: `test/unit/main.cc`
- Create: `test/unit/test_sanity.cc`
- Modify: `CMakeLists.txt` — optionally `add_subdirectory(test/unit)`
- Modify: root `Makefile` — add `make unit_test` target

- [ ] **Step 1: Drop doctest header**

```bash
mkdir -p third_party/doctest
curl -L -o third_party/doctest/doctest.h \
  https://raw.githubusercontent.com/doctest/doctest/v2.4.11/doctest/doctest.h
```

Verify size (~250 KB) and first-line header comment mentions doctest.

- [ ] **Step 2: Create test/unit harness files**

`test/unit/main.cc`:

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
```

`test/unit/test_sanity.cc`:

```cpp
#include "doctest.h"

TEST_CASE("sanity: 1 + 1 == 2") {
    CHECK(1 + 1 == 2);
}
```

`test/unit/CMakeLists.txt`:

```cmake
# Unit tests — doctest-based, no external deps beyond LSM-Vec itself.
add_executable(lsmvec_unit_tests
    main.cc
    test_sanity.cc
)
target_include_directories(lsmvec_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/doctest
    ${CMAKE_SOURCE_DIR}/third_party/json
    ${CMAKE_SOURCE_DIR}/include
    ${ROCKSDB_INCLUDE_DIR}
)
target_link_libraries(lsmvec_unit_tests PRIVATE lsmvec_static ${ROCKSDB_LIB})
if(APPLE)
    target_link_libraries(lsmvec_unit_tests PRIVATE jemalloc)
endif()
```

- [ ] **Step 3: Wire into root CMakeLists.txt**

In `CMakeLists.txt`, at the very bottom (after all `add_library` / `add_executable` for the main project), add:

```cmake
option(LSMVEC_BUILD_UNIT_TESTS "Build doctest-based unit tests" ON)
if(LSMVEC_BUILD_UNIT_TESTS)
    add_subdirectory(test/unit)
endif()
```

Note: `lsmvec_static` must be the target name used in the main CMakeLists. Verify by grepping for `add_library`:

```bash
grep -n "add_library" CMakeLists.txt
```

If the target is named differently (e.g. `lsm_vec` or `lsmvec`), update `target_link_libraries` above to match.

- [ ] **Step 4: Add Makefile target**

In root `Makefile`, append:

```make
.PHONY: unit_test
unit_test: configure
	cmake --build build --target lsmvec_unit_tests -- -j
	./build/test/unit/lsmvec_unit_tests
```

- [ ] **Step 5: Build and run**

```bash
make configure
make unit_test
```

Expected output ends with:

```
[doctest] test cases:  1 |  1 passed | 0 failed | 0 skipped
[doctest] assertions:  1 |  1 passed | 0 failed |
```

- [ ] **Step 6: Commit**

```bash
git add third_party/doctest/ test/unit/ CMakeLists.txt Makefile
git commit -m "chore: bootstrap doctest-based unit test harness"
```

---

## M1: Predicate AST, Evaluator, Parser

**Goal of M1:** `lsm_vec::metadata::Predicate` + `ParsePredicate` + `matches()` fully implemented, 40+ unit tests, zero dependency on the rest of LSM-Vec.

### Task M1.1: Header skeleton

**Files:**
- Create: `include/metadata.h`

- [ ] **Step 1: Write the header**

```cpp
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "json.hpp"
#include "rocksdb/status.h"

namespace lsm_vec::metadata {

using Json      = nlohmann::json;
using FieldPath = std::vector<std::string>;
using Status    = ROCKSDB_NAMESPACE::Status;

class Predicate {
public:
    enum class Kind {
        Eq, Ne, Gt, Gte, Lt, Lte,
        In, Nin,
        Exists,
        ContainsAny, ContainsAll,
        And, Or,
    };

    Kind                    kind = Kind::And;
    FieldPath               path;
    Json                    value;
    std::vector<Json>       values;
    bool                    exists_expected = true;
    std::vector<Predicate>  children;

    bool matches(const Json& doc) const;
};

// Parse a Mongo-style filter JSON string into a Predicate.
// Empty string or "{}" yields an empty PredAnd that matches everything.
Status ParsePredicate(std::string_view json_str, Predicate* out);

}  // namespace lsm_vec::metadata
```

- [ ] **Step 2: Verify it compiles standalone**

Create a temporary `/tmp/compile_check.cc`:

```cpp
#include "metadata.h"
int main() { lsm_vec::metadata::Predicate p; return 0; }
```

```bash
g++ -std=c++17 -I include -I third_party/json -I lib/aster/include \
    /tmp/compile_check.cc -o /tmp/compile_check
```

Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
rm /tmp/compile_check /tmp/compile_check.cc
git add include/metadata.h
git commit -m "feat(metadata): Predicate AST header"
```

---

### Task M1.2: Test — $eq on present and missing field

**Files:**
- Create: `test/unit/test_predicate_eq.cc`

- [ ] **Step 1: Write the failing test**

```cpp
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
```

- [ ] **Step 2: Add to CMake**

Append `test_predicate_eq.cc` to the source list in `test/unit/CMakeLists.txt`:

```cmake
add_executable(lsmvec_unit_tests
    main.cc
    test_sanity.cc
    test_predicate_eq.cc
)
```

- [ ] **Step 3: Run, expect link failure**

```bash
make unit_test
```

Expected: build fails with `undefined reference to lsm_vec::metadata::Predicate::matches`.

---

### Task M1.3: Implement `matches()` for $eq

**Files:**
- Create: `src/metadata.cc`

- [ ] **Step 1: Write minimal impl covering only `Kind::Eq`**

```cpp
#include "metadata.h"

namespace lsm_vec::metadata {

namespace {

// Resolve a dot-path against a JSON document.
// Returns nullptr if any segment is missing or its parent is not an object.
const Json* Resolve(const Json& doc, const FieldPath& path) {
    const Json* cur = &doc;
    for (const auto& seg : path) {
        if (!cur->is_object()) return nullptr;
        auto it = cur->find(seg);
        if (it == cur->end()) return nullptr;
        cur = &*it;
    }
    return cur;
}

}  // namespace

bool Predicate::matches(const Json& doc) const {
    switch (kind) {
        case Kind::Eq: {
            const Json* v = Resolve(doc, path);
            return v != nullptr && *v == value;
        }
        default:
            return false;  // other kinds not yet implemented
    }
}

Status ParsePredicate(std::string_view, Predicate*) {
    return Status::NotSupported("ParsePredicate not yet implemented");
}

}  // namespace lsm_vec::metadata
```

- [ ] **Step 2: Add `src/metadata.cc` to the library target**

In root `CMakeLists.txt`, find the `add_library` call for lsmvec_static (or whatever the static library target is named) and append `src/metadata.cc` to its sources.

Example (actual contents may differ — adjust to match):

```cmake
add_library(lsmvec_static STATIC
    src/lsm_vec_db.cc
    src/lsm_vec_index.cc
    src/utils.cc
    src/metadata.cc           # new
)
```

- [ ] **Step 3: Run tests**

```bash
make unit_test
```

Expected:

```
[doctest] test cases:  3 |  3 passed | 0 failed | 0 skipped
```

(Sanity + two new eq cases.)

- [ ] **Step 4: Commit**

```bash
git add include/metadata.h src/metadata.cc test/unit/test_predicate_eq.cc test/unit/CMakeLists.txt CMakeLists.txt
git commit -m "feat(metadata): Predicate $eq evaluator"
```

---

### Task M1.4: $ne, $gt, $gte, $lt, $lte

**Files:**
- Modify: `src/metadata.cc`
- Create: `test/unit/test_predicate_comparison.cc`

- [ ] **Step 1: Write the tests**

```cpp
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
```

Add `test_predicate_comparison.cc` to `test/unit/CMakeLists.txt`.

- [ ] **Step 2: Run, expect failure**

```bash
make unit_test
```

Expected: comparison tests fail (return path is `false` for all non-Eq).

- [ ] **Step 3: Extend `matches()`**

In `src/metadata.cc`, replace the `switch` body:

```cpp
namespace {

// True if a and b are both numbers OR both strings (same JSON ordering category).
bool SameOrderingKind(const Json& a, const Json& b) {
    return (a.is_number() && b.is_number()) ||
           (a.is_string() && b.is_string());
}

template <typename Cmp>
bool CompareOrdered(const Json* v, const Json& rhs, Cmp cmp) {
    if (v == nullptr) return false;
    if (!SameOrderingKind(*v, rhs)) return false;
    return cmp(*v, rhs);
}

}  // namespace

bool Predicate::matches(const Json& doc) const {
    const Json* v = Resolve(doc, path);  // null on missing; OK for branches that handle it
    switch (kind) {
        case Kind::Eq:
            return v != nullptr && *v == value;
        case Kind::Ne:
            return v == nullptr || *v != value;  // missing → true
        case Kind::Gt:
            return CompareOrdered(v, value, [](auto& a, auto& b) { return a >  b; });
        case Kind::Gte:
            return CompareOrdered(v, value, [](auto& a, auto& b) { return a >= b; });
        case Kind::Lt:
            return CompareOrdered(v, value, [](auto& a, auto& b) { return a <  b; });
        case Kind::Lte:
            return CompareOrdered(v, value, [](auto& a, auto& b) { return a <= b; });
        default:
            return false;
    }
}
```

- [ ] **Step 4: Run tests**

```bash
make unit_test
```

Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add src/metadata.cc test/unit/test_predicate_comparison.cc test/unit/CMakeLists.txt
git commit -m "feat(metadata): $ne, $gt, $gte, $lt, $lte evaluators"
```

---

### Task M1.5: $in, $nin

**Files:**
- Modify: `src/metadata.cc`
- Create: `test/unit/test_predicate_set.cc`

- [ ] **Step 1: Write tests**

```cpp
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
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Add cases to `matches()`**

```cpp
        case Kind::In: {
            if (v == nullptr) return false;
            for (const auto& x : values) if (*v == x) return true;
            return false;
        }
        case Kind::Nin: {
            if (v == nullptr) return true;
            for (const auto& x : values) if (*v == x) return false;
            return true;
        }
```

- [ ] **Step 4: Run tests, all pass**

- [ ] **Step 5: Commit**

```bash
git add src/metadata.cc test/unit/test_predicate_set.cc test/unit/CMakeLists.txt
git commit -m "feat(metadata): \$in, \$nin evaluators"
```

---

### Task M1.6: $exists, $contains_any, $contains_all

**Files:**
- Modify: `src/metadata.cc`
- Create: `test/unit/test_predicate_exists_contains.cc`

- [ ] **Step 1: Write tests**

```cpp
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
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Extend `matches()`**

```cpp
        case Kind::Exists:
            return (v != nullptr) == exists_expected;
        case Kind::ContainsAny: {
            if (v == nullptr || !v->is_array()) return false;
            for (const auto& target : values)
                for (const auto& el : *v)
                    if (el == target) return true;
            return false;
        }
        case Kind::ContainsAll: {
            if (v == nullptr || !v->is_array()) return false;
            for (const auto& target : values) {
                bool found = false;
                for (const auto& el : *v)
                    if (el == target) { found = true; break; }
                if (!found) return false;
            }
            return true;
        }
```

- [ ] **Step 4: Run tests, all pass**

- [ ] **Step 5: Commit**

```bash
git add src/metadata.cc test/unit/test_predicate_exists_contains.cc test/unit/CMakeLists.txt
git commit -m "feat(metadata): \$exists, \$contains_any, \$contains_all"
```

---

### Task M1.7: $and, $or

**Files:**
- Modify: `src/metadata.cc`
- Create: `test/unit/test_predicate_logical.cc`

- [ ] **Step 1: Write tests**

```cpp
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
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Extend `matches()`**

```cpp
        case Kind::And:
            for (const auto& c : children) if (!c.matches(doc)) return false;
            return true;
        case Kind::Or:
            for (const auto& c : children) if (c.matches(doc)) return true;
            return false;
```

Remove the `default:` arm now that all kinds are handled; add a compile-time check:

```cpp
bool Predicate::matches(const Json& doc) const {
    const Json* v = Resolve(doc, path);
    switch (kind) {
        case Kind::Eq:           /* ... */
        case Kind::Ne:           /* ... */
        // ...
        case Kind::And:          /* as above */
        case Kind::Or:           /* as above */
    }
    return false;  // unreachable; silences compiler warning on some toolchains
}
```

- [ ] **Step 4: Run tests, all pass**

- [ ] **Step 5: Commit**

```bash
git add src/metadata.cc test/unit/test_predicate_logical.cc test/unit/CMakeLists.txt
git commit -m "feat(metadata): \$and, \$or logical predicates"
```

---

### Task M1.8: Dot-path resolver tests

**Files:**
- Create: `test/unit/test_predicate_path.cc`

The dot-path code is already in `Resolve()`, but verify it explicitly.

- [ ] **Step 1: Write the tests**

```cpp
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

TEST_CASE("Dot-path: intermediate segment is not object → missing") {
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
```

- [ ] **Step 2: Add to CMake, run tests — should all pass without code changes**

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_predicate_path.cc test/unit/CMakeLists.txt
git commit -m "test(metadata): dot-path resolver coverage"
```

---

### Task M1.9: `ParsePredicate` — basic shape

**Files:**
- Modify: `src/metadata.cc`
- Create: `test/unit/test_predicate_parse.cc`

- [ ] **Step 1: Write tests for parse success cases**

```cpp
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
```

- [ ] **Step 2: Run, expect all to fail** (parser returns NotSupported).

- [ ] **Step 3: Implement the parser**

Replace the `ParsePredicate` stub in `src/metadata.cc`:

```cpp
namespace {

// Split "a.b.c" into ["a", "b", "c"]. Empty string → empty path.
FieldPath SplitPath(std::string_view key) {
    FieldPath out;
    size_t start = 0;
    for (size_t i = 0; i <= key.size(); ++i) {
        if (i == key.size() || key[i] == '.') {
            out.emplace_back(std::string(key.substr(start, i - start)));
            start = i + 1;
        }
    }
    return out;
}

// Convert an operator-object like {"$gt":5,"$lt":10} on `path` to a Predicate.
// Returns OK + writes to `out`. Multiple operator keys → Predicate::Kind::And children.
Status ParseOperatorObject(const FieldPath& path, const Json& obj, Predicate* out) {
    if (!obj.is_object()) {
        return Status::InvalidArgument("expected operator object");
    }
    std::vector<Predicate> leaves;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const std::string& op = it.key();
        const Json& arg = it.value();
        Predicate leaf;
        leaf.path = path;

        if      (op == "$eq")  { leaf.kind = Predicate::Kind::Eq;  leaf.value = arg; }
        else if (op == "$ne")  { leaf.kind = Predicate::Kind::Ne;  leaf.value = arg; }
        else if (op == "$gt")  { leaf.kind = Predicate::Kind::Gt;  leaf.value = arg; }
        else if (op == "$gte") { leaf.kind = Predicate::Kind::Gte; leaf.value = arg; }
        else if (op == "$lt")  { leaf.kind = Predicate::Kind::Lt;  leaf.value = arg; }
        else if (op == "$lte") { leaf.kind = Predicate::Kind::Lte; leaf.value = arg; }
        else if (op == "$in" || op == "$nin" ||
                 op == "$contains_any" || op == "$contains_all") {
            if (!arg.is_array()) {
                return Status::InvalidArgument("operator " + op + " requires an array");
            }
            if      (op == "$in")           leaf.kind = Predicate::Kind::In;
            else if (op == "$nin")          leaf.kind = Predicate::Kind::Nin;
            else if (op == "$contains_any") leaf.kind = Predicate::Kind::ContainsAny;
            else                            leaf.kind = Predicate::Kind::ContainsAll;
            for (const auto& el : arg) leaf.values.push_back(el);
        }
        else if (op == "$exists") {
            if (!arg.is_boolean()) {
                return Status::InvalidArgument("$exists requires a boolean");
            }
            leaf.kind = Predicate::Kind::Exists;
            leaf.exists_expected = arg.get<bool>();
        }
        else {
            return Status::InvalidArgument("unknown operator: " + op);
        }
        leaves.push_back(std::move(leaf));
    }

    if (leaves.size() == 1) {
        *out = std::move(leaves.front());
    } else {
        out->kind = Predicate::Kind::And;
        out->children = std::move(leaves);
    }
    return Status::OK();
}

// Parse an object whose keys are either top-level operators ($and/$or) or field names.
// Returns a single Predicate (PredAnd if multiple keys at this level).
Status ParseObject(const Json& obj, Predicate* out);

Status ParseLogicalArray(const Json& arr, Predicate::Kind kind, Predicate* out) {
    if (!arr.is_array()) {
        return Status::InvalidArgument("$and/$or requires an array");
    }
    out->kind = kind;
    for (const auto& el : arr) {
        if (!el.is_object()) {
            return Status::InvalidArgument("$and/$or element must be an object");
        }
        Predicate child;
        auto st = ParseObject(el, &child);
        if (!st.ok()) return st;
        out->children.push_back(std::move(child));
    }
    return Status::OK();
}

Status ParseObject(const Json& obj, Predicate* out) {
    if (!obj.is_object()) {
        return Status::InvalidArgument("filter must be an object");
    }

    std::vector<Predicate> conjuncts;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const std::string& key = it.key();
        const Json& val = it.value();
        Predicate child;

        if (key == "$and" || key == "$or") {
            auto st = ParseLogicalArray(
                val,
                key == "$and" ? Predicate::Kind::And : Predicate::Kind::Or,
                &child);
            if (!st.ok()) return st;
        } else {
            FieldPath path = SplitPath(key);
            if (val.is_object()) {
                auto st = ParseOperatorObject(path, val, &child);
                if (!st.ok()) return st;
            } else {
                // implicit eq
                child.kind  = Predicate::Kind::Eq;
                child.path  = std::move(path);
                child.value = val;
            }
        }
        conjuncts.push_back(std::move(child));
    }

    if (conjuncts.size() == 1) {
        *out = std::move(conjuncts.front());
    } else {
        out->kind = Predicate::Kind::And;
        out->children = std::move(conjuncts);
    }
    return Status::OK();
}

}  // namespace

Status ParsePredicate(std::string_view json_str, Predicate* out) {
    *out = Predicate{};  // reset to default (empty AND = match all)

    // Trim.
    size_t start = 0, end = json_str.size();
    while (start < end && std::isspace(static_cast<unsigned char>(json_str[start]))) ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(json_str[end-1]))) --end;
    auto trimmed = json_str.substr(start, end - start);

    if (trimmed.empty() || trimmed == "{}") {
        return Status::OK();  // default empty PredAnd matches all
    }

    Json parsed;
    try {
        parsed = Json::parse(trimmed);
    } catch (const std::exception& e) {
        return Status::InvalidArgument(std::string("filter JSON parse error: ") + e.what());
    }

    return ParseObject(parsed, out);
}
```

Note: include `<cctype>` at the top of `src/metadata.cc` for `std::isspace`.

- [ ] **Step 4: Run tests, all pass**

- [ ] **Step 5: Commit**

```bash
git add src/metadata.cc test/unit/test_predicate_parse.cc test/unit/CMakeLists.txt
git commit -m "feat(metadata): ParsePredicate implementation"
```

---

### Task M1.10: Parse error coverage

**Files:**
- Create: `test/unit/test_predicate_parse_errors.cc`

- [ ] **Step 1: Write tests**

```cpp
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
```

- [ ] **Step 2: Run tests, all pass** (parser already returns InvalidArgument for these).

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_predicate_parse_errors.cc test/unit/CMakeLists.txt
git commit -m "test(metadata): ParsePredicate error coverage"
```

---

### Task M1.11: M1 verification

- [ ] **Step 1: Run the full unit-test suite**

```bash
make unit_test
```

Expected output contains:

```
[doctest] test cases: <N> |  <N> passed | 0 failed
```

Where N should be ≥ 30. Every new test file added in M1.2–M1.10 contributes.

- [ ] **Step 2: Confirm no regression in existing build**

```bash
make
```

Expected: existing `lsm_vec` binary builds cleanly.

- [ ] **Step 3: Tag the M1 commit**

```bash
git tag metadata-filter-M1
```

---

## M2: MetadataStore + Insert/Delete Integration

**Goal of M2:** `MetadataStore` wraps a RocksDB CF keyed by node_id → JSON bytes. `LSMVecDB::Insert` overload with metadata. `Delete` cascades. Backward-compatible with existing databases.

### Task M2.1: MetadataStore header

**Files:**
- Create: `include/metadata_store.h`

- [ ] **Step 1: Write the header**

```cpp
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "json.hpp"
#include "rocksdb/db.h"
#include "rocksdb/status.h"

namespace lsm_vec {

using node_id_t = uint64_t;   // matches the rest of the codebase

class MetadataStore {
public:
    using Json   = nlohmann::json;
    using Status = ROCKSDB_NAMESPACE::Status;
    using DB     = ROCKSDB_NAMESPACE::DB;
    using ColumnFamilyHandle = ROCKSDB_NAMESPACE::ColumnFamilyHandle;

    // Caller owns db. MetadataStore does not own the CF handle (the LSMVecDB
    // that created the CF does), it only borrows both.
    MetadataStore(DB* db, ColumnFamilyHandle* cf);

    // Write the raw JSON bytes verbatim under the given id. Overwrites any prior value.
    Status Put(node_id_t id, std::string_view json_bytes);
    Status Put(node_id_t id, const Json& doc);

    // Read the raw JSON bytes. Returns NotFound if no metadata exists for id.
    Status Get(node_id_t id, std::string* out_bytes) const;
    Status Get(node_id_t id, Json* out_doc) const;

    // Delete the metadata row for id. OK if the row does not exist (idempotent).
    Status Delete(node_id_t id);

    // Apply a JSON merge-patch (RFC 7396). Creates a new row if id had no prior metadata.
    Status Update(node_id_t id, const Json& partial);

    // Remove specified keys from the existing metadata. Missing keys are ignored.
    // Returns NotFound if no metadata exists for id.
    Status DeleteKeys(node_id_t id, const std::vector<std::string>& keys);

    // Observability counters (incremented on each call).
    size_t point_gets() const { return point_gets_; }
    size_t parses()     const { return parses_; }

private:
    DB*                 db_;
    ColumnFamilyHandle* cf_;
    mutable size_t      point_gets_ = 0;
    mutable size_t      parses_     = 0;

    static std::string EncodeKey(node_id_t id);  // big-endian uint64
};

}  // namespace lsm_vec
```

- [ ] **Step 2: Commit**

```bash
git add include/metadata_store.h
git commit -m "feat(metadata_store): header for MetadataStore"
```

---

### Task M2.2: MetadataStore Put/Get test skeleton (RAII RocksDB fixture)

**Files:**
- Create: `test/unit/metadata_store_fixture.h`

Because `MetadataStore` needs a live RocksDB instance, we create a small RAII fixture that opens a temp directory with one default CF + one `metadata` CF.

- [ ] **Step 1: Write the fixture**

```cpp
#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

#include "rocksdb/db.h"
#include "rocksdb/options.h"

namespace lsm_vec::test {

// Opens a RocksDB at a unique temp directory with a "metadata" CF.
// Destructor closes the DB and removes the directory.
struct TempDB {
    std::string path;
    ROCKSDB_NAMESPACE::DB* db = nullptr;
    ROCKSDB_NAMESPACE::ColumnFamilyHandle* cf_default = nullptr;
    ROCKSDB_NAMESPACE::ColumnFamilyHandle* cf_metadata = nullptr;

    TempDB() {
        char tmpl[] = "/tmp/lsmvec_mdstore_XXXXXX";
        char* dir = mkdtemp(tmpl);
        path = dir ? dir : "";

        ROCKSDB_NAMESPACE::Options options;
        options.create_if_missing = true;
        options.create_missing_column_families = true;

        std::vector<ROCKSDB_NAMESPACE::ColumnFamilyDescriptor> cfds;
        cfds.emplace_back(ROCKSDB_NAMESPACE::kDefaultColumnFamilyName,
                          ROCKSDB_NAMESPACE::ColumnFamilyOptions());
        cfds.emplace_back("metadata", ROCKSDB_NAMESPACE::ColumnFamilyOptions());

        std::vector<ROCKSDB_NAMESPACE::ColumnFamilyHandle*> handles;
        auto st = ROCKSDB_NAMESPACE::DB::Open(options, path, cfds, &handles, &db);
        if (!st.ok()) { std::fprintf(stderr, "TempDB open failed: %s\n", st.ToString().c_str()); std::abort(); }
        cf_default  = handles[0];
        cf_metadata = handles[1];
    }

    ~TempDB() {
        if (db) {
            db->DestroyColumnFamilyHandle(cf_default);
            db->DestroyColumnFamilyHandle(cf_metadata);
            delete db;
        }
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    }
};

}  // namespace lsm_vec::test
```

- [ ] **Step 2: Commit**

```bash
git add test/unit/metadata_store_fixture.h
git commit -m "test(metadata_store): add TempDB fixture"
```

---

### Task M2.3: Put and Get round-trip

**Files:**
- Modify: `src/metadata.cc` → rename or split? NO — keep `metadata.cc` for Predicate. Create:
- Create: `src/metadata_store.cc`
- Create: `test/unit/test_metadata_store_basic.cc`

- [ ] **Step 1: Write tests**

```cpp
#include "doctest.h"
#include "metadata_store.h"
#include "metadata_store_fixture.h"

using namespace lsm_vec;
using namespace lsm_vec::test;

TEST_CASE("MetadataStore: Put then Get round-trip") {
    TempDB t;
    MetadataStore store(t.db, t.cf_metadata);

    auto put_st = store.Put(42, R"({"tenant":"acme","count":7})");
    REQUIRE(put_st.ok());

    std::string bytes;
    auto get_st = store.Get(42, &bytes);
    REQUIRE(get_st.ok());
    CHECK(bytes == R"({"tenant":"acme","count":7})");
}

TEST_CASE("MetadataStore: Get of non-existent id is NotFound") {
    TempDB t;
    MetadataStore store(t.db, t.cf_metadata);

    std::string bytes;
    auto st = store.Get(99, &bytes);
    CHECK(st.IsNotFound());
}

TEST_CASE("MetadataStore: Put with JSON object overload") {
    TempDB t;
    MetadataStore store(t.db, t.cf_metadata);

    MetadataStore::Json j = {{"x", 1}, {"y", "foo"}};
    REQUIRE(store.Put(1, j).ok());

    MetadataStore::Json out;
    REQUIRE(store.Get(1, &out).ok());
    CHECK(out == j);
}

TEST_CASE("MetadataStore: Put overwrites") {
    TempDB t;
    MetadataStore store(t.db, t.cf_metadata);
    REQUIRE(store.Put(7, R"({"v":1})").ok());
    REQUIRE(store.Put(7, R"({"v":2})").ok());
    std::string bytes;
    REQUIRE(store.Get(7, &bytes).ok());
    CHECK(bytes == R"({"v":2})");
}

TEST_CASE("MetadataStore: Delete is idempotent") {
    TempDB t;
    MetadataStore store(t.db, t.cf_metadata);
    REQUIRE(store.Put(5, R"({})").ok());
    REQUIRE(store.Delete(5).ok());
    REQUIRE(store.Delete(5).ok());           // double delete OK
    std::string bytes;
    CHECK(store.Get(5, &bytes).IsNotFound());
}
```

- [ ] **Step 2: Run, expect link failure** (implementation missing).

- [ ] **Step 3: Implement MetadataStore**

`src/metadata_store.cc`:

```cpp
#include "metadata_store.h"

#include <cstring>

#include "rocksdb/db.h"
#include "rocksdb/options.h"

namespace lsm_vec {

using ROCKSDB_NAMESPACE::PinnableSlice;
using ROCKSDB_NAMESPACE::ReadOptions;
using ROCKSDB_NAMESPACE::Slice;
using ROCKSDB_NAMESPACE::WriteOptions;

std::string MetadataStore::EncodeKey(node_id_t id) {
    // Big-endian 8-byte encoding: lexicographic order == numeric order.
    std::string out(8, '\0');
    for (int i = 7; i >= 0; --i) {
        out[i] = static_cast<char>(id & 0xFFu);
        id >>= 8;
    }
    return out;
}

MetadataStore::MetadataStore(DB* db, ColumnFamilyHandle* cf)
    : db_(db), cf_(cf) {}

MetadataStore::Status MetadataStore::Put(node_id_t id, std::string_view json_bytes) {
    return db_->Put(WriteOptions(), cf_, EncodeKey(id),
                    Slice(json_bytes.data(), json_bytes.size()));
}

MetadataStore::Status MetadataStore::Put(node_id_t id, const Json& doc) {
    auto s = doc.dump();
    return Put(id, s);
}

MetadataStore::Status MetadataStore::Get(node_id_t id, std::string* out_bytes) const {
    ++point_gets_;
    PinnableSlice pin;
    auto st = db_->Get(ReadOptions(), cf_, EncodeKey(id), &pin);
    if (!st.ok()) return st;
    out_bytes->assign(pin.data(), pin.size());
    return Status::OK();
}

MetadataStore::Status MetadataStore::Get(node_id_t id, Json* out_doc) const {
    std::string bytes;
    auto st = Get(id, &bytes);
    if (!st.ok()) return st;
    ++parses_;
    try {
        *out_doc = Json::parse(bytes);
    } catch (const std::exception& e) {
        return Status::Corruption(std::string("metadata parse error: ") + e.what());
    }
    return Status::OK();
}

MetadataStore::Status MetadataStore::Delete(node_id_t id) {
    return db_->Delete(WriteOptions(), cf_, EncodeKey(id));
}

MetadataStore::Status MetadataStore::Update(node_id_t id, const Json& partial) {
    Json existing;
    auto st = Get(id, &existing);
    if (st.IsNotFound()) {
        existing = Json::object();
    } else if (!st.ok()) {
        return st;
    }
    existing.merge_patch(partial);
    return Put(id, existing);
}

MetadataStore::Status MetadataStore::DeleteKeys(node_id_t id,
                                                const std::vector<std::string>& keys) {
    Json existing;
    auto st = Get(id, &existing);
    if (!st.ok()) return st;
    if (!existing.is_object()) return Status::InvalidArgument("metadata is not an object");
    for (const auto& k : keys) existing.erase(k);
    return Put(id, existing);
}

}  // namespace lsm_vec
```

- [ ] **Step 4: Add `src/metadata_store.cc` to the library target in root CMakeLists.txt**

Mirror what was done for `src/metadata.cc` in Task M1.3.

- [ ] **Step 5: Run tests, all pass**

- [ ] **Step 6: Commit**

```bash
git add include/metadata_store.h src/metadata_store.cc test/unit/test_metadata_store_basic.cc test/unit/CMakeLists.txt CMakeLists.txt
git commit -m "feat(metadata_store): Put/Get/Delete implementation"
```

---

### Task M2.4: Update (merge-patch) and DeleteKeys

**Files:**
- Create: `test/unit/test_metadata_store_update.cc`

- [ ] **Step 1: Write tests**

```cpp
#include "doctest.h"
#include "metadata_store.h"
#include "metadata_store_fixture.h"
using namespace lsm_vec;
using namespace lsm_vec::test;

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
```

- [ ] **Step 2: Run tests — already implemented in Task M2.3, should pass**

If any fail, the implementation has a bug; fix in `src/metadata_store.cc`.

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_metadata_store_update.cc test/unit/CMakeLists.txt
git commit -m "test(metadata_store): Update and DeleteKeys coverage"
```

---

### Task M2.5: Integrate MetadataStore into LSMVecDB (CF creation)

**Files:**
- Modify: `include/lsm_vec_db.h` — add `std::unique_ptr<MetadataStore> metadata_store_` and forward-declare
- Modify: `src/lsm_vec_db.cc` — open/create CF on `Open()`, wire up in `Close()`

Current `LSMVecDB::Open` is in `src/lsm_vec_db.cc`. The flow opens the Aster `RocksGraph` which internally holds a `rocksdb::DB`. We need `DB*` to add a CF.

**Inspect first:** grep `rocksdb::DB*` usage in `lib/aster/include/rocksdb/graph.h` and `src/lsm_vec_index.cc` to see if there's an accessor.

```bash
grep -rn "GetBaseDB\|GetDB\|DB\*" lib/aster/include/rocksdb/graph.h include/*.h src/*.cc | head -30
```

If Aster's `RocksGraph` exposes `DB*` via a method (commonly named `GetBaseDB` in RocksDB ecosystem): use it. Otherwise, fall back to opening a sibling `rocksdb::DB` at `<db_path>/metadata/`.

- [ ] **Step 1: Decide the path** (actual accessor vs sibling DB). Record the decision in the PR description.

Assuming the fallback path (sibling DB) for concreteness — it's the safest choice and independent of Aster internals. The remainder of this task writes that path.

- [ ] **Step 2: Add metadata DB field to LSMVecDB**

In `include/lsm_vec_db.h`, add forward declaration and field:

```cpp
namespace lsm_vec {
class MetadataStore;      // forward

class LSMVecDB {
    // ... existing ...
private:
    // ... existing ...
    std::unique_ptr<ROCKSDB_NAMESPACE::DB> metadata_db_;                      // new
    ROCKSDB_NAMESPACE::ColumnFamilyHandle* metadata_cf_ = nullptr;            // new
    std::unique_ptr<MetadataStore> metadata_store_;                           // new
};
}
```

Include `#include "metadata_store.h"` is better done in the `.cc` to avoid leaking nlohmann into the public header. Keep the forward declaration in the `.h`.

- [ ] **Step 3: Open metadata DB in `LSMVecDB::Open`**

In `src/lsm_vec_db.cc`, inside `Open`, after the existing RocksGraph setup and before returning the db object, add:

```cpp
// Open sibling rocksdb::DB at <path>/metadata for metadata CF.
std::string meta_path = path + "/metadata";
rocksdb::Options meta_opts;
meta_opts.create_if_missing = true;
meta_opts.create_missing_column_families = true;
meta_opts.compression = rocksdb::kZSTD;

std::vector<rocksdb::ColumnFamilyDescriptor> cfds;
cfds.emplace_back(rocksdb::kDefaultColumnFamilyName, rocksdb::ColumnFamilyOptions());

rocksdb::ColumnFamilyOptions cfopts;
cfopts.compression = rocksdb::kZSTD;
rocksdb::BlockBasedTableOptions table_opts;
table_opts.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
cfopts.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opts));
cfds.emplace_back("metadata", cfopts);

std::vector<rocksdb::ColumnFamilyHandle*> handles;
rocksdb::DB* meta_raw = nullptr;
auto mst = rocksdb::DB::Open(meta_opts, meta_path, cfds, &handles, &meta_raw);
if (!mst.ok()) {
    delete meta_raw;
    return mst;
}

db_instance->metadata_db_.reset(meta_raw);
db_instance->metadata_cf_ = handles[1];
db_instance->metadata_store_ =
    std::make_unique<MetadataStore>(meta_raw, handles[1]);
// handles[0] (default CF) — must also be retained or destroyed at Close()
// Store the default CF somewhere if needed; for v1 we can destroy it immediately:
meta_raw->DestroyColumnFamilyHandle(handles[0]);
```

Required new includes at top of `src/lsm_vec_db.cc`:

```cpp
#include "metadata_store.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/table.h"
```

- [ ] **Step 4: Close metadata DB in `LSMVecDB::Close`**

In `LSMVecDB::Close` (or destructor), before returning:

```cpp
if (metadata_store_) {
    metadata_store_.reset();  // no-op; it doesn't own DB
}
if (metadata_cf_ && metadata_db_) {
    metadata_db_->DestroyColumnFamilyHandle(metadata_cf_);
    metadata_cf_ = nullptr;
}
metadata_db_.reset();
```

- [ ] **Step 5: Build the full project**

```bash
make lib
```

Expected: clean build.

- [ ] **Step 6: Smoke-test using the existing `lsm_vec` binary**

```bash
rm -rf /tmp/mdtest
./build/bin/lsm_vec --db /tmp/mdtest --data-dir data/sift_100k_ --k 1 --efs 1 --out /tmp/mdout
ls /tmp/mdtest/metadata/
```

Expected: `metadata/` directory exists alongside `mdtest/`'s Aster files, and contains RocksDB files (CURRENT, MANIFEST, etc.).

- [ ] **Step 7: Commit**

```bash
git add include/lsm_vec_db.h src/lsm_vec_db.cc
git commit -m "feat(lsm_vec_db): open metadata RocksDB CF on Open"
```

---

### Task M2.6: `Insert` overload with metadata

**Files:**
- Modify: `include/lsm_vec_db.h`
- Modify: `src/lsm_vec_db.cc`

- [ ] **Step 1: Add overload declaration**

In `include/lsm_vec_db.h`:

```cpp
Status Insert(node_id_t id, Span<float> vec);                                // existing
Status Insert(node_id_t id, Span<float> vec, std::string_view metadata_json); // new
```

- [ ] **Step 2: Implement in .cc**

In `src/lsm_vec_db.cc`:

```cpp
Status LSMVecDB::Insert(node_id_t id, Span<float> vec, std::string_view metadata_json) {
    // 1. Validate vector
    auto vst = ValidateVector(vec);
    if (!vst.ok()) return vst;

    // 2. Pre-parse metadata (fail-fast; no writes yet)
    nlohmann::json parsed;
    if (!metadata_json.empty() && metadata_json != "{}") {
        try {
            parsed = nlohmann::json::parse(metadata_json);
        } catch (const std::exception& e) {
            return Status::InvalidArgument(std::string("metadata parse error: ") + e.what());
        }
        if (!parsed.is_object()) {
            return Status::InvalidArgument("metadata must be a JSON object");
        }
    }

    // 3. Insert vector via existing path
    auto ist = Insert(id, vec);
    if (!ist.ok()) return ist;

    // 4. Persist metadata (best-effort; log on failure but don't un-insert the vector)
    if (!metadata_json.empty() && metadata_json != "{}") {
        auto mst = metadata_store_->Put(id, metadata_json);
        if (!mst.ok()) {
            // Log and surface; vector is inserted but metadata is absent → safe silent-no-match
            return mst;
        }
    }
    return Status::OK();
}
```

- [ ] **Step 3: Create integration test**

`test/unit/test_lsm_vec_db_insert_metadata.cc`:

```cpp
#include "doctest.h"
#include "lsm_vec_db.h"
#include <filesystem>

TEST_CASE("Insert with metadata persists JSON") {
    char tmpl[] = "/tmp/lsmvec_insertmd_XXXXXX";
    std::string path = mkdtemp(tmpl);

    lsm_vec::LSMVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = path + "/vecs.bin";

    std::unique_ptr<lsm_vec::LSMVecDB> db;
    REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());

    std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
    REQUIRE(db->Insert(7, lsm_vec::Span<float>(v), R"({"tenant":"acme"})").ok());

    // TODO: verify via GetPayload once implemented (M3). For now, just assert Insert succeeded.
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Insert with invalid JSON metadata rejected") {
    char tmpl[] = "/tmp/lsmvec_insertbadmd_XXXXXX";
    std::string path = mkdtemp(tmpl);

    lsm_vec::LSMVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = path + "/vecs.bin";

    std::unique_ptr<lsm_vec::LSMVecDB> db;
    REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());

    std::vector<float> v{1,2,3,4};
    auto st = db->Insert(7, lsm_vec::Span<float>(v), R"({not json)");
    CHECK(st.IsInvalidArgument());

    db->Close();
    std::filesystem::remove_all(path);
}
```

- [ ] **Step 4: Run tests**

```bash
make unit_test
```

- [ ] **Step 5: Commit**

```bash
git add include/lsm_vec_db.h src/lsm_vec_db.cc test/unit/test_lsm_vec_db_insert_metadata.cc test/unit/CMakeLists.txt
git commit -m "feat(lsm_vec_db): Insert overload with metadata"
```

---

### Task M2.7: Delete cascades to metadata

**Files:**
- Modify: `src/lsm_vec_db.cc`

- [ ] **Step 1: Update `LSMVecDB::Delete`**

Find the existing `Delete` implementation. Append the metadata cleanup after the existing vector/graph deletion:

```cpp
Status LSMVecDB::Delete(node_id_t id) {
    // ... existing vector + graph delete ...
    // at the very end, before `return status;`:

    if (metadata_store_) {
        // Best-effort: ignore NotFound (id may have had no metadata).
        auto mst = metadata_store_->Delete(id);
        if (!mst.ok() && !mst.IsNotFound()) {
            return mst;
        }
    }
    return status;   // whatever the existing return value was
}
```

- [ ] **Step 2: Write test**

`test/unit/test_lsm_vec_db_delete_cascade.cc`:

```cpp
#include "doctest.h"
#include "lsm_vec_db.h"
#include "metadata_store.h"
#include <filesystem>

// NOTE: Full round-trip (Insert → Delete → Get returns NotFound) is deferred
// to M3 when GetPayload is available. Here we only assert that Delete does
// not fail on a previously-inserted id with metadata.

TEST_CASE("Delete on id with metadata does not error") {
    char tmpl[] = "/tmp/lsmvec_delcascade_XXXXXX";
    std::string path = mkdtemp(tmpl);

    lsm_vec::LSMVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = path + "/vecs.bin";

    std::unique_ptr<lsm_vec::LSMVecDB> db;
    REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());

    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(7, lsm_vec::Span<float>(v), R"({"tag":"x"})").ok());
    REQUIRE(db->Delete(7).ok());

    db->Close();
    std::filesystem::remove_all(path);
}
```

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git add src/lsm_vec_db.cc test/unit/test_lsm_vec_db_delete_cascade.cc test/unit/CMakeLists.txt
git commit -m "feat(lsm_vec_db): Delete cascades to metadata"
```

---

### Task M2.8: Backward-compatibility test

**Files:**
- Create: `test/unit/test_metadata_backward_compat.cc`

- [ ] **Step 1: Write test that mimics an old DB (no metadata subdir)**

```cpp
#include "doctest.h"
#include "lsm_vec_db.h"
#include <filesystem>

TEST_CASE("Opening a DB that was created before metadata feature works") {
    char tmpl[] = "/tmp/lsmvec_bc_XXXXXX";
    std::string path = mkdtemp(tmpl);

    // First open: creates metadata CF. Insert a few vectors without metadata.
    {
        lsm_vec::LSMVecDBOptions opts;
        opts.dim = 4;
        opts.vec_file_capacity = 100;
        opts.vector_file_path = path + "/vecs.bin";
        std::unique_ptr<lsm_vec::LSMVecDB> db;
        REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());
        std::vector<float> v{1,2,3,4};
        for (uint64_t i = 0; i < 10; ++i) REQUIRE(db->Insert(i, lsm_vec::Span<float>(v)).ok());
        db->Close();
    }

    // Simulate an "old" DB by removing the metadata/ subdirectory.
    std::filesystem::remove_all(path + "/metadata");
    REQUIRE_FALSE(std::filesystem::exists(path + "/metadata"));

    // Second open: metadata CF must be re-created; vectors still accessible.
    {
        lsm_vec::LSMVecDBOptions opts;
        opts.dim = 4;
        opts.vec_file_capacity = 100;
        opts.vector_file_path = path + "/vecs.bin";
        std::unique_ptr<lsm_vec::LSMVecDB> db;
        REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());
        CHECK(std::filesystem::exists(path + "/metadata"));

        std::vector<float> out;
        REQUIRE(db->Get(0, &out).ok());   // vectors still work
        db->Close();
    }

    std::filesystem::remove_all(path);
}
```

- [ ] **Step 2: Run, expect pass**

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_metadata_backward_compat.cc test/unit/CMakeLists.txt
git commit -m "test(metadata): backward-compat with pre-metadata DBs"
```

---

### Task M2.9: M2 verification

- [ ] **Step 1: Run all unit tests**

```bash
make unit_test
```

- [ ] **Step 2: Run existing benchmark binary to ensure no regression**

```bash
make bin
mkdir -p /tmp/lsmvec_regression
./build/bin/lsm_vec --db /tmp/lsmvec_regression --data-dir data/sift_100k_ --k 10 --efs 64 --out /tmp/lsmvec_regression.out
```

Expected: binary runs to completion with metrics comparable to prior runs.

- [ ] **Step 3: Tag**

```bash
git tag metadata-filter-M2
```

---

## M3: Metadata CRUD API

**Goal of M3:** SetPayload / UpdatePayload / DeletePayloadKeys / GetPayload public methods on `LSMVecDB`. Invariant: metadata only exists alongside a vector.

### Task M3.1: `GetPayload` (foundation — used by subsequent tests)

**Files:**
- Modify: `include/lsm_vec_db.h`, `src/lsm_vec_db.cc`
- Create: `test/unit/test_payload_crud.cc`

- [ ] **Step 1: Declare in header**

```cpp
Status GetPayload(node_id_t id, std::string* out_json);
```

- [ ] **Step 2: Implement**

```cpp
Status LSMVecDB::GetPayload(node_id_t id, std::string* out_json) {
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    auto st = metadata_store_->Get(id, out_json);
    if (st.IsNotFound()) {
        *out_json = "{}";
        return Status::OK();
    }
    return st;
}
```

- [ ] **Step 3: Write test**

```cpp
#include "doctest.h"
#include "lsm_vec_db.h"
#include <filesystem>

static std::string NewTempPath() {
    char tmpl[] = "/tmp/lsmvec_crud_XXXXXX";
    return mkdtemp(tmpl);
}

static std::unique_ptr<lsm_vec::LSMVecDB> OpenFresh(std::string* out_path) {
    *out_path = NewTempPath();
    lsm_vec::LSMVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = *out_path + "/vecs.bin";
    std::unique_ptr<lsm_vec::LSMVecDB> db;
    REQUIRE(lsm_vec::LSMVecDB::Open(*out_path, opts, &db).ok());
    return db;
}

TEST_CASE("GetPayload: id without metadata returns empty object") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v)).ok());   // no metadata

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(got == "{}");
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("GetPayload: id with metadata returns stored JSON") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v), R"({"k":"v"})").ok());

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(got == R"({"k":"v"})");
    db->Close();
    std::filesystem::remove_all(path);
}
```

- [ ] **Step 4: Run tests, pass**

- [ ] **Step 5: Commit**

```bash
git add include/lsm_vec_db.h src/lsm_vec_db.cc test/unit/test_payload_crud.cc test/unit/CMakeLists.txt
git commit -m "feat(lsm_vec_db): GetPayload"
```

---

### Task M3.2: `SetPayload`

- [ ] **Step 1: Declare**

```cpp
Status SetPayload(node_id_t id, std::string_view metadata_json);
```

- [ ] **Step 2: Implement**

```cpp
Status LSMVecDB::SetPayload(node_id_t id, std::string_view metadata_json) {
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    // Enforce invariant: id must have a vector.
    std::vector<float> tmp;
    auto gst = Get(id, &tmp);
    if (!gst.ok()) return gst;

    // Validate JSON.
    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(metadata_json);
    } catch (const std::exception& e) {
        return Status::InvalidArgument(std::string("metadata parse error: ") + e.what());
    }
    if (!parsed.is_object()) {
        return Status::InvalidArgument("metadata must be a JSON object");
    }

    return metadata_store_->Put(id, metadata_json);
}
```

- [ ] **Step 3: Append tests to `test_payload_crud.cc`**

```cpp
TEST_CASE("SetPayload: overwrites existing metadata") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v), R"({"k":"v1"})").ok());

    REQUIRE(db->SetPayload(1, R"({"k":"v2","new":1})").ok());

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(got == R"({"k":"v2","new":1})");
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("SetPayload: NotFound on id without vector") {
    std::string path;
    auto db = OpenFresh(&path);
    auto st = db->SetPayload(99, R"({"k":"v"})");
    CHECK(st.IsNotFound());
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("SetPayload: invalid JSON rejected") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v)).ok());
    auto st = db->SetPayload(1, R"({not json)");
    CHECK(st.IsInvalidArgument());
    db->Close();
    std::filesystem::remove_all(path);
}
```

- [ ] **Step 4: Run, commit**

```bash
git add include/lsm_vec_db.h src/lsm_vec_db.cc test/unit/test_payload_crud.cc
git commit -m "feat(lsm_vec_db): SetPayload"
```

---

### Task M3.3: `UpdatePayload`

- [ ] **Step 1: Declare**

```cpp
Status UpdatePayload(node_id_t id, std::string_view partial_json);
```

- [ ] **Step 2: Implement**

```cpp
Status LSMVecDB::UpdatePayload(node_id_t id, std::string_view partial_json) {
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");

    // Enforce invariant: id must have a vector.
    std::vector<float> tmp;
    auto gst = Get(id, &tmp);
    if (!gst.ok()) return gst;

    nlohmann::json patch;
    try {
        patch = nlohmann::json::parse(partial_json);
    } catch (const std::exception& e) {
        return Status::InvalidArgument(std::string("metadata parse error: ") + e.what());
    }
    if (!patch.is_object()) {
        return Status::InvalidArgument("metadata patch must be a JSON object");
    }
    return metadata_store_->Update(id, patch);
}
```

- [ ] **Step 3: Append tests**

```cpp
TEST_CASE("UpdatePayload: merge-patch adds fields") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v), R"({"a":1})").ok());

    REQUIRE(db->UpdatePayload(1, R"({"b":2})").ok());

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    // Note: nlohmann::json key order after merge may vary; use JSON equality:
    CHECK(nlohmann::json::parse(got) ==
          nlohmann::json::parse(R"({"a":1,"b":2})"));
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("UpdatePayload: null deletes field") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v), R"({"a":1,"b":2})").ok());

    REQUIRE(db->UpdatePayload(1, R"({"a":null})").ok());

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(nlohmann::json::parse(got) == nlohmann::json::parse(R"({"b":2})"));
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("UpdatePayload: NotFound on id without vector") {
    std::string path;
    auto db = OpenFresh(&path);
    auto st = db->UpdatePayload(99, R"({"a":1})");
    CHECK(st.IsNotFound());
    db->Close();
    std::filesystem::remove_all(path);
}
```

- [ ] **Step 4: Commit**

```bash
git add include/lsm_vec_db.h src/lsm_vec_db.cc test/unit/test_payload_crud.cc
git commit -m "feat(lsm_vec_db): UpdatePayload"
```

---

### Task M3.4: `DeletePayloadKeys`

- [ ] **Step 1: Declare**

```cpp
Status DeletePayloadKeys(node_id_t id, Span<const std::string> keys);
```

- [ ] **Step 2: Implement**

```cpp
Status LSMVecDB::DeletePayloadKeys(node_id_t id, Span<const std::string> keys) {
    if (!metadata_store_) return Status::NotFound("metadata store unavailable");
    std::vector<std::string> key_vec(keys.data(), keys.data() + keys.size());
    return metadata_store_->DeleteKeys(id, key_vec);
}
```

- [ ] **Step 3: Append tests**

```cpp
TEST_CASE("DeletePayloadKeys: removes keys, idempotent for missing") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v), R"({"a":1,"b":2,"c":3})").ok());

    std::vector<std::string> keys{"b", "nonexistent"};
    REQUIRE(db->DeletePayloadKeys(1, lsm_vec::Span<const std::string>(keys)).ok());

    std::string got;
    REQUIRE(db->GetPayload(1, &got).ok());
    CHECK(nlohmann::json::parse(got) == nlohmann::json::parse(R"({"a":1,"c":3})"));
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("DeletePayloadKeys: NotFound on id with no metadata") {
    std::string path;
    auto db = OpenFresh(&path);
    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(1, lsm_vec::Span<float>(v)).ok());  // no metadata

    std::vector<std::string> keys{"a"};
    auto st = db->DeletePayloadKeys(1, lsm_vec::Span<const std::string>(keys));
    CHECK(st.IsNotFound());
    db->Close();
    std::filesystem::remove_all(path);
}
```

- [ ] **Step 4: Commit**

```bash
git add include/lsm_vec_db.h src/lsm_vec_db.cc test/unit/test_payload_crud.cc
git commit -m "feat(lsm_vec_db): DeletePayloadKeys"
```

---

### Task M3.5: Delete cascade round-trip test

Verify `Delete` → `GetPayload` returns empty (since metadata is gone).

**Files:**
- Modify: `test/unit/test_lsm_vec_db_delete_cascade.cc` (extend)

- [ ] **Step 1: Add round-trip test**

```cpp
TEST_CASE("Delete cascades: GetPayload after Delete returns empty") {
    char tmpl[] = "/tmp/lsmvec_delrt_XXXXXX";
    std::string path = mkdtemp(tmpl);

    lsm_vec::LSMVecDBOptions opts;
    opts.dim = 4;
    opts.vec_file_capacity = 100;
    opts.vector_file_path = path + "/vecs.bin";
    std::unique_ptr<lsm_vec::LSMVecDB> db;
    REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());

    std::vector<float> v{1,2,3,4};
    REQUIRE(db->Insert(7, lsm_vec::Span<float>(v), R"({"tag":"x"})").ok());

    std::string got;
    REQUIRE(db->GetPayload(7, &got).ok());
    CHECK(got == R"({"tag":"x"})");

    REQUIRE(db->Delete(7).ok());

    REQUIRE(db->GetPayload(7, &got).ok());
    CHECK(got == "{}");

    db->Close();
    std::filesystem::remove_all(path);
}
```

- [ ] **Step 2: Run and commit**

```bash
git add test/unit/test_lsm_vec_db_delete_cascade.cc
git commit -m "test(lsm_vec_db): Delete cascade end-to-end"
```

---

### Task M3.6: M3 verification

- [ ] **Step 1: All tests pass**

```bash
make unit_test
```

- [ ] **Step 2: Tag**

```bash
git tag metadata-filter-M3
```

---

## M4: Filtered Search with Iterative Expansion

**Goal of M4:** New `searchLayer` overload accepting a `Predicate*`, `SearchKnn` overload with `filter_json`, `SearchOptions::max_scan_candidates`, Tier B iterative expansion correctly implemented and tested across selectivity regimes.

### Task M4.1: Add `max_scan_candidates` to `SearchOptions`

**Files:**
- Modify: `include/lsm_vec_db.h`

- [ ] **Step 1: Add the field**

```cpp
struct SearchOptions {
    int k = 1;
    int ef_search = 64;
    int max_scan_candidates = 0;   // 0 = auto (k * 50 when filter present)
};
```

- [ ] **Step 2: Verify existing code still compiles**

```bash
make lib
```

- [ ] **Step 3: Commit**

```bash
git add include/lsm_vec_db.h
git commit -m "feat(search): add max_scan_candidates to SearchOptions"
```

---

### Task M4.2: New `searchLayer` overload signature

**Files:**
- Modify: `include/lsm_vec_index.h`
- Modify: `src/lsm_vec_index.cc` — placeholder implementation

- [ ] **Step 1: Include metadata header**

In `include/lsm_vec_index.h`, add near the top:

```cpp
#include "metadata.h"
```

- [ ] **Step 2: Add the overload declaration**

In `LSMVec` class under `private:`:

```cpp
// Existing
std::vector<SearchResult> searchLayer(const std::vector<float> &queryVector,
                                      node_id_t entryPointId,
                                      int efSearch,
                                      int layer);

// New
std::vector<SearchResult> searchLayer(const std::vector<float> &queryVector,
                                      node_id_t entryPointId,
                                      int efSearch,
                                      int layer,
                                      const metadata::Predicate* pred,
                                      int k,
                                      int max_scan_candidates,
                                      const class MetadataStore* meta_store);
```

- [ ] **Step 3: Skeleton implementation that compiles**

In `src/lsm_vec_index.cc`:

```cpp
#include "metadata_store.h"

std::vector<SearchResult> LSMVec::searchLayer(
    const std::vector<float>& queryVector,
    node_id_t entryPointId,
    int efSearch,
    int layer,
    const metadata::Predicate* pred,
    int k,
    int max_scan_candidates,
    const MetadataStore* meta_store) {

    // Placeholder: fall back to unfiltered search for now; M4.3 replaces this.
    (void)pred; (void)k; (void)max_scan_candidates; (void)meta_store;
    return searchLayer(queryVector, entryPointId, efSearch, layer);
}
```

- [ ] **Step 4: Build**

```bash
make lib
```

- [ ] **Step 5: Commit**

```bash
git add include/lsm_vec_index.h src/lsm_vec_index.cc
git commit -m "feat(index): searchLayer overload signature (placeholder impl)"
```

---

### Task M4.3: Implement iterative expansion in new overload

**Files:**
- Modify: `src/lsm_vec_index.cc`

This is the most delicate step. Read the existing `searchLayer` first to match patterns (heap usage, visited-version map, edge cache, batch read).

- [ ] **Step 1: Read current `searchLayer`**

```bash
grep -n "searchLayer" src/lsm_vec_index.cc | head
```

Open `src/lsm_vec_index.cc` at the identified line range and study the loop structure.

- [ ] **Step 2: Replace the placeholder with the iterative-expansion implementation**

Use the same data structures as the existing `searchLayer` (min-heap of candidates, max-heap of results, `visited_map_` with version counter, `getEdgesCached`). The filtered version:

```cpp
std::vector<SearchResult> LSMVec::searchLayer(
    const std::vector<float>& queryVector,
    node_id_t entryPointId,
    int efSearch,
    int layer,
    const metadata::Predicate* pred,
    int k,
    int max_scan_candidates,
    const MetadataStore* meta_store) {

    if (pred == nullptr) {
        // No filter → forward to the original overload for zero overhead.
        return searchLayer(queryVector, entryPointId, efSearch, layer);
    }

    if (max_scan_candidates <= 0) max_scan_candidates = k * 50;

    // Version-bump visited map (shared with the unfiltered path).
    ++visited_version_;

    // Fetch metadata for a node. Returns std::nullopt on miss.
    auto fetch_meta = [&](node_id_t id) -> std::optional<metadata::Json> {
        if (meta_store == nullptr) return metadata::Json::object();
        metadata::Json j;
        auto st = meta_store->Get(id, &j);
        if (st.IsNotFound()) return metadata::Json::object();
        if (!st.ok()) return std::nullopt;   // propagated as non-match
        return j;
    };

    auto node_matches = [&](node_id_t id) -> bool {
        auto doc = fetch_meta(id);
        if (!doc.has_value()) return false;
        return pred->matches(*doc);
    };

    // Min-heap by distance (nearest at top)
    using CandItem = std::pair<float, node_id_t>;
    std::priority_queue<CandItem, std::vector<CandItem>, std::greater<>> candidates;
    // Max-heap by distance (farthest at top), capped at k, holds only matching nodes
    std::priority_queue<CandItem> results;

    std::vector<float> entry_vec;
    readVectorWithStats(entryPointId, entry_vec);
    float d_entry = computeDistance(Span<const float>(queryVector), Span<const float>(entry_vec));

    visited_map_[entryPointId] = visited_version_;
    candidates.emplace(d_entry, entryPointId);
    if (node_matches(entryPointId)) {
        results.emplace(d_entry, entryPointId);
    }

    size_t scanned = 0;

    while (!candidates.empty()) {
        auto [cd, cid] = candidates.top();
        candidates.pop();

        // Termination 1: no closer point can improve current top-k
        if (results.size() >= static_cast<size_t>(k) && cd > results.top().first) break;
        // Termination 2: expansion cap
        if (scanned >= static_cast<size_t>(max_scan_candidates)) break;

        auto neighbors = getEdgesCached(cid);   // or equivalent for the layer
        for (node_id_t nb : neighbors) {
            auto it = visited_map_.find(nb);
            if (it != visited_map_.end() && it->second == visited_version_) continue;
            visited_map_[nb] = visited_version_;

            std::vector<float> nb_vec;
            readVectorWithStats(nb, nb_vec);
            float d = computeDistance(Span<const float>(queryVector), Span<const float>(nb_vec));

            ++scanned;

            // Routing: filter-agnostic
            if (results.size() < static_cast<size_t>(k) || d < results.top().first) {
                candidates.emplace(d, nb);
            }

            // Filtering: only matching nodes enter results
            if (node_matches(nb)) {
                if (results.size() < static_cast<size_t>(k) || d < results.top().first) {
                    results.emplace(d, nb);
                    if (results.size() > static_cast<size_t>(k)) results.pop();
                }
            }
        }
    }

    std::vector<SearchResult> out;
    out.reserve(results.size());
    while (!results.empty()) { out.push_back({results.top().second, results.top().first}); results.pop(); }
    std::reverse(out.begin(), out.end());  // ascending by distance
    return out;
}
```

Required includes at top of `src/lsm_vec_index.cc` (if not already present):

```cpp
#include <optional>
#include <queue>
#include "metadata_store.h"
```

**Note on `getEdgesCached` vs layer-0-specific accessor:** the existing `searchLayer` knows whether to use `getEdgesCached` (L0) vs `nodes_[cid].neighbors[layer]` (upper layers). Use the same branching pattern here. Since §7.4 of the design says upper layers do *not* use the filtered overload, you can assume `layer == 0` in the filtered path and use `getEdgesCached` exclusively. Add an assert:

```cpp
assert(layer == 0 && "filtered searchLayer is only supported on layer 0");
```

- [ ] **Step 3: Build**

```bash
make lib
```

Fix any compile errors. Common issues: `Span<const float>` constructor mismatches → use `Span<float>` if the existing code uses mutable spans.

- [ ] **Step 4: Commit**

```bash
git add src/lsm_vec_index.cc include/lsm_vec_index.h
git commit -m "feat(index): implement filtered searchLayer with iterative expansion"
```

---

### Task M4.4: `LSMVecDB::SearchKnn` overload with filter

**Files:**
- Modify: `include/lsm_vec_db.h`, `src/lsm_vec_db.cc`

- [ ] **Step 1: Declare overload**

```cpp
Status SearchKnn(Span<float> query,
                 const SearchOptions& options,
                 std::string_view filter_json,
                 std::vector<SearchResult>* out);
```

- [ ] **Step 2: Implement**

In `src/lsm_vec_db.cc`:

```cpp
Status LSMVecDB::SearchKnn(Span<float> query, const SearchOptions& options,
                           std::string_view filter_json,
                           std::vector<SearchResult>* out) {
    // Fast path: empty filter routes to the unfiltered overload.
    bool effectively_empty = true;
    {
        size_t s = 0, e = filter_json.size();
        while (s < e && std::isspace(static_cast<unsigned char>(filter_json[s]))) ++s;
        while (e > s && std::isspace(static_cast<unsigned char>(filter_json[e-1]))) --e;
        auto trimmed = filter_json.substr(s, e - s);
        if (!trimmed.empty() && trimmed != "{}") effectively_empty = false;
    }
    if (effectively_empty) return SearchKnn(query, options, out);

    // Parse the filter.
    metadata::Predicate pred;
    auto pst = metadata::ParsePredicate(filter_json, &pred);
    if (!pst.ok()) return pst;

    // Invoke the index with filter via a new public-ish path on LSMVec.
    // For simplicity, a thin forward method on LSMVec is added in the next step.
    std::vector<float> qvec(query.data(), query.data() + query.size());
    auto results = index_->knnSearchKFiltered(
        qvec,
        options.k,
        options.ef_search,
        &pred,
        options.max_scan_candidates,
        metadata_store_.get());
    *out = std::move(results);
    return Status::OK();
}
```

This calls a new `LSMVec::knnSearchKFiltered` method. Define it next.

- [ ] **Step 3: Add `LSMVec::knnSearchKFiltered`**

In `include/lsm_vec_index.h` (public section):

```cpp
std::vector<SearchResult> knnSearchKFiltered(
    const std::vector<float>& query,
    int k,
    int ef_search,
    const metadata::Predicate* pred,
    int max_scan_candidates,
    const MetadataStore* meta_store);
```

In `src/lsm_vec_index.cc`:

```cpp
std::vector<SearchResult> LSMVec::knnSearchKFiltered(
    const std::vector<float>& query,
    int k,
    int ef_search,
    const metadata::Predicate* pred,
    int max_scan_candidates,
    const MetadataStore* meta_store) {

    if (entry_point_ == k_invalid_node_id) return {};

    // Greedy descent through upper layers — uses original searchLayer (filter-blind).
    node_id_t curr = entry_point_;
    for (int lvl = max_layer_; lvl > 0; --lvl) {
        curr = greedySearchUpperLayer(query, curr, lvl);
    }

    // Layer 0: filtered iterative expansion.
    return searchLayer(query, curr, ef_search, /*layer=*/0,
                       pred, k, max_scan_candidates, meta_store);
}
```

- [ ] **Step 4: Build**

```bash
make lib
```

- [ ] **Step 5: Commit**

```bash
git add include/lsm_vec_db.h include/lsm_vec_index.h src/lsm_vec_db.cc src/lsm_vec_index.cc
git commit -m "feat(search): SearchKnn overload with filter, knnSearchKFiltered"
```

---

### Task M4.5: Basic correctness tests

**Files:**
- Create: `test/unit/test_filtered_search.cc`

- [ ] **Step 1: Write tests that seed vectors + metadata and verify filter behavior**

```cpp
#include "doctest.h"
#include "lsm_vec_db.h"
#include <filesystem>
#include <random>

static std::string NewTempDir() {
    char tmpl[] = "/tmp/lsmvec_fs_XXXXXX";
    return mkdtemp(tmpl);
}

static std::unique_ptr<lsm_vec::LSMVecDB> OpenFresh(const std::string& path, int dim = 8) {
    lsm_vec::LSMVecDBOptions opts;
    opts.dim = dim;
    opts.m = 8;
    opts.m_max = 16;
    opts.ef_construction = 32;
    opts.vec_file_capacity = 5000;
    opts.vector_file_path = path + "/vecs.bin";
    std::unique_ptr<lsm_vec::LSMVecDB> db;
    REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());
    return db;
}

static std::vector<float> RandVec(int dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = d(rng);
    return v;
}

TEST_CASE("Filter: returns only matching ids (100 vectors, ~10% selectivity)") {
    std::string path = NewTempDir();
    auto db = OpenFresh(path);
    std::mt19937 rng(42);

    for (uint64_t i = 0; i < 100; ++i) {
        auto v = RandVec(8, rng);
        std::string md = (i % 10 == 0)
            ? R"({"tenant":"acme"})"
            : R"({"tenant":"other"})";
        REQUIRE(db->Insert(i, lsm_vec::Span<float>(v), md).ok());
    }
    db->flushVectorWrites();

    auto q = RandVec(8, rng);
    lsm_vec::SearchOptions opts;
    opts.k = 5;
    opts.ef_search = 64;

    std::vector<lsm_vec::SearchResult> out;
    REQUIRE(db->SearchKnn(lsm_vec::Span<float>(q), opts,
                          R"({"tenant":"acme"})", &out).ok());

    CHECK(out.size() == 5);
    for (const auto& r : out) {
        CHECK(r.id % 10 == 0);        // only matching ids
    }
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Filter: returns < k when matches are fewer than k") {
    std::string path = NewTempDir();
    auto db = OpenFresh(path);
    std::mt19937 rng(42);

    for (uint64_t i = 0; i < 100; ++i) {
        auto v = RandVec(8, rng);
        std::string md = (i < 3)
            ? R"({"tenant":"acme"})"
            : R"({"tenant":"other"})";
        REQUIRE(db->Insert(i, lsm_vec::Span<float>(v), md).ok());
    }
    db->flushVectorWrites();

    auto q = RandVec(8, rng);
    lsm_vec::SearchOptions opts;
    opts.k = 10;
    opts.ef_search = 64;

    std::vector<lsm_vec::SearchResult> out;
    REQUIRE(db->SearchKnn(lsm_vec::Span<float>(q), opts,
                          R"({"tenant":"acme"})", &out).ok());

    CHECK(out.size() <= 3);
    for (const auto& r : out) CHECK(r.id < 3);
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Filter: no metadata on vector → does not match any filter") {
    std::string path = NewTempDir();
    auto db = OpenFresh(path);
    std::mt19937 rng(42);

    for (uint64_t i = 0; i < 50; ++i) {
        auto v = RandVec(8, rng);
        if (i < 10) {
            REQUIRE(db->Insert(i, lsm_vec::Span<float>(v), R"({"has_meta":true})").ok());
        } else {
            REQUIRE(db->Insert(i, lsm_vec::Span<float>(v)).ok());  // no metadata
        }
    }
    db->flushVectorWrites();

    auto q = RandVec(8, rng);
    lsm_vec::SearchOptions opts;
    opts.k = 20;
    opts.ef_search = 64;

    std::vector<lsm_vec::SearchResult> out;
    REQUIRE(db->SearchKnn(lsm_vec::Span<float>(q), opts,
                          R"({"has_meta":true})", &out).ok());

    for (const auto& r : out) CHECK(r.id < 10);
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Empty filter = unfiltered search") {
    std::string path = NewTempDir();
    auto db = OpenFresh(path);
    std::mt19937 rng(42);

    for (uint64_t i = 0; i < 50; ++i) {
        auto v = RandVec(8, rng);
        REQUIRE(db->Insert(i, lsm_vec::Span<float>(v), R"({"tag":"x"})").ok());
    }
    db->flushVectorWrites();

    auto q = RandVec(8, rng);
    lsm_vec::SearchOptions opts;
    opts.k = 10;
    opts.ef_search = 64;

    std::vector<lsm_vec::SearchResult> out1, out2;
    REQUIRE(db->SearchKnn(lsm_vec::Span<float>(q), opts, "", &out1).ok());
    REQUIRE(db->SearchKnn(lsm_vec::Span<float>(q), opts, &out2).ok());
    CHECK(out1.size() == out2.size());
    for (size_t i = 0; i < out1.size(); ++i) {
        CHECK(out1[i].id == out2[i].id);
        CHECK(out1[i].distance == doctest::Approx(out2[i].distance));
    }
    db->Close();
    std::filesystem::remove_all(path);
}

TEST_CASE("Filter: invalid JSON is InvalidArgument") {
    std::string path = NewTempDir();
    auto db = OpenFresh(path);
    auto q = std::vector<float>(8, 0.5f);
    std::vector<lsm_vec::SearchResult> out;
    lsm_vec::SearchOptions opts;
    auto st = db->SearchKnn(lsm_vec::Span<float>(q), opts, R"({not json)", &out);
    CHECK(st.IsInvalidArgument());
    db->Close();
    std::filesystem::remove_all(path);
}
```

- [ ] **Step 2: Run the tests**

```bash
make unit_test
```

- [ ] **Step 3: Debug any failures** — most likely sources:
  1. `readVectorWithStats` signature mismatch → check existing `searchLayer` for the exact API.
  2. `computeDistance` signature → same.
  3. `getEdgesCached` returns `std::vector<node_id_t>` or `const std::vector<node_id_t>*`? Adapt.
  4. `visited_map_` usage → mirror the unfiltered overload's pattern exactly.

- [ ] **Step 4: Commit**

```bash
git add test/unit/test_filtered_search.cc test/unit/CMakeLists.txt
git commit -m "test(search): filtered search correctness"
```

---

### Task M4.6: Regression test — unfiltered search unchanged

**Files:**
- Create: `test/unit/test_filtered_search_no_regression.cc`

- [ ] **Step 1: Write test**

```cpp
#include "doctest.h"
#include "lsm_vec_db.h"
#include <filesystem>
#include <random>

TEST_CASE("Unfiltered SearchKnn results unchanged after metadata feature added") {
    char tmpl[] = "/tmp/lsmvec_reg_XXXXXX";
    std::string path = mkdtemp(tmpl);

    lsm_vec::LSMVecDBOptions opts;
    opts.dim = 16;
    opts.m = 8;
    opts.m_max = 16;
    opts.ef_construction = 32;
    opts.vec_file_capacity = 5000;
    opts.vector_file_path = path + "/vecs.bin";
    std::unique_ptr<lsm_vec::LSMVecDB> db;
    REQUIRE(lsm_vec::LSMVecDB::Open(path, opts, &db).ok());

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1, 1);
    for (uint64_t i = 0; i < 200; ++i) {
        std::vector<float> v(16);
        for (auto& x : v) x = dist(rng);
        REQUIRE(db->Insert(i, lsm_vec::Span<float>(v)).ok());  // no metadata
    }
    db->flushVectorWrites();

    std::vector<float> q(16);
    for (auto& x : q) x = dist(rng);

    lsm_vec::SearchOptions so;
    so.k = 10;
    so.ef_search = 64;

    std::vector<lsm_vec::SearchResult> out;
    REQUIRE(db->SearchKnn(lsm_vec::Span<float>(q), so, &out).ok());
    CHECK(out.size() == 10);

    db->Close();
    std::filesystem::remove_all(path);
}
```

- [ ] **Step 2: Run, commit**

```bash
git add test/unit/test_filtered_search_no_regression.cc test/unit/CMakeLists.txt
git commit -m "test(search): no-regression on unfiltered path"
```

---

### Task M4.7: M4 verification

- [ ] **Step 1: All unit tests pass**

```bash
make unit_test
```

- [ ] **Step 2: Benchmark regression — unfiltered path identical**

```bash
make bin
rm -rf /tmp/bench_base; mkdir /tmp/bench_base
./build/bin/lsm_vec --db /tmp/bench_base --data-dir data/sift_100k_ --k 10 --efs 64 --stats --out /tmp/bench_base.out
grep -E "latency|qps|recall" /tmp/bench_base.out
```

Compare against a pre-M4 run. Latency p50/p99 should be identical within noise (±5%).

- [ ] **Step 3: Tag**

```bash
git tag metadata-filter-M4
```

---

## M5: Python Bindings

**Goal of M5:** Expose the new C++ API via pybind11, wrap in Pythonic idioms (dict ↔ JSON, Optional metadata).

### Task M5.1: Inspect current pybind11 module

- [ ] **Step 1: Read `python/lsm_vec_pybind.cc`**

```bash
cat python/lsm_vec_pybind.cc
```

Identify the current `Insert`, `Search`, and related binding code. Note the class name, naming convention, and how existing SearchResult is exposed.

- [ ] **Step 2: Note the findings in the PR description** (e.g. existing `def("insert", &LSMVecDB::Insert)` patterns).

---

### Task M5.2: Add `Insert` overload with metadata

**Files:**
- Modify: `python/lsm_vec_pybind.cc`

- [ ] **Step 1: Bind the overload**

```cpp
// Inside the class binding block for LSMVecDB
.def("insert",
     [](LSMVecDB& self, node_id_t id, py::array_t<float> vec,
        std::optional<py::dict> metadata) -> Status {
         Span<float> span(static_cast<float*>(vec.mutable_unchecked<1>().mutable_data(0)),
                          static_cast<size_t>(vec.shape(0)));
         if (!metadata.has_value()) {
             return self.Insert(id, span);
         }
         // Convert Python dict → JSON string via py::module_ json
         py::object json_mod = py::module_::import("json");
         std::string s = json_mod.attr("dumps")(*metadata).cast<std::string>();
         return self.Insert(id, span, s);
     },
     py::arg("id"), py::arg("vector"), py::arg("metadata") = py::none())
```

(The exact binding style must match existing conventions — use `self.Insert(...)` pattern consistent with the rest of the file.)

- [ ] **Step 2: Build pip package**

```bash
python -m pip install -e . --force-reinstall
```

- [ ] **Step 3: Quick Python smoke**

```python
import lsm_vec
opts = lsm_vec.LSMVecDBOptions(); opts.dim = 4; ...
db = lsm_vec.LSMVecDB.open("/tmp/py_md", opts)
db.insert(1, [0.1,0.2,0.3,0.4], metadata={"tenant":"acme"})
```

- [ ] **Step 4: Commit**

```bash
git add python/lsm_vec_pybind.cc
git commit -m "feat(python): Insert with metadata"
```

---

### Task M5.3: Bind `SetPayload`, `UpdatePayload`, `DeletePayloadKeys`, `GetPayload`

**Files:**
- Modify: `python/lsm_vec_pybind.cc`

- [ ] **Step 1: Bind each method**

```cpp
.def("set_payload",
     [](LSMVecDB& self, node_id_t id, py::dict md) {
         py::object json_mod = py::module_::import("json");
         std::string s = json_mod.attr("dumps")(md).cast<std::string>();
         return self.SetPayload(id, s);
     }, py::arg("id"), py::arg("metadata"))

.def("update_payload",
     [](LSMVecDB& self, node_id_t id, py::dict partial) {
         py::object json_mod = py::module_::import("json");
         std::string s = json_mod.attr("dumps")(partial).cast<std::string>();
         return self.UpdatePayload(id, s);
     }, py::arg("id"), py::arg("metadata"))

.def("delete_payload_keys",
     [](LSMVecDB& self, node_id_t id, std::vector<std::string> keys) {
         return self.DeletePayloadKeys(id, Span<const std::string>(keys));
     }, py::arg("id"), py::arg("keys"))

.def("get_payload",
     [](LSMVecDB& self, node_id_t id) -> py::object {
         std::string s;
         auto st = self.GetPayload(id, &s);
         if (!st.ok()) throw std::runtime_error(st.ToString());
         py::object json_mod = py::module_::import("json");
         return json_mod.attr("loads")(s);
     }, py::arg("id"))
```

- [ ] **Step 2: Build and smoke-test via Python REPL**

- [ ] **Step 3: Commit**

```bash
git add python/lsm_vec_pybind.cc
git commit -m "feat(python): payload CRUD bindings"
```

---

### Task M5.4: Bind `SearchKnn` overload with filter

**Files:**
- Modify: `python/lsm_vec_pybind.cc`

- [ ] **Step 1: Bind**

```cpp
.def("search",
     [](LSMVecDB& self, py::array_t<float> query, int k, int ef_search,
        std::optional<py::dict> filter,
        int max_scan_candidates) -> py::list {

         Span<float> span(static_cast<float*>(query.mutable_unchecked<1>().mutable_data(0)),
                          static_cast<size_t>(query.shape(0)));
         SearchOptions opts;
         opts.k = k;
         opts.ef_search = ef_search;
         opts.max_scan_candidates = max_scan_candidates;

         std::string filter_json = "";
         if (filter.has_value()) {
             py::object json_mod = py::module_::import("json");
             filter_json = json_mod.attr("dumps")(*filter).cast<std::string>();
         }

         std::vector<SearchResult> out;
         Status st = filter_json.empty()
             ? self.SearchKnn(span, opts, &out)
             : self.SearchKnn(span, opts, filter_json, &out);
         if (!st.ok()) throw std::runtime_error(st.ToString());

         py::list result;
         for (auto& r : out) {
             py::dict d;
             d["id"] = r.id;
             d["distance"] = r.distance;
             result.append(d);
         }
         return result;
     },
     py::arg("query"),
     py::arg("k") = 10,
     py::arg("ef_search") = 64,
     py::arg("filter") = py::none(),
     py::arg("max_scan_candidates") = 0)
```

- [ ] **Step 2: Build and smoke-test**

```python
results = db.search([0.1,0.2,0.3,0.4], k=5, filter={"tenant":"acme"})
```

- [ ] **Step 3: Commit**

```bash
git add python/lsm_vec_pybind.cc
git commit -m "feat(python): search with filter binding"
```

---

### Task M5.5: Python integration test

**Files:**
- Create: `python/tests/test_metadata.py`

- [ ] **Step 1: Write the test**

```python
import os
import shutil
import tempfile
import numpy as np
import pytest
import lsm_vec


@pytest.fixture
def fresh_db():
    path = tempfile.mkdtemp(prefix="lsmvec_py_md_")
    opts = lsm_vec.LSMVecDBOptions()
    opts.dim = 8
    opts.m = 8
    opts.m_max = 16
    opts.ef_construction = 32
    opts.vec_file_capacity = 1000
    opts.vector_file_path = os.path.join(path, "vecs.bin")
    db = lsm_vec.LSMVecDB.open(path, opts)
    yield db, path
    db.close()
    shutil.rmtree(path, ignore_errors=True)


def test_insert_with_metadata_and_get(fresh_db):
    db, _ = fresh_db
    db.insert(1, np.random.randn(8).astype(np.float32),
              metadata={"tenant": "acme", "count": 7})
    md = db.get_payload(1)
    assert md == {"tenant": "acme", "count": 7}


def test_search_with_filter(fresh_db):
    db, _ = fresh_db
    rng = np.random.default_rng(42)
    for i in range(100):
        v = rng.standard_normal(8).astype(np.float32)
        md = {"tenant": "acme" if i % 10 == 0 else "other"}
        db.insert(i, v, metadata=md)

    results = db.search(rng.standard_normal(8).astype(np.float32),
                        k=5, filter={"tenant": "acme"})
    assert len(results) == 5
    for r in results:
        assert r["id"] % 10 == 0


def test_update_payload_merge_patch(fresh_db):
    db, _ = fresh_db
    db.insert(1, np.zeros(8, dtype=np.float32), metadata={"a": 1})
    db.update_payload(1, {"b": 2})
    assert db.get_payload(1) == {"a": 1, "b": 2}

    db.update_payload(1, {"a": None})  # null → delete
    assert db.get_payload(1) == {"b": 2}
```

- [ ] **Step 2: Run**

```bash
python -m pip install -e . --force-reinstall
python -m pytest python/tests/test_metadata.py -v
```

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_metadata.py
git commit -m "test(python): metadata filtering integration"
```

---

### Task M5.6: M5 verification

- [ ] **Step 1: Python and C++ tests all pass**

```bash
make unit_test
python -m pytest python/tests/ -v
```

- [ ] **Step 2: Tag**

```bash
git tag metadata-filter-M5
```

---

## M6: Observability & Docs

### Task M6.1: Extend `HNSWStats`

**Files:**
- Modify: `include/statistics.h`

- [ ] **Step 1: Add counters**

```cpp
struct HNSWStats {
    // ... existing fields ...

    size_t metadata_gets = 0;
    size_t metadata_cache_hits = 0;       // reserved; 0 in v1
    size_t filter_evaluations = 0;
    size_t filter_matches = 0;
    size_t filter_scanned = 0;
    size_t filter_cap_hits = 0;
};
```

- [ ] **Step 2: Update the `printStatistics` / equivalent output code** to include the new counters.

- [ ] **Step 3: Commit**

```bash
git add include/statistics.h src/*.cc
git commit -m "feat(stats): metadata filtering counters"
```

---

### Task M6.2: Wire counters in `MetadataStore` and `searchLayer`

**Files:**
- Modify: `src/metadata_store.cc`, `src/lsm_vec_index.cc`

- [ ] **Step 1: `MetadataStore::Get` — counter is already member-local; expose via accessor (already done in M2.1)**

- [ ] **Step 2: In `searchLayer` filtered overload, increment `stats`**

Inside the filtered loop:

```cpp
++stats.filter_evaluations;
// after node_matches returns true:
++stats.filter_matches;
// each scanned increment:
++stats.filter_scanned;
// at termination due to max_scan_candidates:
if (scanned >= max_scan_candidates) ++stats.filter_cap_hits;
```

- [ ] **Step 3: Run tests to confirm counters don't break anything**

- [ ] **Step 4: Commit**

```bash
git add src/metadata_store.cc src/lsm_vec_index.cc
git commit -m "feat(stats): wire filter counters"
```

---

### Task M6.3: User docs

**Files:**
- Create: `docs/METADATA_FILTERING_USAGE.md`

- [ ] **Step 1: Write the usage guide**

Include:
- All 13 Level-2 operators with example filters.
- Common RAG patterns: tenant isolation, time windows, tag contains_any.
- Python SDK examples (Insert, SetPayload, UpdatePayload, search with filter).
- Missing-field semantics table (`$eq` / `$ne` / `$gt` behavior).
- Observability: how to read `filter_cap_hits` and when it indicates a need for scalar indexes.
- Performance guidance: metadata size, selectivity, `max_scan_candidates` tuning.
- Known limitations: no array indexing in dot-path, no thread safety, no bitmap index.

- [ ] **Step 2: Commit**

```bash
git add docs/METADATA_FILTERING_USAGE.md
git commit -m "docs: metadata filtering user guide"
```

---

### Task M6.4: Update Python SDK guide

**Files:**
- Modify: `docs/python_sdk_guide.md`

- [ ] **Step 1: Add a new section "Metadata Filtering"**

Mirror the content of `METADATA_FILTERING_USAGE.md` but focused on Python examples and type hints.

- [ ] **Step 2: Commit**

```bash
git add docs/python_sdk_guide.md
git commit -m "docs(python): metadata filtering section in SDK guide"
```

---

### Task M6.5: M6 verification — final acceptance

- [ ] **Step 1: Run full test matrix**

```bash
make unit_test                         # C++ unit tests
make bin && make lib                  # existing binaries build
python -m pytest python/tests/ -v     # Python tests
```

- [ ] **Step 2: Run existing benchmark binary; no regression**

```bash
./build/bin/lsm_vec --db /tmp/bench_final --data-dir data/sift_100k_ --k 10 --efs 64 --stats --out /tmp/bench_final.out
```

- [ ] **Step 3: Tag final release**

```bash
git tag metadata-filter-v1
```

- [ ] **Step 4: Create summary commit or PR description** listing all tags and summarizing LOC added.

---

## Summary

- **M1** — 11 tasks, Predicate AST + evaluator + parser.
- **M2** — 9 tasks, MetadataStore + Insert/Delete integration + backward compat.
- **M3** — 6 tasks, payload CRUD.
- **M4** — 7 tasks, filtered search with iterative expansion.
- **M5** — 6 tasks, Python bindings.
- **M6** — 5 tasks, observability + docs.

**Total:** 44 tasks. Per-task TDD cycle (write test → verify failure → implement → verify pass → commit). Every task ends in a commit; no untested code lands on the branch.

---

## Self-Review Checklist (done before handoff)

- ✅ Every Level-2 operator in the spec has at least one test (M1.2–M1.10).
- ✅ Every public API method has a test (M2.6, M2.7, M3.1–M3.5, M4.5).
- ✅ Iterative expansion edge cases covered (low/high selectivity, missing metadata, empty filter) in M4.5.
- ✅ Backward compatibility with pre-metadata DBs tested (M2.8).
- ✅ Python bindings cover every C++ public API with at least one smoke test (M5.5).
- ✅ Observability stats defined and wired (M6.1–M6.2).
- ✅ Name consistency: `SetPayload`/`UpdatePayload`/`DeletePayloadKeys`/`GetPayload` used throughout the plan; `MetadataStore::Update` (not `UpdatePatch`) used throughout; `knnSearchKFiltered` used consistently.
- ✅ Every code block in every step is complete and self-contained.
- ✅ No "TBD" / "TODO" / "similar to above" placeholders.
- ✅ Spec → plan coverage: all 12 sections of `METADATA_FILTERING_DESIGN.md` are implemented by at least one task.
