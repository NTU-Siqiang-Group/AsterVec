# LSM-Vec Metadata Filtering — Design

**Status:** Draft · 2026-04-19
**Target release:** Phase 1 (P0 feature per `PRODUCTION_READINESS_ANALYSIS.md`)
**Background reading:** `docs/METADATA_FILTERING_SURVEY.md` (cross-system research)

---

## 1. Context & Goals

LSM-Vec currently has no metadata support — vectors carry only an `id`. This blocks the most common vector-database use case (RAG with per-tenant / per-source / per-timestamp filtering). This document specifies a first-version design that:

- Adds JSON metadata per vector, stored in RocksDB, independently mutable from the vector.
- Supports a Mongo-style filter expression covering the common RAG query shapes (equality, ranges, set membership, array containment, field presence, nested paths, AND/OR).
- Integrates filtering into HNSW search via pre-filter with in-graph predicate check, with iterative candidate expansion to prevent the overfiltering recall cliff.
- Introduces **zero breaking changes** to the existing C++ and Python APIs. All new capability ships as overloads or new methods.
- Leaves a clean extension seam for future scalar-index acceleration (roaring-bitmap per field).

### Non-goals (explicit)

| Not in scope | Why | When |
|---|---|---|
| Scalar/bitmap indexes on metadata fields | Adds ~1500 LOC, insert hot-path changes, RocksDB MergeOperator | Future Phase — when `filter_cap_hits` stat shows users hitting expansion ceiling |
| Full-text search on document text | Separate problem (FTS index, tokenizer, BM25); not metadata | Future |
| ACORN-style payload-aware HNSW edges | Useful only once filtering is a measured hot path | Future |
| Thread safety for metadata mutations | Covered by the broader thread-safety Phase-1 item | Same Phase, different work item |
| Atomic insert across vector + HNSW edges + metadata | Existing Insert is already non-atomic; bigger WAL work-item owns this | Future (Phase 1 WAL work) |
| Schema enforcement | Users declare no schema; metadata is free-form JSON | Not planned |
| Cross-collection multi-tenancy | Single collection only; tenant isolation via `tenant_id` filter | Future |

---

## 2. Decisions Summary

These were settled during brainstorming; each row is a link between an alternative-space and the chosen answer.

| Decision | Alternatives considered | Chosen |
|---|---|---|
| Storage strategy | (A) JSON blob + per-candidate predicate eval; (B) typed scalar indexes with roaring bitmaps; (C) A now, B-compatible seams | **C** — A implementation now, CF naming and API shape leave room to add B later |
| Expression operator scope | L1 (Pinecone minimum); L2 (L1 + `$exists` + array contains + dot-path + implicit eq); L3 (L2 + regex + FTS) | **L2** |
| Metadata lifecycle | (α) strict pairing with vector; (β) independent mutation via separate payload APIs; (γ) upsert-only | **β** |
| Filter-passing API | (1) JSON string; (2) exposed `Filter` class allowing pre-compile; (3) both | **1** |
| Selectivity handling | (A) raw in-graph skip, user tunes `ef_search`; (B) iterative expansion with `max_scan_candidates`; (C) B + brute-force fallback | **B** |

Rationale for each is captured in the survey and brainstorming transcript; this section is a pointer index, not a re-derivation.

---

## 3. Architecture

### 3.1 New modules

```
include/
├── metadata.h          (new) Predicate AST, evaluator, parser
├── metadata_store.h    (new) MetadataStore interface + RocksDB backend
└── lsm_vec_db.h        (modified) new overloads and methods

src/
├── metadata.cc         (new) operator semantics, matcher, parser
├── metadata_store.cc   (new) CF open/read/write, merge_patch wrapper
├── lsm_vec_db.cc       (modified) public API delegations, delete cascade
└── lsm_vec_index.cc    (modified) new searchLayer overload with filter + iterative expansion
```

### 3.2 Responsibilities

- **`metadata::Predicate`** — value type. Method `bool matches(const json& doc) const`. Knows nothing about RocksDB or HNSW. Fully unit-testable.
- **`metadata::ParsePredicate`** — parser from JSON string to `Predicate`. Returns `Status::InvalidArgument` with a descriptive message on error.
- **`MetadataStore`** — thin wrapper over a RocksDB column family. Provides `Get`, `Put`, `Update` (merge-patch), `DeleteKeys`, `Delete`. Does not know about predicates or HNSW. Mockable.
- **`LSMVec::searchLayer` (new overload)** — accepts `const Predicate*` and the two expansion knobs (`k`, `max_scan_candidates`). Only this one HNSW-core function changes.
- **`LSMVecDB`** — composes the three modules, owns the `MetadataStore` instance, provides the public API.

### 3.3 RocksDB column-family layout

```
cf_metadata          v1 uses:  node_id → raw JSON bytes
cf_metadata_idx      v1 unused — reserved for future (field,value) → roaring bitmap
```

The `cf_metadata_*` prefix is deliberate: it binds metadata-related CFs into a namespace so that the future Phase-2 scalar index can be added without CF-name conflicts with existing Aster CFs.

**CF instance choice:** the new CF is added to the existing Aster DB instance (reusing its WAL, block cache, backup). If Aster's encapsulation does not expose `DB*` for CF creation, a fallback plan is to open a sibling `rocksdb::DB` at `<db_path>/metadata/`. Public API is unaffected either way.

### 3.4 Delete cascade

`LSMVecDB::Delete(id)` deletes the metadata entry in addition to the existing vector + HNSW cleanup. Metadata delete is best-effort; if it fails after a successful vector delete, the orphaned metadata row is harmless (no vector with that id exists, so no search can reach the metadata).

---

## 4. Data Model & Storage Layout

### 4.1 Key format

```
key_bytes = big-endian uint64(node_id)      // 8 bytes, fixed width
```

- Fixed-width saves ~30% versus varint over typical 64-bit IDs.
- Big-endian preserves numeric order in RocksDB's `memcmp` sort, enabling range iteration by id for future migration/export tooling.
- No prefix — the CF is the namespace.

### 4.2 Value format

```
value_bytes = raw JSON bytes (as received, after trimming leading/trailing whitespace)
```

- Stored as-is. No re-serialization on write.
- Parsed with `nlohmann::json::parse` on read. Typical RAG metadata is <500 bytes, parse cost < 1 µs.
- Not using MessagePack/CBOR in v1 despite ~30% size savings: preserves `ldb dump` readability, removes a failure mode (format mismatch during migration), simplifies the first version. A later migration to a binary format is a value-layer change only.

### 4.3 Column family options

| Option | Value | Rationale |
|---|---|---|
| `compression` | `kZSTD` | JSON text compresses 3-5× |
| `compression_per_level` | `{none, none, zstd, zstd, ...}` | Keep L0/L1 fast; compress deeper levels |
| `block_cache` | shared with main DB block cache | Hot metadata naturally cached alongside hot vectors/edges |
| `bloom_filter_bits_per_key` | 10 | Every search does point Gets; bloom is mandatory |
| `merge_operator` | not set | v1 uses application-side read-modify-write for UpdatePayload |

### 4.4 Update semantics (UpdatePayload)

```cpp
Status MetadataStore::Update(node_id_t id, const json& partial) {
    json existing;
    auto s = Get(id, &existing);
    if (s.IsNotFound()) existing = json::object();
    else if (!s.ok()) return s;
    existing.merge_patch(partial);                  // RFC 7396 semantics
    return Put(id, existing);
}
```

- JSON Merge-Patch (RFC 7396): setting a key to `null` deletes it; objects merge recursively; arrays and scalars replace entirely.
- `nlohmann::json::merge_patch` is a one-line call.
- Application-level read-modify-write is correct under v1's single-threaded assumption. Thread-safety evolution has two clean paths (row-level mutex or migrate to RocksDB MergeOperator with a custom JSON merge function); either is localized to `MetadataStore`.

### 4.5 Metadata absence

Insert *without* metadata writes no row to `cf_metadata`. Query-side evaluation treats a missing row as the empty object `{}`. This keeps storage cost for non-metadata users at zero and ensures existing databases upgrade losslessly.

---

## 5. Predicate Language (L2)

### 5.1 AST

Tagged struct with explicit `Kind` enum. Rejected `std::variant` due to template bloat, opaque debugger representation, and no measurable runtime benefit for the typical <10-node filter AST.

```cpp
namespace lsm_vec::metadata {

using Json      = nlohmann::json;
using FieldPath = std::vector<std::string>;  // "author.name" → ["author","name"]

class Predicate {
public:
    enum class Kind {
        Eq, Ne, Gt, Gte, Lt, Lte,
        In, Nin,
        Exists,
        ContainsAny, ContainsAll,
        And, Or,
    };

    Kind                   kind;
    FieldPath              path;              // leaf kinds
    Json                   value;             // Eq/Ne/Gt/Gte/Lt/Lte
    std::vector<Json>      values;            // In/Nin/ContainsAny/ContainsAll
    bool                   exists_expected = true;  // Exists
    std::vector<Predicate> children;          // And/Or

    bool matches(const Json& doc) const;
};

Status ParsePredicate(std::string_view json_str, Predicate* out);

}  // namespace lsm_vec::metadata
```

### 5.2 Operator semantics

For every leaf operator, semantics are defined for three input cases: field present and typed, field present but wrong type, field missing.

| Operator | Field missing | Type mismatch |
|---|---|---|
| `$eq`, `$in` | false | false |
| `$ne`, `$nin` | **true** | true |
| `$gt`, `$gte`, `$lt`, `$lte` | false | **false** (e.g. `$gt: 5` on a string value) |
| `$exists: true` | false | n/a |
| `$exists: false` | **true** | n/a |
| `$contains_any`, `$contains_all` | false | false (field exists but is not an array) |

`$ne` / `$nin` returning true on missing fields is the MongoDB convention and matches Pinecone and ChromaDB. Documented explicitly in user docs.

### 5.3 Path resolver

```cpp
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
```

- Any segment that is missing or whose parent is not an object → null (treated as missing).
- Array indices in paths (e.g. `"tags.0"`) are **not** interpreted as array subscript. The literal key `"0"` is looked up instead. For array element queries, use `$contains_any` / `$contains_all`. This matches Pinecone's semantics.

### 5.4 Parser

Input: top-level JSON object. Keys:

- `$and`, `$or` — array of sub-predicate objects.
- Any other key — a field name (dot-path supported). Value is either:
  - A scalar/array/object → implicit `$eq` (full deep equality via `nlohmann::json::operator==`).
  - An object whose keys are operators → one predicate per operator, combined with implicit AND.

Top-level multiple keys are combined with implicit AND (MongoDB convention).

Concrete example:

```json
{
  "tenant_id": "acme",
  "tags":          {"$contains_any": ["python", "ml"]},
  "author.name":   {"$ne": "bot"},
  "created_at":    {"$gt": 1700000000},
  "source_url":    {"$exists": true},
  "$or": [
    {"priority": {"$gte": 5}},
    {"pinned":   true}
  ]
}
```

### 5.5 Parser errors (all `InvalidArgument`)

- Unknown operator key (`$foo`).
- `$in` / `$nin` / `$contains_any` / `$contains_all` value is not an array.
- `$and` / `$or` value is not an array, or array elements are not objects.
- `$exists` value is not a boolean.
- Malformed JSON.

Empty string or `"{}"` are treated as "no filter" (match all) — the evaluator returns true via an empty `PredAnd`. This is not an error.

---

## 6. Public API

### 6.1 Option structs

```cpp
struct SearchOptions {
    int k = 1;
    int ef_search = 64;
    // Tier B iterative expansion upper bound.
    // 0 = auto (k * 50 when filter present; ignored otherwise).
    int max_scan_candidates = 0;
};
```

`LSMVecDBOptions` is unchanged. Metadata CF is always enabled in v1.

### 6.2 Insert

```cpp
// existing — unchanged
Status Insert(node_id_t id, Span<float> vec);

// new overload
Status Insert(node_id_t id, Span<float> vec, std::string_view metadata_json);
```

- Without metadata: no write to `cf_metadata`.
- With metadata: parsed and validated *before* any write. Parse failure → `InvalidArgument`, nothing is written.

### 6.3 Metadata CRUD

```cpp
Status SetPayload(node_id_t id, std::string_view metadata_json);
Status UpdatePayload(node_id_t id, std::string_view partial_json);
Status DeletePayloadKeys(node_id_t id, Span<const std::string> keys);
Status GetPayload(node_id_t id, std::string* out_json);
```

Invariant: metadata exists only alongside a vector.

- `SetPayload`, `UpdatePayload`, `DeletePayloadKeys` on a non-existent id → `NotFound`.
- `GetPayload` on an id with a vector but no metadata → `OK`, `*out_json = "{}"`.
- `Delete(id)` cascades to metadata.

### 6.4 Search

```cpp
// existing — unchanged
Status SearchKnn(Span<float> query, const SearchOptions& options,
                 std::vector<SearchResult>* out);
Status SearchKnn(Span<float> query, std::vector<SearchResult>* out);

// new overload
Status SearchKnn(Span<float> query, const SearchOptions& options,
                 std::string_view filter_json,     // "" or "{}" ⇒ no filter
                 std::vector<SearchResult>* out);
```

- Empty filter routes to the metadata-blind code path with zero additional cost.
- Non-empty filter parses once, builds a `Predicate`, and passes it to `LSMVec::searchLayer`.
- `SearchResult` shape is unchanged (`{id, distance}`). Metadata is not returned by default; users call `GetPayload(id)` if needed. A future `SearchOptions::include_payload` flag can opt into bundled returns without breaking the current signature.

### 6.5 Python bindings

```python
# Insert
db.insert(123, vec)
db.insert(123, vec, metadata={"tenant_id": "X"})

# CRUD
db.set_payload(123, {"tenant_id": "X", "tags": ["py"]})
db.update_payload(123, {"tags": ["py", "ml"]})
db.update_payload(123, {"temp_field": None})       # None deletes the field
db.delete_payload_keys(123, ["temp_field", "old_key"])
md = db.get_payload(123)                            # returns dict

# Search
results = db.search(query, k=10)
results = db.search(query, k=10, filter={"tenant_id": "X"})
```

The Python wrapper handles `dict ↔ JSON string` conversion; the C++ binding layer only sees strings.

### 6.6 Error code reference

| Scenario | Status |
|---|---|
| Filter parse error | `InvalidArgument("filter parse error: ...")` |
| Metadata parse error | `InvalidArgument("metadata parse error: ...")` |
| CRUD on non-existent id | `NotFound` |
| Duplicate Insert | `InvalidArgument` (existing behavior) |
| `GetPayload` with no metadata row | `OK`, out = `"{}"` |
| Search returns fewer than k results | `OK`, `out.size() < k` — expected when filter is too selective |

---

## 7. Search Integration

### 7.1 New searchLayer overload

```cpp
// existing — unchanged
std::vector<SearchResult> searchLayer(
    const std::vector<float>& query, node_id_t entry, int ef_search, int layer);

// new overload
std::vector<SearchResult> searchLayer(
    const std::vector<float>& query, node_id_t entry, int ef_search, int layer,
    const metadata::Predicate* pred,     // nullptr ⇒ no filter
    int k,
    int max_scan_candidates);
```

### 7.2 Iterative-expansion loop (Tier B)

```
candidates: min-heap by distance (routing)
results:    max-heap by distance, capped at k, holds only matching nodes
visited:    set
scanned:    integer, count of candidates whose metadata has been evaluated

// Seed
visited.insert(entry)
d_entry = distance(query, entry)
candidates.push(entry, d_entry)
if pred == nullptr or pred->matches(metadata(entry)):
    results.push(entry, d_entry)

while not candidates.empty():
    curr = candidates.pop_nearest()

    // Convergence: no reachable point can improve top-k
    if results.size() >= k and curr.dist > results.top().dist:
        break

    // Expansion cap: Tier B safety bound
    if scanned >= max_scan_candidates:
        break

    for nb in graph.neighbors(curr):
        if not visited.insert(nb): continue
        d = distance(query, nb)
        scanned += 1

        // Routing: filter-agnostic — all neighbors can serve as bridges
        if results.size() < k or d < results.top().dist:
            candidates.push(nb, d)

        // Filtering: only matching nodes enter results
        bool match = (pred == nullptr) or pred->matches(metadata(nb));
        if match and (results.size() < k or d < results.top().dist):
            results.push(nb, d)
            if results.size() > k: results.pop()

return sorted(results)
```

Key correctness invariant: **routing and filtering are decoupled.** Non-matching nodes stay in `candidates` because they are bridges to matching ones — Tier A's mistake (filter at `candidates` push time) fragments the graph and collapses recall at low selectivities.

### 7.3 `ef_search` semantics with a filter

Without filter: `ef_search` caps the results heap and implicitly bounds `scanned`.

With filter: the results heap is capped at `k`. `ef_search` still influences routing beam aggressiveness (how many neighbors are pushed to `candidates`) but no longer bounds expansion. `max_scan_candidates` is the new expansion cap. This semantic shift is documented in the user-facing docs.

### 7.4 Upper layers

HNSW upper layers only perform greedy navigation and do not produce final results. The filtered overload is only invoked for layer 0. Upper-layer greedy search continues to call the original `searchLayer` — zero change to that code path.

### 7.5 Metadata fetch strategy

`pred->matches(doc)` requires fetching `doc` for each visited neighbor. v1 performs one `MetadataStore::Get` per neighbor:

- Typical visited count per search: 1000–5000 neighbors.
- Per Get: ~5–20 µs (block cache hit much less) + parse ~1–5 µs.
- Aggregate: 5–100 ms per search on metadata I/O, same order of magnitude as the vector-distance compute path.

**Deferred optimizations (not v1):**
- `DB::MultiGet` on a neighbor batch (2–3× amortization).
- Small in-process LRU cache over parsed metadata.

### 7.6 Edge cases

| Case | Behavior |
|---|---|
| Entry point does not match filter | Stays in `candidates` (serves routing), not in `results`. |
| No match in reachable subgraph | Expansion terminates at cap; returns fewer than k. |
| Filter selectivity = 100% | Equivalent to unfiltered search plus a cheap per-candidate `matches()` call whose metadata Gets mostly hit block cache. |
| Unsatisfiable filter (contradiction) | Scans to cap, returns empty. v1 does not detect contradictions; zero-user edge case. |
| `k == 0` | `InvalidArgument` (existing behavior preserved). |

---

## 8. Error Handling, Persistence, Concurrency

### 8.1 Write ordering in Insert

The existing Insert is already not atomic across vector storage and HNSW-edge storage. Metadata fits into the same best-effort model.

Order:

1. Validate arguments (parse metadata JSON, validate vector) — no writes yet. Any failure → return `InvalidArgument`, no side effects.
2. Insert vector (`PagedVectorStorage` or `BasicVectorStorage`) + HNSW edges (`LSMVec::insertNode`).
3. Write metadata (`MetadataStore::Put`) — last.

Failure modes:

- Step 1 fails: nothing written; consistent.
- Step 2 fails: inherited from existing behavior; not newly worsened.
- Step 3 fails: vector exists, metadata is absent. Subsequent queries treat this vector as "metadata missing" — so the filter returns false for any operator except `$exists:false` and `$ne`/`$nin`. This is a **safe silent failure**: no false matches, at worst a stale vector that will never surface in a filtered search.

A future WAL overhaul (separate Phase-1 work item) will unify Insert atomicity. This design does not preempt that work.

### 8.2 WAL and crash recovery

- `cf_metadata` inherits the main DB's WAL configuration. No custom WAL logic.
- Recovery: RocksDB replays WAL on `Open`, reconstructing metadata to the latest `Put` before the crash.
- Backward compatibility: an older DB without `cf_metadata` gets it created on `Open` (idempotent — use `CreateColumnFamilyIfMissing` semantics). Existing vectors implicitly have metadata `{}`.

### 8.3 Concurrency (v1 is single-threaded)

v1 inherits the whole project's single-threaded property. Documented known races for the future thread-safety work:

1. **UpdatePayload RMW.** Two concurrent `UpdatePayload` calls on the same id can lose updates. Fix path: row-level mutex bucketed by `id`, or migrate to RocksDB MergeOperator with a JSON merge function. Localized to `MetadataStore`.
2. **Delete vs Set/UpdatePayload.** Can leave orphaned metadata or touch a deleted id. Fix: row-level mutex, same.
3. **Search vs Insert/Delete.** The HNSW traversal reads metadata while writes modify it. Fix: RocksDB snapshots — search acquires a snapshot at start, reads metadata through it.

All metadata reads and writes pass through `MetadataStore`, so locking is a single-file change when the time comes.

### 8.4 Metadata size and JSON constraints

No hard size limit in v1. Guidance (documented):

| Size | Guidance |
|---|---|
| < 2 KB | Transparent. |
| 2–10 KB | Normal for RAG chunk metadata. |
| 10–100 KB | Works; per-Get cost grows. Consider pruning. |
| > 100 KB | Not recommended. Metadata is not object storage. |

- JSON `null` is allowed and stored. `$eq: null` matches a field whose explicit value is `null`. `$exists: true` matches if the key is present even when the value is `null`. `$exists: false` does not match.
- Nested objects and arrays of arbitrary depth allowed.
- Non-UTF-8 input is rejected at parse time (`nlohmann::json` requirement) — fail fast.

### 8.5 Observability

`HNSWStats` (honored when `LSMVecDBOptions::enable_stats = true`):

```cpp
struct HNSWStats {
    // existing fields...

    // added
    size_t metadata_gets = 0;            // point Gets on cf_metadata
    size_t metadata_cache_hits = 0;      // reserved for future in-process LRU (0 in v1)
    size_t filter_evaluations = 0;       // Predicate::matches calls
    size_t filter_matches = 0;           // matches returning true
    size_t filter_scanned = 0;           // Tier B total scanned count (aggregate over searches)
    size_t filter_cap_hits = 0;          // count of searches that hit max_scan_candidates
};
```

`filter_cap_hits > 0` is the canonical signal that the workload needs scalar indexes — a handoff to the future Phase-2 work.

---

## 9. Milestones

Each milestone is independently compilable, testable, and commit-ready. The main branch stays green after each.

| M | Scope | Days | LOC | Dependencies |
|---|---|---|---|---|
| M1 | `metadata.h/.cc` — Predicate AST, evaluator, parser | 3 | ~350 | none |
| M2 | `metadata_store.h/.cc` — CF open, Get/Put/Delete; Insert/Delete cascade in `LSMVecDB` | 2 | ~250 | M1 |
| M3 | `SetPayload` / `UpdatePayload` / `DeletePayloadKeys` / `GetPayload` | 1 | ~120 | M2 |
| M4 | `SearchKnn` with filter + `searchLayer` overload + Tier B iterative expansion | 4 | ~200 | M1, M2 |
| M5 | Python bindings (`pybind11`) + SDK wrapper | 1 | ~100 | M4 |
| M6 | Stats counters, user docs (`docs/METADATA_FILTERING_USAGE.md`), backward-compat tests | 1 | ~80 | M4 |

**Total: ~12 working days, ~1100 LOC.** Approximately 700 LOC is brand-new code; the remainder is new overloads and cascading delete calls. **No existing code is rewritten** — only overloads added and new delegations wired.

### 9.1 Per-milestone testing gates

**M1** — 40+ unit tests with no RocksDB / HNSW dependency:
- Every operator × (match / no match / missing field / type mismatch).
- Dot-path resolution at several depths.
- Nested `$and` / `$or`.
- Implicit eq equivalent to explicit `$eq` across input types.
- Same-field multi-operator = implicit AND.
- Every parse-error class.
- Boundary: very deep AND/OR trees, very long `$in` lists.

**M2** — integration tests:
- `Insert` + `GetPayload` round-trip.
- `Delete` cascades to metadata (post-delete `GetPayload` → `NotFound`).
- Backward compat: DB created by pre-metadata binary opens cleanly with new code; existing `GetPayload` → `"{}"`.
- Crash recovery: kill between `Insert` and a subsequent search; metadata is preserved on reopen.

**M3** — CRUD coverage:
- `SetPayload` overwrites.
- `UpdatePayload` merge-patch: adds, overwrites, `null` deletes, nested merge.
- `DeletePayloadKeys`: multiple keys, non-existent keys (idempotent).
- `NotFound` cases for all three.

**M4** — §7.6 edge cases plus:
- 100 vectors, filter selecting ~10 → `k=5` returns exactly 5.
- 100 vectors, filter selecting 3 → `k=10` returns exactly 3, no duplicates, no crash.
- 1000 vectors, filter selecting 50 (5%), `max_scan_candidates = k * 50 = 500` → returns full `k=10`.
- 1000 vectors, filter selecting 900 (90%) → returns `k=10` with latency comparable to unfiltered.
- 1000 vectors, filter selecting 1 → returns 0 or 1, no crash.
- Existing unfiltered test suite: 100% passing — zero regression on the blind-search path.
- Benchmark: unfiltered p50/p99 identical (no overhead from the new overload if filter is `nullptr`).

**M5** — Python tests mirroring the C++ integration coverage via the SDK.

**M6** — observability tests assert counters increment correctly; backward-compat tests use golden-file DBs from prior releases.

### 9.2 Release strategy

Each milestone is release-safe on its own:

- After M1: library gains a standalone Predicate facility (not user-visible).
- After M2: `Insert` gains a metadata parameter; users can write metadata but cannot yet filter on it.
- After M3: full CRUD; users can freely manage metadata but still cannot filter.
- After M4: filter search live in C++ API. Feature complete for C++ users.
- After M5: Python API live. Full external availability.
- After M6: stats and docs — production-ready.

---

## 10. Open Questions

Items that should be resolved during implementation but do not block this design:

1. **Aster CF access.** Does the current `RocksGraph` encapsulation expose a `rocksdb::DB*` usable for creating a new CF? If not, the fallback is a sibling `rocksdb::DB` instance at `<db_path>/metadata/`. API-transparent either way. **Resolve in M2.**
2. **Default `max_scan_candidates` multiplier.** The design proposes `k * 50`. This assumes workloads of at least 2% filter selectivity are "common". If empirically many workloads are more selective, the default should be raised. **Tune via M4 benchmarks.**
3. **Metadata per-field size cap.** No cap in v1; document guidance only. A hard cap (e.g. 64 KB matching Milvus) could be useful as a safety rail. **Decide before M2 ships if any cap is introduced.** Current leaning: no cap, document guidance only.
4. **Empty-filter fast path.** The parser returns an empty `PredAnd` for `"{}"`. The search path should detect this and route to the unfiltered `searchLayer` overload rather than invoke the new one with a no-op predicate. Zero-cost short circuit. **Resolve in M4 implementation.**

---

## 11. Glossary

- **Predicate** — an AST representing a Mongo-style filter expression.
- **Allow-set** — not used in v1; the future Phase-2 concept of a precomputed bitmap of matching ids.
- **Pre-filter** — evaluate the predicate (via metadata lookup) during or before search; reject non-matching candidates from the result heap while keeping them in the routing heap. (v1 implementation.)
- **Post-filter** — run unfiltered search, filter results after. Not used; known to risk returning fewer than k.
- **In-graph skip / Tier A** — predicate check in search with no iterative expansion. Not used standalone; v1 uses Tier B.
- **Iterative expansion / Tier B** — continue expanding HNSW candidates past `ef_search` until `k` matches are collected or `max_scan_candidates` is reached. (v1 implementation.)
- **JSON Merge-Patch** — RFC 7396 semantics used by `UpdatePayload`.

---

## 12. References

- `docs/METADATA_FILTERING_SURVEY.md` — cross-system research underpinning the decisions above.
- `docs/PRODUCTION_READINESS_ANALYSIS.md` — places this work in the P0 roadmap.
- Pinecone ICML 2025 paper — pre-filter vs. mid-scan strategy selection: <https://openreview.net/pdf?id=UXq4z6GGYP>.
- Weaviate ACORN blog — two-hop predicate expansion: <https://weaviate.io/blog/speed-up-filtered-vector-search>.
- Qdrant Gridstore article — rationale for moving payload storage off RocksDB: <https://qdrant.tech/articles/gridstore-key-value-storage/>.
- pgvector 0.8 release — iterative index scan for HNSW + filter: <https://www.postgresql.org/about/news/pgvector-080-released-2952/>.
- RFC 7396 (JSON Merge-Patch): <https://datatracker.ietf.org/doc/html/rfc7396>.
