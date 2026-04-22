# Metadata Filtering in Mainstream Vector Databases — Design Survey

A research survey of how production vector databases implement metadata filtering, intended as input to the LSM-Vec design for the "Metadata Filtering (P0)" gap identified in `PRODUCTION_READINESS_ANALYSIS.md`.

The survey covers six systems (Milvus, Qdrant, Weaviate, Pinecone, ChromaDB, pgvector) plus the academic literature (ACORN, Filtered-DiskANN, NHQ, recent surveys). The goal is to understand: **(a) how metadata is modeled and stored on disk, (b) what scalar/inverted indexes are built, (c) how filters interact with HNSW graph traversal, and (d) how each system chooses between pre-filter, post-filter, and in-filter strategies**.

---

## 1. TL;DR — Cross-System Comparison

### 1.1 Storage and schema

| System | Schema | Metadata model | On-disk layout |
|---|---|---|---|
| Milvus | Strict typed + dynamic JSON blob | Bool/Int/Float/Varchar/JSON(≤64KB)/Array/Geo/Timestamp | Columnar binlogs per field, Arrow in-memory |
| Qdrant | Schema-less JSON "payload" | Arbitrary JSON, opt-in typed indexes | Gridstore (appendable) + mmap (immutable); migrating off RocksDB in 1.16 |
| Weaviate | Strict typed schema | Typed properties | LSM object store + LSM inverted index (Roaring) + HNSW graph — three stores per shard |
| Pinecone | Schema-less flat map | string/number/bool/list-of-strings only; no nesting; 40 KB cap | Immutable "slabs" in object storage, LSM-style; roaring bitmap per field/value |
| ChromaDB | Schema-less dict | Scalar types + document text | SQLite (`embedding_metadata` table) + FTS5 + hnswlib index files |
| pgvector | Full Postgres DDL | Any Postgres type, including JSONB | Postgres heap + B-tree/GIN + HNSW index pages |

**Convergence points.** Every production system that has published its internals either (a) stores metadata columnar and builds per-field secondary indexes (Milvus, Qdrant, Pinecone, Weaviate) or (b) inherits the mature columnar/B-tree infrastructure of an RDBMS (pgvector, ChromaDB via SQLite). **Roaring bitmaps** recur across Weaviate and Pinecone as the physical index format because they compose naturally with LSM segments and support fast AND/OR intersection.

### 1.2 Filter expression languages

| System | Style | Example operators |
|---|---|---|
| Milvus | SQL-like EBNF DSL (`expr=` / `filter=`) | `AND/OR/NOT`, `==`, `!=`, `<`, `>`, `IN`, `LIKE`, `json_contains`, `ARRAY_CONTAINS_ANY` |
| Qdrant | JSON, must/should/must_not + leaf conditions | `Match`, `MatchAny`, `Range`, `GeoBoundingBox`, `GeoPolygon`, `IsEmpty`, `nested` |
| Weaviate | GraphQL `where` | `Equal`, `GreaterThan`, `Like`, `ContainsAll/Any`, `WithinGeoRange`, `And/Or/Not` |
| Pinecone | MongoDB-style JSON | `$eq`, `$ne`, `$gt[e]`, `$lt[e]`, `$in`, `$nin`, `$exists`, `$and`, `$or`; `$in` max 10 000 values |
| ChromaDB | MongoDB-style dict + `where_document` | `$eq`, `$ne`, `$gt[e]`, `$lt[e]`, `$in`, `$nin`, `$and`, `$or`, `$contains` |
| pgvector | Raw SQL `WHERE` | Anything Postgres supports (operators, functions, joins) |

**Convergence point.** Two families dominate: **MongoDB-style operators** (Pinecone, ChromaDB) for simplicity and **compositional DSLs** (Milvus EBNF, Qdrant structured JSON) for expressive power. Weaviate's GraphQL is a syntactic wrapper over the same capability set. pgvector delegates entirely to SQL.

### 1.3 Filter strategy selection

| System | Strategies | Selector |
|---|---|---|
| Milvus | Pre-filter bitset + in-graph skip; iterative-filter (lazy); brute-force fallback | Auto, threshold ≈10% match rate for brute-force fallback (Knowhere HNSW) |
| Qdrant | Full scan; payload-index-first; **filterable HNSW** (payload-aware extra edges); **ACORN second-hop** (1.16) | 4-way planner using exact cardinality estimates from payload indexes |
| Weaviate | Sweeping (post-check during traversal); **ACORN-style two-hop** (default in 1.34); flat brute-force cutoff | Heuristic on allow-list size + correlation |
| Pinecone | Pre-filter (scan matching records only) or mid-scan bitmap check | Selectivity-driven; thresholds undisclosed; guarantees exact filter recall |
| ChromaDB | Pre-filter via SQLite → hnswlib filtered search with allow-list callback | Single strategy; overfiltering possible at low selectivity |
| pgvector | Bitmap + post-sort; HNSW + in-filter; iterative HNSW scan (0.8.0); seq scan | Postgres planner using cost estimation + `hnsw.iterative_scan` GUC |

**Convergence point.** The state of the art is converging on a **multi-strategy planner keyed on selectivity**, with three canonical branches:

1. **Very selective** (few rows match, e.g. <1%) → pre-filter, then brute-force scan. HNSW's graph structure is worse than sequential scan at this point.
2. **Moderately selective** (a few to ~20%) → in-graph filter check with an ACORN-style **two-hop expansion** to keep the subgraph connected.
3. **Not selective** (most rows match) → normal HNSW traversal with cheap bitmap lookup per candidate.

A fourth axis — **payload-aware graph edges** — is Qdrant's unique contribution: at build time, extra HNSW neighbor edges are added per indexed payload value, so every node has a route to others sharing that value. Weaviate's 1.34 ACORN achieves a similar effect at query time without rebuilding.

---

## 2. Per-System Deep Dives

### 2.1 Milvus

**Storage.** Strict typed schema with optional dynamic JSON field. Physical layout is columnar: per-field binlog files in segments, Arrow in-memory. JSON ≤64 KB/value, Array ≤4 096 elements. Delete events are written to separate delta binlogs.

**Scalar indexes.** Four main types plus recent additions:

- **INVERTED** — default for general scalar filtering (tokenized term dictionary on Bool/Int/Float/Varchar/JSON/Array).
- **BITMAP** — low-cardinality fields (few dozen distinct values).
- **STL_SORT** — sorted numeric index for range queries.
- **Trie** (marisa-trie) — prefix match on Varchar.
- **NGRAM** (2.6) — accelerates `LIKE '%foo%'`.
- **RTREE** (2.6) — GEOMETRY fields.
- **JSON Path Index** (2.6) — inverted index on specific JSON paths; documented latency improvement from 140 ms to 1.5 ms median.
- **AUTOINDEX** — picks by distribution.

**Filter strategy.** Milvus compiles the filter expression into a **bitset** (one bit per row) using scalar indexes, then passes it to the vector index (HNSW/DiskANN/IVF/CAGRA via Knowhere). During graph traversal any candidate whose bit is 0 is skipped when forming the top-k heap. This is **pre-filter + in-graph skip**, not post-filter.

Two refinements:

- **Brute-force fallback** — if bitset cardinality drops below ~10% of valid rows, Knowhere's HNSW abandons graph traversal and exhaustively scans the matching rows. Known pain point: the threshold is a fraction, not an absolute count, so "10% of 20 M vectors" is still a 2 M brute-force scan (see [milvus#29723](https://github.com/milvus-io/milvus/issues/29723)).
- **Iterative Filter** (2.4.3) — VBase-inspired lazy evaluation for expensive predicates (regex, un-indexed JSON paths). Traverses the graph in batches and evaluates the predicate per candidate on-the-fly.

**Expression example.**
```
filter = "age > 25 AND city IN ['Beijing','Shanghai'] AND product['price'] < 1850 AND json_contains(product['tags'], 'sale')"
```

**Recent changes.** 2.5 added sparse-BM25 full-text. 2.6 added JSON Path Index, JSON Shredding (auto-column JSON sub-fields), NGRAM, RTREE, TIMESTAMPTZ.

### 2.2 Qdrant — the most sophisticated filtered-HNSW implementation

**Payload storage.** Schema-less JSON. Three backends (`SimplePayloadStorage` on RocksDB, `OnDiskPayloadStorage` direct RocksDB, `MmapPayloadStorage`). Version 1.16 replaces RocksDB with **Gridstore** (appendable) and mmap (immutable segments).

**Payload indexes.** Eight typed indexes with configuration flags (`on_disk`, `is_tenant`, `is_principal`, `enable_hnsw`):

| Type | Structure | Filters |
|---|---|---|
| keyword | `MapIndex<String>` posting lists | `Match`, `MatchAny`, `MatchExcept` |
| integer | `NumericIndex<i64>` | `Match` and/or `Range` |
| float | `NumericIndex<f64>` | `Range` |
| bool | `BoolIndex` | `Match` |
| geo | `GeoMapIndex` (geohash) | `GeoBoundingBox`, `GeoRadius`, `GeoPolygon` (with holes) |
| datetime | numeric | `Range` |
| text | `FullTextIndex` (tokenized) | `MatchText`, phrase |
| uuid | `MapIndex<Uuid>` | `Match` |

**Filter language.** Nested JSON with `must`/`should`/`must_not` and leaf conditions including `nested` (which pins sub-conditions to the same array element — important for correctness in arrays of objects).

```json
{
  "must":     [{"key": "city",  "match": {"value": "London"}},
               {"key": "price", "range": {"gte": 100, "lte": 450}}],
  "should":   [{"key": "tags",  "match": {"any": ["sale","new"]}}],
  "must_not": [{"is_empty": {"key": "stock"}}]
}
```

**Filterable HNSW — the core innovation.** Qdrant's engineers analyzed filtered HNSW via **percolation theory**: a graph with average degree `<k>` stays connected when the retained fraction of nodes `p > 1/<k>`. Below this threshold the filtered subgraph fragments into disconnected components and recall collapses.

Their solution: for each indexed payload value, build a separate HNSW subgraph over only the matching points, then **merge those subgraph edges back into the global HNSW graph as additional links**. Every node carries (a) normal HNSW neighbors and (b) payload-aware neighbors sharing one of its indexed values. Two config parameters:

- `hnsw_config.m` — edges in the global graph (default 16).
- `hnsw_config.payload_m` — edges added per indexed payload value (typical 16).

Setting `m=0, payload_m>0` disables cross-tenant global edges entirely and creates **per-tenant subgraphs** — Qdrant's native multi-tenancy primitive.

**Query planner (4-way).** For each segment the planner calls `query_estimator` to compute cardinality from payload indexes, then picks:

| Matches | Strategy |
|---|---|
| Segment ≤ `full_scan_threshold_kb` (default 10 000 kB) | Full scan |
| Low cardinality + payload index | Payload-index-first |
| High cardinality, high selectivity | Filterable HNSW with on-the-fly filter check |
| High cardinality, **low selectivity with weak filters** | **ACORN second-hop expansion** (1.16) |

Because indexes provide exact counts (posting-list lengths, interval counts), the cardinality estimator is tight.

**ACORN in Qdrant 1.16.** When estimated `matches/total < max_selectivity` (default 0.4), the traversal inspects second-hop neighbors if first-hop neighbors fail the filter. Reported numbers: recall **97.2% vs 53.3%** at the cost of ~10× latency (1.25 ms → 13.86 ms). Tuning knob to disable (0.0) or always-on (1.0).

**Known recall cliffs.** Unindexed filter fields lose cardinality estimation; float equality suffers rounding; highly restrictive filters fall back to full scan deliberately.

### 2.3 Weaviate — LSM inverted index + ACORN

**Storage (since v1.5).** Per shard, three stores: object store, inverted index, HNSW vector index. Object store and inverted index are **LSM-tree-backed** (memtable → sorted immutable segments, Bloom-filter-guarded reads, background compaction). Vectors are not segmented.

**Inverted index.** Per property, up to three indexes:

- `indexSearchable` — BM25 keyword map (text only).
- `indexFilterable` — **Roaring Bitmap** per value (equality/set match).
- `indexRangeFilters` — Roaring Bitmap tuned for numeric ranges.

The bitmap index is implemented as **RoaringSet**, natively embedded in the LSM store. Updates are append-only (`additions` + `deletions` bitmaps per segment) to avoid read-modify-write amplification. At query time a `SegmentNode` returns a Roaring Bitmap pointing directly at mmap bytes; worst case a single copy. Reading a 90 M-id bitmap is reported in single-digit milliseconds.

**Filter pipeline.** A `where` predicate compiles into an AND/OR of per-segment roaring bitmaps, unioned into a single **allow-list of uint64 doc IDs** handed to HNSW.

**Strategy.**

- **Sweeping** (legacy): traverse HNSW normally, check allow-list when admitting a candidate. Problem: low correlation wastes distance computations; the filtered subgraph fragments.
- **ACORN** (default in **1.34**, opt-in 1.27+): Weaviate's adaptation of the Stanford ACORN paper. Three moving parts:
  1. **Conditional two-hop expansion** — when a neighbor fails the filter, use its neighbors' neighbors. Treats filtered-out nodes as transparent bridges.
  2. **No re-indexing required** — unlike the paper, Weaviate does *not* densify the graph at build time. Keeps RAM flat and lets existing deployments enable ACORN without rebuilding.
  3. **Seeded entry points** — extra random layer-0 entry points that match the filter, accelerating convergence into the filtered region.
  At 20% selectivity with low query/filter correlation ACORN is ~2× sweeping; on adversarial workloads the improvement is "an order of magnitude". Auto-falls-back to sweeping when filter-graph correlation is high.
- **Flat cutoff**: if the allow-list is below ~15% of shard size, bypass HNSW and exact-scan the allow-list.

**GraphQL filter example.**
```graphql
{ Get { Article(where: {
    operator: And,
    operands: [
      { path: ["wordCount"], operator: GreaterThan, valueInt: 1000 },
      { path: ["title"],     operator: Like,        valueText: "*economy*" }
    ]
  }) { title } } }
```

### 2.4 Pinecone — serverless bitmap slabs

**Metadata model.** Flat JSON map with strict value types: string, number (64-bit float; ints coerced), boolean, **list of strings**. No nested objects. No null (omit the key). Keys cannot start with `$`. Hard cap: **40 KB per record**.

**Filter language.** MongoDB-style: `$eq`, `$ne`, `$gt[e]`, `$lt[e]`, `$in` (max 10 000 values), `$nin`, `$exists`, `$and`, `$or` (only top-level). No LIKE/regex; no nested path dotting.

**Internal architecture (serverless).** Per Pinecone's engineering writeups, vector data is stored as immutable **"slabs"** in object storage, organized LSM-tree style. Each slab carries a **metadata folder containing one roaring bitmap per field/value**, directly adapted from data-warehouse bitmap indexes. The filter evaluator unions/intersects bitmaps at query time.

**Strategy.** Pinecone explicitly states it **never post-filters** (would risk returning fewer than *k* results). Two branches:

- **Pre-filter** — scan only records matching the bitmap. Used for selective filters.
- **Mid-scan filtering** — ANN traversal with inline bitmap check. Used for broad filters.

Their ICML 2025 paper ("Accurate and Efficient Metadata Filtering in Pinecone's Serverless Vector Database") formalizes this as "ad-hoc application vs pre-computed filter representation" and reports exact filter recall. Exact thresholds for auto-switching are undisclosed.

**Namespaces vs metadata.** Pinecone's explicit guidance: use **namespaces for per-tenant isolation** (physically separate slabs) and metadata filters for cross-tenant attribute queries. Their cost example shows a tenant reachable via a 1 GB namespace costs 1 RU, but via metadata filter over a 100 GB shared namespace costs ~100 RU.

**Filter example.**
```json
{ "$and": [
    { "genre":   { "$eq": "drama" } },
    { "year":    { "$gte": 2020 } },
    { "tags":    { "$in": ["oscar","cannes"] } }
] }
```

### 2.5 ChromaDB

**Storage.** Single-node Chroma uses `chroma.sqlite3` as the source of truth for structured data, plus per-collection HNSW binary files (hnswlib-fork). Metadata in SQLite `embedding_metadata` table. Document text indexed via FTS5 in `embedding_fulltext_search`. DuckDB was dropped in early 0.4.

**Filter syntax.** Two arguments:

- `where` — metadata, MongoDB-style (`$eq`, `$ne`, `$gt[e]`, `$lt[e]`, `$in`, `$nin`, `$and`, `$or`).
- `where_document` — raw document text via FTS5 (`$contains`, `$not_contains`, plus `$and`/`$or`).

**Example.**
```python
collection.query(
    query_embeddings=[qvec],
    n_results=10,
    where={"$and": [{"category": {"$in": ["news","blog"]}},
                    {"year":     {"$gte": 2024}}]},
    where_document={"$contains": "vector database"},
)
```

**Strategy.** Pre-filter via SQLite + FTS5 → eligible ID set → hnswlib filtered search (allow-list callback). Simple; suffers the classic **overfiltering recall problem** — hnswlib's ef-bounded traversal can return fewer than *k* hits when the allow-list is sparse. No iterative-scan equivalent. The Rust-core rewrite ("Chroma Distributed") introduces a segmented architecture but has no detailed public internals paper yet.

### 2.6 pgvector

**Model.** Metadata = any Postgres column. Filtering is a standard SQL WHERE clause handled by the Postgres executor.

**Three query shapes.**
1. **Bitmap scan + post-sort by distance** — B-tree/bitmap on metadata → fetch matches → sort by distance operator. Exact, but O(matches · dim) and slow once >~10 K rows.
2. **HNSW index scan + in-filter** — driven by HNSW ORDER BY, filter checked per emitted candidate. Overfiltering risk pre-0.8: `hnsw.ef_search=40` + 10% selectivity → ~4 rows.
3. **Sequential scan + sort** — fallback.

**pgvector 0.8 (Oct 2024) — Iterative Index Scan.** New GUC `hnsw.iterative_scan`:

- `off` (default) — legacy.
- `strict_order` — strictly ordered by distance.
- `relaxed_order` — slight reordering for higher recall.

When enabled, pgvector keeps pulling HNSW candidates past `ef_search` until enough post-filter matches, bounded by `hnsw.max_scan_tuples` (default **20 000**) and `hnsw.scan_mem_multiplier` (default 1). Planner cost estimation was also updated so B-tree is more likely to win on selective filters.

**SQL example.**
```sql
SET hnsw.ef_search = 100;
SET hnsw.iterative_scan = 'relaxed_order';
SET hnsw.max_scan_tuples = 50000;

SELECT id, title
FROM documents
WHERE tenant_id = 42 AND created_at > '2026-01-01'
ORDER BY embedding <=> '[...]'::vector
LIMIT 10;
```

**Limitations.** (a) HNSW build needs graph to fit in `maintenance_work_mem`. (b) `relaxed_order` recall is still selectivity-sensitive. (c) Iterative scan cannot combine with some planner shapes (CTEs, certain joins). (d) No per-label graph — unlike Filtered-DiskANN.

---

## 3. Academic Literature

### 3.1 ACORN (SIGMOD 2024, Stanford)

Patel, Kraft, Guestrin, Zaharia. **"Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data."**

- **Core idea.** Extends HNSW with a predicate-agnostic construction: builds a denser graph (M × γ neighbors) and, at search time, performs **predicate subgraph traversal** — expands neighbors of neighbors on the fly, keeping only those that satisfy the filter. Emulates an "ideal" graph restricted to the passing set.
- **Result.** 2–1000× throughput at fixed recall vs prior baselines across diverse selectivities.
- **Tradeoff.** Larger index (γ expansion factor); no per-label specialization means worst-case queries can touch many nodes. Not free on disk-backed systems.

### 3.2 Filtered-DiskANN (WebConf 2023, Microsoft)

Gollapudi et al. **"Graph Algorithms for Approximate Nearest Neighbor Search with Filters."** Two label-aware Vamana variants:

- **FilteredVamana** — single graph; neighbor selection for vertex `v` considers both geometric distance and shared labels with v's labels, so the graph stays navigable when restricted to any label.
- **StitchedVamana** — small per-label Vamana graphs, unioned then re-pruned.

At recall@90 StitchedVamana is ~6× faster than IVF-inline and ~2× faster than FilteredVamana. Tradeoff: **labels must be known at build time**, very high-cardinality or open-ended predicates are unsupported, StitchedVamana has larger index and longer build.

### 3.3 NHQ (2023)

Wang et al. Builds a **composite graph over fusion vectors** (vector concatenated with encoded attributes) using a joint fusion distance, so filtered ANN reduces to plain ANN. Up to 315× speedup on curated datasets. Restricted to small sets of equality-style attribute constraints.

### 3.4 Recent surveys and benchmarks

- **"Survey of Filtered ANN over Vector-Scalar Hybrid Data"** (arXiv 2505.06501) proposes a **pruning-axis taxonomy**:
  - *Vector-Solely* — pure ANN, no filter awareness.
  - *Vector-Centric Joint* — ACORN, Filtered-DiskANN, NHQ.
  - *Scalar-Centric Joint* — Milvus partitioning.
  - *Scalar-Solely* — pre-filter + brute force.
- **"Attribute Filtering in ANN: An In-depth Experimental Study"** (arXiv 2508.16263) — benchmarks 12 methods across 0.1%–100% selectivity for both label and range filters.
- **"Filtered ANN: A Unified Benchmark"** (arXiv 2509.07789) splits methods into filter-then-search / search-then-filter / hybrid-search. No universal winner. Rough thresholds from the synthesis:
  - **< ~1% selectivity** → pre-filter + brute force dominates.
  - **> ~10% selectivity** → in-filter graph traversal wins.
  - **1–10% ambiguous band** → iterative / predicate-subgraph approaches matter most.
  Exact numbers are dataset- and dimension-dependent; the surveys do not name a universal cutoff.

---

## 4. Cross-Cutting Observations

### 4.1 How metadata is stored — JSON vs typed?

Both models are in production. The pattern:

- **User-facing API is usually JSON-like** (flexibility, schema evolution) — Qdrant, Pinecone, ChromaDB are fully schema-less; Milvus supports both strict + dynamic JSON; Weaviate and pgvector require schemas.
- **Physical index is always typed and columnar.** Even schema-less systems build per-field typed indexes (Qdrant's `MapIndex<String>`, `NumericIndex<i64>`, etc.) lazily when users declare a field as indexable.
- **Raw metadata payload** is stored per-row/per-point in a row-oriented blob: Pinecone's 40 KB cap, Milvus's 64 KB JSON cap, Qdrant's payload object. This blob is returned with search results but not used for filtering directly.

### 4.2 Pre-filter vs post-filter vs in-filter

Canonical taxonomy:

- **Pre-filter** — evaluate predicate first, produce candidate ID set, then do vector search only over that set.
- **Post-filter** — full ANN first, filter results afterward. **Unsound** if `k` results cannot be returned when the filter is selective. Largely abandoned by production systems.
- **In-filter** (aka "filter-while-search", "mid-scan") — evaluate predicate lazily during graph traversal, skipping candidates that fail.

Production systems almost universally combine **pre-filter + in-filter**: materialize a bitset/bitmap up front, then check it inline during HNSW traversal. Pure post-filter is explicitly rejected by Pinecone and missing from Milvus/Qdrant/Weaviate.

**The key nuance** is what happens when the filter makes the graph locally disconnected. This is where ACORN (two-hop expansion), Filtered-DiskANN (label-aware graph edges), and Qdrant's payload-aware edges diverge from each other but all converge on the same problem.

### 4.3 Selectivity-based strategy selection

Production vector DBs now agree that there is no single best strategy — selectivity governs. Rough unified picture:

| Selectivity | Best strategy | Rationale |
|---|---|---|
| < 1% | Pre-filter + brute force over the matching set | Graph structure wasteful; fewer distance computations than a fragmented HNSW walk |
| 1–10% | In-filter HNSW with **two-hop expansion** or **payload-aware edges** | Naive in-graph walk loses recall at percolation cliff |
| 10–40% | Standard HNSW + bitset check | Filter is cheap; graph is still well-connected |
| > 40% | Same as above; filter barely matters | Post-filter essentially equivalent |

Exact thresholds depend on HNSW `M`, dataset geometry, and predicate-vector correlation. Most systems expose a config knob (Milvus `iterative_filter`, Qdrant `max_selectivity`, pgvector `iterative_scan`, Weaviate flat cutoff) rather than hardcoding.

### 4.4 Recall cliffs

Every system has documented recall cliffs:

- **Milvus**: 10%-relative brute-force threshold is too coarse for billion-scale collections ([issue 29723](https://github.com/milvus-io/milvus/issues/29723)).
- **Qdrant**: unindexed filter fields lose cardinality estimation → conservative strategy → lower recall.
- **Weaviate**: low-correlation filters (e.g. time-partitioned queries where filter matches distant region) require ACORN or sweep with large ef.
- **pgvector pre-0.8**: overfiltering when ef_search × selectivity < k.
- **ChromaDB**: same overfiltering, no mitigation yet.

The problem is structural: HNSW was designed for unconstrained nearest-neighbor. Filters violate this assumption, and the mitigations (graph densification, payload-aware edges, two-hop expansion, iterative scan) are all variations on **"restore graph connectivity within the filtered subset"**.

---

## 5. Implications for LSM-Vec

LSM-Vec's architecture maps well to several of these designs:

- **RocksGraph (Aster) already gives us an LSM-backed segment substrate** — the same substrate Weaviate and Pinecone use for bitmap-backed inverted indexes.
- **Columnar per-field storage** aligns naturally with column families in Aster.
- **HNSW core with L0 edges on disk** is the right place to attach payload-aware edges (Qdrant's approach) if we want filterable HNSW.

### 5.1 Design directions to consider

**1. Storage.** JSON-like user-facing API backed by typed columnar indexes in RocksGraph column families. Payload blob per vector (row-oriented) stored separately for result retrieval; filtering uses the indexes.

**2. Index types.** At minimum:
- Keyword / string → inverted posting list (or roaring bitmap per value).
- Integer / float → sorted numeric index for range queries.
- Bool → single bitmap.
- Optional: geo, full-text (deferred).

Roaring bitmaps are the natural format because they (a) compose via AND/OR intersection, (b) embed naturally in LSM segments with `additions` / `deletions` sub-bitmaps per segment (Weaviate's RoaringSet pattern), (c) map neatly onto RocksGraph keys.

**3. Filter expression language.** Start with MongoDB-style operators (`$eq`, `$ne`, `$in`, `$gt/gte/lt/lte`, `$and`, `$or`) — this is the lowest common denominator and matches Pinecone/ChromaDB, which our users are most likely to be migrating from.

**4. Filter strategy.** Three-way planner:

| Estimated selectivity | Strategy |
|---|---|
| < N (absolute count, not percentage) | **Pre-filter + brute-force** over the matched ID set. Use SIMD distance. |
| Mid | **In-filter HNSW + iterative expansion** (pgvector-style `relaxed_order` + optional ACORN-style two-hop on filter miss) |
| Low selectivity (most match) | **Normal HNSW + inline bitmap check** |

Avoid Milvus's fractional threshold — use an **absolute count cutoff** (e.g. `if matches < 10 000 then brute-force`).

**5. Cardinality estimation.** Use exact counts from indexes (posting-list length, range bounds) to estimate per-predicate, then combine with union/intersection bounds (Qdrant's `query_estimator` approach). Much more reliable than Postgres-style cost-based estimation for our small codebase.

**6. Payload-aware edges (later).** Consider Qdrant-style extra HNSW edges per indexed field **only** if profiling shows the recall cliff is a real user problem. The cost is ~2× storage in RocksGraph L0 edges. Do not build this before profiling.

**7. Filterable HNSW without rebuild.** Weaviate's ACORN variant (two-hop expansion on filter miss + seeded entry points, **no graph densification at build time**) is the cheapest option to implement incrementally on top of an existing HNSW implementation. Recommended path for LSM-Vec.

### 5.2 Ordering of work

Realistic phasing (aligned with the P0 ranking in `PRODUCTION_READINESS_ANALYSIS.md`):

1. **Phase 1a — metadata storage + simple filter.** Add a payload column family in RocksGraph storing JSON blobs per vector ID. Add a pre-filter path that evaluates a predicate via sequential scan, returning an ID set. kNN over that ID set by brute force. Correctness first, no performance goal.
2. **Phase 1b — typed indexes.** Inverted index for string/keyword, sorted numeric for int/float, bitmap for bool. MongoDB-style expression parser.
3. **Phase 1c — in-filter HNSW.** Pass the predicate (or materialized bitset) into `searchKnn` traversal. Inline bit check per candidate.
4. **Phase 1d — selectivity planner.** Cardinality estimation from index stats. Absolute-count threshold for brute-force fallback.
5. **Phase 2 (later) — ACORN-style two-hop expansion.** Only if profiling shows recall cliffs at moderate selectivity.
6. **Phase 3 (much later) — payload-aware HNSW edges.** Only if multi-tenancy at scale demands it.

### 5.3 Size/shape defaults to adopt

Pragmatic defaults copied from prior art:

- **Payload size cap per vector** — 40 KB (Pinecone) as a safety rail.
- **`$in` max values** — 10 000 (Pinecone).
- **No nulls** — omit the key (Pinecone).
- **Typed values only at the storage layer**, but allow JSON at the API boundary (Milvus dynamic field pattern).
- **Absolute-count brute-force threshold** (configurable; start with 10 000 matches).

---

## 6. Open Questions

Items the survey could not fully resolve:

- **Exact percolation threshold.** The unified benchmark suggests 1%/10% bands, but these are dataset-dependent. We will need to profile on SIFT/GIST/DEEP before picking defaults.
- **Pinecone's internal pre-filter vs mid-scan threshold.** Not disclosed.
- **Milvus "Dual-Pool" terminology.** Appears in blog paraphrases but not primary sources.
- **ACORN γ factor cost on disk-backed systems.** The Stanford paper benchmarks in-memory. LSM-Vec's L0 edges on disk amplify the cost of a denser graph; this needs measurement before committing.
- **Qdrant's precise merging algorithm for payload-aware subgraphs into the global graph** (implementation detail in `lib/segment/src/index/hnsw_index/`).

---

## 7. Primary Sources

### Milvus
- [Milvus docs — Filtering Explained](https://milvus.io/docs/boolean.md), [Basic Operators](https://milvus.io/docs/basic-operators.md), [JSON Operators](https://milvus.io/docs/json-operators.md), [Array Operators](https://milvus.io/docs/array-operators.md), [Filtered Search](https://milvus.io/docs/filtered-search.md)
- [Scalar Index](https://milvus.io/docs/scalar_index.md), [INVERTED](https://milvus.io/docs/inverted.md), [BITMAP](https://milvus.io/docs/bitmap.md), [STL_SORT](https://milvus.io/docs/stl-sort.md), [JSON Path Index](https://milvus.io/docs/json-indexing.md)
- [How to Filter Efficiently Without Killing Recall (blog)](https://milvus.io/blog/how-to-filter-efficiently-without-killing-recall.md), [Milvus 2.4.3 Metadata Filtering](https://zilliz.com/blog/what-is-new-with-metadata-filtering-in-milvus), [Milvus 2.5 release](https://milvus.io/blog/introduce-milvus-2-5-full-text-search-powerful-metadata-filtering-and-more.md), [Milvus 2.6 release](https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md)
- [SIGMOD '21 Milvus paper](https://www.cs.purdue.edu/homes/csjgwang/pubs/SIGMOD21_Milvus.pdf)
- [Filter threshold issue #29723](https://github.com/milvus-io/milvus/issues/29723)

### Qdrant
- [Filtrable HNSW (article)](https://qdrant.tech/articles/filtrable-hnsw/), [Vector Search Filtering](https://qdrant.tech/articles/vector-search-filtering/), [Filterable HNSW course](https://qdrant.tech/course/essentials/day-2/filterable-hnsw/)
- [Qdrant 1.16 — Tiered Multitenancy](https://qdrant.tech/blog/qdrant-1.16.x/)
- [Indexing docs](https://qdrant.tech/documentation/concepts/indexing/), [Filtering docs](https://qdrant.tech/documentation/concepts/filtering/), [Multitenancy docs](https://qdrant.tech/documentation/manage-data/multitenancy/), [Optimizer docs](https://qdrant.tech/documentation/concepts/optimizer/)
- [Payload Indexing and Filtering (DeepWiki)](https://deepwiki.com/qdrant/qdrant/4-payload-indexing-and-filtering)
- [PR #6148 — mmap payload indexes off RocksDB](https://github.com/qdrant/qdrant/pull/6148)

### Weaviate
- [ACORN blog](https://weaviate.io/blog/speed-up-filtered-vector-search), [Weaviate 1.34 release](https://weaviate.io/blog/weaviate-1-34-release)
- [Filtering concepts](https://docs.weaviate.io/weaviate/concepts/filtering), [Inverted index config](https://docs.weaviate.io/weaviate/config-refs/indexing/inverted-index), [Storage architecture](https://docs.weaviate.io/weaviate/concepts/storage), [GraphQL where filters](https://docs.weaviate.io/weaviate/api/graphql/filters)
- [RoaringSet LSM (issue #2511)](https://github.com/weaviate/weaviate/issues/2511)

### Pinecone
- [How Pinecone Works](https://www.pinecone.io/how-pinecone-works/), [Evolving Pinecone architecture](https://www.pinecone.io/blog/evolving-pinecone-for-knowledgeable-ai/)
- [ICML 2025 — Accurate and Efficient Metadata Filtering](https://www.pinecone.io/research/accurate-and-efficient-metadata-filtering-in-pinecones-serverless-vector-database/), [OpenReview PDF](https://openreview.net/pdf?id=UXq4z6GGYP)
- [Filter by metadata](https://docs.pinecone.io/guides/search/filter-by-metadata), [Understanding metadata](https://docs.pinecone.io/guides/data/understanding-metadata), [Namespaces vs metadata](https://docs.pinecone.io/troubleshooting/namespaces-vs-metadata-filtering)

### ChromaDB / pgvector
- [ChromaDB Storage Layout (Cookbook)](https://cookbook.chromadb.dev/core/storage-layout/), [ChromaDB Filters (Cookbook)](https://cookbook.chromadb.dev/core/filters/), [Metadata Filtering (Docs)](https://docs.trychroma.com/docs/querying-collections/metadata-filtering), [DeepWiki](https://deepwiki.com/chroma-core/chroma)
- [pgvector 0.8.0 release](https://www.postgresql.org/about/news/pgvector-080-released-2952/), [pgvector README](https://github.com/pgvector/pgvector), [pgvector HNSW Index (DeepWiki)](https://deepwiki.com/pgvector/pgvector/5.1-hnsw-index), [Aurora pgvector 0.8 blog](https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql/), [HNSW filtering issue #259](https://github.com/pgvector/pgvector/issues/259)

### Academic
- [ACORN (arXiv 2403.04871)](https://arxiv.org/abs/2403.04871), [ACORN code](https://github.com/stanford-futuredata/ACORN)
- [Filtered-DiskANN PDF](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf), [WebConf '23 ACM link](https://dl.acm.org/doi/10.1145/3543507.3583552)
- [NHQ code](https://github.com/YujianFu97/NHQ)
- [Survey of Filtered ANN (arXiv 2505.06501)](https://arxiv.org/html/2505.06501v1), [Attribute Filtering Experimental Study (arXiv 2508.16263)](https://arxiv.org/html/2508.16263v1), [Filtered ANN Unified Benchmark (arXiv 2509.07789)](https://arxiv.org/html/2509.07789v1)
