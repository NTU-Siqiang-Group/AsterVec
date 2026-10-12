# AsterVec API Reference

This document covers the **engine library** API of AsterVec for both **C++** and
**Python** (`import astervec`) — the embeddable, in-process interface.

```cpp
#include "astervec_db.h"   // C++: the only header you need
```
```python
import astervec            # Python (pybind11 bindings)
```

> For the optional HTTP/REST server (`astervec_http`), see
> [HTTP_API.md](HTTP_API.md) instead.

---

## Table of Contents

- [C++ API](#c-api)
  - [1. Opening and Closing](#1-opening-and-closing)
  - [2. Inserting Vectors](#2-inserting-vectors)
  - [3. Searching](#3-searching)
  - [4. Retrieving, Updating, Deleting](#4-retrieving-updating-deleting)
  - [5. Metadata Payloads](#5-metadata-payloads)
  - [6. Bulk Build](#6-bulk-build)
  - [7. Configuration Reference](#7-configuration-reference)
  - [8. Types](#8-types)
- [Python API](#python-api)
  - [9. Installation](#9-installation)
  - [10. Opening and Closing](#10-opening-and-closing)
  - [11. Inserting Vectors](#11-inserting-vectors)
  - [12. Searching](#12-searching)
  - [13. Retrieving, Updating, Deleting](#13-retrieving-updating-deleting)
  - [14. Metadata Payloads](#14-metadata-payloads)
  - [15. Bulk Build](#15-bulk-build)
  - [16. Configuration Reference](#16-configuration-reference)
  - [17. Metadata Filter Operators](#17-metadata-filter-operators)
  - [18. Error Handling](#18-error-handling)
- [Build and Linking](#build-and-linking)
- [Concurrency Model](#concurrency-model)

> **Note on quantization.** Vectors are stored with 8-bit scalar quantization
> (SQ8): `Get`/`get` returns a dequantized vector that differs from the input by
> up to ~`range/255` per element, and distances/recall are computed on the
> quantized form.

---

# Concurrency Model

`AsterVecDB` is the supported concurrency boundary for embedded use. Normal
operations (`Insert`, `SearchKnn`, `Update`, `Delete`, and payload methods) are
caller-driven: run them from your application's worker pool when you want
parallelism. There is no global "engine thread count" for these methods.

Writes to the same real id are serialized by the DB layer; operations on
different ids and concurrent searches can proceed from different application
threads. `BulkBuild` is different: it is an initial-load-only operation with its
own internal parallelism (`BulkBuildOptions::num_threads` in C++,
`threads=...` in Python) and should run on an empty DB before incremental
updates begin.

For the optional HTTP server, `ASTERVEC_HTTP_THREADS` controls request-level
concurrency. `/v1/build/bulk` can additionally use `X-AsterVec-Threads` for the
bulk-build operation itself.

---

# C++ API

All types live in the `astervec` namespace. Mutating/querying methods return a
`Status` (aliased from RocksDB); call `.ok()` and `.ToString()`.

## 1. Opening and Closing

```cpp
#include "astervec_db.h"
using namespace astervec;

AsterVecDBOptions opts;
opts.dim = 128;                              // required
opts.vector_file_path = "./db/vectors.bin";
opts.reinit = true;                          // true = start fresh

std::unique_ptr<AsterVecDB> db;
Status s = AsterVecDB::Open("./db", opts, &db);
if (!s.ok()) { /* handle s.ToString() */ }

// ... use the database ...
db->Close();
```

### `AsterVecDB::Open`

```cpp
static Status Open(const std::string& path,
                   const AsterVecDBOptions& opts,
                   std::unique_ptr<AsterVecDB>* db);
```

Creates or opens a database directory. `opts.dim` must be > 0. On success `*db`
holds the handle.

### `AsterVecDB::Close`

```cpp
Status Close();
```

Checkpoints (see below) and releases resources. The handle is unusable afterward.

### Point ids

`opts.id_mode` decides how callers identify points. It takes effect when a store is
created and is recorded in it; opening uses the recorded mode, and asking for a
different one returns `InvalidArgument`.

| `IdMode` | Ids | Notes |
|---|---|---|
| `kAuto` (default) | — | Open: the store's mode. Create: `kDense`. |
| `kDense` | integers `0 .. 2^63-1` | The id is the internal slot; keep ids compact. |
| `kKeyed` | any `uint64_t` or string (e.g. a UUID) | The engine maps ids to internal slots itself. |

Every point call takes either a plain integer or a `PointId`:

```cpp
db->Insert(PointId::Str("3f2b9c1e-8d4a-4e7b-9c1d-2a6f5e8b7c90"), vec, R"({"src":"a"})");
db->Insert(PointId::Int(18446744073709551615ULL), vec);   // keyed stores only
db->Delete(PointId::Str("doc-17#chunk-3"));
```

The integer `7` and the string `"7"` are different points.

### Checkpoints and crash recovery

```cpp
Status Checkpoint();                  // make everything written so far durable
RecoveryStats GetRecoveryStats() const;
```

A checkpoint syncs the vector data and both write-ahead logs, then atomically
installs the index metadata; the previous checkpoint is kept and used
automatically if the latest one is damaged. Writes wait while it runs (searches
do not). `flushVectorWrites()`, `BulkBuild()` and `Close()` checkpoint, and
`opts.checkpoint_interval_ops` checkpoints every N writes.

After an unclean shutdown the store reopens to its last checkpoint. With
`opts.enable_oplog = true`, every acknowledged insert, update and delete is also
appended to an operation log and replayed on open, so no acknowledged write is
lost to a process crash. `GetRecoveryStats()` reports what the last open
replayed.

---

## 2. Inserting Vectors

```cpp
std::vector<float> vec(128, 0.5f);
db->Insert(42, vec);                                   // vector only
db->Insert(43, vec, R"({"category":"docs"})");         // with JSON metadata
```

```cpp
Status Insert(node_id_t id, Span<float> vec);
Status Insert(node_id_t id, Span<float> vec, std::string_view metadata_json);
Status Insert(const PointId& id, Span<float> vec, std::string_view metadata_json = {});
```

Inserts a vector and builds its HNSW connections; inserting an existing id
replaces it. `vec` length must equal
`opts.dim`. The optional `metadata_json` is a JSON object stored as the vector's
payload. Returns `InvalidArgument` on a dimension mismatch or malformed metadata.
A `std::vector<float>` converts implicitly to `Span<float>`.

---

## 3. Searching

```cpp
SearchOptions so;
so.k = 10;
so.ef_search = 128;

std::vector<SearchResult> out;
db->SearchKnn(query, so, &out);                         // plain k-NN
db->SearchKnn(query, so, R"({"category":{"$eq":"docs"}})", &out);  // filtered
```

```cpp
Status SearchKnn(Span<float> query, const SearchOptions& options,
                 std::vector<SearchResult>* out);
Status SearchKnn(Span<float> query, const SearchOptions& options,
                 std::string_view filter_json, std::vector<SearchResult>* out);
Status SearchKnn(Span<float> query, std::vector<SearchResult>* out);  // uses opts.k/ef_search
```

Results are sorted by ascending distance, up to `k` entries. `filter_json` is a
Mongo-style predicate (see [§17](#17-metadata-filter-operators)). The last overload
uses the `k` and `ef_search` set on `AsterVecDBOptions` at open time. A search with
nothing to return (empty index, no point matching the filter) returns `OK` with
an empty `out`.

```cpp
Status SearchPoints(Span<float> query, const SearchOptions& options,
                    std::string_view filter_json,     // "" = no filter
                    std::vector<PointHit>* out);
```

`SearchPoints` returns each hit's `PointId`. Use it in keyed stores with string
ids; `SearchKnn` reports integer ids only.

---

## 4. Retrieving, Updating, Deleting

```cpp
Status Get(node_id_t id, std::vector<float>* vec);   // resizes to dim; NotFound if absent
Status Update(node_id_t id, Span<float> vec);        // replace vector (inserts if absent)
Status Delete(node_id_t id);                          // soft delete (tombstone)
std::size_t VectorCount() const;                      // live vector count
```

Each call also takes a `PointId`. `Delete` tombstones the id (excluded from future
searches; survives reopen); deleting an id that was never written is a no-op.
`VectorCount` is exact, including after a restart from a checkpoint.

---

## 5. Metadata Payloads

```cpp
Status GetPayload(node_id_t id, std::string* out_json);
Status SetPayload(node_id_t id, std::string_view metadata_json);     // full replace
Status UpdatePayload(node_id_t id, std::string_view partial_json);   // RFC 7396 merge
Status DeletePayloadKeys(node_id_t id, Span<const std::string> keys);
```

Each call also takes a `PointId`. `UpdatePayload` merge-patches: keys in
`partial_json` overwrite, and a key set to `null` deletes it. `Delete(id)` removes
the vector and its payload together.

---

## 6. Bulk Build

The fastest way to populate an **empty** database: build the whole index in memory
(RNN-Descent), then write it in one pass. IDs are assigned `0..n-1`, or taken
from `ids` (keyed stores; one distinct id per vector).

```cpp
Status BulkBuild(Span<const float> vectors, int n, const BulkBuildOptions& opts);
Status BulkBuild(Span<const float> vectors, const std::vector<PointId>& ids,
                 const BulkBuildOptions& opts);
```

```cpp
struct BulkBuildOptions {
    int num_threads = 0;   // 0 = auto (min(4, hardware_concurrency))
    int rnnd_S  = 16;      // initial random neighbours
    int rnnd_T1 = 4;       // outer iterations
    int rnnd_T2 = 15;      // inner iterations per outer
    int rnnd_R  = 64;      // pool cap / max output degree (truncated to m_max on write)
};
```

`vectors` is `n * dim` contiguous floats (vector `i` at `[i*dim, (i+1)*dim)`).
Requires an empty DB — returns `InvalidArgument` otherwise. For incremental updates
on a non-empty DB, use `Insert` (single calls or a thread pool; the engine is
thread-safe).

---

## 7. Configuration Reference

### AsterVecDBOptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | `int` | `0` | **Required.** Vector dimensionality. |
| `metric` | `DistanceMetric` | `kL2` | `kL2` or `kCosine`. |
| `m` | `int` | `8` | HNSW links per node at layer 0. |
| `m_max` | `int` | `24` | HNSW max neighbors at upper layers. |
| `m_level` | `int` | `1` | Level multiplier for random level generation. |
| `ef_construction` | `float` | `32.0` | Candidate pool during construction. |
| `ef_search` | `int` | `128` | Default candidate pool during search. |
| `k` | `int` | `1` | Default neighbors for the no-options search overload. |
| `vec_file_capacity` | `size_t` | `100000` | Initial vector-file capacity (auto-expands when paged). |
| `paged_max_cached_pages` | `size_t` | `8192` | 4 KB pages in the user-space page cache. |
| `edge_cache_size` | `size_t` | `100000` | In-memory graph edge cache entries. |
| `db_target_size` | `uint64_t` | `~100 GiB` | Aster (RocksDB) target file size hint. |
| `random_seed` | `int` | `12345` | RNG seed for HNSW level generation. |
| `enable_stats` | `bool` | `false` | Collect I/O and timing statistics. |
| `enable_batch_read` | `bool` | `true` | Group vector reads by page during search. |
| `reinit` | `bool` | `false` | `true` = wipe existing data; `false` = reopen. |
| `vector_file_path` | `string` | `""` | Path to the vector data file. |
| `log_file_path` | `string` | `""` | Path to the log file (empty = no file logging). |
| `id_mode` | `IdMode` | `kAuto` | Point-id mode, fixed at create time (see [§1](#point-ids)). |
| `enable_oplog` | `bool` | `false` | Log every write so acknowledged writes survive a process crash. |
| `checkpoint_interval_ops` | `uint64_t` | `0` | Checkpoint every N inserts/updates/deletes (`0` = only on flush/close). |
| `oplog_segment_bytes` | `uint64_t` | `64 MiB` | Operation-log segment size. |

**Tuning:** higher `m` / `ef_construction` → better recall, slower build; higher
`ef_search` → better recall, slower queries; higher `paged_max_cached_pages` → more
RAM, fewer disk reads.

---

## 8. Types

```cpp
enum class DistanceMetric { kL2, kCosine };   // Euclidean / 1 - cosine_similarity
using node_id_t = std::uint64_t;              // 64-bit vector identifier

struct SearchOptions {
    int k = 1;                    // neighbors to return
    int ef_search = 128;          // candidate pool (>= k); higher = better recall
    int max_scan_candidates = 0;  // cap when a filter is set; 0 = auto (k * 50)
};

struct SearchResult { node_id_t id; float distance; };

enum class IdMode { kAuto, kDense, kKeyed };
class PointId {                               // an integer or a string id
  public:
    static PointId Int(uint64_t v);
    static PointId Str(std::string s);
    bool is_int() const;  uint64_t as_int() const;  const std::string& as_str() const;
};
struct PointHit { PointId id; float distance; };   // SearchPoints result
```

`Span<T>` is a lightweight non-owning view over contiguous memory; a
`std::vector<float>` converts implicitly, so you rarely construct one by hand.

---

# Python API

## 9. Installation

```bash
git submodule update --init --recursive
make aster
python -m pip install .
python -c "import astervec; print('OK')"
```

The Python package is published on PyPI as **`aster-vec`** and imported as
`import astervec`. This is the embedded engine module; for the optional REST
server, see [HTTP_API.md](HTTP_API.md).

---

## 10. Opening and Closing

```python
import os, astervec

opts = astervec.AsterVecDBOptions()
opts.dim = 128
opts.vector_file_path = "./db/vectors.bin"
opts.reinit = True

os.makedirs("./db", exist_ok=True)
db = astervec.AsterVecDB.open("./db", opts)
# ...
db.close()
```

`AsterVecDB.open(path: str, opts: AsterVecDBOptions) -> AsterVecDB` — raises `ValueError`
on invalid arguments, `RuntimeError` on I/O errors.

A store created from Python accepts `int`, `str` and `uuid.UUID` ids (keyed mode)
unless `opts.id_mode = astervec.IdMode.Dense`; an existing store keeps the mode it
was created with (`db.id_mode` is `"keyed"` or `"dense"`).

---

## 11. Inserting Vectors

```python
db.insert(0, [0.1] * 128)                                  # list
db.insert(1, np.random.rand(128).astype(np.float32))       # numpy
db.insert(2, [0.1] * 128, metadata={"category": "docs"})   # with metadata
```

```python
db.insert(id: int | str | uuid.UUID, vector: list[float] | np.ndarray,
          metadata: dict | None = None) -> None
```

Integer ids range over `0 .. 2**64-1`; inserting an existing id replaces it; the
integer `7` and the string `"7"` are different points. `vector` length must equal
`opts.dim`. `metadata` is any JSON-serializable object
(serialized internally). Raises `ValueError` on a dimension mismatch.

---

## 12. Searching

```python
# k-NN → list[SearchResult(id, distance)]
results = db.search_knn([0.1] * 128, k=10, ef_search=128)
results = db.search_knn([0.1] * 128, search_opts)          # SearchOptions object
results = db.search_knn([0.1] * 128)                        # uses opts.k / opts.ef_search

# filtered k-NN → list[dict] of {"id", "distance"}
hits = db.search([0.1] * 128, k=10, filter={"category": "docs"})
```

```python
db.search_knn(query, opts: SearchOptions) -> list[SearchResult]
db.search_knn(query, k: int, ef_search: int) -> list[SearchResult]
db.search_knn(query) -> list[SearchResult]
db.search(query, k=10, ef_search=128, filter=None, max_scan_candidates=0) -> list[dict]
```

`query` is a list or 1-D float32 numpy array. `search_knn` returns `SearchResult`
objects; `search` accepts a Mongo-style `filter` dict (see
[§17](#17-metadata-filter-operators)) and returns plain dicts (convenient for
pandas). `max_scan_candidates` bounds the filtered scan (`0` = auto, `k * 50`).

`search` returns each id as it was inserted (`int` or `str`; a `uuid.UUID` comes
back as its string form). `search_knn` reports integer ids only, so use `search`
with string ids. A search with nothing to return gives `[]`.

---

## 13. Retrieving, Updating, Deleting

```python
vec = db.get(id)            # -> np.ndarray (float32); raises KeyError if absent
db.update(id, vector)       # replace vector (inserts if absent)
db.delete(id)               # soft delete; no-op for an unknown id
```

---

## 14. Metadata Payloads

```python
db.set_payload(1, {"category": "docs", "price": 79})   # full replace
db.update_payload(1, {"price": 69})                    # merge-patch (RFC 7396)
db.update_payload(1, {"stale": None})                  # None deletes a field
db.delete_payload_keys(1, ["temp", "debug"])           # remove specific keys
md = db.get_payload(1)                                  # -> dict ({} if none)
```

---

## 15. Bulk Build

```python
import numpy as np
report = db.bulk_build(np.random.rand(100_000, 128).astype(np.float32), threads=4)
# -> {"n": 100000, "elapsed_ms": ..., "vectors_per_sec": ..., "threads": 4}
```

```python
db.bulk_build(vectors: np.ndarray, threads: int = 0, ids: list | None = None) -> dict
```

`vectors` is a 2-D `(n, dim)` float32 array (or a list of equal-length rows).
**Initial-load only**: the DB must be empty; ids are assigned `0..n-1` unless
`ids` gives one distinct `int`/`str`/`uuid.UUID` per row (keyed stores). For
incremental updates afterward use `insert`.

---

## 16. Configuration Reference

### AsterVecDBOptions

All properties are read-write.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `dim` | `int` | `0` | **Required.** Vector dimensionality. |
| `metric` | `DistanceMetric` | `L2` | `astervec.L2` or `astervec.Cosine`. |
| `m` | `int` | `8` | HNSW links per node (layer 0). |
| `m_max` | `int` | `24` | Max neighbors at upper layers. |
| `m_level` | `int` | `1` | Level multiplier. |
| `ef_construction` | `float` | `32.0` | Construction candidate pool. |
| `ef_search` | `int` | `128` | Default search candidate pool. |
| `k` | `int` | `1` | Default neighbors for the bare `search_knn`. |
| `vec_file_capacity` | `int` | `100000` | Initial vector-file capacity. |
| `paged_max_cached_pages` | `int` | `8192` | Page cache size (4 KB pages). |
| `db_target_size` | `int` | `107374182400` | Aster target file size (bytes). |
| `random_seed` | `int` | `12345` | RNG seed. |
| `enable_stats` | `bool` | `False` | Collect statistics. |
| `enable_batch_read` | `bool` | `True` | Batch vector reads by page. |
| `reinit` | `bool` | `False` | Wipe existing data on open. |
| `vector_file_path` | `str` | `""` | Vector storage file path. |
| `log_file_path` | `str` | `""` | Log file path. |
| `id_mode` | `IdMode` | `Auto` | `Auto`: new stores keyed, existing keep theirs; `Dense`; `Keyed`. |
| `enable_oplog` | `bool` | `False` | Log every write so acknowledged writes survive a process crash. |
| `checkpoint_interval_ops` | `int` | `0` | Checkpoint every N inserts/updates/deletes (`0` = only on flush/close). |
| `oplog_segment_bytes` | `int` | `67108864` | Operation-log segment size. |

### SearchOptions

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `int` | `1` | Number of nearest neighbors. |
| `ef_search` | `int` | `128` | Candidate pool (>= k). |
| `max_scan_candidates` | `int` | `0` | Cap when a filter is set; `0` = auto (`k * 50`). |

### DistanceMetric

```python
astervec.DistanceMetric.L2      # Euclidean (L2) distance
astervec.DistanceMetric.Cosine  # 1 - cosine similarity
```

### IdMode

```python
astervec.IdMode.Auto    # default: new stores keyed, existing stores keep their mode
astervec.IdMode.Dense   # integer ids used directly as slots
astervec.IdMode.Keyed   # int, str or uuid.UUID ids
```

### Other methods

| Method | Description |
|--------|-------------|
| `db.checkpoint()` | Make everything written so far durable (raises on failure). |
| `db.flush_vector_writes()` | Same as `checkpoint()`, but logs errors instead of raising. |
| `db.recovery_stats() -> dict` | What the last open replayed from the operation log. |
| `db.trim_memory()` | Ask the allocator to return idle heap to the OS (glibc `malloc_trim`; no-op elsewhere). |
| `db.delete_stats() -> dict` | Tombstone/Bloom observability counters. |

---

## 17. Metadata Filter Operators

Filters are JSON predicate trees (passed as a Python dict or JSON string).

| Operator | Meaning |
|---|---|
| `$eq`, `$ne` | equals / not equals |
| `$gt`, `$gte`, `$lt`, `$lte` | numeric comparison |
| `$in`, `$nin` | value is / isn't in the array |
| `$exists` | key present — boolean (`{"$exists": false}` = absent) |
| `$contains_any`, `$contains_all` | array contains any / all of the values |
| `$and`, `$or` | logical combination (arrays of sub-predicates) |

A bare scalar is an implicit `$eq` (`{"category": "docs"}`). Nested keys use dot
paths (`{"author.name": {"$eq": "ann"}}`). There is no standalone `$not` — negate
at the field level (`$ne`, `$nin`, `$exists: false`).

```python
db.search(query, k=10, filter={
    "$and": [
        {"category": {"$eq": "docs"}},
        {"price": {"$lt": 100}},
    ],
})
```

---

## 18. Error Handling

| C++ `Status` | Python exception | When |
|--------------|------------------|------|
| `InvalidArgument` / `NotSupported` | `ValueError` | dimension mismatch, bad params, unsupported metric |
| `NotFound` | `KeyError` | id not found (`get`, `set_payload`, `update_payload`); searches return `[]` instead |
| other | `RuntimeError` | I/O / storage errors |

---

# Build and Linking

```bash
git submodule update --init --recursive
make aster      # build Aster (RocksDB fork) → lib/aster/librocksdb.a
make            # build libastervec.a, libastervec.so/.dylib, and binaries
```

Link against the AsterVec library plus Aster's RocksDB and zstd (the only required
codec — snappy/lz4/bz2/zlib are disabled in the Aster build):

```bash
g++ -std=c++17 -O2 -Iinclude my_app.cc \
    -Lbuild/lib -lastervec \
    -Llib/aster -lrocksdb -lzstd -lpthread -ldl \
    -o my_app
```

On macOS, also add `-ljemalloc`.

### Python package

```bash
make aster
python -m pip install .   # compiles the extension via scikit-build-core
```
