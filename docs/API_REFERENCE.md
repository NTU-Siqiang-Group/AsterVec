# LSM-Vec API Reference

This document covers the public APIs of the LSM-Vec library for both **C++** and **Python**.

Only one header is needed for C++:

```cpp
#include "lsm_vec_db.h"
```

For Python:

```python
import lsm_vec
```

---

## Table of Contents

- [C++ API](#c-api)
  - [1. Opening and Closing a Database](#1-opening-and-closing-a-database)
  - [2. Inserting Vectors](#2-inserting-vectors)
  - [3. Searching](#3-searching)
  - [4. Retrieving, Updating, and Deleting](#4-retrieving-updating-and-deleting)
  - [5. Configuration Reference](#5-configuration-reference)
  - [6. Types](#6-types)
- [Python API](#python-api)
  - [7. Installation](#7-installation)
  - [8. Opening and Closing a Database](#8-opening-and-closing-a-database)
  - [9. Inserting Vectors](#9-inserting-vectors)
  - [10. Searching](#10-searching)
  - [11. Retrieving, Updating, and Deleting](#11-retrieving-updating-and-deleting)
  - [12. Configuration Reference](#12-configuration-reference)
  - [13. Error Handling](#13-error-handling)
- [Build and Linking](#build-and-linking)

---

# C++ API

All types live in the `lsm_vec` namespace. Every mutating or querying method
returns a `Status` object (aliased from RocksDB). Call `.ok()` to check for
success and `.ToString()` for an error message.

## 1. Opening and Closing a Database

```cpp
#include "lsm_vec_db.h"
using namespace lsm_vec;

// Configure
LSMVecDBOptions opts;
opts.dim = 128;                              // required
opts.vector_file_path = "./db/vectors.bin";  // required
opts.reinit = true;                          // true = start fresh

// Open
std::unique_ptr<LSMVecDB> db;
Status s = LSMVecDB::Open("./db", opts, &db);
if (!s.ok()) { /* handle error */ }

// ... use the database ...

// Close
db->Close();
```

### `LSMVecDB::Open`

```cpp
static Status Open(const std::string& path,
                   const LSMVecDBOptions& opts,
                   std::unique_ptr<LSMVecDB>* db);
```

Creates or opens a database at the given directory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `const std::string&` | Directory for database files. Created automatically if it does not exist. |
| `opts` | `const LSMVecDBOptions&` | Configuration. `opts.dim` must be > 0. |
| `db` | `std::unique_ptr<LSMVecDB>*` | On success, `*db` holds the opened database handle. |

### `LSMVecDB::Close`

```cpp
Status Close();
```

Flushes pending writes and releases all resources. The handle is unusable after
this call.

---

## 2. Inserting Vectors

```cpp
std::vector<float> vec(128, 0.5f);
db->Insert(42, Span<float>(vec));
```

### `LSMVecDB::Insert`

```cpp
Status Insert(node_id_t id, Span<float> vec);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `node_id_t` (`uint64_t`) | Unique identifier for this vector. |
| `vec` | `Span<float>` | Vector data. Length must equal `opts.dim`. A `std::vector<float>` converts implicitly. |

Inserts a new vector and builds its HNSW graph connections. Returns
`InvalidArgument` if the dimension mismatches.

---

## 3. Searching

```cpp
std::vector<float> query(128, 0.1f);

SearchOptions search_opts;
search_opts.k = 10;           // number of neighbors
search_opts.ef_search = 128;  // candidate pool size (higher = better recall)

std::vector<SearchResult> results;
db->SearchKnn(Span<float>(query), search_opts, &results);

for (const auto& r : results) {
    std::cout << "id=" << r.id << " dist=" << r.distance << "\n";
}
```

### `LSMVecDB::SearchKnn`

```cpp
Status SearchKnn(Span<float> query,
                 const SearchOptions& options,
                 std::vector<SearchResult>* out);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `Span<float>` | Query vector. Length must equal `opts.dim`. |
| `options` | `const SearchOptions&` | Search parameters (see below). |
| `out` | `std::vector<SearchResult>*` | Results sorted by ascending distance, up to `k` entries. |

### SearchOptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `k` | `int` | `1` | Number of nearest neighbors to return. |
| `ef_search` | `int` | `64` | Candidate pool size during search. Must be >= `k`. Higher values improve recall at the cost of latency. |

### SearchResult

| Field | Type | Description |
|-------|------|-------------|
| `id` | `node_id_t` | Identifier of the matched vector. |
| `distance` | `float` | Distance from the query vector. |

---

## 4. Retrieving, Updating, and Deleting

### `LSMVecDB::Get`

```cpp
Status Get(node_id_t id, std::vector<float>* vec);
```

Retrieves the vector for a given ID. `vec` is resized to `opts.dim` on success.
Returns `NotFound` if the ID does not exist.

### `LSMVecDB::Update`

```cpp
Status Update(node_id_t id, Span<float> vec);
```

Replaces the vector data for an existing ID and rebuilds its graph connections.
Returns `NotFound` if the ID does not exist.

### `LSMVecDB::Delete`

```cpp
Status Delete(node_id_t id);
```

Marks a vector as deleted. Deleted vectors are excluded from future search
results.

### `LSMVecDB::printStatistics`

```cpp
void printStatistics() const;
```

Prints I/O and timing statistics to stdout. Only meaningful when
`opts.enable_stats = true`.

---

## 5. Configuration Reference

### LSMVecDBOptions

Pass this struct to `LSMVecDB::Open()`.

```cpp
LSMVecDBOptions opts;
opts.dim = 128;
opts.metric = DistanceMetric::kL2;
// ... set other fields as needed ...
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | `int` | `0` | **Required.** Dimensionality of vectors. |
| `metric` | `DistanceMetric` | `kL2` | Distance metric (`kL2` or `kCosine`). |
| `m` | `int` | `8` | HNSW: bi-directional links created per node at layer 0. |
| `m_max` | `int` | `16` | HNSW: max neighbors per node at upper layers. |
| `ef_construction` | `float` | `32.0` | HNSW: candidate pool size during index construction. |
| `vec_file_capacity` | `size_t` | `100000` | Initial vector file capacity. Auto-expands with PagedVectorStorage. |
| `paged_max_cached_pages` | `size_t` | `4096` | Number of 4 KB pages in the user-space page cache. |
| `vector_storage_type` | `int` | `1` | `0` = BasicVectorStorage (flat file), `1` = PagedVectorStorage (paged + cached). |
| `db_target_size` | `uint64_t` | `~100 GiB` | Target file size hint for Aster (RocksDB). |
| `random_seed` | `int` | `12345` | RNG seed for HNSW level generation. |
| `enable_stats` | `bool` | `false` | Collect I/O and timing statistics. |
| `enable_batch_read` | `bool` | `true` | Group vector reads by page during search. |
| `reinit` | `bool` | `false` | `true` = wipe existing data; `false` = reopen. |
| `vector_file_path` | `string` | `""` | Path for vector storage file. |
| `log_file_path` | `string` | `""` | Path for log file (empty = no file logging). |

**Tuning tips:**

- Higher `m` and `ef_construction` &rarr; better recall, slower indexing.
- Higher `ef_search` &rarr; better recall, slower queries.
- Higher `paged_max_cached_pages` &rarr; more RAM, fewer disk reads.

---

## 6. Types

### DistanceMetric

```cpp
enum class DistanceMetric {
    kL2,      // Euclidean distance
    kCosine,  // 1 - cosine_similarity
};
```

### node_id_t

```cpp
using node_id_t = std::uint64_t;
```

Unique 64-bit vector identifier.

### Span\<T\>

A lightweight, non-owning view over contiguous memory (similar to C++20
`std::span`). You rarely need to construct one explicitly because
`std::vector<float>` converts to `Span<float>` implicitly.

```cpp
std::vector<float> vec(128);
db->Insert(0, vec);  // implicit Span<float>(vec)
```

---

# Python API

## 7. Installation

```bash
git submodule update --init --recursive
make aster
python -m pip install .
```

Verify:

```bash
python -c "import lsm_vec; print('OK')"
```

---

## 8. Opening and Closing a Database

```python
import os
import lsm_vec

opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128
opts.vector_file_path = "./db/vectors.bin"
opts.reinit = True

os.makedirs("./db", exist_ok=True)
db = lsm_vec.LSMVecDB.open("./db", opts)

# ... use the database ...

db.close()
```

### `LSMVecDB.open`

```python
db = lsm_vec.LSMVecDB.open(path: str, opts: LSMVecDBOptions) -> LSMVecDB
```

Opens or creates a database. Raises `ValueError` on invalid arguments,
`RuntimeError` on I/O errors.

### `db.close`

```python
db.close() -> None
```

---

## 9. Inserting Vectors

Accepts both Python lists and NumPy arrays.

```python
db.insert(0, [0.1] * 128)

import numpy as np
db.insert(1, np.random.rand(128).astype(np.float32))
```

### `db.insert`

```python
db.insert(id: int, vector: list[float] | numpy.ndarray) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `int` | Unique vector identifier. |
| `vector` | `list[float]` or `numpy.ndarray` (float32, 1-D) | Must have length `opts.dim`. |

Raises `ValueError` on dimension mismatch.

---

## 10. Searching

```python
# With SearchOptions
search_opts = lsm_vec.SearchOptions()
search_opts.k = 10
search_opts.ef_search = 128
results = db.search_knn([0.1] * 128, search_opts)

# Or with k and ef_search directly
results = db.search_knn(query_array, k=10, ef_search=128)

for r in results:
    print(f"id={r.id}  distance={r.distance:.4f}")
```

### `db.search_knn`

```python
# Option A: with SearchOptions
db.search_knn(query, opts: SearchOptions) -> list[SearchResult]

# Option B: with explicit parameters
db.search_knn(query, k: int, ef_search: int) -> list[SearchResult]
```

`query` can be a `list[float]` or a 1-D `numpy.ndarray` of float32.

### SearchOptions

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `int` | `1` | Number of nearest neighbors. |
| `ef_search` | `int` | `64` | Candidate pool size. Must be >= `k`. |

### SearchResult

| Property | Type | Description |
|----------|------|-------------|
| `id` | `int` | Vector identifier (read-only). |
| `distance` | `float` | Distance from query (read-only). |

---

## 11. Retrieving, Updating, and Deleting

### `db.get`

```python
vec = db.get(id: int) -> numpy.ndarray  # float32, 1-D
```

Raises `KeyError` if the ID does not exist.

### `db.update`

```python
db.update(id: int, vector: list[float] | numpy.ndarray) -> None
```

Raises `KeyError` if the ID does not exist.

### `db.delete`

```python
db.delete(id: int) -> None
```

---

## 12. Configuration Reference

### LSMVecDBOptions

All properties are read-write.

```python
opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128
opts.metric = lsm_vec.DistanceMetric.Cosine
```

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `dim` | `int` | `0` | **Required.** Vector dimensionality. |
| `metric` | `DistanceMetric` | `L2` | `L2` or `Cosine`. |
| `m` | `int` | `8` | HNSW connections per node. |
| `m_max` | `int` | `16` | Max neighbors at upper layers. |
| `ef_construction` | `float` | `32.0` | Construction-time candidate pool. |
| `vec_file_capacity` | `int` | `100000` | Initial vector file capacity. |
| `paged_max_cached_pages` | `int` | `4096` | Page cache size (4 KB pages). |
| `vector_storage_type` | `int` | `1` | `0` = basic, `1` = paged. |
| `db_target_size` | `int` | `~100 GiB` | Aster target file size. |
| `random_seed` | `int` | `12345` | RNG seed. |
| `enable_stats` | `bool` | `False` | Collect statistics. |
| `enable_batch_read` | `bool` | `True` | Batch vector reads by page. |
| `reinit` | `bool` | `False` | Wipe existing data on open. |
| `vector_file_path` | `str` | `""` | Vector storage file path. |
| `log_file_path` | `str` | `""` | Log file path. |

### DistanceMetric

```python
lsm_vec.DistanceMetric.L2      # Euclidean distance
lsm_vec.DistanceMetric.Cosine  # 1 - cosine similarity
```

---

## 13. Error Handling

| C++ Status | Python Exception | When |
|------------|-----------------|------|
| `InvalidArgument` | `ValueError` | Dimension mismatch, bad parameters. |
| `NotFound` | `KeyError` | Vector ID not found. |
| Other errors | `RuntimeError` | I/O failures, database errors. |

---

# Build and Linking

## Build from Source

```bash
git submodule update --init --recursive
make aster    # build Aster (RocksDB fork)
make          # build liblsmvec.a, liblsmvec.so/dylib, and test binary
```

## Outputs

| File | Description |
|------|-------------|
| `build/lib/liblsmvec.a` | Static library |
| `build/lib/liblsmvec.so` / `.dylib` | Shared library |
| `build/bin/lsm_vec` | Test / benchmark binary |

## Linking

```bash
g++ -std=c++17 -O2 -Iinclude my_app.cc \
    -Lbuild/lib -llsmvec \
    -lrocksdb -lzstd -lsnappy -llz4 -lbz2 -lz -lpthread -ldl \
    -o my_app
```

On macOS, also add `-ljemalloc` if applicable.

## Python Package

```bash
make aster
python -m pip install .
```

This compiles the C++ code into a Python extension module via scikit-build-core.
No separate `make lib` step is needed.
