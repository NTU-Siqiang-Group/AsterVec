# LSM-Vec Python SDK Guide

## Prerequisites

- **Conda** (Miniconda or Anaconda)
- **CMake** >= 3.10
- **C++17 compiler** (GCC 8+ or Clang 10+)
- **Aster** (RocksDB fork) built under `lib/aster/`
- System libraries: `zstd`, `snappy`, `lz4`, `bz2`, `zlib`

### macOS

```bash
brew install cmake zstd snappy lz4 bzip2
```

### Ubuntu / Debian

```bash
sudo apt-get install -y build-essential cmake \
  libzstd-dev libsnappy-dev liblz4-dev libbz2-dev zlib1g-dev
```

## Installation

### 1. Build Aster (required before pip install)

`pip install .` runs CMake internally, which requires the Aster RocksDB library
(`lib/aster/librocksdb.a`) to already exist. You do **not** need `make lib` —
only `make aster`.

```bash
git submodule update --init --recursive
make aster
```

### 2. Create a conda environment

```bash
conda create -n lainn python=3.12 -y
conda activate lainn
```

### 3. Install the Python package

```bash
pip install .
```

This uses `scikit-build-core` + `pybind11` under the hood to compile the C++
library and Python bindings in one step. The build fetches pybind11 automatically
via CMake FetchContent. All lsmvec code is compiled and statically linked into
the Python module, so no separate `liblsmvec.so` is needed at runtime.

To verify the installation:

```bash
python -c "import lsm_vec; print('OK')"
```

## Quick Start

```python
import lsm_vec
import os

# Configure the database
opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128                              # vector dimensionality (required)
opts.vector_file_path = "./run/db/vectors.bin"

# Ensure the DB directory exists
os.makedirs("./run/db", exist_ok=True)

# Open (creates a new database if the directory is empty)
db = lsm_vec.LSMVecDB.open("./run/db/", opts)

# Insert a vector
db.insert(1, [0.1] * 128)

# Search
search_opts = lsm_vec.SearchOptions()
search_opts.k = 10
results = db.search_knn([0.1] * 128, search_opts)
for r in results:
    print(f"id={r.id}  distance={r.distance}")

# Clean up
db.close()
```

## API Reference

### `lsm_vec.LSMVecDBOptions`

Configuration object passed to `LSMVecDB.open()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | `int` | `0` | Vector dimensionality (required) |
| `metric` | `DistanceMetric` | `L2` | Distance metric (`L2` or `Cosine`) |
| `m` | `int` | `8` | HNSW: number of established connections |
| `m_max` | `int` | `16` | HNSW: max neighbors per layer |
| `m_level` | `int` | `1` | HNSW: level normalization factor |
| `ef_construction` | `float` | `64.0` | HNSW: candidate pool size during construction |
| `vec_file_capacity` | `int` | `100000` | Initial vector file capacity |
| `paged_max_cached_pages` | `int` | `256` | Page cache capacity (number of 4KB pages) |
| `vector_storage_type` | `int` | `1` | `0` = BasicVectorStorage, `1` = PagedVectorStorage |
| `db_target_size` | `int` | `107374182400` | RocksDB target file size (bytes) |
| `random_seed` | `int` | `12345` | Seed for HNSW level generation |
| `enable_stats` | `bool` | `False` | Enable performance statistics |
| `enable_batch_read` | `bool` | `False` | Enable batched vector reads in search |
| `reinit` | `bool` | `False` | Reinitialize (wipe) existing database on open |
| `vector_file_path` | `str` | `""` | Path to the vector storage file |
| `log_file_path` | `str` | `""` | Path to the log file |

### `lsm_vec.SearchOptions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `k` | `int` | `1` | Number of nearest neighbors to return |
| `ef_search` | `int` | `64` | Candidate pool size during search (higher = more accurate, slower) |

### `lsm_vec.SearchResult`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Node ID of the result |
| `distance` | `float` | Distance from query vector |

### `lsm_vec.DistanceMetric`

| Value | Description |
|-------|-------------|
| `lsm_vec.L2` | Euclidean (L2) distance |
| `lsm_vec.Cosine` | Cosine distance |

### `lsm_vec.LSMVecDB`

| Method | Signature | Description |
|--------|-----------|-------------|
| `open` | `LSMVecDB.open(path: str, opts: LSMVecDBOptions) -> LSMVecDB` | Open or create a database at the given directory |
| `insert` | `db.insert(id: int, vector: list[float] \| np.ndarray)` | Insert a vector with the given ID |
| `update` | `db.update(id: int, vector: list[float] \| np.ndarray)` | Update vector for an existing ID |
| `delete` | `db.delete(id: int)` | Delete a vector by ID |
| `get` | `db.get(id: int) -> np.ndarray` | Retrieve the vector for a given ID |
| `search_knn` | `db.search_knn(query, opts: SearchOptions) -> list[SearchResult]` | K-nearest-neighbor search |
| `search_knn` | `db.search_knn(query, k: int, ef_search: int) -> list[SearchResult]` | K-NN search (shorthand) |
| `close` | `db.close()` | Close the database and flush to disk |

The `query` and `vector` parameters accept either a Python list of floats or a 1-D NumPy `float32` array.

## Usage with NumPy

```python
import numpy as np
import lsm_vec

opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128
opts.vector_file_path = "./run/db/vectors.bin"

db = lsm_vec.LSMVecDB.open("./run/db/", opts)

# Insert with NumPy array
vec = np.random.rand(128).astype(np.float32)
db.insert(42, vec)

# Search with NumPy array
query = np.random.rand(128).astype(np.float32)
results = db.search_knn(query, k=5, ef_search=100)
for r in results:
    print(f"id={r.id}  distance={r.distance:.4f}")

db.close()
```

## Troubleshooting

### `ImportError: dlopen ... Library not loaded: @rpath/liblsmvec.dylib`

This means the Python extension was linked against the shared library instead of the static library. Make sure the CMakeLists.txt has:

```cmake
target_link_libraries(lsm_vec_python PRIVATE lsmvec_static)
```

Then reinstall:

```bash
pip install . --force-reinstall
```

### `externally-managed-environment` error

Your system Python is managed by the OS package manager (Homebrew/apt). Use a conda environment instead:

```bash
conda create -n lainn python=3.12 -y
conda activate lainn
pip install .
```

### `pip install .` fails with `TypeVar() got unexpected keyword argument`

Your Python version is too old for `scikit-build-core`. Use Python 3.10 or newer:

```bash
conda create -n lainn python=3.12 -y
conda activate lainn
pip install .
```

### Build fails with `Aster RocksDB library or headers not found`

Aster hasn't been built yet. Run:

```bash
git submodule update --init --recursive
make aster
```

### Build fails with `libsnappy not found` (or lz4, bz2)

Install the missing system library:

- **macOS**: `brew install snappy lz4 bzip2`
- **Ubuntu**: `sudo apt-get install libsnappy-dev liblz4-dev libbz2-dev`

## Metadata Filtering

Each vector can carry an arbitrary JSON metadata document. Searches can be
restricted with a MongoDB-style filter expression passed as a Python dict.

### Insert with metadata

```python
import numpy as np
import lsm_vec

opts = lsm_vec.LSMVecDBOptions()
opts.dim = 768
db = lsm_vec.LSMVecDB.open("/path/to/db", opts)

embedding = np.random.randn(768).astype(np.float32)
db.insert(
    id=1,
    vector=embedding,
    metadata={"tenant": "acme", "tags": ["ml", "py"], "created_at": 1700000000},
)
```

Both Python lists and NumPy `float32` arrays are accepted as the vector. The
`metadata` argument is any JSON-serializable Python value (typically a dict).

### Search with a filter

```python
results = db.search(
    query=embedding,
    k=10,
    filter={"tenant": "acme", "created_at": {"$gt": 1699000000}},
)
# results: list[dict] — [{"id": int, "distance": float}, ...]
```

Filter operators supported in v1 (Level 2 set):

- Comparison: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
- Membership: `$in`, `$nin`
- Existence: `$exists: True / False`
- Array: `$contains_any`, `$contains_all`
- Logical: `$and`, `$or` (nested composition)
- Implicit `$eq` for scalar and object values
- Dot-path keys for nested fields (e.g. `"author.name"`)

### Tuning expansion budget

`max_scan_candidates` bounds how many neighbors the filtered search visits
before giving up. Default is `k * 50`. Raise for highly selective filters on
large corpora; lower for throughput when the filter matches most rows.

```python
results = db.search(
    query=embedding,
    k=10,
    filter={"tenant": "acme"},
    max_scan_candidates=2000,
)
```

### Manage metadata independently

```python
db.set_payload(1, {"tenant": "acme", "version": 2})     # overwrite
db.update_payload(1, {"version": 3})                    # merge-patch
db.update_payload(1, {"old_field": None})               # delete field via merge
db.delete_payload_keys(1, ["temp", "debug"])            # remove specific keys
md = db.get_payload(1)                                  # -> dict
```

`db.delete(id)` removes both the vector and its metadata.

### NumPy compatibility

All vector arguments accept `np.ndarray` with `dtype=np.float32` (preferred) or
any Python sequence that can be converted. Results come back as plain Python
dicts with integer `id` and float `distance`, so they are trivial to feed into
pandas:

```python
import pandas as pd
df = pd.DataFrame(db.search(query=embedding, k=50, filter={"tenant": "acme"}))
```

### Where to look next

See `docs/METADATA_FILTERING_USAGE.md` for the complete operator reference,
missing-field semantics table, performance guidance, and observability
counters (`metadata_gets`, `filter_evaluations`, `filter_matches`,
`filter_scanned`, `filter_cap_hits`).
