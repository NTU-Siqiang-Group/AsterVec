# LSM-Vec -- Memory Friendly, High Performance VDB
<p align="center">
  <img src="docs/assets/lsm-vec-poster.jpg" alt="LSM-Vec" width="350">
</p>

<p align="center">
  <a href="https://volatill.github.io/lsm-vec-site">Website</a> ·
  <a href="https://volatill.github.io/lsm-vec-site/docs.html">Quick Start</a> ·
  <a href="https://volatill.github.io/lsm-vec-site/docs.html">Docs</a> ·
  <a href="https://volatill.github.io/lsm-vec-site/benchmark.html">Benchmark</a>
</p>

LSM-Vec is a persistent vector database for approximate nearest-neighbor (ANN)
search. It combines an [HNSW](https://arxiv.org/abs/1603.09320) graph index
with **[Aster](https://github.com/NTU-Siqiang-Group/Aster)**, a native graph-oriented LSM-tree storage engine.

## Why LSM-Vec?

### Minimal Memory Overhead
Unlike many vector databases that keep large index state in memory, LSM-Vec is fully disk-oriented.
Its memory footprint remains small and predictable even at large data scale. 


### Graph-Oriented LSM-Tree Storage

LSM-Vec stores the majority of the HNSW index within **Aster**, which extends
RocksDB with a graph data model (`RocksGraph`). This graph-oriented LSM-tree structure enables LSM-Vec to achieve search and update performance comparable to in-memory vector databases.

### Embeddable and Easy to Use

LSM-Vec is offered as a lightweight C++ library with Python bindings. 
LSM-Vec can be built with just a few lines of commands.
Then users can simply link the library or import the module to get started.



## Features

- HNSW graph index with fully configurable hyperparameters (`M`, `Mmax`, `Ml`, `efConstruction`)
- Layer-0 edges persisted in Aster RocksGraph (LSM-tree backed)
- Upper-layer edges kept in-memory for fast navigation
- Two vector storage backends:
  - **BasicVectorStorage** -- contiguous flat file, offset by logical ID
  - **PagedVectorStorage** -- 4 KB page-managed layout with a user-space page cache (FIFO eviction)
- Batch vector read -- groups neighbor reads by page to reduce I/O during search
- Persistent metadata -- database can be closed and reopened without re-indexing
- L2 (Euclidean) and Cosine distance metrics with SIMD acceleration (AVX2/SSE2)
- Python SDK via pybind11 (`pip install .`)
- Per-vector JSON payloads with filter-aware k-NN search (predicates over equality, comparison, exists/contains, set membership, logical and/or/not, dotted paths)
- Soft delete + upsert -- `Delete()` is tombstoned and `Update()` rewires ids in place; the index reopens without re-indexing and exposes tombstone / Bloom-filter counters for maintenance

## Repository layout

```
LAINN/
  include/           # C++ headers (public API + internals)
  src/               # C++ source files + Python example
  python/            # pybind11 binding source
  test/              # C++ test binary entry point + Python quick-start
  data/              # dataset preparation scripts
  lib/
    aster/           # Aster submodule (RocksDB fork)
  CMakeLists.txt
  Makefile
  pyproject.toml     # Python packaging (scikit-build-core)
```


---

## Prerequisites

### Compiler & tools

- C++17 compiler (GCC 8+ or Clang 10+)
- CMake >= 3.10
- GNU Make
- Boost (headers only)

### System libraries

**Ubuntu / Debian**

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake libboost-dev \
  libzstd-dev libsnappy-dev liblz4-dev libbz2-dev zlib1g-dev
```

**macOS (Homebrew)**

```bash
brew install cmake boost zstd snappy lz4 bzip2
```

---


## Building (C++)

### Step 1 -- Build Aster

Aster is included as a Git submodule. Initialize it and build the static
library:

```bash
git submodule update --init --recursive
make aster
```

This produces `lib/aster/librocksdb.a`. The build typically takes a few minutes
and uses all available cores automatically.

### Step 2 -- Build the LSM-Vec libraries

```bash
make lib
```

Outputs:

| File | Description |
|------|-------------|
| `build/lib/liblsmvec.a` | Static library |
| `build/lib/liblsmvec.so` (Linux) / `liblsmvec.dylib` (macOS) | Shared library |

### Step 3 -- Build the test binary

```bash
make bin
```

The executable `lsm_vec` is placed in `build/bin/`. It reads a vector dataset,
builds the HNSW index, runs k-NN queries, and compares results against a ground
truth file.

### One-shot build

```bash
make          # equivalent to: make lib bin
```

### Cleaning

```bash
make clean    # removes the build/ directory
```

### Running the unit test suite

```bash
make unit_test    # builds and runs build/test/unit/lsmvec_unit_tests
```

Covers the Bloom filter, payload CRUD, predicate parsing, delete persistence, and filtered search.

---

## Running the Test Binary

### Prepare a dataset

A helper script downloads the SIFT1M corpus and creates a 100 k-vector subset
with pre-computed ground truth:

```bash
cd data
python prepare_sift_100k.py
cd ..
```

This produces three files under `data/`:

| File | Format | Contents |
|------|--------|----------|
| `sift_100k_input.fvecs` | fvecs | 100 000 base vectors (128-d) |
| `sift_100k_query.fvecs` | fvecs | 10 000 query vectors (128-d) |
| `sift_100k_groundtruth.ivecs` | ivecs | Exact top-100 neighbors per query |

### Run with `--data-dir` (recommended)

If all three files share a common directory and follow the naming convention
`input.fvecs` / `query.fvecs` / `groundtruth.ivecs`, point `--data-dir` at that
directory:

```bash
./build/bin/lsm_vec \
  --db ./run/db \
  --data-dir ./data/sift_100k_ \
  --out ./run/output.txt
```

If the files carry a shared prefix, pass `--name`:

```bash
# expects: data/sift_100k_input.fvecs, data/sift_100k_query.fvecs, ...
./build/bin/lsm_vec \
  --db ./run/db \
  --data-dir ./data/ \
  --name sift_100k \
  --out ./run/output.txt
```

### Run with explicit file paths

```bash
./build/bin/lsm_vec \
  --db ./run/db \
  --base  ./data/sift_100k_input.fvecs \
  --query ./data/sift_100k_query.fvecs \
  --truth ./data/sift_100k_groundtruth.ivecs \
  --out   ./run/output.txt
```

---

## CLI Reference

Run `./build/bin/lsm_vec --help` for the full list.

### Required

| Flag | Description |
|------|-------------|
| `--db <path>` | Database directory (created automatically if absent) |
| Dataset | Either `--data-dir` (+ optional `--name`) **or** all three of `--base`, `--query`, `--truth` |

### HNSW Parameters

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--M <int>` | `-m` | 8 | Number of bi-directional links per node |
| `--Mmax <int>` | `-x` | 24 | Max neighbors per node per layer |
| `--Ml <int>` | `-l` | 1 | Level multiplier for random level generation |
| `--efc <float>` | `-e` | 32 | Candidate pool size during construction (ef_construction) |
| `--k <int>` | `-k` | 1 | Number of nearest neighbors to retrieve |
| `--efs <int>` | `-f` | 128 | Candidate pool size during search (ef_search) |

### Storage

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--vec <path>` | `-v` | `<db>/vector.log` | Path to the vector data file |
| `--vec-storage <int>` | `-V` | 1 | `0` = BasicVectorStorage, `1` = PagedVectorStorage |
| `--paged-cache-pages <int>` | | 8192 | Number of 4 KB pages kept in the user-space page cache |
| `--db-target-size <bytes>` | `-s` | 107374182400 | RocksDB target file size (100 GiB) |

### Runtime / Feature Switches

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-read <0\|1>` | 1 | Batch vector reads during search (groups by page for fewer I/O ops) |
| `--stats <0\|1>` | 0 | Print performance statistics after the run |
| `--reinit <0\|1>` | 1 | Wipe and reinitialize the database on open (set to 0 to reopen an existing DB) |
| `--edge-policy <eager\|lazy\|none>` | eager | Edge update strategy |

### Output

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--out <path>` | `-o` | `output.txt` | File to write query results |

### Example with parameters

```bash
./build/bin/lsm_vec \
  --db ./run/db \
  --data-dir ./data/sift_100k_ \
  --M 8 --Mmax 24 --efc 32 --k 10 --efs 128 \
  --vec-storage 1 --paged-cache-pages 8192 \
  --batch-read \
  --stats \
  --out ./run/output.txt
```

---

## Vector Storage Backends

### BasicVectorStorage (`--vec-storage 0`)

- Vectors stored contiguously: `offset = id * dim * sizeof(float)`
- No user-space caching; relies entirely on the OS page cache
- Simple and fast for sequential ID access patterns

### PagedVectorStorage (`--vec-storage 1`, default)

- Vectors organized into 4 KB pages; no vector crosses a page boundary
- Vectors sharing the same HNSW Level-1 entry point are co-located on the same page,
  improving spatial locality during graph traversal
- User-space FIFO page cache (`--paged-cache-pages` controls the capacity)
- **Batch read** (`--batch-read`): during k-NN search, all unvisited neighbor
  vectors in a layer-0 search step are grouped by page and read together,
  reducing redundant page loads

---

## Embedding LSM-Vec in Your Application

Include headers from `include/` and link against `liblsmvec.a` (static) or
`liblsmvec.so` / `liblsmvec.dylib` (shared).

Transitive link dependencies: `rocksdb` (Aster), `zstd`, `snappy`, `lz4`,
`bz2`, `z`, `pthread`, `dl`. On macOS, `jemalloc` is also required.

```cpp
#include "lsm_vec_db.h"

lsm_vec::LSMVecDBOptions opts;
opts.dim = 128;
opts.vector_file_path = "./db/vectors.bin";

std::unique_ptr<lsm_vec::LSMVecDB> db;
auto s = lsm_vec::LSMVecDB::Open("./db", opts, &db);

// Insert
std::vector<float> vec(128, 0.1f);
db->Insert(0, vec);

// Search (uses k and ef_search from opts)
std::vector<lsm_vec::SearchResult> results;
db->SearchKnn(vec, &results);

// Close
db->Close();
```

---

## Python SDK

LSM-Vec provides a Python module (`lsm_vec`) via pybind11. It supports Python
lists and NumPy arrays as vector inputs.


### Installation

**Prerequisite:** Aster must be built first (see [Step 1](#step-1----build-aster)). And please install `ninja-build` at first.


```bash
git submodule update --init --recursive   # if not done already
make aster                                 # builds lib/aster/librocksdb.a
python -m pip install .                    # builds and installs the lsm_vec module
```

`python -m pip install .` handles the entire compilation internally via
scikit-build-core. You do **not** need to run `make lib` beforehand.

To verify:

```bash
python -c "import lsm_vec; print('OK')"
```

### Quick Start

A ready-to-run example is provided at **`test/python_example.py`**:

```bash
python test/python_example.py
```

```python
import lsm_vec
import os

opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128
db_dir = "./run/db/"
opts.vector_file_path = os.path.join(db_dir, "vectors.bin")
opts.reinit = True

db = lsm_vec.LSMVecDB.open(db_dir, opts)

db.insert(1, [0.1] * 128)

# Search (uses k and ef_search from opts)
results = db.search_knn([0.1] * 128)
print(results[0].id, results[0].distance)
```

### Python API Reference

#### `lsm_vec.LSMVecDBOptions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | int | 0 | Vector dimensionality (required) |
| `metric` | DistanceMetric | `L2` | `lsm_vec.L2` or `lsm_vec.Cosine` |
| `m` | int | 8 | HNSW M parameter |
| `m_max` | int | 24 | Max neighbors per layer |
| `m_level` | int | 1 | Level multiplier |
| `ef_construction` | float | 32.0 | Construction-time candidate pool size |
| `vec_file_capacity` | int | 100000 | Initial vector file capacity |
| `paged_max_cached_pages` | int | 8192 | Page cache capacity (number of pages) |
| `vector_storage_type` | int | 1 | 0 = Basic, 1 = Paged |
| `db_target_size` | int | 107374182400 | RocksDB target file size (bytes) |
| `random_seed` | int | 12345 | Seed for HNSW level generation |
| `enable_stats` | bool | False | Print statistics |
| `enable_batch_read` | bool | True | Batch vector reads during search |
| `reinit` | bool | False | Wipe DB on open |
| `k` | int | 1 | Default number of nearest neighbors for search |
| `ef_search` | int | 128 | Default search-time candidate pool size |
| `vector_file_path` | str | "" | Path to the vector data file |
| `log_file_path` | str | "" | Path to the log file |

#### `lsm_vec.SearchOptions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `k` | int | 1 | Number of nearest neighbors |
| `ef_search` | int | 128 | Search-time candidate pool size |
| `max_scan_candidates` | int | 0 | Cap on candidates scanned when a filter is set; 0 = auto (`k * 50`) |

#### `lsm_vec.SearchResult`

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Vector ID |
| `distance` | float | Distance from query |

#### `lsm_vec.LSMVecDB`

| Method | Description |
|--------|-------------|
| `LSMVecDB.open(path, opts)` | Open or create a database |
| `db.insert(id, vector, metadata=None)` | Insert a vector with an optional Python dict, serialized to JSON |
| `db.update(id, vector)` | Update an existing vector (in-place upsert via id remap) |
| `db.delete(id)` | Soft-delete a vector (tombstoned; survives reopen) |
| `db.get(id) -> np.ndarray` | Retrieve a vector by ID |
| `db.search_knn(query)` | Search using `k` and `ef_search` from `LSMVecDBOptions` |
| `db.search_knn(query, k, ef_search, filter=None, max_scan_candidates=0)` | Search with explicit params and an optional metadata filter (Python dict) |
| `db.search_knn(query, opts)` | Search with a `SearchOptions` object |
| `db.set_payload(id, metadata)` | Replace the metadata document for `id` |
| `db.update_payload(id, metadata)` | RFC 7396 merge-patch the metadata; `None` values delete fields |
| `db.delete_payload_keys(id, keys)` | Remove specific keys from the metadata document |
| `db.get_payload(id) -> dict` | Return the metadata document (`{}` if none) |
| `db.flush_vector_writes()` | Flush pending vector writes to disk |
| `db.close()` | Flush and close the database |

### NumPy Example

```python
import numpy as np
import lsm_vec

opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128
opts.k = 5
opts.ef_search = 100
opts.vector_file_path = "./run/db/vectors.bin"
opts.reinit = True

os.makedirs("./run/db/", exist_ok=True)
db = lsm_vec.LSMVecDB.open("./run/db/", opts)

vec = np.random.rand(128).astype(np.float32)
db.insert(42, vec)

query = np.random.rand(128).astype(np.float32)
results = db.search_knn(query)
for r in results:
    print(f"id={r.id}  distance={r.distance:.4f}")

db.close()
```

### Metadata payloads & filter-aware search

Each vector can carry a JSON document. The same dict is accepted at insert
time or via the dedicated payload endpoints, and any JSON-serializable
predicate can be passed as a `filter=` to `search_knn`.

```python
db.insert(1, vec, metadata={"category": "shoes", "price": 89})
db.set_payload(1, {"category": "shoes", "price": 79})    # full replace
db.update_payload(1, {"price": 69})                       # merge-patch (RFC 7396)
db.get_payload(1)                                         # -> {"category": "shoes", "price": 69}

# Filter-aware search: pass any JSON-serializable predicate dict.
results = db.search_knn(query, k=10, filter={"category": "shoes"})
```

See `docs/METADATA_FILTERING_USAGE.md` for the full predicate grammar
(equality, comparison, exists/contains, set membership, logical and/or/not,
dotted paths).

---

## Troubleshooting

### `Aster RocksDB library or headers not found`

Aster hasn't been built yet:

```bash
git submodule update --init --recursive
make aster
```

### `undefined reference to rocksdb::RocksGraph::AddEdge` (or similar)

You are linking against system RocksDB instead of Aster. Make sure CMake
resolves the library from `lib/aster/`, not `/usr/lib/`. Check the CMake output
for the line `Using RocksDB library at ...`.

### `libzstd not found` / `libsnappy not found` / `liblz4 not found` / `libbz2 not found`

Install the missing library:

- **Ubuntu:** `sudo apt-get install libzstd-dev libsnappy-dev liblz4-dev libbz2-dev`
- **macOS:** `brew install zstd snappy lz4 bzip2`

### `FetchContent` fails to download pybind11 during `pip install .`

In regions with restricted access to GitHub, CMake's `FetchContent` may fail to
clone the pybind11 repository. As a workaround, install pybind11 locally first
and switch `CMakeLists.txt` to use `find_package` instead:

1. Install pybind11 via conda or pip:

```bash
conda install -c conda-forge pybind11   # or: pip install pybind11
```

2. In `CMakeLists.txt`, replace the `FetchContent` block (inside the
   `if(LSMVEC_BUILD_PYTHON)` section) with `find_package`:

```cmake
    # include(FetchContent)
    # FetchContent_Declare(
    #     pybind11
    #     GIT_REPOSITORY https://github.com/pybind/pybind11.git
    #     GIT_TAG v2.13.6
    # )
    # FetchContent_MakeAvailable(pybind11)
    find_package(pybind11 REQUIRED)
```

Then re-run `python -m pip install .`.

### `externally-managed-environment` when running `pip install .`

Your system Python is managed by the OS package manager. Use a virtual
environment or conda:

```bash
conda create -n lainn python=3.12 -y
conda activate lainn
python -m pip install .
```

### `cannot allocate memory in static TLS block` (Linux, jemalloc)

This occurs when the Python module is loaded via `dlopen` and jemalloc's
thread-local storage cannot be allocated. Preload jemalloc at process startup:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 python test/python_example.py
```

To make this automatic in a conda environment:

```bash
conda activate lainn
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
echo 'export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2' \
  > $CONDA_PREFIX/etc/conda/activate.d/jemalloc.sh
echo 'unset LD_PRELOAD' \
  > $CONDA_PREFIX/etc/conda/deactivate.d/jemalloc.sh
```

### `GLIBCXX_3.4.30 not found` (Linux, conda)

Conda ships an older `libstdc++`. Update it:

```bash
conda install -c conda-forge libstdcxx-ng>=12
```


### ImportError: liblsmvec.so: cannot open shared object file: No such file or directory

Update library paths:

```bash
export LD_LIBRARY_PATH=/path/to/LSM_Vec/build/lib:$LD_LIBRARY_PATH
```
