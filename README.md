# LSM-Vec

LSM-Vec is a research prototype that builds an HNSW-style index on top of **Aster**
(a RocksDB fork providing a graph API). Vectors are stored on disk via a
pluggable vector storage layer.

## Features

* HNSW graph construction with configurable hyperparameters (`M`, `Mmax`, `Ml`, `efConstruction`)
* Layer-0 edges stored in **Aster RocksGraph**
* Upper layers stored in memory
* Two vector storage backends:
  * **BasicVectorStorage**: contiguous by logical ID
  * **PagedVectorStorage**: 4KB-page managed layout + FIFO page cache (user-space)

## Repository layout

```
LSM-Vec/
  include/
  src/
  test/
  lib/
    aster/        # Aster submodule
  CMakeLists.txt
  Makefile
```

## Dependencies

### System packages (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake \
  libzstd-dev libsnappy-dev liblz4-dev libbz2-dev zlib1g-dev
```

## Build

### 1) Build Aster

```bash
git submodule update --init --recursive
make aster
```

### 2) Build embeddable libraries

```bash
make lib
```

Outputs:

```
build/lib/liblsmvec.a
build/lib/liblsmvec.so
```

### 3) Build the example/test binary

```bash
make bin
```

The target `lsm_vec` is built from `test/test.cc` and is placed in `build/bin/`
by default. If your generator overrides output paths, check the CMake build logs
for the exact location.

## Embedding

Include headers from `include/` and link against `liblsmvec.a` or
`liblsmvec.so`, plus the transitive dependencies:
`rocksdb`, `zstd`, `snappy`, `lz4`, `bz2`, `z`, `jemalloc`, `pthread`, `dl`.
When using the shared library, make sure the runtime can locate `liblsmvec.so`.

## Testing

The example/test entry point is the binary built from `test/test.cc`. It accepts
CLI flags defined in `include/config.h` and expects a dataset on disk.

### Example: prepare a mini dataset

You can generate a small SIFT dataset under `data/` and then run the examples
below.

```bash
python data/prepare_sift_100k.py
```

### Required flags

* `--db <path>`: DB directory (required)
* dataset files: either via `--data-dir` (recommended) or explicit `--base/--query/--truth`

### Option A: use `--data-dir` (recommended)

If you pass only `--data-dir`, the tool expects:

* `<data-dir>/input.fvecs`
* `<data-dir>/query.fvecs`
* `<data-dir>/groundtruth.ivecs`

Example:

```bash
./build/bin/lsm_vec \
  --db ./run/db \
  --data-dir ./data/sift_100k \
  --out ./run/output.txt
```

### Option B: explicit file paths

```bash
./build/bin/lsm_vec \
  --db ./run/db \
  --base ./data/sift_100k_input.fvecs \
  --query ./data/sift_100k_query.fvecs \
  --truth ./data/sift_100k_groundtruth.ivecs \
  --out ./run/output.txt
```

## CLI options

### Graph / index parameters

* `--M <int>` (default: 8)
* `--Mmax <int>` (default: 16)
* `--Ml <int>` (default: 1)
* `--efc <float>` (default: 64)

Example:

```bash
./build/bin/lsm_vec \
  --db ./run/db \
  --data-dir ./data/sift_100k \
  --M 16 --Mmax 32 --Ml 1 --efc 200 \
  --out ./run/output.txt
```

### Storage / paths

* `--db <path>`: DB directory (required)
* `--vec <path>`: vector file path (default: `<db>/vector.log`)
* `--vec-storage <int>`:
  * `0` = BasicVectorStorage (default)
  * `1` = PagedVectorStorage (4KB pages + FIFO cache)
* `--paged-cache-pages <count>`: page cache capacity in pages (default: 256)
* `--out <path>`: output file (default: `output.txt`)

## Notes on vector storage

### BasicVectorStorage

* Writes/reads by offset = `id * dim * sizeof(float)`
* Performance depends heavily on OS page cache and access pattern

### PagedVectorStorage

* Manages vectors by 4KB pages
* Vectors never cross page boundaries
* Page-level caching with FIFO eviction (`paged_max_cached_pages` inside `Config.h`)
* Supports optional page prefetch by IDs

## Python plugin (SDK/API)

LSM-Vec includes an optional pybind11-based Python module. Enable it by passing
`-DLSMVEC_BUILD_PYTHON=ON` when configuring CMake. The resulting extension is
named `lsm_vec` and is written to `build/python/`.

```bash
cmake -S . -B build -DLSMVEC_BUILD_PYTHON=ON
cmake --build build
```

Example usage:

```python
import lsm_vec

opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128
opts.vector_file_path = "./run/db/vector.log"

db = lsm_vec.LSMVecDB.open("./run/db", opts)
db.insert(0, [0.1] * 128)

search_opts = lsm_vec.SearchOptions()
search_opts.k = 5
results = db.search_knn([0.1] * 128, search_opts)
print([(r.id, r.distance) for r in results])
```

### pip install .

You can also build and install the module via `pip` (uses `pyproject.toml` and
scikit-build-core under the hood):

```bash
pip install .
```

If you want an editable install while iterating locally:

```bash
pip install -e .
```

## Troubleshooting

#### 1) Link errors about RocksGraph methods (e.g. `undefined reference to rocksdb::RocksGraph::AddEdge`)

This usually means you linked against **system RocksDB** instead of Aster’s build.

Make sure:

* Aster is built under `lib/aster`
* CMake resolves the RocksDB library from `lib/aster` (not `/usr/lib/...`)

#### 2) ZSTD undefined references (e.g. `ZSTD_versionNumber`)

You are missing `-lzstd` at link time, or `libzstd-dev` is not installed.

* Install `libzstd-dev`
* Ensure `target_link_libraries(... zstd ...)` is present (your current CMake does this)
