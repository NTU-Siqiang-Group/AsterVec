# LSM-Vec Python SDK Guide

This guide covers the **engine** Python module, `import lsm_vec` (pybind11
bindings) — the embeddable, in-process interface. For the dense per-method
reference see [API_REFERENCE.md](API_REFERENCE.md); for the optional HTTP/REST
server see [HTTP_API.md](HTTP_API.md).

> The PyPI package is **`lsm-vec`** (`import lsm_vec`). It is *not* the same as
> `lsmvec-client`, which is a separate HTTP client for the `lsm_vec_http` server.

## Prerequisites

- **Python** 3.9+ (a virtualenv or conda environment is recommended)
- **CMake** ≥ 3.10, **C++17 compiler** (GCC 8+ / Clang 10+), GNU Make
- **Boost** (headers only) and **zstd** — the only required compression library
  (Aster is built with snappy / lz4 / bzip / zlib disabled)
- **Aster** (RocksDB fork) built under `lib/aster/` (see step 1 below)

```bash
# Ubuntu / Debian
sudo apt-get install -y build-essential cmake libboost-dev libzstd-dev

# macOS (Homebrew)
brew install cmake boost zstd jemalloc
```

## Installation

### 1. Build Aster (required before `pip install`)

`pip install .` runs CMake internally, which needs Aster's static library
(`lib/aster/librocksdb.a`) to already exist. You do **not** need `make lib` — only
`make aster`.

```bash
git submodule update --init --recursive
make aster
```

### 2. (Optional) create an environment

```bash
conda create -n lainn python=3.12 -y && conda activate lainn
# or: python -m venv .venv && source .venv/bin/activate
```

### 3. Install the package

```bash
python -m pip install .
python -c "import lsm_vec; print('OK')"
```

`scikit-build-core` + `pybind11` compile the C++ engine into the extension module
in one step (pybind11 is fetched automatically via CMake). The engine is statically
linked into the module, so no separate `liblsmvec.so` is needed at runtime.

## Quick Start

```python
import os, lsm_vec

opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128                                  # vector dimensionality (required)
opts.vector_file_path = "./run/db/vectors.bin"
opts.reinit = True                              # start fresh

os.makedirs("./run/db", exist_ok=True)
db = lsm_vec.LSMVecDB.open("./run/db/", opts)

db.insert(1, [0.1] * 128, metadata={"category": "docs"})

for r in db.search_knn([0.1] * 128, k=10, ef_search=128):
    print(r.id, r.distance)

db.close()
```

Both Python lists and 1-D NumPy `float32` arrays are accepted everywhere a vector
is expected.

## Core API

```python
db = lsm_vec.LSMVecDB.open(path, opts)        # open or create

db.insert(id, vector, metadata=None)          # insert (+ optional dict metadata)
db.update(id, vector)                          # replace a vector
db.delete(id)                                  # soft delete
vec = db.get(id)                               # -> np.ndarray (float32)

db.search_knn(query, k=..., ef_search=...)     # -> list[SearchResult(id, distance)]
db.search_knn(query, search_opts)              # with a SearchOptions object
db.search(query, k=10, ef_search=128,          # -> list[dict] {"id","distance"}
          filter=None, max_scan_candidates=0)  #    with an optional metadata filter

db.set_payload(id, metadata)                   # full replace
db.update_payload(id, partial)                 # RFC 7396 merge (None deletes a field)
db.delete_payload_keys(id, ["k1", "k2"])       # remove specific keys
md = db.get_payload(id)                         # -> dict ({} if none)

db.bulk_build(vectors_2d, threads=0)           # initial load (empty DB); -> report dict
db.flush_vector_writes()                        # flush to disk
db.close()                                      # flush + close
```

> **Quantization (SQ8).** Vectors are stored with 8-bit scalar quantization;
> `get` returns a dequantized vector that differs from the input by up to
> ~`range/255` per element, and distances/recall use the quantized form.

### Bulk build (initial load)

The fastest way to populate an **empty** database — builds the whole index in
memory (RNN-Descent) and writes it in one pass; ids are assigned `0..n-1`.

```python
import numpy as np
report = db.bulk_build(np.random.rand(100_000, 128).astype(np.float32), threads=4)
print(report)   # {"n": 100000, "elapsed_ms": ..., "vectors_per_sec": ..., "threads": 4}
```

For incremental updates on a non-empty DB, use `insert` (single calls, or many
concurrent calls from a thread pool — the engine is thread-safe).

## Configuration (`LSMVecDBOptions`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | `int` | `0` | **Required.** Vector dimensionality. |
| `metric` | `DistanceMetric` | `L2` | `lsm_vec.L2` or `lsm_vec.Cosine`. |
| `m` | `int` | `8` | HNSW links per node (layer 0). |
| `m_max` | `int` | `24` | Max neighbors at upper layers. |
| `m_level` | `int` | `1` | Level multiplier. |
| `ef_construction` | `float` | `32.0` | Construction candidate pool. |
| `ef_search` | `int` | `128` | Default search candidate pool. |
| `k` | `int` | `1` | Default neighbors for the bare `search_knn`. |
| `vec_file_capacity` | `int` | `100000` | Initial vector-file capacity. |
| `paged_max_cached_pages` | `int` | `8192` | Page cache size (4 KB pages). |
| `vector_storage_type` | `int` | `1` | `0` = basic, `1` = paged. |
| `enable_batch_read` | `bool` | `True` | Batch vector reads by page during search. |
| `reinit` | `bool` | `False` | Wipe existing data on open. |
| `vector_file_path` | `str` | `""` | Vector storage file path. |

`SearchOptions`: `k` (default `1`), `ef_search` (default `128`),
`max_scan_candidates` (default `0` = auto, `k * 50` when a filter is set).
See [API_REFERENCE.md §16](API_REFERENCE.md#16-configuration-reference) for the
full list.

## Metadata filtering

Each vector can carry a JSON metadata document, and searches can be restricted with
a Mongo-style filter passed as a Python dict.

```python
import numpy as np
emb = np.random.randn(768).astype(np.float32)
db.insert(1, emb, metadata={"tenant": "acme", "tags": ["ml", "py"], "ts": 1700000000})

results = db.search(
    emb, k=10,
    filter={"tenant": "acme", "ts": {"$gt": 1699000000}},
)   # -> [{"id": int, "distance": float}, ...]
```

Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`,
`$exists`, `$contains_any`, `$contains_all`, `$and`, `$or`. A bare scalar is an
implicit `$eq`; nested keys use dot paths (`"author.name"`); there is no `$not`
(negate at the field level). Full reference:
[API_REFERENCE.md §17](API_REFERENCE.md#17-metadata-filter-operators).

`max_scan_candidates` bounds how many neighbors a filtered search visits before
giving up (default `k * 50`). Raise it for highly selective filters on large
corpora; lower it for throughput when the filter matches most rows.

```python
results = db.search(emb, k=10, filter={"tenant": "acme"}, max_scan_candidates=2000)
```

Results come back as plain dicts (`{"id", "distance"}`), so they drop straight into
pandas:

```python
import pandas as pd
df = pd.DataFrame(db.search(emb, k=50, filter={"tenant": "acme"}))
```

## Troubleshooting

### `Aster RocksDB library or headers not found`

Aster hasn't been built. Run:

```bash
git submodule update --init --recursive
make aster
```

### `libzstd not found`

Install zstd (the only required codec): `apt-get install libzstd-dev` (Ubuntu) or
`brew install zstd` (macOS).

### `externally-managed-environment` on `pip install .`

Your system Python is managed by the OS package manager. Use a virtualenv or conda
environment, then `python -m pip install .`.

### `pip install .` fails with `TypeVar() got unexpected keyword argument`

Your Python is too old for `scikit-build-core`. Use Python 3.10+.

### `ImportError: ... liblsmvec ... not loaded` / TLS errors (Linux, jemalloc)

If the extension can't allocate jemalloc's thread-local storage, preload it:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 python your_app.py
```

### `GLIBCXX_3.4.30 not found` (Linux, conda)

Conda ships an older libstdc++; update it:

```bash
conda install -c conda-forge "libstdcxx-ng>=12"
```
