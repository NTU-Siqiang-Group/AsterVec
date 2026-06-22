# LSM-Vec — the open-source core of Asteroid

<p align="center">
  <img src="docs/assets/lsm-vec-poster.jpg" alt="LSM-Vec" width="350">
</p>

<p align="center">
  <b>An embeddable, memory-friendly vector storage engine — your index lives mostly on disk.</b>
</p>

<p align="center">
  <a href="https://lsmvec.com">Website</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#build-from-source">Build</a> ·
  <a href="docs/API_REFERENCE.md">Python / C++ API</a> ·
  <a href="docs/HTTP_API.md">HTTP API</a>
</p>

LSM-Vec is a persistent, disk-oriented vector database for approximate
nearest-neighbor (ANN) search. It pairs an [HNSW](https://arxiv.org/abs/1603.09320)
graph index with **[Aster](https://github.com/NTU-Siqiang-Group/Aster)**, a
graph-oriented LSM-tree storage engine (a RocksDB fork), so that most of the
index lives on disk — keeping memory small and predictable as your data grows.

You use it in one of two ways:

- **Embed it** *(primary)* — link the C++ library, or `import lsm_vec` in Python,
  and run the engine **in your own process**. No server, no network hop.
- **Serve it** *(optional)* — the repo also builds a small standalone REST server,
  [`lsm_vec_http`](#run-as-an-http-service-optional), if you'd rather talk to it
  over HTTP.

> **Naming.** *Asteroid* is the product; **LSM-Vec** is its open-source engine —
> this repository. The Python module is `import lsm_vec`. A managed, hosted option
> is available as **Asteroid Cloud** ([lsmvec.com](https://lsmvec.com)).

## Why LSM-Vec?

- **Minimal memory footprint.** Unlike vector databases that hold the whole index
  in RAM, LSM-Vec is disk-oriented — memory stays small and predictable at scale.
- **Graph-oriented LSM-tree storage.** Layer-0 HNSW edges are persisted inside
  **Aster** (`RocksGraph`); only the small upper layers stay in memory. This
  graph-in-LSM design is the core of the engine: it keeps the largest graph layer
  and vector data off the heap, supports efficient graph updates, and manages
  variable per-node edge counts compactly — preserving fast graph traversal and
  updates while the bulk of the index stays on disk.
- **A lightweight engine for on-device retrieval.** LSM-Vec keeps large vector
  indexes mostly on disk, making persistent search practical for personal AI
  agents and desktop RAG.

## Features

- **Embeddable** — C++ library + Python bindings (`import lsm_vec`); runs in-process.
- **HNSW search** — fully configurable (`m`, `m_max`, `ef_construction`, `ef_search`);
  **L2** and **Cosine** metrics with SIMD acceleration (AVX2/SSE2).
- **Graph-in-LSM** — layer-0 edges persisted in Aster `RocksGraph`; upper layers
  in memory.
- **Metadata payloads + filtered search** — attach a JSON document per vector and
  restrict queries with Mongo-style predicates (`$eq`, `$gt`, `$in`, `$and`, …).
- **SQ8 quantization** — vectors stored with 8-bit scalar quantization for a small
  on-disk footprint.
- **Fast ingestion** — per-vector `insert`/`upsert`, batch insert, and an
  **in-memory bulk build** (RNN-Descent) for initial loads.
- **Paged vector storage** — 4 KB page layout with a user-space page cache and
  batch (by-page) neighbor reads during search.
- **Persistent** — close and reopen without re-indexing.
- **Optional HTTP server** — one-binary `lsm_vec_http` REST service + Docker image.

## Quick Start

LSM-Vec is embedded directly in your application. (Prefer a network service? See
[Run as an HTTP service](#run-as-an-http-service-optional).)

### Python (`import lsm_vec`)

```bash
git submodule update --init --recursive
make aster
python -m pip install .          # builds + installs the lsm_vec module
```

```python
import lsm_vec

opts = lsm_vec.LSMVecDBOptions()
opts.dim = 128
opts.vector_file_path = "./db/vectors.bin"
opts.reinit = True               # start fresh

db = lsm_vec.LSMVecDB.open("./db", opts)

db.insert(1, [0.1] * 128, metadata={"category": "docs"})

# k-NN search → list[SearchResult(id, distance)]
for r in db.search_knn([0.1] * 128, k=10, ef_search=128):
    print(r.id, r.distance)

# filtered search → list[dict] of {"id", "distance"}
hits = db.search([0.1] * 128, k=10, filter={"category": "docs"})

db.close()
```

Both Python lists and NumPy `float32` arrays are accepted. See the
[Python / C++ API reference](docs/API_REFERENCE.md) and the
[Python SDK guide](docs/python_sdk_guide.md).

### C++

Include headers from `include/` and link `liblsmvec.a` (static) or
`liblsmvec.so`/`.dylib`. Transitive deps: `rocksdb` (Aster), `zstd`, `pthread`,
`dl` (plus `jemalloc` on macOS).

```cpp
#include "lsm_vec_db.h"
using namespace lsm_vec;

LSMVecDBOptions opts;
opts.dim = 128;
opts.vector_file_path = "./db/vectors.bin";
opts.reinit = true;

std::unique_ptr<LSMVecDB> db;
LSMVecDB::Open("./db", opts, &db);

std::vector<float> v(128, 0.1f);
db->Insert(1, v);

SearchOptions so; so.k = 10; so.ef_search = 128;
std::vector<SearchResult> results;
db->SearchKnn(v, so, &results);

db->Close();
```

## How it works

```
LSMVecDB (public API — lsm_vec_db.h)
  └─ LSMVec (HNSW index)
       ├─ RocksGraph (Aster)  — layer-0 graph edges on disk (LSM-tree)
       ├─ nodes_ map          — upper-layer edges in memory
       └─ IVectorStorage      — raw vectors on disk (paged + cached)
```

Layer-0 edges are persisted in Aster's LSM-tree; the much smaller upper layers are
kept in memory for fast navigation. Vectors are stored by `PagedVectorStorage`
(4 KB pages, FIFO page cache) with neighbors co-located by HNSW entry point for
locality.

## Build from source

### Prerequisites

- C++17 compiler (GCC 8+ / Clang 10+), CMake ≥ 3.10, GNU Make, Boost (headers only)
- **zstd** — the only required compression library (Aster is built with snappy /
  lz4 / bzip / zlib disabled)
- jemalloc on macOS

**Ubuntu / Debian**

```bash
sudo apt-get install -y build-essential cmake libboost-dev libzstd-dev
```

**macOS (Homebrew)**

```bash
brew install cmake boost zstd jemalloc
```

### Build

```bash
git submodule update --init --recursive   # fetch the Aster submodule
make aster                                 # build lib/aster/librocksdb.a (required first)
make                                       # build build/lib/liblsmvec.{a,so} + the test binary (build/bin/lsm_vec)
python -m pip install .                    # (optional) build + install the Python module
make unit_test                             # (optional) run the test suite
```

`make` produces the static/shared libraries and the `lsm_vec` test/benchmark binary.
To build the optional HTTP server, see
[Run as an HTTP service](#run-as-an-http-service-optional). `python -m pip install .`
compiles the bindings via scikit-build-core (no separate `make lib` needed).

## Run as an HTTP service (optional)

`lsm_vec_http` is a thin, single-process REST wrapper around the same engine — for
when you want a network service instead of (or alongside) embedding. It does **not**
authenticate requests itself; run it behind your own reverse proxy if you need TLS
or an API key.

```bash
# Build the server target (configured by default after `make`):
cmake --build build --target lsm_vec_http -j
LSMVEC_DATA_DIR=./data ./build/bin/lsm_vec_http        # serves on :8000

# Or with Docker:
docker build -t lsmvec:latest .                        # needs lib/aster populated first
docker run -d --name lsmvec -p 8000:8000 -v "$(pwd)/data:/data" lsmvec:latest
```

```bash
# Create the index, insert, search, inspect — over HTTP (toy 2-D vectors):
H='content-type: application/json'
curl -s -X PUT localhost:8000/v1/index -H "$H" -d '{"dim":2,"metric":"l2"}'
curl -s localhost:8000/v1/vectors      -H "$H" -d '{"id":1,"vector":[0.1,0.2]}'
curl -s localhost:8000/v1/search       -H "$H" -d '{"vector":[0.1,0.2],"k":10}'
curl -s localhost:8000/v1/stats        # {"vectors":1,"dim":2,"metric":"l2","memory_bytes":...}
```

See the full **[HTTP API reference](docs/HTTP_API.md)** for every endpoint, the
metadata-filter language, and configuration.

## Configuration

Pass an `LSMVecDBOptions` to `open`. Common fields:

| Field | Default | Description |
|-------|---------|-------------|
| `dim` | `0` | **Required.** Vector dimensionality. |
| `metric` | `L2` | `L2` or `Cosine`. |
| `m` | `8` | HNSW links per node (layer 0). |
| `m_max` | `24` | HNSW max neighbors, upper layers. |
| `ef_construction` | `32.0` | Candidate pool during construction. |
| `ef_search` | `128` | Default candidate pool during search. |
| `vector_storage_type` | `1` | `0` = flat file, `1` = paged + cached. |
| `paged_max_cached_pages` | `8192` | Page cache size (4 KB pages). |
| `reinit` | `false` | `true` = wipe on open; `false` = reopen. |
| `vector_file_path` | `""` | Path to the vector data file. |

Higher `m` / `ef_construction` → better recall, slower build. Higher `ef_search` →
better recall, slower queries. See [API_REFERENCE.md](docs/API_REFERENCE.md) for
the complete list; the HTTP server is configured with the matching `LSMVEC_*`
environment variables (see [HTTP_API.md](docs/HTTP_API.md)).

## Storage & quantization

- **PagedVectorStorage** (`vector_storage_type = 1`, default) — 4 KB pages; no
  vector crosses a page boundary; vectors sharing an HNSW entry point are
  co-located; a user-space FIFO page cache plus by-page batch reads cut I/O.
- **BasicVectorStorage** (`vector_storage_type = 0`) — contiguous flat file,
  `offset = id * dim * sizeof(float)`; relies on the OS page cache.
- **SQ8** — vectors are stored with 8-bit scalar quantization; `get` returns a
  dequantized vector (≈`range/255` per element), and distances/recall use the
  quantized form.

## Test binary / benchmarking

`make` also builds `build/bin/lsm_vec`, a CLI harness that loads a dataset, builds
the index, runs k-NN queries, and compares against ground truth — useful for
benchmarking (not the way you'd use the engine in an app).

```bash
cd data && python prepare_sift_100k.py && cd ..
./build/bin/lsm_vec --db ./run/db --data-dir ./data/sift_100k_ \
  --M 8 --Mmax 24 --efc 32 --k 10 --efs 128 --stats --out ./run/output.txt
```

Run `./build/bin/lsm_vec --help` for all flags (HNSW params, storage backend,
batch read, etc.).

## Troubleshooting

- **`Aster RocksDB library or headers not found`** — build Aster first:
  `git submodule update --init --recursive && make aster`.
- **`libzstd not found`** — install zstd (the only required codec):
  `apt-get install libzstd-dev` / `brew install zstd`.
- **`FetchContent` can't download pybind11 during `pip install .`** — install
  pybind11 (`pip install pybind11` or conda) and switch the `FetchContent` block in
  `CMakeLists.txt` to `find_package(pybind11 REQUIRED)`.
- **`externally-managed-environment` on `pip install .`** — use a virtualenv/conda
  environment.
- **`cannot allocate memory in static TLS block` (Linux, jemalloc)** — preload
  jemalloc: `LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 python your_app.py`.
- **`liblsmvec.so: cannot open shared object file`** — add the build dir to the
  loader path: `export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH`.

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).
