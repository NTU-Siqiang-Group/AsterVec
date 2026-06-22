# AsterVec — memory-friendly vector storage engine

<p align="center">
  <img src="docs/assets/aster-vec-poster.jpg" alt="AsterVec" width="350">
</p>

<p align="center">
  <b>A lightweight, memory-friendly vector storage engine for on-device retrieval.</b>
</p>

<p align="center">
  <a href="https://lsmvec.com">Website</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#build-from-source">Build</a> ·
  <a href="docs/API_REFERENCE.md">Python / C++ API</a> ·
  <a href="docs/HTTP_API.md">HTTP API</a>
</p>

AsterVec is an embeddable vector engine for AI applications that need persistent
retrieval close to the app. It keeps large vector indexes mostly on disk using
**[Aster](https://github.com/NTU-Siqiang-Group/Aster)**, a graph-oriented LSM-tree storage engine,
so agents and desktop RAG systems can run vector search without assuming a large always-on database server.

Use AsterVec as local retrieval infrastructure in two modes:

- **Embed it** *(primary)* — run the engine in-process for local agent memory,
  RAG indexes, and app-owned vector storage.
- **Serve it** *(optional)* — run `astervec_http` when an app needs a local REST
  boundary.

> **Naming.** Asteroid is the product; AsterVec is the open-source engine in this
> repo. The Python module is `astervec`; Asteroid Cloud is the managed service.

> **Rename note.** AsterVec was previously named **LSM-Vec** (version `0.1.0`). The
> latest version uses `astervec` for the Python package/import, C++ API, binaries,
> Docker image, and HTTP config. The old `lsm-vec` PyPI package stays as a deprecated
> alias, and the server still accepts the legacy `LSMVEC_*` env vars and `X-LSMVec-*`
> headers, so existing deployments keep working.

## Why AsterVec?

- **Minimal memory footprint.** Unlike vector databases that hold the whole index
  in RAM, AsterVec is disk-oriented — memory stays small and predictable at scale.
- **Built for on-device retrieval.** Disk-backed indexing keeps persistent vector
  search practical for agents, desktop RAG, and small servers.
- **Graph-in-LSM storage.** AsterVec persists the largest HNSW layer in Aster
  `RocksGraph`, keeping only the upper navigation layers in memory.
- **Designed as an engine.** Embed it into your app when retrieval state should stay
  local, private, and application-owned.

## Features

- **Local retrieval engine** — embed persistent vector search directly in Python or
  C++ applications.
- **Agent/RAG-ready metadata** — attach JSON payloads and filter retrieval with
  Mongo-style predicates (`$eq`, `$gt`, `$in`, `$and`, ...).
- **Disk-backed HNSW** — layer-0 edges live in Aster `RocksGraph`; upper layers stay
  in memory.
- **Compact vector storage** — SQ8 quantization and paged storage reduce disk and
  memory pressure.
- **Ingestion paths** — insert/upsert, batch insert, and in-memory bulk build for
  initial retrieval indexes.
- **Optional service mode** — expose the same engine through `astervec_http` when a
  REST boundary is useful.

## Quick Start

AsterVec is embedded directly in your application. (Prefer a network service? See
[Run as an HTTP service](#run-as-an-http-service-optional).)

### Python (`import astervec`)

```bash
git submodule update --init --recursive
make aster
python -m pip install .          # builds + installs the astervec module
```

```python
import astervec

opts = astervec.AsterVecDBOptions()
opts.dim = 128
opts.vector_file_path = "./db/vectors.bin"
opts.reinit = True               # start fresh

db = astervec.AsterVecDB.open("./db", opts)

db.insert(1, [0.1] * 128, metadata={"source": "notes", "type": "memory"})

# k-NN search → list[SearchResult(id, distance)]
for r in db.search_knn([0.1] * 128, k=10, ef_search=128):
    print(r.id, r.distance)

# filtered search → list[dict] of {"id", "distance"}
hits = db.search([0.1] * 128, k=10, filter={"source": "notes"})

db.close()
```

Both Python lists and NumPy `float32` arrays are accepted. See the
[Python / C++ API reference](docs/API_REFERENCE.md) and the
[Python SDK guide](docs/python_sdk_guide.md).

### C++

Include headers from `include/` and link `libastervec.a` (static) or
`libastervec.so`/`.dylib`. Transitive deps: `rocksdb` (Aster), `zstd`, `pthread`,
`dl` (plus `jemalloc` on macOS).

```cpp
#include "astervec_db.h"
using namespace astervec;

AsterVecDBOptions opts;
opts.dim = 128;
opts.vector_file_path = "./db/vectors.bin";
opts.reinit = true;

std::unique_ptr<AsterVecDB> db;
AsterVecDB::Open("./db", opts, &db);

std::vector<float> v(128, 0.1f);
db->Insert(1, v);

SearchOptions so; so.k = 10; so.ef_search = 128;
std::vector<SearchResult> results;
db->SearchKnn(v, so, &results);

db->Close();
```

## How it works

```
AsterVecDB (public API — astervec_db.h)
  └─ AsterVec (HNSW index)
       ├─ RocksGraph (Aster)  — layer-0 graph edges on disk (LSM-tree)
       ├─ nodes_ map          — upper-layer edges in memory
       └─ IVectorStorage      — raw vectors on disk (paged + cached)
```

AsterVec keeps the hot navigation path small. Upper HNSW layers stay in memory,
while layer-0 edges and vector data are stored on disk. `PagedVectorStorage` stores
vectors in 4 KB pages and caches recently used pages during search.

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
make                                       # build build/lib/libastervec.{a,so} + the test binary (build/bin/astervec)
python -m pip install .                    # (optional) build + install the Python module
make unit_test                             # (optional) run the test suite
```

`make` produces the static/shared libraries and the `astervec` test/benchmark binary.
To build the optional HTTP server, see
[Run as an HTTP service](#run-as-an-http-service-optional). `python -m pip install .`
compiles the bindings via scikit-build-core (no separate `make lib` needed).

## Run as an HTTP service (optional)

`astervec_http` exposes the same local engine over REST. Use it when an agent,
desktop app, or local service needs a process boundary instead of embedding directly.
It does **not** authenticate requests itself; run it behind your own reverse proxy if
you need TLS or an API key.

```bash
# Build the server target (configured by default after `make`):
cmake --build build --target astervec_http -j
ASTERVEC_DATA_DIR=./data ./build/bin/astervec_http        # serves on :8000

# Or with Docker:
docker build -t astervec:latest .                        # needs lib/aster populated first
docker run -d --name astervec -p 8000:8000 -v "$(pwd)/data:/data" astervec:latest
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

Pass an `AsterVecDBOptions` to `open`. Common fields:

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
the complete list.

For service mode, use `ASTERVEC_*` environment variables; see
[HTTP_API.md](docs/HTTP_API.md).

## Storage & quantization

- **PagedVectorStorage** — stores vectors in 4 KB pages and caches recently used
  pages, reducing repeated disk reads during graph search.
- **BasicVectorStorage** — stores vectors in a contiguous flat file and relies on the
  OS page cache.
- **SQ8** — stores vectors as 8-bit scalar-quantized values; `get` returns
  dequantized vectors, and search runs on the quantized representation.

## Test binary / benchmarking

`make` also builds `build/bin/astervec`, a CLI harness that loads a dataset, builds
the index, runs k-NN queries, and compares against ground truth — useful for
benchmarking (not the way you'd use the engine in an app).

```bash
cd data && python prepare_sift_100k.py && cd ..
./build/bin/astervec --db ./run/db --data-dir ./data/sift_100k_ \
  --M 8 --Mmax 24 --efc 32 --k 10 --efs 128 --stats --out ./run/output.txt
```

Run `./build/bin/astervec --help` for all flags (HNSW params, storage backend,
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
- **`libastervec.so: cannot open shared object file`** — add the build dir to the
  loader path: `export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH`.

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).
