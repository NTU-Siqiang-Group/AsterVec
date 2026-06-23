# Changelog

All notable changes to AsterVec are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [0.2.0]

Features added to the engine since the initial public release.

### Added
- **HTTP/REST server** (`astervec_http`) — a one-binary service wrapper around the
  engine — plus a Docker image. See [docs/HTTP_API.md](docs/HTTP_API.md).
- **Index lifecycle** over HTTP: `PUT` / `GET` / `DELETE /v1/index` to create,
  inspect, and wipe the index (user-set dimension and metric).
- **`GET /v1/stats`** — a runtime stats endpoint:
  `{vectors, dim, metric, memory_bytes}`.
- **Metadata payloads + filtered search** — per-vector JSON documents and
  Mongo-style query predicates (`$eq`, `$ne`, comparisons, `$in`/`$nin`,
  `$exists`, `$contains_any`/`$contains_all`, `$and`/`$or`).
- **Batch insert** — `POST /v1/vectors/batch` (validated as a whole; up to 10k
  items / 64 MB per request).
- **In-memory bulk build** (RNN-Descent) for fast initial loads, exposed as
  `bulk_build` in the Python and C++ APIs and `POST /v1/build/bulk` over HTTP
  (with an optional payload tail).
- **SQ8 scalar quantization** for compact on-disk vectors (`get` returns a
  dequantized vector; distances use the quantized form).
- **Python (pybind) additions**: `insert(metadata=...)`, payload CRUD
  (`set_payload` / `update_payload` / `delete_payload_keys` / `get_payload`),
  `search(..., filter=...)`, `bulk_build`, `trim_memory`, `delete_stats`.

### Changed
- **Renamed the engine from LSM-Vec to AsterVec.** Version `0.1.0` used the LSM-Vec
  name; this version uses `astervec` for the PyPI package/import, C++ API
  (`AsterVecDB`, `namespace astervec`), binaries, Docker image, and docs. The old
  `lsm-vec` PyPI package remains as a deprecated alias. Legacy `LSMVEC_*` env vars
  and `X-LSMVec-*` headers are still accepted for compatibility.
- Aster is now built **zstd-only**; snappy / lz4 / bzip / zlib are no longer
  required to build or run.
- Input guards: over-capacity ids and out-of-range dimensions are rejected with a
  `400` instead of crashing; non-string metric values are rejected.

### Notes
- The **graph-in-LSM** design (HNSW layer-0 edges persisted in Aster `RocksGraph`)
  and the paged, cached vector storage are the core of the engine.

## [0.1.0] - 2026-04-02

Initial public release under the **LSM-Vec** name.

### Added
- Embeddable C++ vector engine with Python bindings (`pip install .`).
- HNSW approximate nearest-neighbor search with configurable `M`, `Mmax`,
  `m_level`, `ef_construction`, `ef_search`, and `k`.
- Disk-oriented graph storage: layer-0 HNSW edges persisted in Aster `RocksGraph`,
  with upper navigation layers kept in memory.
- Persistent vector storage with paged 4 KB layout and page cache.
- Batch vector reads to reduce repeated disk I/O during graph search.
- L2 and Cosine distance metrics with SIMD-accelerated distance computation.
- Basic lifecycle APIs: insert, update, delete, get, search, close/reopen.
- Benchmark/test CLI for loading vector datasets, running k-NN queries, and
  comparing against ground truth.

### Notes
- The project was published as `lsm-vec`, imported as `lsm_vec`, and exposed the
  C++ API as `LSMVecDB` in `namespace lsm_vec`.
- HTTP service mode, metadata payloads, filtered search, batch insert, bulk build,
  SQ8, and the AsterVec rename were added later in `0.2.0`.
