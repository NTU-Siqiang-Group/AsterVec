# Changelog

All notable changes to LSM-Vec are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Features added to the engine since the initial public release.

### Added
- **HTTP/REST server** (`lsm_vec_http`) — a one-binary service wrapper around the
  engine — plus a Docker image. See [docs/HTTP_API.md](docs/HTTP_API.md).
- **Index lifecycle** over HTTP: `PUT` / `GET` / `DELETE /v1/index` to create,
  inspect, and wipe the index (user-set dimension and metric).
- **`GET /v1/stats`** — a runtime stats snapshot:
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
- Aster is now built **zstd-only**; snappy / lz4 / bzip / zlib are no longer
  required to build or run.
- Input guards: over-capacity ids and out-of-range dimensions are rejected with a
  `400` instead of crashing; non-string metric values are rejected.

### Notes
- The **graph-in-LSM** design (HNSW layer-0 edges persisted in Aster `RocksGraph`)
  and the paged, cached vector storage are the core of the engine.
