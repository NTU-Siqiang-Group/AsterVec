# Changelog

All notable changes to AsterVec are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Any point id.** Stores can be created in keyed mode, where ids are any
  64-bit unsigned integer or any string (UUIDs included); the engine maps them
  to compact internal slots. C++: `PointId`, `IdMode`, `SearchPoints`,
  `BulkBuild(vectors, ids, opts)`. Python: ids may be `int`, `str` or
  `uuid.UUID`, `search()` returns them as inserted, `bulk_build(..., ids=...)`,
  `db.id_mode`.
- **Checkpoints.** `Checkpoint()` / `db.checkpoint()` makes everything written
  so far durable in one atomic step and keeps the previous checkpoint as a
  fallback. After an unclean shutdown a store reopens to its last checkpoint.
  `checkpoint_interval_ops` checkpoints automatically every N writes.
- **Operation log** (`enable_oplog`, off by default): every acknowledged
  insert, update and delete survives a process crash and is replayed on open;
  `recovery_stats()` reports the replay. HTTP: `ASTERVEC_OPLOG`,
  `ASTERVEC_CHECKPOINT_INTERVAL_OPS`, `ASTERVEC_OPLOG_SEGMENT_BYTES`.

### Changed
- A search with nothing to return (empty index, no point matching the filter)
  returns an empty result instead of `NotFound` / `KeyError`.
- New stores created from Python use keyed ids by default; integer ids work
  exactly as before, and existing stores keep their mode.
- `flush_vector_writes()` and `close()` now checkpoint.
- Python `update`, `delete` and `get` take 64-bit ids (previously 32-bit).

### Fixed
- Vectors written by `update` could be overwritten by later inserts after the
  store was reopened.
- Concurrent writers could crash the process or corrupt vector records.

### Compatibility
- Index metadata is now format v4. Stores written by this version cannot be
  opened by 0.2.2 or earlier; older stores open unchanged.

## [0.2.2] - 2026-08-31

Wheels-only release: Python 3.14 and 3.15 support. No engine changes.

### Added
- **Python 3.14 and 3.15 wheels** on all four platforms (Linux
  x86_64/aarch64, macOS arm64/x86_64). The cp315 wheels are built against
  the CPython 3.15 RC ABI and work with the 3.15.0 final release.
  Free-threaded builds (`cp314t`/`cp315t`) are not yet provided; engine
  thread-safety under no-GIL needs its own review first.

### Changed
- Bundled pybind11 bumped from 2.13.6 to 3.1.0 (required for CPython
  3.14+); wheel pipeline moved from cibuildwheel 2.21 to 4.2.0. Python
  3.9-3.13 wheels are unchanged in behavior.

## [0.2.1] - 2026-08-26

Packaging, performance, and reliability release.

### Fixed
- **Update-path data corruption**: updating a vector whose page buffer had
  already been flushed created a zero-filled write buffer holding one slot;
  its next full-page flush silently wiped every other vector on that 4 KB
  page. Update-heavy (churn) workloads were affected — recall under a
  sustained insert/update/delete benchmark improves from ~0.95 to ~0.99 at
  100K and ~0.93 to ~0.96 at 1M with the fix. Indexes that received updates
  or deletes with earlier versions may contain zeroed co-paged vectors;
  rebuilding from source embeddings fully repairs them.

### Changed
- **Performance:** internal cache and search-path rework (edge-cache slot
  arena + open-addressed table, varint-encoded cached adjacency, flat
  direct-mapped page-cache index, bounded pooled write buffers, dense
  visited tracking). Measured on SIFT-1M at default settings: query
  throughput +28%, insert throughput +27% (+40% under churn), with recall
  unchanged; steady-state RSS down ~10-20 MB.

### Added
- **Prebuilt Linux aarch64 wheels** — `pip install aster-vec` now works in Docker
  on Apple Silicon, on AWS Graviton, and on Raspberry Pi 5.
- **Prebuilt macOS Intel wheels** (x86_64, macOS 13+).
- **CI** — unit-test workflow on Linux and macOS runs on every push and PR.
- README: CI / PyPI / Python-version / license badges; the quick start now
  installs with a single `pip install aster-vec`.
- New `AsterVecDBOptions` tunables: `edge_cache_size` (now exposed to Python),
  `block_cache_bytes` (previously hardcoded at 32 MB), and `write_buf_cap`
  (bounds live section write buffers; default 8192 = 32 MB pooled).
- `reset_statistics()` API (C++/Python) for phase-separated stats reporting.
- `docs/TROUBLESHOOTING.md` — install/build/runtime fixes collected in one
  place (moved out of the README).

### Fixed (packaging)
- PyPI project page: the logo renders again (absolute image URL) and the sidebar
  links to Documentation, Changelog, and Issues.
- macOS wheels statically link a source-built zstd, so the wheel's minimum
  macOS version no longer depends on the build runner's Homebrew bottles.
- Wheels no longer contain CPU-specific (`-march=native`) RocksDB code, which
  could crash with `Illegal instruction` on machines with older CPUs than the
  build runner. Aster is now built with `PORTABLE=1` for all shipped artifacts.

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
