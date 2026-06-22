# LSM-Vec HTTP API

LSM-Vec is primarily an **embeddable** vector storage engine (see the
[README](../README.md) and [API_REFERENCE.md](API_REFERENCE.md)). For cases
where you'd rather talk to it over the network, the repo also builds a small
standalone REST server, **`lsm_vec_http`** — a thin service wrapper around the
same engine. This document is its reference.

> **Auth & deployment.** `lsm_vec_http` is a **single-tenant, single-process**
> server and does **not** authenticate requests itself. Run it behind your own
> reverse proxy (e.g. nginx/Caddy) if you need TLS, an API key, or rate limiting.
> The examples below talk to a local server directly. `/health` and `/ready` are
> the only endpoints you'd typically expose unauthenticated.

## Running the server

```bash
# From a source build (after `make aster && make`; the HTTP target is on by default):
cmake --build build --target lsm_vec_http -j
LSMVEC_DATA_DIR=./data LSMVEC_PORT=8000 ./build/bin/lsm_vec_http

# Or via Docker:
docker run -d --name lsmvec -p 8000:8000 -v "$(pwd)/data:/data" lsmvec:latest
```

The dimension can be fixed at startup with `LSMVEC_DIM`, or left unset and
created later over the API with `PUT /v1/index`. See
[Configuration](#configuration) for all knobs.

## Conventions

- Base URL in these examples: `http://localhost:8000`.
- Request and response bodies are JSON (except the binary bulk-build body).
- Errors return `{"error": "...", "code": "..."}` with a standard HTTP status.
- IDs are unsigned 64-bit integers. Values above 2^53−1 may be sent as JSON
  strings to avoid float rounding; the server accepts both forms.
- Vectors are stored with **8-bit scalar quantization (SQ8)**: a `GET` returns a
  dequantized vector that differs from the input by up to ~`range/255` per
  element, and distances/recall are computed on the quantized form.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness — process is up |
| `GET` | `/ready` | Readiness — index initialized and responsive |
| `PUT` | `/v1/index` | Create the index (set dimension + metric) |
| `GET` | `/v1/index` | Inspect index state |
| `DELETE` | `/v1/index` | Delete the index (wipes all data) |
| `GET` | `/v1/stats` | Index stats snapshot |
| `POST` | `/v1/vectors` | Insert a vector (+ optional metadata) |
| `GET` | `/v1/vectors/:id` | Fetch a vector by id |
| `PUT` | `/v1/vectors/:id` | Upsert (replace) a vector |
| `DELETE` | `/v1/vectors/:id` | Delete a vector |
| `GET` | `/v1/vectors/:id/payload` | Fetch metadata payload |
| `PUT` | `/v1/vectors/:id/payload` | Replace metadata payload |
| `PATCH` | `/v1/vectors/:id/payload` | Merge-patch metadata (RFC 7396) |
| `POST` | `/v1/search` | k-NN search (+ optional metadata filter) |
| `POST` | `/v1/vectors/batch` | Insert many vectors in one request |
| `POST` | `/v1/build/bulk` | One-shot bulk index build (empty index only) |

### Health & readiness

```bash
curl -s http://localhost:8000/health   # {"status":"ok"}  (always 200)
curl -s http://localhost:8000/ready    # 200 once an index exists; 503 before
```

`/ready` returns `503` until an index has been created (via `LSMVEC_DIM` at
startup or `PUT /v1/index`). This is what a container healthcheck / load balancer
should poll.

### Index lifecycle

```bash
# Create the index once, at your embedding dimension.
curl -s -X PUT http://localhost:8000/v1/index \
  -H 'content-type: application/json' \
  -d '{"dim": 128, "metric": "l2"}'
# 200 {"dim":128,"metric":"l2","initialized":true}

curl -s http://localhost:8000/v1/index
# 200 {"initialized":true,"dim":128,"metric":"l2"}   (dim/metric null if uninit)

curl -s -X DELETE http://localhost:8000/v1/index
# 200 {"initialized":false}   — destructive: removes all vectors, payloads, graph
```

- `dim`: integer in `[1, 4000]`. `metric` (optional): `"l2"` (default) or `"cosine"`.
- `PUT` returns `409` if the index is already initialized, `400` on a bad dim or
  metric. `DELETE` returns `500 delete_failed` if any file removal fails.

### GET /v1/stats

```json
{ "vectors": 1000, "dim": 128, "metric": "l2", "memory_bytes": 9306112 }
```

`vectors` is the live count (exact in steady state and across graceful restarts;
approximate only after an unclean crash). `memory_bytes` is the process resident
anonymous memory (`0` on macOS).

### POST /v1/vectors

```bash
curl -s http://localhost:8000/v1/vectors \
  -H 'content-type: application/json' \
  -d '{"id": 1, "vector": [0.1, 0.2, "..."], "metadata": {"title": "intro"}}'
# 201 {"id": 1}
```

- `vector`: array of floats; length must equal the index dimension.
- `metadata` (optional): any JSON object. On insert, a non-empty object sets the
  payload, an explicit `{}` clears it, and omitting it leaves any existing payload
  unchanged.

### GET / PUT / DELETE /v1/vectors/:id

```bash
curl -s http://localhost:8000/v1/vectors/1
# 200 {"id":1,"vector":[...]}   (404 if not found; SQ8-dequantized — see Conventions)

curl -s -X PUT http://localhost:8000/v1/vectors/1 \
  -H 'content-type: application/json' -d '{"vector": [0.3, 0.4, "..."]}'
# 200 {"id":1}   — insert-or-replace the vector (payload untouched)

curl -s -X DELETE http://localhost:8000/v1/vectors/1
# 204   — soft delete; the id stops appearing in search and GET returns 404
```

### Payload endpoints

```bash
curl -s http://localhost:8000/v1/vectors/1/payload          # GET → metadata object
curl -s -X PUT  .../v1/vectors/1/payload -d '{"k":"v"}'     # replace
curl -s -X PATCH .../v1/vectors/1/payload -d '{"k":null}'   # RFC 7396 merge (null deletes)
```

### POST /v1/search

```bash
curl -s http://localhost:8000/v1/search \
  -H 'content-type: application/json' \
  -d '{"vector": [0.1, 0.2, "..."], "k": 10,
       "filter": {"$and": [{"category": {"$eq": "docs"}}]}}'
# 200 {"results": [{"id": 2, "distance": 0.41}, {"id": 7, "distance": 0.55}]}
```

- `k` (default 10): neighbors to return.
- `ef_search` (optional): search breadth; higher = better recall, slower
  (default = the server's `LSMVEC_EFS_DEFAULT`, 128).
- `filter` (optional): metadata predicate (see [Filters](#metadata-filters)).
- `max_scan_candidates` (optional): cap on candidates visited when a filter is
  set; `0` = auto (`k * 50`).

Results are sorted by ascending distance. An empty index returns `{"results": []}`.

### POST /v1/vectors/batch

Insert many already-embedded vectors in one request (validated as a whole — a
bad item writes nothing).

```bash
curl -s http://localhost:8000/v1/vectors/batch \
  -H 'content-type: application/json' \
  -d '{"items": [
        {"id": 10, "vector": [/* dim floats */], "metadata": {"src": "docs"}},
        {"id": 11, "vector": [/* dim floats */]}
      ]}'
# 201 {"inserted": 2}
```

Limits: up to **10,000 items** and **64 MB** per request (`413` if exceeded).

### POST /v1/build/bulk

One-shot in-memory build of the whole index (RNN-Descent), then a single-pass
disk write — 2–3× faster and higher-recall than a loop of inserts. **Initial-load
only**: the index must already be **created** (`PUT /v1/index`) but still **empty**.
IDs are assigned `0..n-1`.

The body is a raw little-endian **float32** blob (not JSON); shape goes in headers:

```
POST /v1/build/bulk
Content-Type: application/octet-stream
X-LSMVec-N: 100000
X-LSMVec-Dim: 128
X-LSMVec-Threads: 4        (optional; default 1)

<N * Dim * 4 bytes of float32>
[optional JSON tail: an array of N payload objects/nulls, mapped to ids 0..n-1]
```

```json
{ "n": 100000, "elapsed_ms": 10020.5, "vectors_per_sec": 9979,
  "threads": 4, "payloads_written": 0 }
```

Errors: `409` if the index is not initialized or already holds data; `400` on a
dim/header/payload mismatch; `413` if the body exceeds the bulk limit (~4 GB).
The Python client wraps this as `bulk_build(...)`.

## Metadata filters

Filters are a JSON predicate tree. Supported operators:

| Operator | Meaning |
|---|---|
| `$eq`, `$ne` | equals / not equals |
| `$gt`, `$gte`, `$lt`, `$lte` | numeric comparison |
| `$in`, `$nin` | value is / isn't in the given array |
| `$exists` | key present — boolean (`{"$exists": false}` = key absent) |
| `$contains_any`, `$contains_all` | the field's array contains any / all of the values |
| `$and`, `$or` | logical combination (arrays of sub-predicates) |

Negation is expressed at the field level (`$ne`, `$nin`, `$exists: false`); there
is no standalone `$not`. Nested keys use dot paths (`{"attrs.color": {"$eq": "red"}}`).
A bare scalar is an implicit `$eq` (`{"category": "docs"}`).

```json
{ "$and": [ {"price": {"$lt": 100}}, {"in_stock": {"$eq": true}} ] }
```

## Configuration

All server settings come from environment variables:

| Variable | Default | Description |
|---|---|---|
| `LSMVEC_PORT` | `8000` | HTTP listen port |
| `LSMVEC_DATA_DIR` | `/data` | Data directory |
| `LSMVEC_HTTP_THREADS` | `1` | Request worker threads |
| `LSMVEC_DIM` | `0` | Index dimension (`0` = set later via `PUT /v1/index`) |
| `LSMVEC_METRIC` | `l2` | `l2` or `cosine` |
| `LSMVEC_M` | `8` | HNSW M (links per node, layer 0) |
| `LSMVEC_MMAX` | `24` | HNSW max neighbors, upper layers |
| `LSMVEC_EFC` | `32` | HNSW ef_construction (build quality) |
| `LSMVEC_EFS_DEFAULT` | `128` | Default ef_search for queries |
| `LSMVEC_EDGE_CACHE_SIZE` | `100000` | In-memory graph edge cache |
| `LSMVEC_VEC_FILE_CAPACITY` | `1000000` | Initial vector-file capacity (auto-expands) |
| `LSMVEC_PAGED_MAX_CACHED_PAGES` | `8192` | Page cache size (4 KB pages) |
| `LSMVEC_ENABLE_STATS` | `false` | Collect internal timing/IO stats |
| `LSMVEC_LOG_LEVEL` | `info` | `trace`/`debug`/`info`/`warn`/`error` |

## Error codes

| HTTP | Typical `code` | Meaning |
|---|---|---|
| 400 | `bad_json`, `missing_field`, `wrong_dim`, `bad_vector`, `bad_id` | malformed request |
| 404 | `not_found`, `db_error` | unknown route / id not found |
| 409 | `index_not_initialized`, `index_already_initialized` | index lifecycle conflict |
| 413 | — | request body exceeds the limit |
| 500 | `delete_failed`, `db_error` | server-side storage error |
