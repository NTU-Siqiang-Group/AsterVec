<p align="center">
  <img src="https://raw.githubusercontent.com/NTU-Siqiang-Group/AsterVec/main/docs/assets/aster-vec-logo-text.png" alt="AsterVec" width="350">
</p>

<p align="center">
  <b>Memory-friendly vector engine for on-device AI memory.</b>
</p>

<p align="center">
  <a href="#why-astervec">Why</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#documentation">Docs</a>
</p>

<p align="center">
  <a href="https://github.com/NTU-Siqiang-Group/AsterVec/actions/workflows/ci.yml"><img src="https://github.com/NTU-Siqiang-Group/AsterVec/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/aster-vec/"><img src="https://img.shields.io/pypi/v/aster-vec" alt="PyPI"></a>
  <a href="https://pypi.org/project/aster-vec/"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+"></a>
  <a href="https://github.com/NTU-Siqiang-Group/AsterVec/blob/main/LICENSE"><img src="https://img.shields.io/github/license/NTU-Siqiang-Group/AsterVec" alt="License"></a>
</p>

AsterVec is a vector search engine that keeps its index on disk and puts
as little as possible in memory. RAM holds only a small navigation
structure and caches, so an application can search millions of embeddings
within a few hundred MB of memory. It is built for local AI agents and
desktop RAG.

## Why AsterVec

- **Minimize your memory usage.** Other vector engines are designed
  around RAM and take as much of it as your data demands. AsterVec is
  designed around disk: the index lives there, and memory holds only the
  minimal part needed to navigate it. Set a memory budget and it holds,
  at 100K vectors or at 10 million.
- **Insert-friendly.** AsterVec assumes your data keeps growing. The
  index absorbs new vectors as they arrive, updates and deletes
  included, and never stops for a rebuild.
- **Embeddable.** A thread-safe Python and C++ library that runs inside
  your app, with no separate service to deploy. An HTTP server ships in
  the box if you want one.

## Quick start

```bash
pip install aster-vec
```

```python
import astervec

opts = astervec.AsterVecDBOptions()
opts.dim = 128
opts.vector_file_path = "./db/vectors.bin"
db = astervec.AsterVecDB.open("./db", opts)

db.insert(1, [0.1] * 128, metadata={"source": "notes"})

hits = db.search([0.1] * 128, k=10, filter={"source": "notes"})
for h in hits:
    print(h["id"], h["distance"])

db.close()
```

The index takes inserts, updates and deletes in place. There is no
rebuild step:

```python
db.update(7, new_vector)          # replace a vector in place
db.delete(3)                      # removed from results immediately
db.bulk_build(embeddings)         # fast initial load, NumPy accepted
```

## Performance

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NTU-Siqiang-Group/AsterVec/main/docs/assets/perf-snapshot-dark.png">
    <img src="https://raw.githubusercontent.com/NTU-Siqiang-Group/AsterVec/main/docs/assets/perf-snapshot-light.png" alt="AsterVec vs Chroma and LanceDB at 100K vectors: 2.2 times less memory, 5.2 times more queries per second, 7.3 times more inserts per second, recall on par" width="880">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NTU-Siqiang-Group/AsterVec/main/docs/assets/perf-snapshot-1m-dark.png">
    <img src="https://raw.githubusercontent.com/NTU-Siqiang-Group/AsterVec/main/docs/assets/perf-snapshot-1m-light.png" alt="AsterVec vs Chroma at 1M vectors: 4.1 times less memory, faster inserts, recall on par, query speed comparable while serving from disk" width="880">
  </picture>
</p>

Measured on SIFT at ef_search 32, same machine, same HNSW parameters,
default settings for every engine. At 100K the working set is mostly
cached, so AsterVec is effectively in memory too, and it compares well
across the board. At 1M it is genuinely serving from disk. The memory gap widens there,
and query speed stays roughly comparable. That is what the
disk-oriented design is for: comparable performance while most of the
index stays out of RAM. Full results and methodology:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## How it works

At the heart of AsterVec is one architectural decision: **on disk, the
graph and the vectors are stored separately**, because the two parts are
updated and accessed in essentially different patterns.

- **Graph** — edges churn in small, scattered writes, so they live in
  [Aster](https://github.com/NTU-Siqiang-Group/Aster), a graph-oriented
  LSM-tree that absorbs them out of place.
- **Vectors** — they rarely change once written, so they live in packed
  pages built for bulk reads.

```
Your app's process
  └─ AsterVec
       ├─ Memory — upper navigation layers + caches   (small, bounded)
       └─ Disk — two separate stores, updated independently:
            ├─ Graph store   — base-layer edges in Aster, a graph-oriented LSM-tree, built for small, frequent link updates
            └─ Vector store  — related vectors stored in nearby pages, compressed
```

Keeping the two apart is what makes updates cheap. When graph and
vectors share one disk record, recording a single new edge can move
about 8 KB of vector data that did not change. In AsterVec, that edge
lands in the graph store alone. On the vector side, related vectors are
packed onto the same pages, so one read serves many candidate
evaluations.

## Documentation

| Guide | |
|---|---|
| Python & C++ API reference | [docs/API_REFERENCE.md](docs/API_REFERENCE.md) |
| Python SDK guide | [docs/python_sdk_guide.md](docs/python_sdk_guide.md) |
| Configuration options | [docs/API_REFERENCE.md §7](docs/API_REFERENCE.md#7-configuration-reference) |
| HTTP server (optional) | [docs/HTTP_API.md](docs/HTTP_API.md) |
| Troubleshooting | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| Build from source | [CONTRIBUTING.md](CONTRIBUTING.md) |

The optional `astervec_http` server exposes the same engine over HTTP.
See [docs/HTTP_API.md](docs/HTTP_API.md).

## Contributing

Contributions are welcome — [CONTRIBUTING.md](CONTRIBUTING.md) has build
steps, tests, and PR expectations.

## License

Apache-2.0 — see [LICENSE](LICENSE).
