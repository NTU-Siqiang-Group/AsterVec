#!/usr/bin/env python3
"""Embed LSM-Vec in your Python process (the primary way to use the engine).

Run after building + installing the engine module:
    git submodule update --init --recursive
    make aster
    python -m pip install .
    python examples/embed_quickstart.py
"""
import os
import lsm_vec

DIM = 8
DB_DIR = "./run/embed_db"

# ---- open (creates a fresh DB) ----
opts = lsm_vec.LSMVecDBOptions()
opts.dim = DIM
opts.metric = lsm_vec.DistanceMetric.L2
opts.vector_file_path = os.path.join(DB_DIR, "vectors.bin")
opts.reinit = True  # start fresh each run

os.makedirs(DB_DIR, exist_ok=True)
db = lsm_vec.LSMVecDB.open(DB_DIR, opts)

# ---- insert vectors (+ optional JSON metadata) ----
for i in range(1, 6):
    vec = [float(i)] * DIM
    db.insert(i, vec, metadata={"category": "docs" if i % 2 else "blog", "n": i})

# ---- k-NN search -> list[SearchResult(id, distance)] ----
query = [1.1] * DIM
print("nearest to query:")
for r in db.search_knn(query, k=3, ef_search=64):
    print(f"  id={r.id}  distance={r.distance:.4f}")

# ---- filtered search -> list[dict] {"id","distance"} ----
print('filtered (category == "docs"):')
for hit in db.search(query, k=5, filter={"category": {"$eq": "docs"}}):
    print(f"  id={hit['id']}  distance={hit['distance']:.4f}")

# ---- payload CRUD ----
db.update_payload(1, {"n": 99})          # merge-patch (RFC 7396)
print("payload for id=1:", db.get_payload(1))

db.close()
print("OK")
