import os
import shutil
import tempfile

import numpy as np
import pytest

import lsm_vec


@pytest.fixture
def empty_db():
    path = tempfile.mkdtemp(prefix="lsmvec_py_bb_")
    opts = lsm_vec.LSMVecDBOptions()
    opts.dim = 8
    opts.m = 8
    opts.m_max = 16
    opts.ef_construction = 32
    opts.vec_file_capacity = 10000
    opts.vector_file_path = os.path.join(path, "vecs.bin")
    db = lsm_vec.LSMVecDB.open(path, opts)
    try:
        yield db
    finally:
        db.close()
        shutil.rmtree(path, ignore_errors=True)


def test_bulk_build_reports_and_builds_searchable_index(empty_db):
    db = empty_db
    rng = np.random.default_rng(0)
    n, dim = 200, 8
    vectors = rng.standard_normal((n, dim)).astype(np.float32)

    report = db.bulk_build(vectors, threads=2)
    assert report["n"] == n
    assert report["threads"] == 2
    assert report["elapsed_ms"] >= 0.0
    assert report["vectors_per_sec"] >= 0.0

    db.flush_vector_writes()

    # IDs are assigned 0..n-1; querying a stored vector retrieves it (the index
    # is built and searchable).
    results = db.search(vectors[0], k=5)
    assert len(results) == 5
    ids = [r["id"] for r in results]
    assert all(0 <= i < n for i in ids)
    assert 0 in ids  # exact-match self-retrieval


def test_bulk_build_rejects_non_2d(empty_db):
    db = empty_db
    with pytest.raises(Exception):
        db.bulk_build(np.zeros(8, dtype=np.float32))  # 1-D -> value_error
