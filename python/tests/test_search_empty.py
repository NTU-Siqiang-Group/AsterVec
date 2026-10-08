"""A search with nothing to return gives [] instead of raising."""
import os
import shutil
import tempfile

import numpy as np
import pytest

import astervec


@pytest.fixture
def fresh_db():
    path = tempfile.mkdtemp(prefix="astervec_py_empty_")
    opts = astervec.AsterVecDBOptions()
    opts.dim = 8
    opts.vec_file_capacity = 1000
    opts.vector_file_path = os.path.join(path, "vecs.bin")
    db = astervec.AsterVecDB.open(path, opts)
    try:
        yield db
    finally:
        db.close()
        shutil.rmtree(path, ignore_errors=True)


def test_empty_index(fresh_db):
    q = np.ones(8, dtype=np.float32)
    assert fresh_db.search(q, k=5) == []
    assert fresh_db.search(q, k=5, filter={"tag": "a"}) == []


def test_all_deleted(fresh_db):
    rng = np.random.default_rng(1)
    for i in range(20):
        fresh_db.insert(i, rng.standard_normal(8).astype(np.float32))
    for i in range(20):
        fresh_db.delete(i)
    assert fresh_db.search(np.ones(8, dtype=np.float32), k=5) == []


def test_filter_matches_nothing(fresh_db):
    rng = np.random.default_rng(2)
    for i in range(20):
        fresh_db.insert(i, rng.standard_normal(8).astype(np.float32), metadata={"tag": "a"})
    q = rng.standard_normal(8).astype(np.float32)
    assert fresh_db.search(q, k=5, filter={"tag": "zzz"}) == []
    assert len(fresh_db.search(q, k=5, filter={"tag": "a"})) == 5
