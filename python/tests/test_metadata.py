import os
import shutil
import tempfile

import numpy as np
import pytest

import astervec


@pytest.fixture
def fresh_db():
    path = tempfile.mkdtemp(prefix="astervec_py_md_")
    opts = astervec.AsterVecDBOptions()
    opts.dim = 8
    opts.m = 8
    opts.m_max = 16
    opts.ef_construction = 32
    opts.vec_file_capacity = 1000
    opts.vector_file_path = os.path.join(path, "vecs.bin")
    db = astervec.AsterVecDB.open(path, opts)
    try:
        yield db, path
    finally:
        db.close()
        shutil.rmtree(path, ignore_errors=True)


def test_insert_with_metadata_and_get(fresh_db):
    db, _ = fresh_db
    db.insert(
        1,
        np.random.randn(8).astype(np.float32),
        metadata={"tenant": "acme", "count": 7},
    )
    md = db.get_payload(1)
    assert md == {"tenant": "acme", "count": 7}


def test_search_with_filter(fresh_db):
    db, _ = fresh_db
    rng = np.random.default_rng(42)
    for i in range(100):
        v = rng.standard_normal(8).astype(np.float32)
        md = {"tenant": "acme" if i % 10 == 0 else "other"}
        db.insert(i, v, metadata=md)
    db.flush_vector_writes()

    results = db.search(
        rng.standard_normal(8).astype(np.float32),
        k=5,
        filter={"tenant": "acme"},
    )
    assert len(results) == 5
    for r in results:
        assert r["id"] % 10 == 0


def test_update_payload_merge_patch(fresh_db):
    db, _ = fresh_db
    db.insert(1, np.zeros(8, dtype=np.float32), metadata={"a": 1})
    db.update_payload(1, {"b": 2})
    assert db.get_payload(1) == {"a": 1, "b": 2}

    db.update_payload(1, {"a": None})  # null -> delete
    assert db.get_payload(1) == {"b": 2}
