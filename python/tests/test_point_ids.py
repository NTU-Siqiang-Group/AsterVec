"""Any point id: int (0 .. 2**64-1), str or uuid.UUID."""
import os
import shutil
import tempfile
import uuid

import numpy as np
import pytest

import astervec

DIM = 8


def options(path, mode=None):
    opts = astervec.AsterVecDBOptions()
    opts.dim = DIM
    opts.vec_file_capacity = 64
    opts.vector_file_path = os.path.join(path, "vecs.bin")
    if mode is not None:
        opts.id_mode = mode
    return opts


@pytest.fixture
def path():
    p = tempfile.mkdtemp(prefix="astervec_py_ids_")
    try:
        yield p
    finally:
        shutil.rmtree(p, ignore_errors=True)


def vec(seed):
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


def test_new_python_stores_are_keyed(path):
    db = astervec.AsterVecDB.open(path, options(path))
    assert db.id_mode == "keyed"
    db.close()


def test_every_kind_of_id_round_trips(path):
    u = uuid.uuid4()
    ids = [0, 2**64 - 1, "doc-1#chunk-2", u, "7", 7]
    db = astervec.AsterVecDB.open(path, options(path))
    for i, pid in enumerate(ids):
        db.insert(pid, vec(i), metadata={"i": i})
    for i in range(40):
        db.insert(f"filler-{i}", vec(100 + i))
    db.close()

    db = astervec.AsterVecDB.open(path, options(path))
    assert db.id_mode == "keyed"
    for i, pid in enumerate(ids):
        expect = str(pid) if isinstance(pid, uuid.UUID) else pid
        hits = db.search(vec(i), k=1)
        assert hits[0]["id"] == expect
        assert type(hits[0]["id"]) is type(expect)
        assert db.get_payload(pid) == {"i": i}
        assert db.get(pid).shape == (DIM,)
    # The int 7 and the string "7" are different points.
    assert db.get_payload(7) == {"i": 5}
    assert db.get_payload("7") == {"i": 4}

    db.update("doc-1#chunk-2", vec(999))
    assert db.search(vec(999), k=1)[0]["id"] == "doc-1#chunk-2"
    db.update_payload(u, {"tag": "x"})
    assert db.search(vec(3), k=1, filter={"tag": "x"})[0]["id"] == str(u)
    db.delete(u)
    with pytest.raises(KeyError):
        db.get(u)
    db.delete("never-written")  # idempotent
    db.close()


def test_bad_ids_are_rejected(path):
    db = astervec.AsterVecDB.open(path, options(path))
    with pytest.raises(TypeError):
        db.insert(True, vec(1))
    with pytest.raises(TypeError):
        db.insert(1.5, vec(1))
    with pytest.raises(ValueError):
        db.insert(-1, vec(1))
    with pytest.raises(ValueError):
        db.insert(2**64, vec(1))
    db.close()


def test_dense_stores_keep_integer_ids(path):
    db = astervec.AsterVecDB.open(path, options(path, astervec.IdMode.Dense))
    assert db.id_mode == "dense"
    db.insert(3, vec(3))
    with pytest.raises(ValueError):
        db.insert("a", vec(1))
    db.close()
    # Reopening with the default options keeps the store dense.
    db = astervec.AsterVecDB.open(path, options(path))
    assert db.id_mode == "dense"
    assert db.search(vec(3), k=1)[0]["id"] == 3
    db.close()
    with pytest.raises(ValueError):
        astervec.AsterVecDB.open(path, options(path, astervec.IdMode.Keyed))


def test_bulk_build_with_ids(path):
    db = astervec.AsterVecDB.open(path, options(path))
    n = 200
    data = np.stack([vec(i) for i in range(n)])
    ids = [str(uuid.UUID(int=i + 1)) for i in range(n)]
    report = db.bulk_build(data, threads=2, ids=ids)
    assert report["n"] == n
    found = sum(db.search(vec(i), k=1)[0]["id"] == ids[i] for i in range(0, n, 10))
    assert found >= 19
    db.insert("after", vec(5000))
    assert db.search(vec(5000), k=1)[0]["id"] == "after"
    db.close()
