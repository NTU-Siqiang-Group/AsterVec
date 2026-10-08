#!/usr/bin/env python3
"""Crash-injection harness for AsterVec crash recovery.

A child process opens a store and runs a deterministic stream of upserts and
deletes, printing one line per acknowledged operation. The parent kill -9s it
at a random point, reopens the store, and checks every id against a model:

* log on  (--oplog 1): every acknowledged operation is reflected; the one in
  flight at the kill may or may not be.
* log off (--oplog 0): the store equals the state at the last acknowledged
  checkpoint (or the one in flight at the kill).

Each series of runs reuses one store, so later runs crash a store that was
itself recovered from a crash.

Usage:
  PYTHONPATH=build-py/python python3 test/crash/crash_harness.py --oplog 1 --runs 200
"""
import argparse
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

DIM = 32
ID_SPACE = 3000
TOL = 0.05  # SQ8 round-trip error is far below this


def vector(seed):
    return np.random.default_rng(seed).normal(size=DIM).astype("float32")


def op_at(series_seed, i):
    """Operation i of a series: (kind, id, vector seed)."""
    r = random.Random(series_seed * 1_000_003 + i)
    kind = "del" if r.random() < 0.15 else "put"
    return kind, r.randrange(ID_SPACE), series_seed * 10_000_000 + i


def options(astervec, path, oplog, interval):
    o = astervec.AsterVecDBOptions()
    o.dim = DIM
    o.vec_file_capacity = ID_SPACE
    o.vector_file_path = path + "/vecs.bin"
    o.log_file_path = path + "/engine.log"
    o.enable_oplog = oplog
    o.checkpoint_interval_ops = interval
    return o


# ---------------------------------------------------------------- child
def child(args):
    import astervec
    db = astervec.AsterVecDB.open(args.path, options(astervec, args.path, args.oplog, 0))
    out = sys.stdout
    i = args.start
    while True:
        kind, vid, vseed = op_at(args.series_seed, i)
        if kind == "put":
            db.insert(vid, vector(vseed).tolist(), metadata={"s": vseed})
        else:
            db.delete(vid)
        out.write(f"op {i}\n")
        out.flush()
        i += 1
        if i % args.ckpt_every == 0:
            out.write(f"ckpt-begin {i}\n")
            out.flush()
            db.checkpoint()
            out.write(f"ckpt {i}\n")
            out.flush()


# --------------------------------------------------------------- parent
def model_until(series_seed, n_ops, base=None):
    """State after ops [0, n_ops): id -> vector seed (absent = deleted)."""
    state = dict(base or {})
    for i in range(n_ops):
        kind, vid, vseed = op_at(series_seed, i)
        if kind == "put":
            state[vid] = vseed
        else:
            state.pop(vid, None)
    return state


def read_store(astervec, db):
    got = {}
    for vid in range(ID_SPACE):
        try:
            got[vid] = db.get(vid)
        except KeyError:
            pass
    return got


def matches(got, state, oplog, db):
    if set(got) != set(state):
        return False
    for vid, v in got.items():
        if np.max(np.abs(v - vector(state[vid]))) > TOL:
            return False
        if oplog and db.get_payload(vid) != {"s": state[vid]}:
            return False
    return True


def verify(astervec, db, series_seed, acked_ops, ckpts, in_flight_ckpt, oplog):
    got = read_store(astervec, db)
    if oplog:
        # Every acked op, plus maybe the one in flight.
        candidates = [acked_ops, acked_ops + 1]
    else:
        candidates = [ckpts[-1] if ckpts else 0]
        if in_flight_ckpt is not None:
            candidates.append(in_flight_ckpt)
    for n in candidates:
        if matches(got, model_until(series_seed, n), oplog, db):
            matched = n
            break
    else:
        return f"state matches none of {candidates} (have {len(got)} live ids)", None

    # Search must work and find live vectors.
    live = list(got)
    rng = random.Random(series_seed)
    sample = rng.sample(live, min(100, len(live)))
    hits = 0
    for vid in sample:
        res = db.search(got[vid].tolist(), k=1)
        hits += bool(res) and res[0]["id"] == vid
    for _ in range(20):
        db.search(np.random.default_rng(rng.randrange(1 << 30)).normal(size=DIM).tolist(), k=10)
    if sample and hits < 0.95 * len(sample):
        return f"exact-match search found {hits}/{len(sample)}", None
    return None, matched


def run_series(args, series_seed, n_runs, report):
    import astervec
    path = tempfile.mkdtemp(prefix="astervec_crash_")
    try:
        done_ops = 0      # ops applied before this run's child started
        ckpts = []
        for _ in range(n_runs):
            cmd = [sys.executable, __file__, "--child", "--path", path,
                   "--oplog", str(int(args.oplog)), "--series-seed", str(series_seed),
                   "--start", str(done_ops), "--ckpt-every", str(args.ckpt_every)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 env=dict(os.environ))
            time.sleep(random.uniform(args.min_s, args.max_s))
            os.kill(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            acked = done_ops
            in_flight_ckpt = None
            for line in out.decode().splitlines():
                parts = line.split()
                if parts[0] == "op":
                    acked = int(parts[1]) + 1
                    in_flight_ckpt = None
                elif parts[0] == "ckpt-begin":
                    in_flight_ckpt = int(parts[1])
                elif parts[0] == "ckpt":
                    ckpts.append(int(parts[1]))
                    in_flight_ckpt = None

            t0 = time.time()
            db = astervec.AsterVecDB.open(path, options(astervec, path, args.oplog, 0))
            open_s = time.time() - t0
            err, matched = verify(astervec, db, series_seed, acked, ckpts,
                                  in_flight_ckpt, args.oplog)
            stats = db.recovery_stats()
            db.close()
            report(err, acked - done_ops, stats, open_s)
            if err:
                return False
            # The store now holds the model state after `matched` ops, and
            # close() made that the checkpoint. The next child resumes there
            # (log off: the lost ops are simply issued again).
            done_ops = matched
            ckpts.append(matched)
        return True
    finally:
        shutil.rmtree(path, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--path")
    ap.add_argument("--oplog", type=int, default=1)
    ap.add_argument("--series-seed", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=400)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--runs-per-series", type=int, default=20)
    ap.add_argument("--min-s", type=float, default=0.5)
    ap.add_argument("--max-s", type=float, default=2.5)
    args = ap.parse_args()
    args.oplog = bool(args.oplog)
    if args.child:
        child(args)
        return

    failures = 0
    total = 0
    replayed = []
    ops = []

    def report(err, n_ops, stats, open_s):
        nonlocal failures, total
        total += 1
        ops.append(n_ops)
        replayed.append((stats["replayed_ops"], open_s))
        status = "FAIL " + err if err else "ok"
        print(f"run {total:3d}: {n_ops:5d} acked ops, replayed {stats['replayed_ops']:5d} "
              f"in {stats['seconds']:.2f}s (open {open_s:.2f}s) {status}", flush=True)
        if err:
            failures += 1

    series = 0
    while total < args.runs:
        series += 1
        n = min(args.runs_per_series, args.runs - total)
        if not run_series(args, series, n, report):
            break
    print(f"\n{'oplog on' if args.oplog else 'oplog off'}: {total - failures}/{total} runs passed; "
          f"{sum(ops)} acked ops; max replay {max(r for r, _ in replayed) if replayed else 0} ops")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
