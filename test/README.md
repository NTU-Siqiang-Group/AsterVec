# Test Sample Build

This directory contains a standalone test sample (`test.cc`) and a dedicated Makefile
to build it without CMake.

Unlike the older setup (compiling AsterVec sources directly), this test links against
the compiled static library.

## Prerequisites

1) Build Aster (RocksDB fork) from the repo root:

```bash
git submodule update --init --recursive
make aster
```

* Ensure system dependencies are installed (see the repo root `README.md`).

## Build

From this `test` directory:

```bash
make
```

This produces `./astervec_test`.

## Run

Example usage (same flags as the main binary):

```bash
./astervec_test \
  --db ../run/db \
  --data-dir ../data/sift_100k \
  --out ../run/output.txt
```

## Clean

```bash
make clean
```
