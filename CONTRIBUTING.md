# Contributing to AsterVec

Thanks for your interest in AsterVec, an embeddable vector engine for local retrieval
workloads. This guide covers building from source, running tests, and submitting
changes.

## Development setup

```bash
git clone --recurse-submodules https://github.com/NTU-Siqiang-Group/LSM-Vec.git
cd AsterVec

# If you cloned without --recurse-submodules:
git submodule update --init --recursive

make aster        # build the Aster (RocksDB fork) static lib — required first
make              # build libastervec.{a,so}, the test binary, and astervec_http
make unit_test    # build + run the test suite
```

**Prerequisites:** a C++17 compiler (GCC 8+ / Clang 10+), CMake ≥ 3.10, GNU Make,
Boost (headers only), and **zstd** (the only required compression library — Aster
is built zstd-only). On macOS also install `jemalloc`. See the
[README](README.md#build-from-source) for the package commands.

To work on the Python bindings: `python -m pip install .` (after `make aster`).

## Project layout

```
include/   public C++ headers + internals      python/    pybind11 bindings
src/       C++ sources (engine + astervec_http)  test/      test binary + unit tests
lib/aster/ Aster submodule (RocksDB fork)       docs/      user-facing docs
examples/  runnable quickstarts                 data/      dataset prep scripts
```

## Tests

```bash
make unit_test                 # doctest-based unit suite
make unit_test_asan            # AddressSanitizer build
make unit_test_tsan            # ThreadSanitizer build
make unit_test_ubsan           # UndefinedBehaviorSanitizer build
```

Please make sure `make unit_test` passes before opening a pull request, and add
tests for new behavior where practical.

## Coding conventions

- C++17; keep code in the `astervec` namespace (Aster types via `ROCKSDB_NAMESPACE`).
- Match the style of the surrounding code (naming, comment density, idioms).
- Error handling uses `Status` (aliased from RocksDB); return descriptive statuses.

## Pull requests

1. Branch from `main`; keep each PR focused on one change.
2. Build cleanly and ensure the test suite passes.
3. Describe what changed and why; reference any related issue.
4. Update the relevant docs (`README.md`, `docs/`) when you change behavior or APIs.

## Reporting issues

Open a GitHub issue with steps to reproduce, your platform/compiler, and the
expected vs. actual behavior. For build problems, include the failing command and
its output.

## License

By contributing, you agree that your contributions are licensed under the
[Apache-2.0 License](LICENSE), the same license as the project.
