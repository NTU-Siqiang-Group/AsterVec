# LSM-Vec HTTP server image.
#
# Two-stage build: compile the engine + HTTP server, then ship a slim runtime.
#
# Build (from repo root):
#   git submodule update --init --recursive   # populate lib/aster
#   docker build -t lsmvec:latest .
#
# Run:
#   docker run -d --name lsmvec \
#     -p 8000:8000 \
#     -v "$(pwd)/data:/data" \
#     -e LSMVEC_DIM=128 -e LSMVEC_METRIC=l2 \
#     lsmvec:latest

# ---- build stage ----
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libboost-dev libzstd-dev libsnappy-dev liblz4-dev libbz2-dev zlib1g-dev \
    libjemalloc-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# lib/aster/ must already be populated in the build context. The
# .dockerignore excludes .git/, so an in-container `git submodule update`
# would fail — fail fast with a useful message instead.
RUN test -f lib/aster/CMakeLists.txt \
    || (echo ""; \
        echo "ERROR: lib/aster/CMakeLists.txt is missing from the build context."; \
        echo "Run 'git submodule update --init --recursive' on the host before"; \
        echo "invoking 'docker build'. The .dockerignore excludes .git/ so the"; \
        echo "container cannot init submodules itself."; \
        echo ""; \
        exit 1)
RUN make aster
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLSMVEC_BUILD_HTTP=ON
RUN cmake --build build --target lsm_vec_http -j
# Strip symbols to keep the runtime image small.
RUN strip --strip-unneeded build/lib/liblsmvec.so build/bin/lsm_vec_http

# ---- runtime stage ----
FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzstd1 libsnappy1v5 liblz4-1 libbz2-1.0 zlib1g libjemalloc2 \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
# Binary depends on the LSM-Vec shared library; ship both.
COPY --from=build /src/build/bin/lsm_vec_http /usr/local/bin/
COPY --from=build /src/build/lib/liblsmvec.so /usr/local/lib/
RUN ldconfig

# Default config — every value is also overridable by env var.
ENV LSMVEC_DATA_DIR=/data \
    LSMVEC_PORT=8000 \
    LSMVEC_HTTP_THREADS=1 \
    LSMVEC_METRIC=l2

EXPOSE 8000
VOLUME ["/data"]
STOPSIGNAL SIGTERM

# The container runs as root; rely on container-level isolation
# (--read-only root FS, --cap-drop=ALL, --pids-limit, memory/cpu limits)
# at `docker run` time. Switch to a non-root user if your host volume
# ownership model requires it.
RUN mkdir -p /data

HEALTHCHECK --interval=15s --timeout=3s --retries=3 --start-period=10s \
  CMD curl -fsS http://localhost:8000/ready || exit 1

ENTRYPOINT ["/usr/local/bin/lsm_vec_http"]
