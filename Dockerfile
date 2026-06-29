# AsterVec HTTP server image.
#
# Packages the optional standalone REST server (astervec_http). The engine itself
# is embeddable (a C++ library / Python module); this image is only for running it
# as a network service. See docs/HTTP_API.md.
#
# Build (from the repo root):
#   git submodule update --init --recursive   # populate lib/aster
#   docker build -t astervec:latest .
#
# Run:
#   docker run -d --name astervec \
#     -p 8000:8000 \
#     -v "$(pwd)/data:/data" \
#     -e ASTERVEC_DIM=128 -e ASTERVEC_METRIC=l2 \
#     astervec:latest

# ---- build stage ----
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libboost-dev libzstd-dev \
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
        -DASTERVEC_BUILD_HTTP=ON
RUN cmake --build build --target astervec_http -j
# Strip symbols to keep the runtime image small.
RUN strip --strip-unneeded build/bin/astervec_http

# ---- runtime stage ----
FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzstd1 libjemalloc2 \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
# astervec_http is statically linked against the engine — ship just the binary.
COPY --from=build /src/build/bin/astervec_http /usr/local/bin/

# Default config — every value is also overridable by env var (see docs/HTTP_API.md).
ENV ASTERVEC_DATA_DIR=/data \
    ASTERVEC_PORT=8000 \
    ASTERVEC_HTTP_THREADS=1 \
    ASTERVEC_METRIC=l2

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

ENTRYPOINT ["/usr/local/bin/astervec_http"]
