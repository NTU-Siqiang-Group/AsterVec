#!/usr/bin/env bash
# Drive the optional HTTP server (astervec_http) over curl: create an index,
# insert, search, inspect. See docs/HTTP_API.md for the full reference.
#
# Start the server first, e.g.:
#   ASTERVEC_DATA_DIR=./data ./build/bin/astervec_http
#   # or: docker run -d -p 8000:8000 -v "$(pwd)/data:/data" astervec:latest
#
# Then run:  ./examples/http_quickstart.sh
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
H='content-type: application/json'

echo "== health =="
curl -fsS "$BASE/health"; echo

echo "== create the index (dim=4, L2) =="
curl -fsS -X PUT "$BASE/v1/index" -H "$H" -d '{"dim": 4, "metric": "l2"}'; echo

echo "== insert a few vectors (+ metadata) =="
curl -fsS "$BASE/v1/vectors" -H "$H" -d '{"id": 1, "vector": [0.1, 0.1, 0.1, 0.1], "metadata": {"category": "docs"}}'; echo
curl -fsS "$BASE/v1/vectors" -H "$H" -d '{"id": 2, "vector": [0.9, 0.8, 0.7, 0.6], "metadata": {"category": "blog"}}'; echo

echo "== search (k=2) =="
curl -fsS "$BASE/v1/search" -H "$H" -d '{"vector": [0.1, 0.1, 0.1, 0.12], "k": 2}'; echo

echo "== filtered search (category == docs) =="
curl -fsS "$BASE/v1/search" -H "$H" \
  -d '{"vector": [0.1, 0.1, 0.1, 0.12], "k": 5, "filter": {"category": {"$eq": "docs"}}}'; echo

echo "== stats =="
curl -fsS "$BASE/v1/stats"; echo
