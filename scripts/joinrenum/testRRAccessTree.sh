#!/usr/bin/env bash
# =============================================================================
# testRRAccessTree.sh — AJB RRAccessTree test runner
#
# Origin: upstream/joinrenum/testRRAccessTree.sh (0 lines, empty)
# AJB adaptation (100% new): compile + run, capture path-compression
#   diagnostics from [AJB_STATE] tags.
# =============================================================================
set -euo pipefail

AJB_ROOT="${AJB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SRC="${AJB_ROOT}/src/joinrenum"
EXE="/tmp/ajb_test_rrat_quick"

echo "[AJB_TEST] building test_rr_access_tree..." >&2
g++ -std=c++17 -O2 -g -DAJB_DEBUG \
    -I"${SRC}" \
    "${SRC}/tests/test_rr_access_tree.cpp" \
    -o "$EXE" -lglpk 2>&1

T0=$(date +%s%N)
cd "${SRC}/db" && timeout 60 "$EXE" 2>&1 | tee /tmp/ajb_rrat_output.txt
T1=$(date +%s%N)
MS=$(( (T1 - T0) / 1000000 ))

STATE_LINES=$(grep -c '^\[AJB_STATE\]' /tmp/ajb_rrat_output.txt 2>/dev/null || echo 0)
echo "[AJB_TEST] elapsed_ms=${MS} state_dumps=${STATE_LINES}" >&2
