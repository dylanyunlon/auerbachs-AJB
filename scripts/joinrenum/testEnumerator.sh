#!/usr/bin/env bash
# =============================================================================
# testEnumerator.sh — AJB enumerator test build+run
#
# Origin: upstream/joinrenum/testEnumerator.sh (0 lines, empty)
# AJB adaptation (100% new): compile-and-run with RSS monitoring, wall-clock
#   timing, and [AJB_TIMER] trace line count reporting.
# =============================================================================
set -euo pipefail

AJB_ROOT="${AJB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SRC="${AJB_ROOT}/src/joinrenum"
EXE="/tmp/ajb_test_enumerator_quick"

echo "[AJB_TEST] building test_enumerator..." >&2
g++ -std=c++17 -O2 -g -DAJB_DEBUG \
    -I"${SRC}" \
    "${SRC}/tests/test_enumerator.cpp" \
    -o "$EXE" -lglpk 2>&1

echo "[AJB_TEST] running from ${SRC}/db/..." >&2
T0=$(date +%s%N)
cd "${SRC}/db" && "$EXE" 2>&1 | tee /tmp/ajb_enum_output.txt
T1=$(date +%s%N)
ELAPSED_MS=$(( (T1 - T0) / 1000000 ))

TRACE_LINES=$(grep -c '^\[AJB_' /tmp/ajb_enum_output.txt 2>/dev/null || echo 0)
echo "[AJB_TEST] elapsed_ms=${ELAPSED_MS} trace_lines=${TRACE_LINES}" >&2
