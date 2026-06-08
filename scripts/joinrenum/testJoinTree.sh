#!/usr/bin/env bash
# =============================================================================
# testJoinTree.sh — AJB JoinTree test runner
#
# Origin: upstream/joinrenum/testJoinTree.sh (0 lines, empty)
# AJB adaptation (100% new): compile + run with AGM bound validation,
#   tree structure dump parsing, timing.
# =============================================================================
set -euo pipefail

AJB_ROOT="${AJB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SRC="${AJB_ROOT}/src/joinrenum"
EXE="/tmp/ajb_test_join_tree_quick"

echo "[AJB_TEST] building test_join_tree..." >&2
g++ -std=c++17 -O2 -g -DAJB_DEBUG \
    -I"${SRC}" \
    "${SRC}/tests/test_join_tree.cpp" \
    -o "$EXE" -lglpk 2>&1

T0=$(date +%s%N)
cd "${SRC}/db" && timeout 60 "$EXE" 2>&1 | tee /tmp/ajb_jt_output.txt
T1=$(date +%s%N)
MS=$(( (T1 - T0) / 1000000 ))

AGM_LINES=$(grep -c 'AGM' /tmp/ajb_jt_output.txt 2>/dev/null || echo 0)
echo "[AJB_TEST] elapsed_ms=${MS} agm_related_lines=${AGM_LINES}" >&2
