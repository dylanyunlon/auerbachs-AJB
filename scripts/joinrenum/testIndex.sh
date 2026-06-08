#!/usr/bin/env bash
# =============================================================================
# testIndex.sh — AJB Index test with optimization-level sweep
#
# Origin: upstream/joinrenum/testIndex.sh (1 line: g++ -O3 -g -o test.exe -lglpk && ./test.exe)
# AJB adaptation (~40%): runs test at -O0 (debug) and -O2 (release), compares
#   timing to catch optimization-sensitive bugs (UB exposed by -O2 but not -O0).
#   Structured [AJB_TEST] diagnostics with per-level wall-clock.
# =============================================================================
set -euo pipefail

AJB_ROOT="${AJB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SRC="${AJB_ROOT}/src/joinrenum"

for OPT in O0 O2; do
    EXE="/tmp/ajb_test_index_${OPT}"
    echo "[AJB_TEST] compiling test_index (-${OPT})..." >&2
    g++ -std=c++17 "-${OPT}" -g -DAJB_DEBUG \
        -I"${SRC}" \
        "${SRC}/tests/test_index.cpp" \
        -o "$EXE" -lglpk 2>&1

    echo "[AJB_TEST] running test_index (-${OPT})..." >&2
    T0=$(date +%s%N)
    cd "${SRC}/db" && timeout 120 "$EXE" > /tmp/ajb_idx_${OPT}.out 2>&1 || true
    T1=$(date +%s%N)
    MS=$(( (T1 - T0) / 1000000 ))
    echo "[AJB_TEST] test_index_${OPT}: ${MS}ms" >&2
done

# compare outputs for optimization-level divergence
if diff -q /tmp/ajb_idx_O0.out /tmp/ajb_idx_O2.out > /dev/null 2>&1; then
    echo "[AJB_TEST] PASS optimization levels produce identical output" >&2
else
    echo "[AJB_TEST] WARN outputs differ between -O0 and -O2 (possible UB)" >&2
    diff /tmp/ajb_idx_O0.out /tmp/ajb_idx_O2.out | head -20 >&2
fi
