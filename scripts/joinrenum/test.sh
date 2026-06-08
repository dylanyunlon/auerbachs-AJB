#!/usr/bin/env bash
# =============================================================================
# test.sh — AJB joinrenum test runner
#
# Origin: upstream/joinrenum/test.sh (0 lines, empty placeholder)
# AJB adaptation (100% new): discovers test_* executables in build/tests/,
#   runs each with timeout, captures [AJB_*] trace lines, reports pass/fail
#   with per-test timing. Exit code = number of failures.
# =============================================================================
set -euo pipefail

AJB_ROOT="${AJB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BUILD_DIR="${AJB_ROOT}/build"
TEST_DIR="${BUILD_DIR}/tests"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"
DB_DIR="${AJB_ROOT}/src/joinrenum/db"

if [ ! -d "$TEST_DIR" ]; then
    echo "[AJB_TEST] ERROR test_dir_missing=${TEST_DIR}" >&2
    echo "[AJB_TEST] run build.sh first" >&2
    exit 1
fi

PASS=0
FAIL=0
SKIP=0
RESULTS=()

echo "[AJB_TEST] ============================================" >&2
echo "[AJB_TEST] AJB joinrenum test suite" >&2
echo "[AJB_TEST] test_dir=${TEST_DIR}" >&2
echo "[AJB_TEST] timeout=${TIMEOUT_SEC}s" >&2
echo "[AJB_TEST] ============================================" >&2

for exe in "${TEST_DIR}"/ajb_test_*; do
    [ -x "$exe" ] || continue
    name=$(basename "$exe")

    T0=$(date +%s%N)
    echo "[AJB_TEST] RUN ${name}" >&2

    # run from db directory so relative paths work
    set +e
    output=$(cd "$DB_DIR" && timeout "$TIMEOUT_SEC" "$exe" 2>&1)
    rc=$?
    set -e

    T1=$(date +%s%N)
    elapsed_ms=$(( (T1 - T0) / 1000000 ))

    # extract AJB trace lines for diagnostics
    trace_count=$(echo "$output" | grep -c '^\[AJB_' || true)

    if [ $rc -eq 0 ]; then
        echo "[AJB_TEST] PASS ${name} (${elapsed_ms}ms, ${trace_count} trace lines)" >&2
        PASS=$((PASS + 1))
        RESULTS+=("PASS  ${name}  ${elapsed_ms}ms")
    elif [ $rc -eq 124 ]; then
        echo "[AJB_TEST] TIMEOUT ${name} (>${TIMEOUT_SEC}s)" >&2
        FAIL=$((FAIL + 1))
        RESULTS+=("TIMEOUT  ${name}  >${TIMEOUT_SEC}s")
    else
        echo "[AJB_TEST] FAIL ${name} (exit=${rc}, ${elapsed_ms}ms)" >&2
        # print last 20 lines of output for diagnosis
        echo "$output" | tail -20 >&2
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  ${name}  exit=${rc}")
    fi
done

echo "" >&2
echo "[AJB_TEST] ============================================" >&2
echo "[AJB_TEST] SUMMARY: ${PASS} passed, ${FAIL} failed" >&2
for r in "${RESULTS[@]}"; do
    echo "[AJB_TEST]   ${r}" >&2
done
echo "[AJB_TEST] ============================================" >&2

exit "$FAIL"
