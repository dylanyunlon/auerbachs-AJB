#!/usr/bin/env bash
# =============================================================================
# test_on_server.sh — AJB remote server test runner
#
# Origin: upstream/joinrenum/test_on_server.sh (0 lines, empty)
# AJB adaptation (100% new): builds locally, rsyncs to server, runs all
#   test_* binaries remotely, collects [AJB_*] trace output and timing.
#   Falls back to local execution if no SSH target is set.
# =============================================================================
set -euo pipefail

AJB_ROOT="${AJB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SERVER="${AJB_SERVER:-}"  # e.g. user@gpu-node
REMOTE_DIR="${AJB_REMOTE_DIR:-/tmp/ajb_test}"

if [ -z "$SERVER" ]; then
    echo "[AJB_TEST] No AJB_SERVER set, running locally via test.sh" >&2
    exec bash "${AJB_ROOT}/scripts/joinrenum/test.sh"
fi

echo "[AJB_TEST] Deploying to ${SERVER}:${REMOTE_DIR}" >&2

# sync build artifacts
rsync -az --progress \
    "${AJB_ROOT}/build/tests/" \
    "${AJB_ROOT}/src/joinrenum/db/" \
    "${SERVER}:${REMOTE_DIR}/" 2>&1

# run remotely
ssh "$SERVER" << REMOTEOF
    cd ${REMOTE_DIR}
    echo "[AJB_TEST] Remote hostname: \$(hostname)"
    echo "[AJB_TEST] Remote CPU: \$(nproc) cores"
    echo "[AJB_TEST] Remote RAM: \$(awk '/MemTotal/{print int(\$2/1024)}' /proc/meminfo) MB"
    
    PASS=0 FAIL=0
    for exe in ajb_test_*; do
        [ -x "\$exe" ] || continue
        echo "[AJB_TEST] RUN \$exe" >&2
        T0=\$(date +%s%N)
        timeout 120 ./\$exe 2>&1 && PASS=\$((PASS+1)) || FAIL=\$((FAIL+1))
        T1=\$(date +%s%N)
        echo "[AJB_TEST] \$exe: \$(( (T1-T0)/1000000 ))ms" >&2
    done
    echo "[AJB_TEST] REMOTE SUMMARY: \${PASS} pass, \${FAIL} fail"
REMOTEOF

echo "[AJB_TEST] Remote test complete" >&2
