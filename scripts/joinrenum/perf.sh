#!/usr/bin/env bash
# =============================================================================
# perf.sh — AJB-instrumented perf flamegraph pipeline
#
# Origin: upstream/joinrenum/perf.sh (5 lines, hardcoded paths)
# AJB adaptation (~35%): DWARF-mode call graph (--call-graph dwarf replaces
#   frame-pointer -g for inlined C++ code), auto-discover FlameGraph toolkit
#   via PATH/$AJB_ROOT/../../FlameGraph, CPU governor check for stable perf
#   data, RSS/timing [AJB_PERF] tags before and after recording.
# =============================================================================
set -euo pipefail

AJB_ROOT="${AJB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PERF_DIR="${AJB_ROOT}/perf"
mkdir -p "$PERF_DIR"

TARGET="${1:-testEnumerator.cpp}"
TARGET_BASE="${TARGET%.cpp}"
EXE="${PERF_DIR}/${TARGET_BASE}.perf.exe"

# --- AJB: CPU governor diagnostic ---
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
echo "[AJB_PERF] cpu_governor=${GOV}" >&2
if [ "$GOV" != "performance" ] && [ "$GOV" != "unknown" ]; then
    echo "[AJB_PERF] WARN governor=${GOV} (recommend 'performance' for stable results)" >&2
fi

# --- compile with debug info for DWARF unwinding ---
echo "[AJB_PERF] compiling ${TARGET} → ${EXE}" >&2
g++ "${AJB_ROOT}/src/joinrenum/tests/${TARGET_BASE}.cpp" \
    -I"${AJB_ROOT}/src/joinrenum" \
    -O2 -g -fno-omit-frame-pointer -std=c++17 \
    -o "$EXE" -lglpk 2>&1

RSS_BEFORE=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
echo "[AJB_PERF] mem_available_mb=${RSS_BEFORE} (before recording)" >&2

# --- AJB: DWARF call-graph mode for accurate C++ stacks ---
PERF_DATA="${PERF_DIR}/perf.data"
echo "[AJB_PERF] recording with DWARF call-graph..." >&2
T_START=$(date +%s%N)
perf record --call-graph dwarf -e cpu-clock -o "$PERF_DATA" "$EXE" 2>&1 || {
    echo "[AJB_PERF] WARN dwarf failed, fallback to frame-pointer mode" >&2
    perf record -g -e cpu-clock -o "$PERF_DATA" "$EXE" 2>&1
}
T_END=$(date +%s%N)
ELAPSED_MS=$(( (T_END - T_START) / 1000000 ))
echo "[AJB_PERF] recording_ms=${ELAPSED_MS}" >&2

# --- auto-discover FlameGraph ---
FLAMEGRAPH_DIR=""
for candidate in \
    "${AJB_ROOT}/../FlameGraph" \
    "${AJB_ROOT}/../../FlameGraph" \
    "${HOME}/FlameGraph" \
    "/opt/FlameGraph"; do
    if [ -f "${candidate}/stackcollapse-perf.pl" ]; then
        FLAMEGRAPH_DIR="$candidate"
        break
    fi
done

perf script -i "$PERF_DATA" > "${PERF_DIR}/perf.unfold" 2>&1

if [ -n "$FLAMEGRAPH_DIR" ]; then
    echo "[AJB_PERF] FlameGraph found at ${FLAMEGRAPH_DIR}" >&2
    "${FLAMEGRAPH_DIR}/stackcollapse-perf.pl" "${PERF_DIR}/perf.unfold" \
        > "${PERF_DIR}/perf.folded" 2>&1
    "${FLAMEGRAPH_DIR}/flamegraph.pl" "${PERF_DIR}/perf.folded" \
        > "${PERF_DIR}/perf.svg" 2>&1
    echo "[AJB_PERF] flamegraph=${PERF_DIR}/perf.svg" >&2
else
    echo "[AJB_PERF] WARN FlameGraph not found — raw perf.unfold available at ${PERF_DIR}/perf.unfold" >&2
    echo "[AJB_PERF] install: git clone https://github.com/brendangregg/FlameGraph" >&2
fi

RSS_AFTER=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
echo "[AJB_PERF] mem_available_mb=${RSS_AFTER} (after recording)" >&2
echo "[AJB_PERF] DONE target=${TARGET}" >&2
