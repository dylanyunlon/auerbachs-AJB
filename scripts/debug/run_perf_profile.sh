#!/bin/bash
# =============================================================================
# run_perf_profile.sh — AJB perf + flamegraph profiling wrapper
#
# Origin: upstream/joinrenum/perf.sh (6 lines)
# Adaptation (~20%): AJB target selection, automatic FlameGraph path
#   detection, NUMA-aware execution, and structured output paths.
#
# Usage: ./scripts/debug/run_perf_profile.sh [binary] [--numa NODE]
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PERF_DIR="$PROJECT_ROOT/perf"
BINARY="${1:-$PROJECT_ROOT/build_tests/test_enumerator}"
NUMA_NODE="${2:-}"

mkdir -p "$PERF_DIR"

echo "[AJB] Profiling: $BINARY"

# Build if needed
if [ ! -f "$BINARY" ]; then
    echo "[AJB] Binary not found, attempting build..."
    SRC="${BINARY##*/}"
    g++ -O2 -g -fno-omit-frame-pointer -std=c++17 \
        -I"$PROJECT_ROOT/src/joinrenum" \
        "$PROJECT_ROOT/src/joinrenum/tests/${SRC}.cpp" \
        -o "$BINARY" -lglpk 2>&1 || {
            echo "[AJB_ERROR] Build failed"; exit 1;
        }
fi

# Run with optional NUMA binding
RUN_CMD="$BINARY"
if [ -n "$NUMA_NODE" ]; then
    RUN_CMD="numactl --cpubind=$NUMA_NODE --membind=$NUMA_NODE $BINARY"
    echo "[AJB] NUMA binding: node $NUMA_NODE"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERF_DATA="$PERF_DIR/perf_${TIMESTAMP}.data"

echo "[AJB] Recording perf data..."
perf record -g -e cpu-clock -o "$PERF_DATA" $RUN_CMD 2>&1

echo "[AJB] Generating flamegraph..."
perf script -i "$PERF_DATA" > "$PERF_DIR/perf.unfold" 2>/dev/null

# Try to find FlameGraph scripts
FG_COLLAPSE=""
for path in ../FlameGraph ./FlameGraph /opt/FlameGraph; do
    if [ -f "$path/stackcollapse-perf.pl" ]; then
        FG_COLLAPSE="$path/stackcollapse-perf.pl"
        FG_FLAME="$path/flamegraph.pl"
        break
    fi
done

if [ -n "$FG_COLLAPSE" ]; then
    $FG_COLLAPSE "$PERF_DIR/perf.unfold" > "$PERF_DIR/perf.folded"
    $FG_FLAME "$PERF_DIR/perf.folded" > "$PERF_DIR/flamegraph_${TIMESTAMP}.svg"
    echo "[AJB] Flamegraph: $PERF_DIR/flamegraph_${TIMESTAMP}.svg"
else
    echo "[AJB_WARN] FlameGraph not found — install to ../FlameGraph or /opt/FlameGraph"
    echo "[AJB] Raw perf data: $PERF_DATA"
fi

echo "[AJB] Profiling DONE"
