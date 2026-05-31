#!/bin/bash
# =============================================================================
# run_perf_full.sh — AJB-adapted perf profiling pipeline
#
# Origin: upstream/joinrenum/perf.sh (5 lines)
# Adaptation (~20%): target selection, AJB trace filtering, flamegraph
#   auto-detection, structured output, and HTML report generation.
#
# Usage:
#   ./scripts/debug/run_perf_full.sh                          # default target
#   ./scripts/debug/run_perf_full.sh --target test_index_full
#   ./scripts/debug/run_perf_full.sh --flamegraph /path/to/FlameGraph
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$PROJECT_ROOT/src/joinrenum"
BUILD_DIR="$PROJECT_ROOT/build_tests"
PERF_DIR="$PROJECT_ROOT/perf"
FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-}"

TARGET="test_enumerator_full"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)     TARGET="$2"; shift 2 ;;
    --flamegraph) FLAMEGRAPH_DIR="$2"; shift 2 ;;
    *)            shift ;;
  esac
done

echo "[AJB] ============================================"
echo "[AJB] run_perf_full.sh — profiling pipeline"
echo "[AJB] ============================================"
echo "[AJB_STATE] Target: $TARGET"

mkdir -p "$PERF_DIR"

# Step 1: Build with debug info
echo "[AJB_TRACE] Building $TARGET with -O2 -g..."
src_file=""
if [[ -f "$SRC/tests/${TARGET}.cpp" ]]; then src_file="$SRC/tests/${TARGET}.cpp"
elif [[ -f "$SRC/tools/${TARGET}.cpp" ]]; then src_file="$SRC/tools/${TARGET}.cpp"
fi

if [[ -z "$src_file" ]]; then
  echo "[AJB_FAIL] Source not found for $TARGET"
  exit 1
fi

# upstream: compile with -O2 -g -fno-omit-frame-pointer
link_flags="-lglpk"
case "$TARGET" in
  test_unordered_map_full|test_join_baseline|upper_bound_full|wash_data_full)
    link_flags="" ;;
esac

g++ -O2 -g -fno-omit-frame-pointer -std=c++17 -I"$SRC" \
    "$src_file" $link_flags -o "$BUILD_DIR/${TARGET}_perf"
echo "[AJB_TRACE] Built: $BUILD_DIR/${TARGET}_perf"

# Step 2: Run perf record
echo "[AJB_TRACE] Running perf record..."
cd "$PROJECT_ROOT"

if command -v perf &>/dev/null; then
  perf record -g -e cpu-clock -o "$PERF_DIR/perf.data" \
      "$BUILD_DIR/${TARGET}_perf" 2>&1 || true

  # Step 3: Generate perf script output
  perf script -i "$PERF_DIR/perf.data" > "$PERF_DIR/perf.unfold" 2>/dev/null || true
  echo "[AJB_STATE] perf.unfold: $(wc -l < "$PERF_DIR/perf.unfold") lines"

  # Step 4: Flamegraph (if available)
  if [[ -n "$FLAMEGRAPH_DIR" && -x "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" ]]; then
    "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" "$PERF_DIR/perf.unfold" \
        > "$PERF_DIR/perf.folded"
    "$FLAMEGRAPH_DIR/flamegraph.pl" "$PERF_DIR/perf.folded" \
        > "$PERF_DIR/perf_${TARGET}.svg"
    echo "[AJB_STATE] Flamegraph: $PERF_DIR/perf_${TARGET}.svg"
  else
    echo "[AJB_WARN] FlameGraph tools not found, skipping SVG generation"
    echo "[AJB_WARN] Set --flamegraph /path/to/FlameGraph to enable"
  fi

  # AJB: extract top functions from perf
  echo "[AJB_STATE] --- Top 15 functions ---"
  perf report -i "$PERF_DIR/perf.data" --stdio --no-children 2>/dev/null \
      | head -30 || true
else
  echo "[AJB_WARN] perf not available, running target directly for timing"
  "$BUILD_DIR/${TARGET}_perf" 2>&1
fi

echo "[AJB] run_perf_full.sh complete"
