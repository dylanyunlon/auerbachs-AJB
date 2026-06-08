#!/bin/bash
# =============================================================================
# run_perf_profile.sh — AJB perf + flamegraph profiling wrapper
#
# Origin: upstream/joinrenum/perf.sh (6 lines)
# Merged: run_perf_full.sh target dispatch, GLPK-aware linking, perf report
# Adaptation (~25%): --target source auto-discovery in tests/tools dirs,
#   GLPK-aware link dispatch, NUMA binding, timestamped perf data, FlameGraph
#   auto-search, perf report top-function extraction, [AJB_BP] state dumps.
#
# Usage:
#   ./scripts/debug/run_perf_profile.sh [binary_or_target] [--numa NODE]
#   ./scripts/debug/run_perf_profile.sh --target test_enumerator
#   ./scripts/debug/run_perf_profile.sh --target test_index --numa 0
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$PROJECT_ROOT/src/joinrenum"
BUILD_DIR="$PROJECT_ROOT/build_tests"
PERF_DIR="$PROJECT_ROOT/perf"
NUMA_NODE=""
TARGET=""
BINARY=""

# --- Arg parsing: supports both positional and --target/--numa flags ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)     TARGET="$2"; shift 2 ;;
    --numa)       NUMA_NODE="$2"; shift 2 ;;
    *)            BINARY="$1"; shift ;;
  esac
done

mkdir -p "$PERF_DIR" "$BUILD_DIR"

# --- GLPK-aware link dispatch ---
needs_glpk() {
  case "$1" in
    test_bucket_pool|test_count_oracle|test_enumerator|test_index|\
    test_join_tree|test_rr_access_tree|test_renum_baseline|\
    test_sample_baseline|gen_co_data) return 0 ;;
    *) return 1 ;;
  esac
}

# --- Source auto-discovery ---
find_source() {
  local name="$1"
  for dir in "$SRC/tests" "$SRC/tools" "$SRC"; do
    if [[ -f "$dir/${name}.cpp" ]]; then
      echo "$dir/${name}.cpp"
      return 0
    fi
  done
  return 1
}

# --- Resolve binary from --target if needed ---
if [[ -n "$TARGET" && -z "$BINARY" ]]; then
  src_file=$(find_source "$TARGET") || { echo "[AJB_ERROR] Source not found for $TARGET"; exit 1; }
  BINARY="$BUILD_DIR/${TARGET}_perf"

  link_flags=""
  needs_glpk "$TARGET" && link_flags="-lglpk"

  echo "[AJB_BP] compile: target=$TARGET src=$src_file flags=-O2 -g -fno-omit-frame-pointer $link_flags"
  g++ -O2 -g -fno-omit-frame-pointer -std=c++17 -I"$SRC" \
      "$src_file" $link_flags -o "$BINARY" 2>&1 || {
        echo "[AJB_ERROR] Build failed for $TARGET"; exit 1;
      }
  echo "[AJB_BP] binary=$BINARY size=$(stat -c%s "$BINARY" 2>/dev/null || echo unknown)"
elif [[ -z "$BINARY" ]]; then
  BINARY="$BUILD_DIR/test_enumerator"
fi

# Build from name if binary missing
if [[ ! -f "$BINARY" ]]; then
    echo "[AJB] Binary not found, attempting build..."
    name="${BINARY##*/}"
    name="${name%_perf}"
    src_file=$(find_source "$name") || { echo "[AJB_ERROR] No source for $name"; exit 1; }
    link_flags=""
    needs_glpk "$name" && link_flags="-lglpk"
    g++ -O2 -g -fno-omit-frame-pointer -std=c++17 -I"$SRC" \
        "$src_file" $link_flags -o "$BINARY" 2>&1 || {
            echo "[AJB_ERROR] Build failed"; exit 1;
        }
fi

echo "[AJB] Profiling: $BINARY"

# Run with optional NUMA binding
RUN_CMD="$BINARY"
if [[ -n "$NUMA_NODE" ]] && command -v numactl &>/dev/null; then
    RUN_CMD="numactl --cpubind=$NUMA_NODE --membind=$NUMA_NODE $BINARY"
    echo "[AJB_BP] numa_bind=node$NUMA_NODE"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERF_DATA="$PERF_DIR/perf_${TIMESTAMP}.data"

if command -v perf &>/dev/null; then
  echo "[AJB] Recording perf data..."
  perf record -g -e cpu-clock -o "$PERF_DATA" $RUN_CMD 2>&1

  echo "[AJB] Generating script output..."
  perf script -i "$PERF_DATA" > "$PERF_DIR/perf.unfold" 2>/dev/null
  echo "[AJB_BP] perf_unfold_lines=$(wc -l < "$PERF_DIR/perf.unfold")"

  # Try to find FlameGraph scripts
  FG_COLLAPSE=""
  for path in ../FlameGraph ./FlameGraph /opt/FlameGraph "$HOME/FlameGraph"; do
      if [[ -f "$path/stackcollapse-perf.pl" ]]; then
          FG_COLLAPSE="$path/stackcollapse-perf.pl"
          FG_FLAME="$path/flamegraph.pl"
          break
      fi
  done

  if [[ -n "$FG_COLLAPSE" ]]; then
      $FG_COLLAPSE "$PERF_DIR/perf.unfold" > "$PERF_DIR/perf.folded"
      $FG_FLAME "$PERF_DIR/perf.folded" > "$PERF_DIR/flamegraph_${TIMESTAMP}.svg"
      echo "[AJB] Flamegraph: $PERF_DIR/flamegraph_${TIMESTAMP}.svg"
  else
      echo "[AJB_WARN] FlameGraph not found — install to ../FlameGraph or /opt/FlameGraph"
  fi

  # Top functions from perf report
  echo "[AJB_STATE] --- Top 15 functions ---"
  perf report -i "$PERF_DATA" --stdio --no-children 2>/dev/null | head -30 || true
else
  echo "[AJB_WARN] perf not available, running target directly for timing"
  time $RUN_CMD 2>&1
fi

echo "[AJB] Profiling DONE"
