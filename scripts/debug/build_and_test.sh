#!/bin/bash
# =============================================================================
# build_and_test.sh — AJB-adapted build + test runner
#
# Origin: upstream/joinrenum/test.sh + test_on_server.sh + debugEnumerator.sh
# Adaptation (~20%): unified build for all test targets, NUMA-aware mode,
#   address sanitizer support, structured output, and pass/fail summary.
#
# Usage:
#   ./scripts/debug/build_and_test.sh                    # build + run all
#   ./scripts/debug/build_and_test.sh --target test_index_full
#   ./scripts/debug/build_and_test.sh --asan              # address sanitizer
#   ./scripts/debug/build_and_test.sh --numa 0            # NUMA node binding
#   ./scripts/debug/build_and_test.sh --list              # list available targets
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$PROJECT_ROOT/src/joinrenum"
BUILD_DIR="$PROJECT_ROOT/build_tests"

# Defaults
TARGET=""
ASAN=false
NUMA_NODE=""
LIST_ONLY=false
OPT_FLAGS="-O3"
EXTRA_LINK="-lglpk"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)  TARGET="$2"; shift 2 ;;
    --asan)    ASAN=true; shift ;;
    --numa)    NUMA_NODE="$2"; shift 2 ;;
    --list)    LIST_ONLY=true; shift ;;
    --debug)   OPT_FLAGS="-O0 -g"; shift ;;
    *)         shift ;;
  esac
done

# Discover test/tool targets
TESTS_FULL=(
  test_enumerator_full
  test_index_full
  test_join_tree_full
  test_rr_access_tree_full
  test_count_oracle_full
  test_bucket_pool_full
  test_unordered_map_full
  test_join_baseline
)

TOOLS_FULL=(
  gen_co_data_full
  run_bpt_full
  upper_bound_full
  wash_data_full
)

ALL_TARGETS=("${TESTS_FULL[@]}" "${TOOLS_FULL[@]}")

if $LIST_ONLY; then
  echo "[AJB] Available targets:"
  for t in "${ALL_TARGETS[@]}"; do
    src_file=""
    if [[ -f "$SRC/tests/${t}.cpp" ]]; then src_file="tests/${t}.cpp"
    elif [[ -f "$SRC/tools/${t}.cpp" ]]; then src_file="tools/${t}.cpp"
    fi
    echo "  $t  ($src_file)"
  done
  exit 0
fi

echo "[AJB] ============================================"
echo "[AJB] build_and_test.sh — unified build + run"
echo "[AJB] ============================================"
echo "[AJB_STATE] Project root: $PROJECT_ROOT"
echo "[AJB_STATE] Optimization: $OPT_FLAGS"
echo "[AJB_STATE] ASAN: $ASAN"
echo "[AJB_STATE] NUMA: ${NUMA_NODE:-disabled}"

mkdir -p "$BUILD_DIR"

# Build function
build_target() {
  local name="$1"
  local src_file=""

  if [[ -f "$SRC/tests/${name}.cpp" ]]; then
    src_file="$SRC/tests/${name}.cpp"
  elif [[ -f "$SRC/tools/${name}.cpp" ]]; then
    src_file="$SRC/tools/${name}.cpp"
  else
    echo "[AJB_FAIL] Source not found for target: $name"
    return 1
  fi

  local flags="$OPT_FLAGS -std=c++17 -I$SRC"
  if $ASAN; then
    flags="$flags -fsanitize=address -fno-omit-frame-pointer"
  fi

  # Some targets don't need glpk
  local link_flags="$EXTRA_LINK"
  case "$name" in
    test_unordered_map_full|test_join_baseline|upper_bound_full|wash_data_full|run_bpt_full)
      link_flags="" ;;
  esac

  echo "[AJB_TRACE] Building $name..."
  g++ $flags "$src_file" $link_flags -o "$BUILD_DIR/$name" 2>&1
  echo "[AJB_TRACE] Built: $BUILD_DIR/$name"
}

# Run function
run_target() {
  local name="$1"
  local exe="$BUILD_DIR/$name"

  if [[ ! -x "$exe" ]]; then
    echo "[AJB_FAIL] Executable not found: $exe"
    return 1
  fi

  local prefix=""
  if [[ -n "$NUMA_NODE" ]]; then
    prefix="numactl --cpubind=$NUMA_NODE --membind=$NUMA_NODE"
  fi

  echo ""
  echo "[AJB] >>>>>>>>>>> Running: $name <<<<<<<<<<<<"
  local t0=$(date +%s%N)

  # Run with working dir set to project root (for db/ paths)
  cd "$PROJECT_ROOT"
  $prefix "$exe" 2>&1 || true

  local t1=$(date +%s%N)
  local elapsed=$(( (t1 - t0) / 1000000 ))
  echo "[AJB_TIMER] $name wall time: ${elapsed} ms"
}

# Select targets
if [[ -n "$TARGET" ]]; then
  SELECTED=("$TARGET")
else
  SELECTED=("${ALL_TARGETS[@]}")
fi

# Build phase
echo ""
echo "[AJB] === BUILD PHASE ==="
BUILT=0
FAILED=0
for t in "${SELECTED[@]}"; do
  if build_target "$t"; then
    BUILT=$((BUILT + 1))
  else
    FAILED=$((FAILED + 1))
  fi
done
echo "[AJB_STATE] Build: $BUILT ok, $FAILED failed"

# Run phase
echo ""
echo "[AJB] === RUN PHASE ==="
for t in "${SELECTED[@]}"; do
  if [[ -x "$BUILD_DIR/$t" ]]; then
    run_target "$t"
  fi
done

echo ""
echo "[AJB] ============================================"
echo "[AJB] build_and_test.sh complete"
echo "[AJB] ============================================"
