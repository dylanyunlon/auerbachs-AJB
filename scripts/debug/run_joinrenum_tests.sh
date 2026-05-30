#!/bin/bash
# =============================================================================
# run_joinrenum_tests.sh — AJB JoinREnum test suite runner
#
# Compiles and runs all joinrenum tests with AJB debug instrumentation.
# Each test produces structured [AJB_TIMER], [AJB_STATE], [AJB_RESULTS]
# output for easy grep/parsing.
#
# Usage: ./scripts/debug/run_joinrenum_tests.sh [--verbose] [--only TEST]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DIR="$PROJECT_ROOT/src/joinrenum/tests"
TOOL_DIR="$PROJECT_ROOT/src/joinrenum/tools"
DB_DIR="$PROJECT_ROOT/src/joinrenum/db"
BUILD_DIR="$PROJECT_ROOT/build_tests"

VERBOSE=""
ONLY=""

for arg in "$@"; do
  case $arg in
    --verbose|-v) VERBOSE="-v" ;;
    --only=*)     ONLY="${arg#*=}" ;;
  esac
done

mkdir -p "$BUILD_DIR"

# Colors for terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

run_test() {
    local name="$1"
    local src="$2"
    local extra_flags="${3:-}"
    local run_args="${4:-}"

    if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then
        ((SKIP++))
        return
    fi

    echo -e "${CYAN}━━━ Building: $name ━━━${NC}"

    local bin="$BUILD_DIR/$name"
    local compile_cmd="g++ -O2 -std=c++17 -I$PROJECT_ROOT/src/joinrenum $extra_flags $src -o $bin -lglpk 2>&1"

    if ! eval $compile_cmd; then
        echo -e "${YELLOW}[SKIP]${NC} $name — compilation failed (may need glpk or other deps)"
        ((SKIP++))
        return
    fi

    echo -e "${CYAN}━━━ Running: $name $run_args ━━━${NC}"
    cd "$DB_DIR/.." 2>/dev/null || cd "$PROJECT_ROOT"

    if $bin $run_args $VERBOSE 2>&1; then
        echo -e "${GREEN}[PASS]${NC} $name"
        ((PASS++))
    else
        echo -e "${RED}[FAIL]${NC} $name (exit code $?)"
        ((FAIL++))
    fi
    echo ""
}

echo "=============================================="
echo " AJB JoinREnum Test Suite"
echo " Project: $PROJECT_ROOT"
echo " Date: $(date -Iseconds)"
echo "=============================================="
echo ""

# --- Unit tests ---
run_test "test_bucket_pool"    "$TEST_DIR/test_bucket_pool.cpp"       "-I$PROJECT_ROOT/src/joinrenum"
run_test "test_unordered_map"  "$TEST_DIR/test_unordered_map.cpp"     "" "500000"

# --- Tests requiring db/ data ---
run_test "test_join_tree"      "$TEST_DIR/test_join_tree.cpp"         "-I$PROJECT_ROOT/src/joinrenum"
run_test "test_index"          "$TEST_DIR/test_index.cpp"             "-I$PROJECT_ROOT/src/joinrenum" "100000"
run_test "test_rr_access_tree" "$TEST_DIR/test_rr_access_tree.cpp"   "-I$PROJECT_ROOT/src/joinrenum"
run_test "test_count_oracle"   "$TEST_DIR/test_count_oracle.cpp"     "-I$PROJECT_ROOT/src/joinrenum"
run_test "test_enumerator"     "$TEST_DIR/test_enumerator.cpp"       "-I$PROJECT_ROOT/src/joinrenum"

# --- Tools (run as smoke tests) ---
run_test "upper_bound_demo"    "$TOOL_DIR/upper_bound_demo.cpp"       ""
run_test "run_bpt"             "$TOOL_DIR/run_bpt.cpp"                "-I$PROJECT_ROOT/src/joinrenum" "5"

echo "=============================================="
echo -e " Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC}"
echo "=============================================="

exit $FAIL
