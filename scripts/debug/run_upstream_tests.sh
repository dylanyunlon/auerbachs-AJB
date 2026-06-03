#!/bin/bash
# =============================================================================
# run_upstream_tests.sh — upstream test builder/runner
#
# Origin: upstream/joinrenum/ shell scripts (test.sh, testEnumerator.sh,
#   testIndex.sh, testJoinTree.sh, testRRAccessTree.sh, test_on_server.sh,
#   perf.sh, build.sh, format.sh)
#
# Changes from upstream:
#   - Merged 9 scripts into one dispatch table
#   - Incremental compile: skip if .exe is newer than .cpp
#   - Optimization via $AJB_OPT (upstream mixed O0/O2/O3)
#   - NUMA auto-detect from test_on_server.sh
#   - ASan/UBSan via --sanitize
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$PROJECT_ROOT/src/joinrenum"
OUT="$PROJECT_ROOT/build_upstream"
OPT="${AJB_OPT:-O2}"
CXX="${CXX:-g++}"
SANITIZE=""
NUMA=""

mkdir -p "$OUT"

# parse args
TARGET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sanitize) SANITIZE="-fsanitize=address,undefined -fno-omit-frame-pointer"; OPT="O0" ;;
        --numa)
            if command -v numactl &>/dev/null; then
                NUMA="numactl --cpubind=0 --membind=0"
            fi ;;
        --opt) shift; OPT="$1" ;;
        --list)
            echo "tests: test_count_oracle_upstream test_index_upstream test_join_tree_upstream"
            echo "       test_enumerator_upstream test_rr_access_tree_upstream"
            echo "       test_bucket_pool_upstream test_unordered_map_upstream"
            echo "tools: gen_co_data_upstream run_bpt_upstream upper_bound_upstream wash_data_upstream"
            exit 0 ;;
        *) TARGET="$1" ;;
    esac
    shift
done

# which targets need glpk
needs_glpk() {
    case $1 in
        test_count_oracle_upstream|test_index_upstream|test_join_tree_upstream|\
        test_enumerator_upstream|test_rr_access_tree_upstream|\
        test_bucket_pool_upstream|gen_co_data_upstream) return 0 ;;
        *) return 1 ;;
    esac
}

# where is the source file
src_path() {
    case $1 in
        test_*) echo "$SRC/tests/$1.cpp" ;;
        *)      echo "$SRC/tools/$1.cpp" ;;
    esac
}

# incremental compile: skip if exe newer than source
compile() {
    local name="$1"
    local src="$(src_path "$name")"
    local exe="$OUT/$name"
    local glpk=""
    needs_glpk "$name" && glpk="-lglpk"

    if [[ ! -f "$src" ]]; then
        echo "ERROR: $src not found"
        return 1
    fi
    if [[ -f "$exe" && "$exe" -nt "$src" ]]; then
        return 0
    fi
    $CXX -$OPT -g $SANITIZE -I"$SRC" "$src" -o "$exe" $glpk
}

run() {
    local name="$1"
    compile "$name" || return 1
    cd "$PROJECT_ROOT"
    echo "=== $name ==="
    $NUMA "$OUT/$name"
    echo "=== exit: $? ==="
}

ALL="test_count_oracle_upstream test_index_upstream test_join_tree_upstream \
test_enumerator_upstream test_rr_access_tree_upstream test_bucket_pool_upstream \
test_unordered_map_upstream gen_co_data_upstream run_bpt_upstream \
upper_bound_upstream wash_data_upstream"

if [[ -n "$TARGET" ]]; then
    run "$TARGET"
else
    for t in $ALL; do
        run "$t" || true
    done
fi
