#!/usr/bin/env bash
# =============================================================================
# build_joinrenum_targets.sh — Unified build + run for joinrenum tests/tools
#
# Origin: upstream/joinrenum/ shell scripts (7 files, each 1-2 lines):
#   debugEnumerator.sh, testEnumerator.sh, testIndex.sh, testJoinTree.sh,
#   testRRAccessTree.sh, test.sh, test_on_server.sh
# Algorithm changes (~30%):
#   1. Target dispatch: single script routes to any target via argv
#      (upstream: separate script per target, hardcoded filenames)
#   2. Sanitizer selection: --asan/--tsan/--ubsan flags
#      (upstream: only debugEnumerator.sh had -fsanitize=address)
#   3. Incremental build: compare .cpp mtime vs .exe mtime, skip if fresh
#      (upstream: always recompile unconditionally)
#   4. NUMA binding: auto-detect numactl, bind to node 0 if available
#      (upstream: only test_on_server.sh had hardcoded numactl)
#   5. ccache detection: prepend ccache to g++ if found
#      (upstream: bare g++ always)
#   6. Post-build run with stderr capture to .trace.log
#      (upstream: just ./test.exe with no output capture)
#
# Usage:
#   ./build_joinrenum_targets.sh                    # build+run all targets
#   ./build_joinrenum_targets.sh index              # build+run testIndex only
#   ./build_joinrenum_targets.sh enumerator --asan  # build with AddressSanitizer
#   ./build_joinrenum_targets.sh --list             # show available targets
#   ./build_joinrenum_targets.sh --build-only       # compile without running
#   ./build_joinrenum_targets.sh all --numa         # all targets with NUMA bind
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RENUM_DIR="$(cd "$SCRIPT_DIR/../../src/joinrenum" && pwd)"
BUILD_DIR="${RENUM_DIR}/_build"
DB_DIR="${RENUM_DIR}/db"

# --- target registry ---
# map short name → source file (relative to RENUM_DIR)
# upstream had: testEnumerator.sh → g++ testEnumerator.cpp
#               testIndex.sh → g++ testIndex.cpp   etc.
declare -A TARGET_MAP=(
    [enumerator]="tests/test_enumerator_full.cpp"
    [index]="tests/test_index_full.cpp"
    [join_tree]="tests/test_join_tree_full.cpp"
    [rr_access]="tests/test_rr_access_tree_full.cpp"
    [count_oracle]="tests/test_count_oracle_full.cpp"
    [bucket_pool]="tests/test_bucket_pool_full.cpp"
    [unordered_map]="tests/test_unordered_map_full.cpp"
    [join_baseline]="tests/test_join_baseline_full.cpp"
    [test]="test.cpp"
    [testjoin]="testjoin.cpp"
    [gen_co_data]="tools/gen_co_data_full.cpp"
    [run_bpt]="tools/run_bpt_full.cpp"
    [upper_bound]="tools/upper_bound_full.cpp"
    [wash_data]="tools/wash_data_full.cpp"
)

# --- default flags (upstream used -O3 -g for most, -O2 for perf, -O0 for debug) ---
CXX_BASE="-std=c++17 -O3 -g -I${RENUM_DIR}"
LINK_FLAGS="-lglpk"
SANITIZER=""
RUN_AFTER_BUILD=1
USE_NUMA=0

# --- parse arguments ---
TARGETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --asan)     SANITIZER="-fsanitize=address -fno-omit-frame-pointer -O0" ;;
        --tsan)     SANITIZER="-fsanitize=thread -O1" ;;
        --ubsan)    SANITIZER="-fsanitize=undefined -fno-omit-frame-pointer" ;;
        --numa)     USE_NUMA=1 ;;
        --build-only) RUN_AFTER_BUILD=0 ;;
        --list)
            echo "Available targets:"
            for t in $(echo "${!TARGET_MAP[@]}" | tr ' ' '\n' | sort); do
                echo "  $t  →  ${TARGET_MAP[$t]}"
            done
            exit 0
            ;;
        --help|-h)
            sed -n '2,/^set /{ /^#/s/^# \?//p }' "$0"
            exit 0
            ;;
        *)
            if [[ "$1" == "all" ]]; then
                TARGETS=("${!TARGET_MAP[@]}")
            elif [[ -v "TARGET_MAP[$1]" ]]; then
                TARGETS+=("$1")
            else
                echo "Unknown target: $1 (use --list to see available)" >&2
                exit 1
            fi
            ;;
    esac
    shift
done

# default: build all
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("${!TARGET_MAP[@]}")
fi

# --- detect ccache (algorithm change 5) ---
CXX="g++"
if command -v ccache &>/dev/null; then
    CXX="ccache g++"
    echo "[build] ccache detected, using: $CXX"
fi

# --- detect numactl (algorithm change 4) ---
NUMA_PREFIX=""
if [[ $USE_NUMA -eq 1 ]]; then
    if command -v numactl &>/dev/null; then
        # bind to NUMA node 0 — upstream test_on_server.sh hardcoded this
        NUMA_PREFIX="numactl --cpubind=0 --membind=0"
        echo "[build] NUMA binding: $NUMA_PREFIX"
    else
        echo "[build] --numa requested but numactl not found, ignoring" >&2
    fi
fi

mkdir -p "$BUILD_DIR"

# --- build + run loop ---
PASS=0
FAIL=0
SKIP=0

for target in $(echo "${TARGETS[@]}" | tr ' ' '\n' | sort); do
    src="${RENUM_DIR}/${TARGET_MAP[$target]}"
    exe="${BUILD_DIR}/${target}.exe"
    trace_log="${BUILD_DIR}/${target}.trace.log"

    if [[ ! -f "$src" ]]; then
        echo "[build] SKIP $target: source not found ($src)" >&2
        ((SKIP++)) || true
        continue
    fi

    # --- incremental build: compare mtime (algorithm change 3) ---
    # upstream: always recompile. here: skip if exe is newer than source
    needs_build=1
    if [[ -f "$exe" ]]; then
        src_mtime=$(stat -c %Y "$src" 2>/dev/null || echo 0)
        exe_mtime=$(stat -c %Y "$exe" 2>/dev/null || echo 0)
        if [[ $exe_mtime -gt $src_mtime && -z "$SANITIZER" ]]; then
            echo "[build] FRESH $target (exe newer than source)"
            needs_build=0
        fi
    fi

    if [[ $needs_build -eq 1 ]]; then
        echo "[build] COMPILE $target: $src"
        # upstream: bare g++ with hardcoded flags
        # changed: ccache + sanitizer + structured error capture
        compile_cmd="$CXX $CXX_BASE $SANITIZER $src -o $exe $LINK_FLAGS"
        echo "[build]   cmd: $compile_cmd"
        if ! eval "$compile_cmd" 2>&1; then
            echo "[build] FAIL $target: compilation error" >&2
            ((FAIL++)) || true
            continue
        fi
        echo "[build]   exe: $(stat -c '%s bytes' "$exe")"
    fi

    if [[ $RUN_AFTER_BUILD -eq 0 ]]; then
        ((PASS++)) || true
        continue
    fi

    # --- run with stderr capture (algorithm change 6) ---
    echo "[build] RUN $target"
    run_cmd="$NUMA_PREFIX $exe"
    # cd to RENUM_DIR so db/ paths resolve (upstream scripts ran from same dir)
    pushd "$RENUM_DIR" > /dev/null
    set +e
    eval "$run_cmd" 2>"$trace_log"
    rc=$?
    set -e
    popd > /dev/null

    if [[ $rc -eq 0 ]]; then
        echo "[build] PASS $target (exit=$rc)"
        ((PASS++)) || true
    else
        echo "[build] FAIL $target (exit=$rc)" >&2
        ((FAIL++)) || true
    fi

    # debug: show AJB trace summary from stderr log
    if [[ -f "$trace_log" ]]; then
        ajb_lines=$(grep -c '^\[AJB' "$trace_log" 2>/dev/null || echo 0)
        timer_lines=$(grep -c '^\[AJB_TIMER\]' "$trace_log" 2>/dev/null || echo 0)
        warn_lines=$(grep -c '^\[AJB_WARN\]' "$trace_log" 2>/dev/null || echo 0)
        echo "[build]   trace: ${ajb_lines} AJB lines, ${timer_lines} timers, ${warn_lines} warnings"
        echo "[build]   log: $trace_log"
    fi
done

echo ""
echo "==========================================="
echo "[build] SUMMARY: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
echo "==========================================="

[[ $FAIL -eq 0 ]]
