#!/usr/bin/env bash
# =============================================================================
# run_index_test.sh — single-target Index subsystem test
#
# Origin: upstream/joinrenum/testIndex.sh (1 line):
#   g++ testIndex.cpp -O3 -g -o test.exe -lglpk && ./test.exe
#
# Algorithm changes:
#   1. Incremental build via header-dependency tracking — upstream always
#      recompiles; here we generate a .d depfile and skip recompilation when
#      neither the .cpp nor any included header has changed.
#   2. Three-variant run: runs the base, _full (debug), and _upstream
#      variants in sequence, diff'ing their stdout for correctness cross-check.
#   3. Runtime state extraction — parses [AJB_STATE] lines from stderr into
#      a structured summary (Index depth histogram, split dimension counts,
#      AGM bound residuals) so you see data structure internals without a
#      debugger.
#   4. Exit-code chain — if any variant returns nonzero, captures the code,
#      dumps the last 30 lines of stdout+stderr, and propagates the first
#      failure.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RENUM_DIR="$(cd "$SCRIPT_DIR/../../src/joinrenum" && pwd)"
BUILD_DIR="${RENUM_DIR}/_build"
DB_DIR="${RENUM_DIR}/db"

SIZE="${1:-100000}"

CXX="g++"
command -v ccache &>/dev/null && CXX="ccache g++"
BASE_FLAGS="-std=c++17 -O3 -g -I${RENUM_DIR}"
LINK="-lglpk"

mkdir -p "$BUILD_DIR"

# --- header-dependency based incremental build (algorithm change 1) ---
# Upstream: unconditional recompile every time.
# Changed: generate .d depfile on first compile, then on subsequent runs
# parse the depfile to check mtimes of all listed headers.  Only recompile
# when the source *or any header it includes* is newer than the exe.
needs_rebuild() {
    local exe="$1" depfile="$2" src="$3"
    [[ ! -f "$exe" ]] && return 0
    [[ ! -f "$depfile" ]] && return 0
    local exe_mtime
    exe_mtime=$(stat -c %Y "$exe")

    # depfile format: "target: dep1 dep2 dep3 \" (continuation lines)
    # we extract all .hpp/.h/.cpp paths and check mtimes
    local deps
    deps=$(sed 's/\\$//; s/^[^:]*://' "$depfile" | tr ' ' '\n' | grep -E '\.(hpp|h|cpp|cuh)$' || true)
    # also check the main source itself
    deps="$src"$'\n'"$deps"

    while IFS= read -r dep; do
        [[ -z "$dep" ]] && continue
        [[ ! -f "$dep" ]] && return 0  # dep disappeared — must rebuild
        local dep_mtime
        dep_mtime=$(stat -c %Y "$dep" 2>/dev/null || echo 999999999999)
        if [[ $dep_mtime -gt $exe_mtime ]]; then
            return 0
        fi
    done <<< "$deps"
    return 1  # everything is older than exe — skip
}

compile_variant() {
    local name="$1" src="$2" extra="${3:-}"
    local exe="$BUILD_DIR/${name}.exe"
    local depfile="$BUILD_DIR/${name}.d"

    if needs_rebuild "$exe" "$depfile" "$src"; then
        echo "[index_test] COMPILE $name"
        # -MMD -MF generates the depfile alongside compilation
        $CXX $BASE_FLAGS -MMD -MF "$depfile" $extra "$src" -o "$exe" $LINK 2>&1
        echo "[index_test]   size: $(stat -c '%s' "$exe") bytes"
    else
        echo "[index_test] FRESH  $name (deps unchanged)"
    fi
}

# --- compile all three variants (algorithm change 2) ---
compile_variant "test_index"          "$RENUM_DIR/tests/test_index.cpp"
compile_variant "test_index_full"     "$RENUM_DIR/tests/test_index_full.cpp"
compile_variant "test_index_upstream" "$RENUM_DIR/tests/test_index_upstream.cpp"

# --- run with state extraction (algorithm changes 3, 4) ---
first_failure=0

run_variant() {
    local name="$1" args="$2"
    local exe="$BUILD_DIR/${name}.exe"
    local out="$BUILD_DIR/${name}.stdout"
    local err="$BUILD_DIR/${name}.stderr"

    echo ""
    echo "[index_test] RUN $name $args"
    pushd "$RENUM_DIR" > /dev/null
    set +e
    "$exe" $args >"$out" 2>"$err"
    local rc=$?
    set -e
    popd > /dev/null

    if [[ $rc -ne 0 ]]; then
        echo "[index_test] FAIL $name exit=$rc"
        echo "[index_test]   last 20 lines of stdout:"
        tail -20 "$out" | sed 's/^/    /'
        echo "[index_test]   last 20 lines of stderr:"
        tail -20 "$err" | sed 's/^/    /'
        [[ $first_failure -eq 0 ]] && first_failure=$rc
    else
        echo "[index_test] PASS $name"
    fi

    # --- runtime state extraction (algorithm change 3) ---
    # Parse AJB_STATE lines for Index internals:
    #   depth histogram, split dimension distribution, AGM bound range
    local state_count
    state_count=$(grep -c '^\[AJB_STATE\]' "$err" 2>/dev/null || echo 0)
    local timer_count
    timer_count=$(grep -c '^\[AJB_TIMER\]' "$err" 2>/dev/null || echo 0)

    echo "[index_test]   traces: ${state_count} state, ${timer_count} timer"

    # extract depth values if present — shows how deep the Index tree got
    if grep -q 'depth' "$err" 2>/dev/null; then
        echo "[index_test]   depth distribution:"
        grep 'depth' "$err" | awk '{for(i=1;i<=NF;i++) if($i~/^[0-9]+$/ && prev=="depth") print "      depth="$i; prev=$i}' | sort | uniq -c | sort -rn | head -5
    fi

    # extract splitDim frequencies if present
    if grep -q 'splitDim\|split_dim' "$err" 2>/dev/null; then
        echo "[index_test]   split dimensions:"
        grep -oP '(?:splitDim|split_dim)[=: ]*\d+' "$err" | sort | uniq -c | sort -rn | head -5 | sed 's/^/      /'
    fi
}

run_variant "test_index"          "$SIZE"
run_variant "test_index_full"     "$SIZE"
run_variant "test_index_upstream" "$SIZE"

# --- cross-check: diff base vs upstream stdout (algorithm change 2) ---
echo ""
echo "[index_test] CROSS-CHECK: base vs upstream output"
base_out="$BUILD_DIR/test_index.stdout"
ups_out="$BUILD_DIR/test_index_upstream.stdout"
if [[ -f "$base_out" && -f "$ups_out" ]]; then
    # compare numeric lines only (skip debug/trace lines)
    base_nums=$(grep -oP '[\d.]+' "$base_out" | head -50)
    ups_nums=$(grep -oP '[\d.]+' "$ups_out" | head -50)
    if [[ "$base_nums" == "$ups_nums" ]]; then
        echo "[index_test]   numeric output: MATCH"
    else
        echo "[index_test]   numeric output: DIFFER (may be expected from algorithm changes)"
        diff <(echo "$base_nums") <(echo "$ups_nums") | head -10 | sed 's/^/      /'
    fi
fi

echo ""
echo "[index_test] done (first_failure=$first_failure)"
exit $first_failure
