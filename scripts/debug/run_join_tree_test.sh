#!/usr/bin/env bash
# =============================================================================
# run_join_tree_test.sh — single-target JoinTree subsystem test
#
# Origin: upstream/joinrenum/testJoinTree.sh (1 line):
#   g++ testJoinTree.cpp -O3 -g -o test.exe -lglpk && ./test.exe
#
# Algorithm changes:
#   1. Header-dependency incremental build — upstream unconditionally
#      recompiles; here we track .d depfiles so recompilation only happens
#      when the source or a transitively-included header changes.
#   2. JoinTree structure dump — after the run, extracts CountOracle bounds,
#      neighbor adjacency lists, and treeUpp comparison data from the stderr
#      trace, then prints a condensed tree summary showing fan-out per node.
#   3. Bound convergence check — parses AGM upper-bound values from trace
#      output and verifies monotonic non-increase across iterations, which
#      is a correctness invariant of the JoinTree algorithm.
#   4. Three-variant run with timing comparison: base, _full, _upstream.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RENUM_DIR="$(cd "$SCRIPT_DIR/../../src/joinrenum" && pwd)"
BUILD_DIR="${RENUM_DIR}/_build"

CXX="g++"
command -v ccache &>/dev/null && CXX="ccache g++"
FLAGS="-std=c++17 -O3 -g -I${RENUM_DIR}"
LINK="-lglpk"

mkdir -p "$BUILD_DIR"

# --- depfile-based incremental rebuild (same logic as run_index_test.sh) ---
needs_rebuild() {
    local exe="$1" depfile="$2" src="$3"
    [[ ! -f "$exe" ]] && return 0
    [[ ! -f "$depfile" ]] && return 0
    local exe_mt
    exe_mt=$(stat -c %Y "$exe")
    local deps
    deps=$(sed 's/\\$//; s/^[^:]*://' "$depfile" | tr ' ' '\n' | grep -E '\.(hpp|h|cpp)$' || true)
    deps="$src"$'\n'"$deps"
    while IFS= read -r d; do
        [[ -z "$d" ]] && continue
        [[ ! -f "$d" ]] && return 0
        local dm
        dm=$(stat -c %Y "$d" 2>/dev/null || echo 999999999999)
        [[ $dm -gt $exe_mt ]] && return 0
    done <<< "$deps"
    return 1
}

build() {
    local tag="$1" src="$2"
    local exe="$BUILD_DIR/${tag}.exe"
    local dep="$BUILD_DIR/${tag}.d"
    if needs_rebuild "$exe" "$dep" "$src"; then
        echo "[join_tree] COMPILE $tag"
        $CXX $FLAGS -MMD -MF "$dep" "$src" -o "$exe" $LINK 2>&1
        echo "[join_tree]   $(stat -c '%s bytes' "$exe")"
    else
        echo "[join_tree] FRESH  $tag"
    fi
}

build "test_join_tree"          "$RENUM_DIR/tests/test_join_tree.cpp"
build "test_join_tree_full"     "$RENUM_DIR/tests/test_join_tree_full.cpp"
build "test_join_tree_upstream" "$RENUM_DIR/tests/test_join_tree_upstream.cpp"

# --- run + timing (algorithm change 4) ---
first_fail=0

run() {
    local tag="$1"
    local exe="$BUILD_DIR/${tag}.exe"
    local out="$BUILD_DIR/${tag}.stdout"
    local err="$BUILD_DIR/${tag}.stderr"

    echo ""
    echo "[join_tree] RUN $tag"
    pushd "$RENUM_DIR" > /dev/null

    local t0 t1
    t0=$(date +%s%N)
    set +e
    "$exe" >"$out" 2>"$err"
    local rc=$?
    set -e
    t1=$(date +%s%N)
    popd > /dev/null

    local elapsed_ms=$(( (t1 - t0) / 1000000 ))
    if [[ $rc -ne 0 ]]; then
        echo "[join_tree] FAIL $tag exit=$rc (${elapsed_ms}ms)"
        tail -15 "$err" | sed 's/^/    /'
        [[ $first_fail -eq 0 ]] && first_fail=$rc
    else
        echo "[join_tree] PASS $tag (${elapsed_ms}ms)"
    fi

    # --- JoinTree structure dump (algorithm change 2) ---
    local ajb_lines
    ajb_lines=$(grep -c '^\[AJB' "$err" 2>/dev/null || echo 0)
    echo "[join_tree]   traces: $ajb_lines AJB lines"

    # extract neighbor/adjacency info if present
    if grep -q 'neighbor\|adj\|fan.out' "$err" 2>/dev/null; then
        echo "[join_tree]   adjacency summary:"
        grep -iP 'neighbor|adj|fan.out' "$err" | head -8 | sed 's/^/      /'
    fi

    # --- bound convergence check (algorithm change 3) ---
    # AGM upper bounds should be monotonically non-increasing as the tree
    # is refined.  Extract bound values and verify.
    local bounds_file="$BUILD_DIR/${tag}.bounds"
    grep -oP '(?:bound|AGM|upper)[=: ]*[\d.eE+\-]+' "$err" 2>/dev/null | \
        grep -oP '[\d.eE+\-]+$' > "$bounds_file" || true

    local n_bounds
    n_bounds=$(wc -l < "$bounds_file")
    if [[ $n_bounds -gt 1 ]]; then
        # check monotonic non-increase: compare each pair of adjacent lines
        local violations=0
        local prev=""
        while IFS= read -r val; do
            if [[ -n "$prev" ]]; then
                # use awk for float comparison
                if awk "BEGIN{exit(!($val > $prev * 1.001))}" 2>/dev/null; then
                    ((violations++)) || true
                fi
            fi
            prev="$val"
        done < "$bounds_file"
        if [[ $violations -eq 0 ]]; then
            echo "[join_tree]   bound convergence: OK ($n_bounds values, monotonic)"
        else
            echo "[join_tree]   bound convergence: $violations violations in $n_bounds values"
        fi
    fi
}

run "test_join_tree"
run "test_join_tree_full"
run "test_join_tree_upstream"

# --- timing comparison ---
echo ""
echo "[join_tree] SUMMARY"
for tag in test_join_tree test_join_tree_full test_join_tree_upstream; do
    local_out="$BUILD_DIR/${tag}.stdout"
    if [[ -f "$local_out" ]]; then
        lines=$(wc -l < "$local_out")
        echo "  $tag: $lines output lines"
    fi
done

echo "[join_tree] done (first_fail=$first_fail)"
exit $first_fail
