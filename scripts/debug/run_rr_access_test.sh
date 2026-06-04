#!/usr/bin/env bash
# =============================================================================
# run_rr_access_test.sh — single-target RRAccessTree subsystem test
#
# Origin: upstream/joinrenum/testRRAccessTree.sh (1 line):
#   g++ testRRAccessTree.cpp -O3 -g -o test.exe -lglpk && ./test.exe
#
# Algorithm changes:
#   1. Header-dependency incremental build (same depfile approach as Index).
#   2. Dimension-distribution profiling — collects dimension-access counts
#      from trace output and reports which dimensions are hot (heavily
#      queried) vs cold, which reveals imbalanced query patterns.
#   3. Latency percentile extraction — if timing traces are present, sorts
#      them and reports p50/p90/p99 access latencies, giving you the same
#      kind of tail-latency visibility you'd get from a production profiler.
#   4. Three-variant run with structural output comparison.
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
    local exe="$BUILD_DIR/${tag}.exe" dep="$BUILD_DIR/${tag}.d"
    if needs_rebuild "$exe" "$dep" "$src"; then
        echo "[rr_access] COMPILE $tag"
        $CXX $FLAGS -MMD -MF "$dep" "$src" -o "$exe" $LINK 2>&1
    else
        echo "[rr_access] FRESH  $tag"
    fi
}

build "test_rr_access_tree"          "$RENUM_DIR/tests/test_rr_access_tree.cpp"
build "test_rr_access_tree_full"     "$RENUM_DIR/tests/test_rr_access_tree_full.cpp"
build "test_rr_access_tree_upstream" "$RENUM_DIR/tests/test_rr_access_tree_upstream.cpp"

first_fail=0

run() {
    local tag="$1"
    local exe="$BUILD_DIR/${tag}.exe"
    local out="$BUILD_DIR/${tag}.stdout"
    local err="$BUILD_DIR/${tag}.stderr"

    echo ""
    echo "[rr_access] RUN $tag"
    pushd "$RENUM_DIR" > /dev/null
    local t0 t1
    t0=$(date +%s%N)
    set +e
    "$exe" >"$out" 2>"$err"
    local rc=$?
    set -e
    t1=$(date +%s%N)
    popd > /dev/null

    local ms=$(( (t1 - t0) / 1000000 ))
    if [[ $rc -ne 0 ]]; then
        echo "[rr_access] FAIL $tag exit=$rc (${ms}ms)"
        tail -15 "$err" | sed 's/^/    /'
        [[ $first_fail -eq 0 ]] && first_fail=$rc
    else
        echo "[rr_access] PASS $tag (${ms}ms)"
    fi

    # --- dimension-distribution profiling (algorithm change 2) ---
    # Count how often each dimension index appears in access traces.
    # A uniform distribution means balanced queries; skewed means the
    # RRAccessTree is being hit lopsidedly on certain attributes.
    local dim_file="$BUILD_DIR/${tag}.dims"
    grep -oP '(?:dim|dimension|attr)[=: ]*\d+' "$err" 2>/dev/null | \
        grep -oP '\d+$' | sort -n | uniq -c | sort -rn > "$dim_file" || true
    local n_dims
    n_dims=$(wc -l < "$dim_file")
    if [[ $n_dims -gt 0 ]]; then
        echo "[rr_access]   dimension access distribution ($n_dims distinct):"
        head -5 "$dim_file" | awk '{printf "      dim=%s  count=%s\n", $2, $1}'
        # compute skew ratio: max_count / min_count
        local max_c min_c
        max_c=$(head -1 "$dim_file" | awk '{print $1}')
        min_c=$(tail -1 "$dim_file" | awk '{print $1}')
        if [[ $min_c -gt 0 ]]; then
            local skew
            skew=$(awk "BEGIN{printf \"%.1f\", $max_c/$min_c}")
            echo "[rr_access]   skew ratio: ${skew}x (max/min)"
        fi
    fi

    # --- latency percentile extraction (algorithm change 3) ---
    # If the _full variant emits per-query timing (e.g. "[AJB_TIMER] query_ns=1234"),
    # extract values, sort, and compute p50/p90/p99.
    local timing_file="$BUILD_DIR/${tag}.latencies"
    grep -oP '(?:query_ns|access_us|latency)[=: ]*[\d.]+' "$err" 2>/dev/null | \
        grep -oP '[\d.]+$' | sort -n > "$timing_file" || true
    local n_samples
    n_samples=$(wc -l < "$timing_file")
    if [[ $n_samples -ge 10 ]]; then
        # percentile by line position: p = line_at(N * fraction)
        local p50_idx=$(( n_samples * 50 / 100 + 1 ))
        local p90_idx=$(( n_samples * 90 / 100 + 1 ))
        local p99_idx=$(( n_samples * 99 / 100 + 1 ))
        local p50 p90 p99
        p50=$(sed -n "${p50_idx}p" "$timing_file")
        p90=$(sed -n "${p90_idx}p" "$timing_file")
        p99=$(sed -n "${p99_idx}p" "$timing_file")
        echo "[rr_access]   latency ($n_samples samples): p50=$p50 p90=$p90 p99=$p99"
    fi
}

run "test_rr_access_tree"
run "test_rr_access_tree_full"
run "test_rr_access_tree_upstream"

# --- structural output comparison (algorithm change 4) ---
echo ""
echo "[rr_access] CROSS-CHECK"
for pair in "base:test_rr_access_tree,upstream:test_rr_access_tree_upstream"; do
    IFS=',' read -r a_spec b_spec <<< "$pair"
    a_tag="${a_spec#*:}"; b_tag="${b_spec#*:}"
    a_out="$BUILD_DIR/${a_tag}.stdout"
    b_out="$BUILD_DIR/${b_tag}.stdout"
    if [[ -f "$a_out" && -f "$b_out" ]]; then
        a_lines=$(wc -l < "$a_out")
        b_lines=$(wc -l < "$b_out")
        echo "  ${a_spec%%:*} ($a_lines lines) vs ${b_spec%%:*} ($b_lines lines)"
    fi
done

echo ""
echo "[rr_access] done (first_fail=$first_fail)"
exit $first_fail
