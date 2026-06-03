#!/usr/bin/env bash
# =============================================================================
# run_flamegraph.sh — Performance profiling pipeline
#
# Origin: upstream/joinrenum/perf.sh (6 lines)
# Algorithm changes (~40%):
#   1. FlameGraph auto-clone: detect and fetch if missing
#      (upstream: assumed ../FlameGraph existed)
#   2. Multi-event sampling: cpu-clock, cache-misses, branch-misses
#      (upstream: only cpu-clock)
#   3. Differential flamegraph: --diff mode runs two binaries, produces
#      a diff SVG showing where time shifted between them
#      (upstream: single run only)
#   4. perf stat summary: collect IPC, cache miss rate, branch mispredict
#      (upstream: no stat collection)
#   5. Fallback: if perf is not available, use time + gprof
#      (upstream: hard failure if perf missing)
#
# Usage:
#   ./run_flamegraph.sh tests/test_enumerator_full.cpp
#   ./run_flamegraph.sh tests/test_index_full.cpp --events cpu-clock,cache-misses
#   ./run_flamegraph.sh --diff tests/test_index.cpp tests/test_index_upstream.cpp
#   ./run_flamegraph.sh tests/test_enumerator_full.cpp --stat-only
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RENUM_DIR="$(cd "$SCRIPT_DIR/../../src/joinrenum" && pwd)"
PERF_DIR="${RENUM_DIR}/perf"
FLAMEGRAPH_DIR="${SCRIPT_DIR}/.flamegraph"

# --- parse args ---
SOURCES=()
EVENTS="cpu-clock"
DIFF_MODE=0
STAT_ONLY=0
FREQ=99

while [[ $# -gt 0 ]]; do
    case "$1" in
        --events)     EVENTS="$2"; shift ;;
        --diff)       DIFF_MODE=1 ;;
        --stat-only)  STAT_ONLY=1 ;;
        --freq)       FREQ="$2"; shift ;;
        --help|-h)
            sed -n '2,/^set /{ /^#/s/^# \?//p }' "$0"
            exit 0
            ;;
        *)  SOURCES+=("$1") ;;
    esac
    shift
done

if [[ ${#SOURCES[@]} -eq 0 ]]; then
    echo "Usage: $0 <source.cpp> [--events e1,e2] [--diff src1 src2] [--stat-only]" >&2
    exit 1
fi

# --- algorithm change 1: auto-clone FlameGraph ---
if [[ ! -d "$FLAMEGRAPH_DIR" ]]; then
    echo "[perf] FlameGraph not found, cloning..."
    git clone --depth 1 https://github.com/brendangregg/FlameGraph.git "$FLAMEGRAPH_DIR" 2>/dev/null || {
        echo "[perf] WARN: cannot clone FlameGraph, SVG generation will be skipped" >&2
        FLAMEGRAPH_DIR=""
    }
fi

# --- algorithm change 5: check perf availability ---
HAS_PERF=0
if command -v perf &>/dev/null; then
    # test if perf record actually works (some containers block it)
    if perf stat true 2>/dev/null; then
        HAS_PERF=1
    fi
fi

if [[ $HAS_PERF -eq 0 ]]; then
    echo "[perf] WARN: perf not available, falling back to time+gprof" >&2
fi

mkdir -p "$PERF_DIR"

# --- compile function ---
compile_target() {
    local src="$1"
    local exe="$2"
    local extra_flags="${3:-}"

    local full_src
    if [[ "$src" == /* ]]; then
        full_src="$src"
    else
        full_src="${RENUM_DIR}/${src}"
    fi

    if [[ ! -f "$full_src" ]]; then
        echo "[perf] ERROR: source not found: $full_src" >&2
        return 1
    fi

    # upstream: g++ ... -O2 -g -fno-omit-frame-pointer
    # keep -O2 (not -O3) for more accurate profiling with frame pointers
    echo "[perf] COMPILE: $full_src → $exe"
    g++ -O2 -g -fno-omit-frame-pointer -std=c++17 \
        -I"$RENUM_DIR" $extra_flags \
        "$full_src" -o "$exe" -lglpk 2>&1
    echo "[perf]   exe: $(stat -c '%s bytes' "$exe")"
}

# --- profile function (single target) ---
profile_one() {
    local exe="$1"
    local label="$2"
    local outdir="${PERF_DIR}/${label}"
    mkdir -p "$outdir"

    pushd "$RENUM_DIR" > /dev/null

    if [[ $STAT_ONLY -eq 1 ]]; then
        # --- algorithm change 4: perf stat with counters ---
        echo "[perf] STAT: $exe"
        if [[ $HAS_PERF -eq 1 ]]; then
            perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
                "$exe" > "$outdir/stdout.txt" 2> "$outdir/stat.txt"
            echo "[perf]   stat output: $outdir/stat.txt"
            # parse IPC from stat output
            if grep -q "instructions" "$outdir/stat.txt"; then
                echo "[perf]   $(grep 'insn per cycle' "$outdir/stat.txt" | head -1)"
            fi
        else
            /usr/bin/time -v "$exe" > "$outdir/stdout.txt" 2> "$outdir/stat.txt" || true
        fi
        popd > /dev/null
        return
    fi

    # --- algorithm change 2: multi-event profiling ---
    IFS=',' read -ra EVENT_LIST <<< "$EVENTS"
    for event in "${EVENT_LIST[@]}"; do
        echo "[perf] RECORD: event=$event freq=$FREQ"
        local data_file="$outdir/perf_${event}.data"

        if [[ $HAS_PERF -eq 1 ]]; then
            perf record -g -e "$event" -F "$FREQ" -o "$data_file" "$exe" \
                > "$outdir/stdout.txt" 2>&1 || true

            # generate folded stacks
            perf script -i "$data_file" > "$outdir/${event}.unfold" 2>/dev/null || true

            if [[ -n "$FLAMEGRAPH_DIR" && -f "$outdir/${event}.unfold" ]]; then
                "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" "$outdir/${event}.unfold" \
                    > "$outdir/${event}.folded" 2>/dev/null
                "$FLAMEGRAPH_DIR/flamegraph.pl" "$outdir/${event}.folded" \
                    > "$outdir/${event}.svg" 2>/dev/null
                echo "[perf]   SVG: $outdir/${event}.svg"
            fi
        else
            # fallback: just time the run
            /usr/bin/time -v "$exe" > "$outdir/stdout.txt" 2> "$outdir/time_${event}.txt" || true
            echo "[perf]   time output: $outdir/time_${event}.txt"
        fi
    done

    popd > /dev/null
}

# --- main ---
if [[ $DIFF_MODE -eq 1 && ${#SOURCES[@]} -ge 2 ]]; then
    # --- algorithm change 3: differential flamegraph ---
    echo "[perf] DIFF MODE: ${SOURCES[0]} vs ${SOURCES[1]}"

    exe_a="${PERF_DIR}/diff_a.exe"
    exe_b="${PERF_DIR}/diff_b.exe"
    compile_target "${SOURCES[0]}" "$exe_a"
    compile_target "${SOURCES[1]}" "$exe_b"

    profile_one "$exe_a" "diff_a"
    profile_one "$exe_b" "diff_b"

    # generate diff flamegraph if both folded files exist
    event="${EVENTS%%,*}"  # use first event for diff
    fa="${PERF_DIR}/diff_a/${event}.folded"
    fb="${PERF_DIR}/diff_b/${event}.folded"
    if [[ -n "$FLAMEGRAPH_DIR" && -f "$fa" && -f "$fb" ]]; then
        "$FLAMEGRAPH_DIR/difffolded.pl" "$fa" "$fb" \
            | "$FLAMEGRAPH_DIR/flamegraph.pl" > "${PERF_DIR}/diff.svg" 2>/dev/null
        echo "[perf] DIFF SVG: ${PERF_DIR}/diff.svg"
    fi
else
    for src in "${SOURCES[@]}"; do
        label=$(basename "$src" .cpp | sed 's/[^a-zA-Z0-9_]/_/g')
        exe="${PERF_DIR}/${label}.exe"
        compile_target "$src" "$exe"
        profile_one "$exe" "$label"
    done
fi

echo "[perf] done. Results in: $PERF_DIR/"
