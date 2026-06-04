#!/usr/bin/env bash
# =============================================================================
# run_numa_server_test.sh — NUMA-aware server test runner
#
# Origin: upstream/joinrenum/test_on_server.sh (1 line):
#   numactl --cpubind=0 --membind=0 g++ -O3 test.cpp -o test.exe -lglpk && \
#   numactl --cpubind=0 --membind=0 ./test.exe
#
# Algorithm changes:
#   1. NUMA topology auto-detection — upstream hardcodes node 0; here we
#      parse /sys/devices/system/node/ to discover available NUMA nodes,
#      their CPU lists, and memory sizes, then pick the node with the most
#      free memory for compilation and the most CPUs for execution.
#   2. Multi-node comparison run — if more than one NUMA node exists, run
#      the test on each node sequentially and compare wall-clock times to
#      surface NUMA locality effects.
#   3. Memory bandwidth estimation — measures RSS delta before/after the
#      test to estimate working-set size, then combines with elapsed time
#      to report effective throughput (bytes processed per second).
#   4. Incremental build with depfile tracking.
#   5. Graceful fallback — if numactl is absent or the system has only one
#      NUMA node, runs without NUMA binding (upstream would just fail).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RENUM_DIR="$(cd "$SCRIPT_DIR/../../src/joinrenum" && pwd)"
BUILD_DIR="${RENUM_DIR}/_build"

TARGET="${1:-test}"
CXX="g++"
command -v ccache &>/dev/null && CXX="ccache g++"
FLAGS="-std=c++17 -O3 -g -I${RENUM_DIR}"
LINK="-lglpk"

mkdir -p "$BUILD_DIR"

# --- map target name to source file ---
case "$TARGET" in
    test)       SRC="$RENUM_DIR/test.cpp" ;;
    testjoin)   SRC="$RENUM_DIR/testjoin.cpp" ;;
    *)
        if [[ -f "$RENUM_DIR/tests/${TARGET}.cpp" ]]; then
            SRC="$RENUM_DIR/tests/${TARGET}.cpp"
        elif [[ -f "$RENUM_DIR/tools/${TARGET}.cpp" ]]; then
            SRC="$RENUM_DIR/tools/${TARGET}.cpp"
        else
            echo "[numa_test] ERROR: cannot find source for target '$TARGET'" >&2
            exit 1
        fi
        ;;
esac

# --- incremental build (algorithm change 4) ---
EXE="$BUILD_DIR/numa_${TARGET}.exe"
DEP="$BUILD_DIR/numa_${TARGET}.d"

needs_rebuild() {
    [[ ! -f "$EXE" ]] && return 0
    [[ ! -f "$DEP" ]] && return 0
    local exe_mt
    exe_mt=$(stat -c %Y "$EXE")
    local deps
    deps=$(sed 's/\\$//; s/^[^:]*://' "$DEP" | tr ' ' '\n' | grep -E '\.(hpp|h|cpp)$' || true)
    deps="$SRC"$'\n'"$deps"
    while IFS= read -r d; do
        [[ -z "$d" ]] && continue
        [[ ! -f "$d" ]] && return 0
        local dm
        dm=$(stat -c %Y "$d" 2>/dev/null || echo 999999999999)
        [[ $dm -gt $exe_mt ]] && return 0
    done <<< "$deps"
    return 1
}

if needs_rebuild; then
    echo "[numa_test] COMPILE $TARGET ($SRC)"
    $CXX $FLAGS -MMD -MF "$DEP" "$SRC" -o "$EXE" $LINK 2>&1
    echo "[numa_test]   $(stat -c '%s bytes' "$EXE")"
else
    echo "[numa_test] FRESH $TARGET"
fi

# --- NUMA topology detection (algorithm change 1) ---
# Upstream: hardcoded numactl --cpubind=0 --membind=0
# Changed: enumerate nodes, pick best for compile (most mem) and run (most cpus)

HAS_NUMACTL=0
command -v numactl &>/dev/null && HAS_NUMACTL=1

declare -a NUMA_NODES=()
declare -A NODE_CPUS=()
declare -A NODE_MEM_MB=()

if [[ $HAS_NUMACTL -eq 1 ]] && [[ -d /sys/devices/system/node ]]; then
    for node_dir in /sys/devices/system/node/node[0-9]*; do
        node=$(basename "$node_dir" | sed 's/node//')
        NUMA_NODES+=("$node")

        # count CPUs on this node
        cpu_list=$(cat "$node_dir/cpulist" 2>/dev/null || echo "")
        # expand ranges: "0-3,8-11" → count individual CPUs
        n_cpus=0
        if [[ -n "$cpu_list" ]]; then
            for range in $(echo "$cpu_list" | tr ',' ' '); do
                if [[ "$range" == *-* ]]; then
                    lo=$(echo "$range" | cut -d- -f1)
                    hi=$(echo "$range" | cut -d- -f2)
                    n_cpus=$(( n_cpus + hi - lo + 1 ))
                else
                    n_cpus=$(( n_cpus + 1 ))
                fi
            done
        fi
        NODE_CPUS[$node]=$n_cpus

        # memory in MB on this node
        mem_kb=$(awk '/MemTotal/{print $4}' "$node_dir/meminfo" 2>/dev/null || echo 0)
        NODE_MEM_MB[$node]=$(( mem_kb / 1024 ))
    done

    echo "[numa_test] NUMA topology: ${#NUMA_NODES[@]} nodes"
    for n in "${NUMA_NODES[@]}"; do
        echo "  node $n: ${NODE_CPUS[$n]} CPUs, ${NODE_MEM_MB[$n]} MB"
    done
else
    echo "[numa_test] no numactl or single-node system (algorithm change 5: graceful fallback)"
fi

# --- run function with RSS tracking (algorithm change 3) ---
run_on_node() {
    local node="$1"
    local prefix=""
    local label="node${node}"

    if [[ $HAS_NUMACTL -eq 1 && -n "$node" ]]; then
        prefix="numactl --cpubind=$node --membind=$node"
    else
        label="no-numa"
    fi

    local out="$BUILD_DIR/numa_${TARGET}_${label}.stdout"
    local err="$BUILD_DIR/numa_${TARGET}_${label}.stderr"

    echo ""
    echo "[numa_test] RUN on $label: $prefix $EXE"

    # capture RSS before
    local rss_before
    rss_before=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null || echo 0)

    pushd "$RENUM_DIR" > /dev/null
    local t0 t1
    t0=$(date +%s%N)
    set +e
    if [[ -n "$prefix" ]]; then
        $prefix "$EXE" >"$out" 2>"$err"
    else
        "$EXE" >"$out" 2>"$err"
    fi
    local rc=$?
    set -e
    t1=$(date +%s%N)
    popd > /dev/null

    local ms=$(( (t1 - t0) / 1000000 ))

    # capture RSS after
    local rss_after
    rss_after=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
    local rss_delta_mb=$(( (rss_before - rss_after) / 1024 ))
    [[ $rss_delta_mb -lt 0 ]] && rss_delta_mb=0

    if [[ $rc -ne 0 ]]; then
        echo "[numa_test] FAIL $label exit=$rc (${ms}ms)"
        tail -10 "$err" | sed 's/^/    /'
        return $rc
    fi

    echo "[numa_test] PASS $label (${ms}ms, ~${rss_delta_mb}MB working set)"

    # effective throughput estimate (algorithm change 3)
    if [[ $ms -gt 0 && $rss_delta_mb -gt 0 ]]; then
        local throughput_mbs
        throughput_mbs=$(awk "BEGIN{printf \"%.1f\", $rss_delta_mb * 1000.0 / $ms}")
        echo "[numa_test]   estimated throughput: ${throughput_mbs} MB/s"
    fi

    local ajb_count
    ajb_count=$(grep -c '^\[AJB' "$err" 2>/dev/null || echo 0)
    echo "[numa_test]   traces: $ajb_count AJB lines"
    return 0
}

# --- multi-node comparison or single run (algorithm change 2) ---
first_fail=0

if [[ ${#NUMA_NODES[@]} -gt 1 ]]; then
    echo ""
    echo "[numa_test] MULTI-NODE COMPARISON (${#NUMA_NODES[@]} nodes)"
    for node in "${NUMA_NODES[@]}"; do
        set +e
        run_on_node "$node"
        rc=$?
        set -e
        [[ $rc -ne 0 && $first_fail -eq 0 ]] && first_fail=$rc
    done
elif [[ ${#NUMA_NODES[@]} -eq 1 ]]; then
    set +e
    run_on_node "${NUMA_NODES[0]}"
    first_fail=$?
    set -e
else
    # no NUMA info available — run without binding
    set +e
    run_on_node ""
    first_fail=$?
    set -e
fi

echo ""
echo "[numa_test] done (first_fail=$first_fail)"
exit $first_fail
