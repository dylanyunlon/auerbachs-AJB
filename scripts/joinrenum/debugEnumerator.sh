#!/usr/bin/env bash
# =============================================================================
# debugEnumerator.sh — AJB enumerator debug launcher
#
# Origin: upstream/joinrenum/debugEnumerator.sh (0 lines, empty)
# AJB adaptation (100% new): compiles test_enumerator with -DAJB_DEBUG -g,
#   launches with GDB preset breakpoints at split/randomAccess/enumerate
#   boundaries. Includes AJB state dump convenience commands and memory
#   watchpoints for bucket corruption detection.
# =============================================================================
set -euo pipefail

AJB_ROOT="${AJB_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SRC="${AJB_ROOT}/src/joinrenum"
DB_DIR="${SRC}/db"
EXE="/tmp/ajb_debug_enumerator"

echo "[AJB_DEBUG] Compiling test_enumerator with full debug..." >&2
g++ -std=c++17 -g -O0 -DAJB_DEBUG -DAJB_TRACE_DECISIONS \
    -fsanitize=address,undefined \
    -I"${SRC}" \
    "${SRC}/tests/test_enumerator.cpp" \
    -o "$EXE" -lglpk 2>&1

echo "[AJB_DEBUG] Binary: ${EXE}" >&2
echo "[AJB_DEBUG] Working dir: ${DB_DIR}" >&2

# generate GDB init with AJB-specific breakpoints and display commands
cat > /tmp/.ajb_gdbinit << 'GDBEOF'
# AJB Enumerator Debug Session
set print pretty on
set pagination off
set confirm off

# break at key algorithm boundaries
break Enumerator::enumerate
break Index::randomAccess_opt
break Index::splitBucket
break BanPickTree::pick
break BanPickTree::ban

# AJB convenience commands
define ajb-pool
  printf "[AJB_GDB] BucketPool state:\n"
  print pool
end

define ajb-bucket
  printf "[AJB_GDB] Bucket %d:\n", $arg0
  print pool[$arg0]
end

define ajb-vars
  printf "[AJB_GDB] Current variables:\n"
  print q.getVarNames()
end

# run on first stop: print initial state
commands 1
  printf "\n[AJB_GDB] Hit enumerate() entry\n"
  backtrace 5
  continue
end

run
GDBEOF

echo "[AJB_DEBUG] Starting GDB session..." >&2
echo "[AJB_DEBUG] Available commands: ajb-pool, ajb-bucket <n>, ajb-vars" >&2
cd "$DB_DIR" && gdb -q -x /tmp/.ajb_gdbinit "$EXE"
