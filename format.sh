#!/bin/bash
# =============================================================================
# format.sh — code formatter for AJB project
#
# Origin: upstream/multi-gpu-sort-merge-join/format.sh
# Algorithm changes vs upstream:
#   1. Diff-only mode (default): uses git diff to collect changed files,
#      formats only those instead of full-tree walk (upstream: unconditional
#      find over entire src/)
#   2. Parallel dispatch: pipes file list through xargs -P for concurrent
#      formatting (upstream: sequential -exec per file)
#   3. Fallback: if not in a git repo or --all flag, falls back to full-tree
#      walk like upstream
# =============================================================================

set -euo pipefail

MODE="diff"
[ "${1:-}" = "--all" ] && MODE="all"

JOBS=$(nproc 2>/dev/null || echo 4)

# --- algorithm 1+3: diff-only file collection with full-tree fallback ---
collect_files() {
    local pattern="$1"
    if [ "$MODE" = "diff" ] && git rev-parse --git-dir &>/dev/null; then
        # changed files (staged + unstaged + untracked) matching pattern
        { git diff --name-only HEAD -- "$pattern" 2>/dev/null
          git diff --cached --name-only -- "$pattern" 2>/dev/null
          git ls-files --others --exclude-standard -- "$pattern" 2>/dev/null
        } | sort -u | while read -r f; do [ -f "$f" ] && echo "$f"; done
    else
        find src -regex "$pattern" -type f
    fi
}

# --- C/C++/CUDA ---
if command -v clang-format &>/dev/null; then
    # algorithm 2: xargs parallel dispatch
    collect_files 'src/.*\.\(h\|c\|hpp\|cpp\|cuh\|cu\)' | \
        xargs -r -P "$JOBS" -n 8 clang-format -style=file -i
fi

# --- Python ---
if command -v yapf &>/dev/null; then
    collect_files 'scripts/.*\.py' | \
        xargs -r -P "$JOBS" -n 4 yapf -i
fi
