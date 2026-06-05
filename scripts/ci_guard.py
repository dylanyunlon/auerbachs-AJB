#!/usr/bin/env python3
"""
CI Guard: block PRs that bypass in-place rewrite rules.

Checks:
  1. Banned suffixes (_v2, _port, _new, _copy, etc.)
  2. Duplicate basenames exceeding upstream count
  3. New file naming (must use ajb_ prefix or tests/tools dir)
  4. upstream/ must not be modified
  5. Same-name file diff rate >= 20%

Usage:
  python3 scripts/ci_guard.py
  python3 scripts/ci_guard.py --diff-only HEAD~1
  python3 scripts/ci_guard.py --min-rate 0.20 --verbose

Exit: 0=pass 1=violations
"""

import argparse
import difflib
import os
import re
import subprocess
import sys
from pathlib import Path

BANNED_SUFFIXES = [
    r'_v\d+', r'_port', r'_ported', r'_new', r'_copy',
    r'_alt', r'_mod', r'_modified', r'_revised',
    r'_fixed', r'_patched', r'_rewrite', r'_rewritten',
    r'_refactor', r'_refactored', r'_backup', r'_bak',
    r'_old', r'_orig', r'_original', r'_tmp', r'_temp',
    r'_draft', r'_wip', r'_fork', r'_forked',
    r'_clone', r'_cloned', r'_duplicate', r'_dup',
    r'_2', r'_ii',
]

BANNED_PATTERN = re.compile(
    r'(?i)(' + '|'.join(BANNED_SUFFIXES) + r')\.[a-zA-Z]+$'
)

SOURCE_EXTENSIONS = {'.cu', '.cuh', '.cpp', '.hpp', '.h', '.py', '.sh'}

ALLOWED_NEW_FILE_PATTERNS = [
    r'^src/.*/ajb_', r'^src/ajb_',
    r'^src/.*/tests/', r'^src/.*/tools/',
    r'^scripts/', r'^docs/', r'^paper/', r'^\.github/',
    r'_full\.[a-zA-Z]+$', r'_upstream\.[a-zA-Z]+$',
]


def find_all_source_files(directory):
    results = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix in SOURCE_EXTENSIONS:
                results.append(os.path.join(root, f))
    return sorted(results)


def check_banned_suffixes(files):
    return [f for f in files if BANNED_PATTERN.search(os.path.basename(f))]


def check_duplicate_basenames(src_files, upstream_dirs):
    from collections import defaultdict

    upstream_counts = defaultdict(int)
    for udir in upstream_dirs:
        for f in find_all_source_files(udir):
            upstream_counts[os.path.basename(f)] += 1

    src_name_to_paths = defaultdict(list)
    for f in src_files:
        rel = os.path.relpath(f, 'src')
        parts = rel.split(os.sep)
        if any(p in ('tests', 'tools', 'debug', 'db') for p in parts):
            continue
        src_name_to_paths[os.path.basename(f)].append(f)

    violations = []
    for name, paths in src_name_to_paths.items():
        allowed_count = max(upstream_counts.get(name, 0), 1)
        if len(paths) > allowed_count:
            violations.append((name, paths, allowed_count))
    return violations


def check_new_file_naming(src_files, upstream_basenames):
    allowed_re = [re.compile(p) for p in ALLOWED_NEW_FILE_PATTERNS]
    violations = []
    for f in src_files:
        if os.path.basename(f) in upstream_basenames:
            continue
        rel_path = os.path.relpath(f, '.')
        if any(r.search(rel_path) for r in allowed_re):
            continue
        violations.append(f)
    return violations


def check_upstream_untouched(diff_base=None):
    cmd = ['git', 'diff', '--name-only',
           diff_base or 'origin/main', '--', 'upstream/']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return [f for f in result.stdout.strip().split('\n') if f]
    except Exception:
        return []


def compute_diff_rate(file_a, file_b):
    try:
        with open(file_a, 'r', errors='replace') as f:
            lines_a = f.readlines()
        with open(file_b, 'r', errors='replace') as f:
            lines_b = f.readlines()
    except FileNotFoundError:
        return None
    if not lines_a and not lines_b:
        return 0.0
    sm = difflib.SequenceMatcher(None, lines_a, lines_b)
    matching = sum(block.size for block in sm.get_matching_blocks())
    total = max(len(lines_a), len(lines_b))
    return 1.0 - (matching / total) if total else 0.0


def check_diff_rates(src_dir, upstream_dirs, min_rate=0.20):
    upstream_map = {}
    for udir in upstream_dirs:
        for f in find_all_source_files(udir):
            upstream_map[os.path.basename(f)] = f

    violations, checked = [], []
    for f in find_all_source_files(src_dir):
        basename = os.path.basename(f)
        if basename not in upstream_map:
            continue
        rate = compute_diff_rate(f, upstream_map[basename])
        if rate is None:
            continue
        checked.append((f, upstream_map[basename], rate))
        if rate < min_rate:
            violations.append((f, upstream_map[basename], rate))
    return violations, checked


def get_new_files_in_diff(diff_base):
    cmd = ['git', 'diff', '--name-only', '--diff-filter=A', diff_base]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return [f for f in result.stdout.strip().split('\n') if f]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description='CI Guard')
    parser.add_argument('--diff-only', metavar='BASE')
    parser.add_argument('--min-rate', type=float, default=0.20)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    repo_root = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True, text=True
    ).stdout.strip()
    if repo_root:
        os.chdir(repo_root)

    src_dir = 'src'
    upstream_dirs = [
        'upstream/multi-gpu-sort-merge-join/src',
        'upstream/multi-gpu-sort-merge-join/scripts',
        'upstream/joinrenum',
    ]

    upstream_basenames = set()
    for udir in upstream_dirs:
        for f in find_all_source_files(udir):
            upstream_basenames.add(os.path.basename(f))

    if args.diff_only:
        src_files = [f for f in get_new_files_in_diff(args.diff_only)
                     if f.startswith('src/') and Path(f).suffix in SOURCE_EXTENSIONS]
    else:
        src_files = find_all_source_files(src_dir)

    all_src_files = find_all_source_files(src_dir)
    errors = 0

    # -- Check 1: Banned suffixes --
    print("=== Check 1: Banned suffixes ===")
    banned = check_banned_suffixes(src_files)
    if banned:
        for f in banned:
            print(f"  FAIL BANNED: {f}")
        errors += len(banned)
    else:
        print("  OK")

    # -- Check 2: Duplicate basenames --
    print("\n=== Check 2: Duplicate basenames ===")
    dupes = check_duplicate_basenames(all_src_files, upstream_dirs)
    if dupes:
        for name, paths, allowed in dupes:
            print(f"  FAIL {name}: {len(paths)} copies (upstream has {allowed})")
            for p in paths:
                print(f"       {p}")
        errors += len(dupes)
    else:
        print("  OK")

    # -- Check 3: New file naming --
    print("\n=== Check 3: New file naming ===")
    bad_names = check_new_file_naming(src_files, upstream_basenames)
    if bad_names:
        for f in bad_names:
            print(f"  WARN {f} -- need ajb_ prefix or tests/tools dir")
        errors += len(bad_names)
    else:
        print("  OK")

    # -- Check 4: upstream/ untouched --
    print("\n=== Check 4: upstream/ untouched ===")
    upstream_changed = check_upstream_untouched(args.diff_only)
    if upstream_changed:
        for f in upstream_changed:
            print(f"  FAIL MODIFIED: {f}")
        errors += len(upstream_changed)
    else:
        print("  OK")

    # -- Check 5: Diff rate --
    print(f"\n=== Check 5: Diff rate >= {args.min_rate:.0%} ===")
    rate_violations, rate_checked = check_diff_rates(
        src_dir, upstream_dirs, args.min_rate
    )
    if args.verbose:
        for f, uf, rate in rate_checked:
            status = "OK" if rate >= args.min_rate else "FAIL"
            print(f"  {status} {rate:5.1%}  {f}")
    if rate_violations:
        for f, uf, rate in rate_violations:
            print(f"  FAIL {rate:5.1%} < {args.min_rate:.0%}: {f}")
        errors += len(rate_violations)
    else:
        print(f"  OK  all {len(rate_checked)} files pass")

    # -- Summary --
    print(f"\n{'=' * 50}")
    if errors:
        print(f"FAILED: {errors} violation(s). PR blocked.")
        sys.exit(1)
    else:
        print(f"PASSED: all checks OK. {len(rate_checked)} files validated.")
        sys.exit(0)


if __name__ == '__main__':
    main()
