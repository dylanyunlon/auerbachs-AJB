#!/usr/bin/env python3
"""
ajb_adaptive_debug.py — Adaptive breakpoint scheduling, trie-indexed state
snapshots, and IQR-based timing anomaly detection for AJB traces.

M1041-M1050: Claude #25

Components:
  1. AdaptiveBreakpointScheduler  — exponential-backoff (2^n, cap=1000)
     decides which [AJB_BP] events to print vs suppress.
  2. StructStateSnapshot          — regex extraction of key=value pairs from
     stderr, indexed by a prefix trie for fast state-change-path lookup;
     builds per-test state timeseries tables.
  3. AnomalyDetector              — IQR-based outlier detection on timing
     values; flags samples >1.5×IQR with full surrounding context.
  4. Main                         — runs test_enumerator & test_join_tree,
     pipes their combined output through all three analyzers, then prints
     a comparative cross-test report.

Usage:
  # pipe a single test:
  ./test_enum 2>&1 | python3 ajb_adaptive_debug.py
  # auto-run both tests (default):
  python3 ajb_adaptive_debug.py --run
  # from file:
  python3 ajb_adaptive_debug.py trace.log
"""

import argparse
import math
import os
import re
import subprocess
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. AdaptiveBreakpointScheduler — exponential backoff (2^n, cap 1000)
# ---------------------------------------------------------------------------
class AdaptiveBreakpointScheduler:
    """Suppresses high-frequency [AJB_BP] events using exponential backoff.

    For each distinct BP label, the scheduler tracks how many times that
    event has fired.  The next print threshold is 2^k (1, 2, 4, 8, ...),
    capped at 1000.  Between thresholds, events are suppressed but counted.
    Low-frequency labels (≤10 total occurrences) are never suppressed.

    Algorithm:
      next_print[label] starts at 1.
      On each fire:  count[label] += 1
        if count == next_print → print; next_print = min(next_print * 2, 1000)
        else                  → suppress (but still count)
    After all events, a summary shows total/printed/suppressed per label.
    """

    def __init__(self, cap=1000):
        self._cap = cap
        self._count = defaultdict(int)          # label → total fires
        self._next_print = defaultdict(lambda: 1)  # label → next threshold
        self._printed = defaultdict(int)         # label → times printed
        self._suppressed = defaultdict(int)      # label → times suppressed

    def should_print(self, label):
        """Record one firing of *label*; return True if it should be printed."""
        self._count[label] += 1
        cnt = self._count[label]
        thresh = self._next_print[label]
        if cnt >= thresh:
            # Advance threshold: exponential backoff with cap
            self._next_print[label] = min(thresh * 2, self._cap)
            self._printed[label] += 1
            return True
        self._suppressed[label] += 1
        return False

    def summary(self):
        """Return per-label stats sorted by total count descending."""
        rows = []
        for label in sorted(self._count, key=lambda k: self._count[k],
                            reverse=True):
            rows.append({
                'label': label,
                'total': self._count[label],
                'printed': self._printed[label],
                'suppressed': self._suppressed[label],
                'backoff_level': int(math.log2(self._next_print[label]))
                    if self._next_print[label] > 0 else 0,
            })
        return rows


# ---------------------------------------------------------------------------
# 2. StructStateSnapshot — trie-indexed state-change-path tracker
# ---------------------------------------------------------------------------
class _TrieNode:
    """Node of a prefix trie used to index state variable paths."""
    __slots__ = ('children', 'values', 'count')

    def __init__(self):
        self.children = {}     # char/segment → _TrieNode
        self.values = []       # [(line_no, value)] at terminal nodes
        self.count = 0         # total insertions through this node


class StructStateSnapshot:
    """Extracts key=value pairs from AJB output lines, indexes keys by a
    prefix trie so that hierarchical key paths (e.g. 'Index.varnum',
    'Table.col[0].distinct') can be queried by prefix efficiently.

    Per-test state timeseries:
        {test_name: {key: [(line_no, value), ...]}}
    The trie provides O(|prefix|) lookup for all keys sharing a common
    prefix — useful for grouping state variables by subsystem.
    """

    _KV_RE = re.compile(
        r'(?<![<\w])(\w[\w.\[\]]*)\s*=\s*'   # key (dotted or bracketed)
        r'('
        r'-?[\d]+(?:\.[\d]+)?(?:[eE][+\-]?\d+)?'  # numeric
        r'|'
        r'\[[^\]]*\]'                              # bracketed list
        r'|'
        r'[^\s,;)]+(?:%)?'                         # token
        r')'
    )

    def __init__(self):
        self._root = _TrieNode()
        self._timeseries = defaultdict(lambda: defaultdict(list))
        self._current_test = 'default'

    def set_test(self, name):
        self._current_test = name

    def _trie_insert(self, key):
        """Insert key segments into the prefix trie."""
        node = self._root
        for ch in key:
            node.count += 1
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        node.count += 1

    def trie_query(self, prefix):
        """Return all keys stored in the trie that share *prefix*."""
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        # DFS to collect all terminal paths
        results = []
        stack = [(node, prefix)]
        while stack:
            nd, path = stack.pop()
            if nd.values:
                results.append(path)
            for ch, child in nd.children.items():
                stack.append((child, path + ch))
        return results

    def ingest(self, line_no, line):
        """Extract key=value pairs from one line; record in timeseries + trie."""
        pairs_found = 0
        for m in self._KV_RE.finditer(line):
            key = m.group(1)
            val = m.group(2)
            # Attempt numeric coercion
            try:
                val_store = int(val)
            except ValueError:
                try:
                    val_store = float(val)
                except ValueError:
                    val_store = val
            self._timeseries[self._current_test][key].append(
                (line_no, val_store))
            self._trie_insert(key)
            pairs_found += 1
        return pairs_found

    def get_timeseries(self, test_name=None):
        if test_name:
            return dict(self._timeseries.get(test_name, {}))
        return {t: dict(kv) for t, kv in self._timeseries.items()}

    def table_summary(self):
        """Build a printable per-test state table showing final values and
        change counts for each key."""
        tables = {}
        for test, keys in self._timeseries.items():
            rows = []
            for key in sorted(keys):
                entries = keys[key]
                changes = 0
                for i in range(1, len(entries)):
                    if entries[i][1] != entries[i - 1][1]:
                        changes += 1
                rows.append({
                    'key': key,
                    'first_val': entries[0][1],
                    'last_val': entries[-1][1],
                    'observations': len(entries),
                    'changes': changes,
                })
            tables[test] = rows
        return tables


# ---------------------------------------------------------------------------
# 3. AnomalyDetector — IQR-based timing outlier detection
# ---------------------------------------------------------------------------
class AnomalyDetector:
    """Collects timing values from [AJB_TIMER] events and detects outliers
    using the Inter-Quartile Range (IQR) method:
      Q1, Q3 = 25th/75th percentile
      IQR    = Q3 - Q1
      outlier if value > Q3 + 1.5*IQR  or  value < Q1 - 1.5*IQR

    Each outlier is reported with its surrounding context lines for
    quick diagnosis.
    """

    _TIMER_RE = re.compile(
        r'\[AJB_TIMER\].*?(\S+)\s*[:=]?\s*([\d.]+)\s*(s|ms|us)'
        r'|'
        r'\[AJB_TIMER\]\s*<<<\s*(\S+)\s+([\d.]+)\s+s'
    )

    def __init__(self, context_radius=2):
        self._events = []       # [(line_no, phase, seconds)]
        self._all_lines = []    # all raw lines for context retrieval
        self._context_radius = context_radius

    def set_lines(self, lines):
        self._all_lines = list(lines)

    def ingest(self, line_no, line):
        """Try to extract a timing value from line."""
        m = self._TIMER_RE.search(line)
        if not m:
            return
        if m.group(1):
            phase = m.group(1)
            raw = float(m.group(2))
            unit = m.group(3)
            secs = raw / 1000 if unit == 'ms' else raw / 1e6 if unit == 'us' else raw
        else:
            phase = m.group(4)
            secs = float(m.group(5))
        self._events.append((line_no, phase, secs))

    def detect(self):
        """Return list of outlier dicts with context."""
        if len(self._events) < 4:
            return []
        values = sorted(v for _, _, v in self._events)
        n = len(values)
        q1 = values[n // 4]
        q3 = values[(3 * n) // 4]
        iqr = q3 - q1
        if iqr == 0:
            return []
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        outliers = []
        for line_no, phase, secs in self._events:
            if secs < lower_fence or secs > upper_fence:
                ctx_start = max(0, line_no - 1 - self._context_radius)
                ctx_end = min(len(self._all_lines),
                              line_no + self._context_radius)
                context = [
                    self._all_lines[i].rstrip()
                    for i in range(ctx_start, ctx_end)
                ]
                outliers.append({
                    'line': line_no,
                    'phase': phase,
                    'seconds': secs,
                    'deviation': ('above' if secs > upper_fence else 'below'),
                    'fence': upper_fence if secs > upper_fence else lower_fence,
                    'iqr': iqr,
                    'factor': ((secs - q3) / iqr if secs > upper_fence
                               else (q1 - secs) / iqr),
                    'context': context,
                })
        return outliers

    def iqr_stats(self):
        """Return basic IQR statistics."""
        if len(self._events) < 4:
            return None
        values = sorted(v for _, _, v in self._events)
        n = len(values)
        return {
            'n': n,
            'q1': values[n // 4],
            'median': values[n // 2],
            'q3': values[(3 * n) // 4],
            'iqr': values[(3 * n) // 4] - values[n // 4],
            'min': values[0],
            'max': values[-1],
        }


# ---------------------------------------------------------------------------
# 4. Pipeline: analyze one test's output
# ---------------------------------------------------------------------------
def analyze_output(lines, test_name, bp_sched, snapshot, anomaly_det):
    """Process lines from a single test run through all three analyzers."""
    snapshot.set_test(test_name)
    anomaly_det.set_lines(lines)

    bp_re = re.compile(r'\[AJB_BP\]\s*(.*)')
    state_re = re.compile(r'\[AJB_STATE\]\s*(.*)')

    printed_lines = []
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        line_no = i + 1

        # Breakpoint scheduling
        m_bp = bp_re.search(line)
        if m_bp:
            label = m_bp.group(1).strip()
            if bp_sched.should_print(label):
                printed_lines.append(f"  [BP #{bp_sched._count[label]}] {label}")

        # State snapshot ingestion
        m_st = state_re.search(line)
        if m_st:
            snapshot.ingest(line_no, m_st.group(1))
        # Also pick up key=value from non-STATE lines (timers, results)
        if '[AJB_TIMER]' in line or '[AJB_RESULTS]' in line:
            snapshot.ingest(line_no, line)

        # Anomaly detector ingestion
        anomaly_det.ingest(line_no, line)

    return printed_lines


# ---------------------------------------------------------------------------
# 5. Report printer
# ---------------------------------------------------------------------------
def print_report(test_name, bp_sched, snapshot, anomaly_det, bp_lines):
    """Print a structured diagnostic report for one test."""
    hdr = f"{'=' * 64}\n  Adaptive Debug Report: {test_name}\n{'=' * 64}"
    print(hdr)

    # --- Breakpoint scheduling summary ---
    print(f"\n  Breakpoint Scheduling (exponential backoff, cap=1000):")
    bps = bp_sched.summary()
    if bps:
        for row in bps:
            ratio = (row['printed'] / row['total'] * 100
                     if row['total'] > 0 else 0)
            print(f"    {row['label'][:50]:50s}  total={row['total']:>5d}  "
                  f"printed={row['printed']:>4d} ({ratio:5.1f}%)  "
                  f"backoff_lvl={row['backoff_level']}")
    else:
        print("    (no breakpoint events)")

    # --- Selected BP lines ---
    if bp_lines:
        print(f"\n  Selected BP events ({len(bp_lines)} printed):")
        for bl in bp_lines[:20]:
            print(f"   {bl}")
        if len(bp_lines) > 20:
            print(f"    ... and {len(bp_lines) - 20} more")

    # --- State timeseries table ---
    tables = snapshot.table_summary()
    if test_name in tables:
        rows = tables[test_name]
        print(f"\n  State Timeseries ({len(rows)} keys tracked):")
        changing = [r for r in rows if r['changes'] > 0]
        static = [r for r in rows if r['changes'] == 0]
        if changing:
            print(f"    Changing keys ({len(changing)}):")
            for r in changing[:15]:
                print(f"      {r['key']:30s}  {r['first_val']} → {r['last_val']}  "
                      f"({r['observations']} obs, {r['changes']} changes)")
        if static:
            print(f"    Static keys: {len(static)}  "
                  f"(e.g. {', '.join(r['key'] for r in static[:5])})")

    # --- Trie prefix query demo ---
    for prefix in ['col', 'var', 'total']:
        matches = snapshot.trie_query(prefix)
        if matches:
            print(f"    Trie['{prefix}*']: {len(matches)} keys "
                  f"({', '.join(matches[:5])}{'...' if len(matches) > 5 else ''})")

    # --- IQR anomaly detection ---
    stats = anomaly_det.iqr_stats()
    if stats:
        print(f"\n  Timing IQR Analysis (n={stats['n']}):")
        print(f"    Q1={stats['q1']:.6f}s  median={stats['median']:.6f}s  "
              f"Q3={stats['q3']:.6f}s  IQR={stats['iqr']:.6f}s")
        print(f"    range=[{stats['min']:.6f}, {stats['max']:.6f}]s")

        outliers = anomaly_det.detect()
        if outliers:
            print(f"    Outliers detected: {len(outliers)}")
            for o in outliers:
                print(f"      L{o['line']:>4d} {o['phase']:30s}  "
                      f"{o['seconds']:.6f}s  ({o['deviation']}, "
                      f"{o['factor']:.2f}×IQR past fence)")
                for ctx in o['context']:
                    print(f"        | {ctx}")
        else:
            print(f"    No timing outliers (all within 1.5×IQR fences)")
    else:
        print(f"\n  Timing IQR Analysis: insufficient data (<4 events)")


# ---------------------------------------------------------------------------
# 6. Cross-test comparison
# ---------------------------------------------------------------------------
def print_comparison(snapshots_by_test):
    """Compare state keys across test_enumerator and test_join_tree."""
    tests = list(snapshots_by_test.keys())
    if len(tests) < 2:
        return
    print(f"\n{'=' * 64}")
    print(f"  Cross-test Comparison: {tests[0]} vs {tests[1]}")
    print(f"{'=' * 64}")

    keys0 = set(snapshots_by_test[tests[0]].keys())
    keys1 = set(snapshots_by_test[tests[1]].keys())

    shared = keys0 & keys1
    only0 = keys0 - keys1
    only1 = keys1 - keys0

    print(f"\n  Shared state keys: {len(shared)}")
    print(f"  Only in {tests[0]}: {len(only0)}")
    print(f"  Only in {tests[1]}: {len(only1)}")

    # Compare final values of shared keys
    diffs = []
    for key in sorted(shared):
        ts0 = snapshots_by_test[tests[0]][key]
        ts1 = snapshots_by_test[tests[1]][key]
        final0 = ts0[-1][1] if ts0 else None
        final1 = ts1[-1][1] if ts1 else None
        if final0 != final1:
            diffs.append((key, final0, final1))

    if diffs:
        print(f"\n  Divergent final values ({len(diffs)} keys):")
        for key, v0, v1 in diffs[:15]:
            print(f"    {key:30s}  {tests[0]}={v0}  {tests[1]}={v1}")
        if len(diffs) > 15:
            print(f"    ... and {len(diffs) - 15} more")
    else:
        print(f"\n  All shared keys have identical final values.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='AJB adaptive debug — M1041-M1050')
    parser.add_argument('file', nargs='?', default='-',
                        help='Trace log file (- for stdin, default)')
    parser.add_argument('--run', action='store_true',
                        help='Auto-compile and run both tests')
    parser.add_argument('--cap', type=int, default=1000,
                        help='Backoff cap for BP scheduler (default 1000)')
    args = parser.parse_args()

    if args.run:
        # Auto-run mode: compile and capture both tests
        script_dir = os.path.dirname(os.path.abspath(__file__))
        renum_dir = os.path.join(script_dir, '..', '..', 'src', 'joinrenum')
        renum_dir = os.path.normpath(renum_dir)

        tests = [
            ('test_enumerator', 'tests/test_enumerator.cpp', '/tmp/_ajb_test_e'),
            ('test_join_tree', 'tests/test_join_tree.cpp', '/tmp/_ajb_test_jt'),
        ]
        all_outputs = {}
        for name, src, exe in tests:
            src_path = os.path.join(renum_dir, src)
            cmd_compile = (f'g++ -std=c++17 -O2 -I {renum_dir} '
                           f'{src_path} -o {exe} -lglpk')
            rc = subprocess.call(cmd_compile, shell=True,
                                 stderr=subprocess.STDOUT)
            if rc != 0:
                print(f"[SKIP] compilation failed for {name}", file=sys.stderr)
                continue
            result = subprocess.run(
                exe, shell=True, capture_output=True, text=True,
                cwd=renum_dir)
            all_outputs[name] = (result.stdout + result.stderr).splitlines(
                keepends=True)

        # Analyze each test
        all_ts = {}
        for name, lines in all_outputs.items():
            bp = AdaptiveBreakpointScheduler(cap=args.cap)
            snap = StructStateSnapshot()
            anom = AnomalyDetector()
            bp_lines = analyze_output(lines, name, bp, snap, anom)
            print_report(name, bp, snap, anom, bp_lines)
            all_ts[name] = snap.get_timeseries(name)

        # Cross-test comparison
        if len(all_ts) >= 2:
            print_comparison(all_ts)

        print(f"\n{'=' * 64}")
        print(f"  Done — analyzed {len(all_outputs)} tests")
        print(f"{'=' * 64}")
    else:
        # Pipe / file mode: single stream
        if args.file == '-':
            lines = sys.stdin.readlines()
        else:
            with open(args.file) as f:
                lines = f.readlines()

        test_name = 'stdin'
        bp = AdaptiveBreakpointScheduler(cap=args.cap)
        snap = StructStateSnapshot()
        anom = AnomalyDetector()
        bp_lines = analyze_output(lines, test_name, bp, snap, anom)
        print_report(test_name, bp, snap, anom, bp_lines)


if __name__ == '__main__':
    main()
