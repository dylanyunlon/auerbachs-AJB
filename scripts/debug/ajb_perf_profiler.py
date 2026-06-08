#!/usr/bin/env python3
"""
ajb_perf_profiler.py — AJB Performance Profiler (M1106-M1110)

Flamegraph text tree from [AJB_TIMER] events, multi-run Welford variance,
hot path annotation, and two-trace diff mode.

Usage:
    python3 ajb_perf_profiler.py trace1.log [trace2.log ...]
    python3 ajb_perf_profiler.py --diff trace_a.log trace_b.log
"""

import sys
import re
import math
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Timer event parsing
# =============================================================================

# Matches all observed [AJB_TIMER] formats:
#   [AJB_TIMER] name: 0.02 ms
#   [AJB_TIMER][Cat] name: 0.02ms
#   [AJB_TIMER][Cat] name=0.02ms
#   [AJB_TIMER][Cat] name total=0.02ms (extras)
TIMER_RE = re.compile(
    r'\[AJB_TIMER\]'
    r'(?:\[([^\]]+)\])?'     # optional [Category]
    r'\s+'
    r'(\S+)'                   # timer name
    r'(?:\s+\S+)?'            # optional extra word (e.g. "total")
    r'[=:\s]+'                 # = or : separator
    r'([\d.]+)'                # numeric value
    r'\s*'
    r'(ms|s)'                   # unit
)

NESTED_RE = re.compile(
    r'\[AJB_TIMER\]\s+(\S+)/(\S+)[=:\s]+([\d.]+)\s*(ms|s)'
)


@dataclass
class TimerEvent:
    name: str
    duration_ms: float
    parent: Optional[str] = None


def parse_trace(filepath: str) -> List[TimerEvent]:
    """Parse [AJB_TIMER] events from a log file."""
    events = []
    try:
        with open(filepath, 'r', errors='replace') as f:
            for line in f:
                # Try nested format first: parent/child
                m = NESTED_RE.search(line)
                if m:
                    dur = float(m.group(3))
                    if m.group(4) == "s":
                        dur *= 1000.0
                    events.append(TimerEvent(
                        name=m.group(2),
                        duration_ms=dur,
                        parent=m.group(1)
                    ))
                    continue

                m = TIMER_RE.search(line)
                if m:
                    category = m.group(1) or ""
                    name = m.group(2)
                    val = float(m.group(3))
                    unit = m.group(4)
                    if category:
                        name = f"{category}/{name}"
                    dur_ms = val if unit == "ms" else val * 1000.0
                    events.append(TimerEvent(name=name, duration_ms=dur_ms))
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    return events


# =============================================================================
# Flamegraph text tree (M1106-M1107)
# =============================================================================

@dataclass
class TreeNode:
    name: str
    total_ms: float = 0.0
    count: int = 0
    children: Dict[str, 'TreeNode'] = field(default_factory=dict)


def build_tree(events: List[TimerEvent]) -> TreeNode:
    """Build a call hierarchy tree from timer events."""
    root = TreeNode(name="<root>")

    # Aggregate by name
    aggregated: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    parent_map: Dict[str, str] = {}

    for evt in events:
        old_total, old_count = aggregated[evt.name]
        aggregated[evt.name] = (old_total + evt.duration_ms, old_count + 1)
        if evt.parent:
            parent_map[evt.name] = evt.parent

    # Build tree from parent relationships
    nodes: Dict[str, TreeNode] = {}
    for name, (total, count) in aggregated.items():
        nodes[name] = TreeNode(name=name, total_ms=total, count=count)

    # Link children to parents
    for name, parent_name in parent_map.items():
        if parent_name in nodes:
            nodes[parent_name].children[name] = nodes[name]

    # Top-level nodes (no parent)
    for name, node in nodes.items():
        if name not in parent_map:
            root.children[name] = node
            root.total_ms += node.total_ms

    return root


def print_tree(node: TreeNode, total_time_ms: float, depth: int = 0,
               prefix: str = "", is_last: bool = True, hot_threshold: float = 0.1):
    """Print ASCII flamegraph tree with hot path annotation."""
    if depth > 0:
        connector = "└── " if is_last else "├── "
        pct = (node.total_ms / total_time_ms * 100) if total_time_ms > 0 else 0

        # Hot path coloring (ANSI red for >10% of total)
        is_hot = (pct >= hot_threshold * 100)
        color_start = "\033[91m" if is_hot else ""
        color_end = "\033[0m" if is_hot else ""
        hot_marker = " 🔥" if is_hot else ""

        print(f"{prefix}{connector}{color_start}{node.name}: "
              f"{node.total_ms:.2f} ms ({pct:.1f}%) "
              f"[n={node.count}]{hot_marker}{color_end}")

    # Sort children by total_ms descending
    children = sorted(node.children.values(), key=lambda c: c.total_ms, reverse=True)
    for i, child in enumerate(children):
        is_child_last = (i == len(children) - 1)
        child_prefix = prefix + ("    " if is_last else "│   ") if depth > 0 else ""
        print_tree(child, total_time_ms, depth + 1, child_prefix, is_child_last,
                   hot_threshold)


# =============================================================================
# Multi-run Welford online variance (M1108)
# =============================================================================

@dataclass
class WelfordAccumulator:
    """Welford's online algorithm for streaming mean/variance."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared differences from mean

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    @property
    def cv(self) -> float:
        return self.stddev / self.mean if abs(self.mean) > 1e-12 else 0.0


def welford_multi_run(trace_files: List[str]) -> Dict[str, WelfordAccumulator]:
    """Compute per-timer Welford stats across multiple trace files."""
    accumulators: Dict[str, WelfordAccumulator] = defaultdict(WelfordAccumulator)

    for filepath in trace_files:
        events = parse_trace(filepath)
        # Aggregate per-timer within this run
        run_totals: Dict[str, float] = defaultdict(float)
        for evt in events:
            run_totals[evt.name] += evt.duration_ms

        for name, total in run_totals.items():
            accumulators[name].update(total)

    return dict(accumulators)


def print_welford_report(accumulators: Dict[str, WelfordAccumulator]):
    """Print Welford multi-run variance report."""
    print("\n" + "=" * 72)
    print("  WELFORD MULTI-RUN VARIANCE REPORT")
    print("=" * 72)
    print(f"  {'Timer':<30} {'Mean (ms)':>12} {'Std (ms)':>12} {'CV':>8} {'N':>6}")
    print("-" * 72)

    sorted_timers = sorted(accumulators.items(), key=lambda x: x[1].mean, reverse=True)
    for name, acc in sorted_timers:
        cv_str = f"{acc.cv:.3f}" if acc.n > 1 else "n/a"
        std_str = f"{acc.stddev:.2f}" if acc.n > 1 else "n/a"
        # Hot path marker
        hot = " 🔥" if acc.n > 1 and acc.cv > 0.2 else ""
        print(f"  {name:<30} {acc.mean:>12.2f} {std_str:>12} {cv_str:>8} {acc.n:>6}{hot}")

    print("=" * 72)


# =============================================================================
# Two-trace diff mode (M1109-M1110)
# =============================================================================

def diff_traces(file_a: str, file_b: str):
    """Compare two traces and highlight >2σ deviations."""
    events_a = parse_trace(file_a)
    events_b = parse_trace(file_b)

    # Aggregate per timer
    totals_a: Dict[str, float] = defaultdict(float)
    totals_b: Dict[str, float] = defaultdict(float)
    counts_a: Dict[str, int] = defaultdict(int)
    counts_b: Dict[str, int] = defaultdict(int)

    for evt in events_a:
        totals_a[evt.name] += evt.duration_ms
        counts_a[evt.name] += 1
    for evt in events_b:
        totals_b[evt.name] += evt.duration_ms
        counts_b[evt.name] += 1

    all_timers = sorted(set(totals_a.keys()) | set(totals_b.keys()))

    print("\n" + "=" * 80)
    print(f"  TRACE DIFF: {file_a} vs {file_b}")
    print("=" * 80)
    print(f"  {'Timer':<25} {'A (ms)':>10} {'B (ms)':>10} {'Δ (ms)':>10} "
          f"{'Δ%':>8} {'Flag':>6}")
    print("-" * 80)

    for name in all_timers:
        a_val = totals_a.get(name, 0.0)
        b_val = totals_b.get(name, 0.0)
        delta = b_val - a_val
        pct = (delta / a_val * 100) if abs(a_val) > 1e-9 else float('inf')

        # Flag significant deviations (>50% change or >2σ if we had std)
        # Since we only have 2 points, use a simple threshold
        is_significant = abs(pct) > 50 and abs(delta) > 1.0
        flag = "\033[91m!!!\033[0m" if is_significant else ""

        pct_str = f"{pct:+.1f}%" if abs(pct) < 10000 else "new"
        print(f"  {name:<25} {a_val:>10.2f} {b_val:>10.2f} {delta:>+10.2f} "
              f"{pct_str:>8} {flag}")

    print("=" * 80)

    # Summary
    total_a = sum(totals_a.values())
    total_b = sum(totals_b.values())
    print(f"\n  Total: A={total_a:.2f} ms, B={total_b:.2f} ms, "
          f"Δ={total_b-total_a:+.2f} ms ({(total_b-total_a)/total_a*100:+.1f}%)"
          if total_a > 0 else "")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AJB Performance Profiler — flamegraph, Welford, and diff")
    parser.add_argument("traces", nargs="*", help="Trace log files")
    parser.add_argument("--diff", action="store_true",
                        help="Two-trace diff mode (requires exactly 2 files)")
    parser.add_argument("--hot-threshold", type=float, default=0.10,
                        help="Hot path threshold (fraction of total, default 0.10)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI color output")
    args = parser.parse_args()

    if not args.traces:
        parser.print_help()
        sys.exit(1)

    if args.no_color:
        # Strip ANSI codes by monkey-patching print
        import functools
        _orig_print = print
        ansi_re = re.compile(r'\033\[[0-9;]*m')
        @functools.wraps(_orig_print)
        def _clean_print(*pargs, **kwargs):
            pargs = tuple(ansi_re.sub('', str(a)) for a in pargs)
            _orig_print(*pargs, **kwargs)
        import builtins
        builtins.print = _clean_print

    if args.diff:
        if len(args.traces) != 2:
            print("[ERROR] --diff requires exactly 2 trace files", file=sys.stderr)
            sys.exit(1)
        diff_traces(args.traces[0], args.traces[1])
        return

    # Single trace: flamegraph tree
    if len(args.traces) == 1:
        events = parse_trace(args.traces[0])
        if not events:
            print(f"[WARN] No [AJB_TIMER] events found in {args.traces[0]}")
            return

        tree = build_tree(events)
        print(f"\n{'='*60}")
        print(f"  FLAMEGRAPH TEXT TREE: {args.traces[0]}")
        print(f"  Total events: {len(events)}, Total time: {tree.total_ms:.2f} ms")
        print(f"{'='*60}")
        print_tree(tree, tree.total_ms, hot_threshold=args.hot_threshold)
        print()

    # Multiple traces: Welford variance report + tree for first
    if len(args.traces) > 1:
        # First trace tree
        events = parse_trace(args.traces[0])
        if events:
            tree = build_tree(events)
            print(f"\n{'='*60}")
            print(f"  FLAMEGRAPH (first trace): {args.traces[0]}")
            print(f"{'='*60}")
            print_tree(tree, tree.total_ms, hot_threshold=args.hot_threshold)

        # Welford across all traces
        accumulators = welford_multi_run(args.traces)
        print_welford_report(accumulators)


if __name__ == "__main__":
    main()
