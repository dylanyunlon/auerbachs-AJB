#!/usr/bin/env python3
"""
draw_results.py — AJB vs Upstream comparison plotter

Origin: upstream/joinrenum/draw.py (17 lines)
Adaptation: complete algorithmic rewrite for M961-M965.

New capabilities:
  - Dual Y-axis: left axis = timer_ms, right axis = trace_count
  - Grouped bar chart: test_name groups with 3 bars (ajb/full/upstream)
  - Outlier annotation: points >2σ from group mean are red-circled
  - Agg backend: matplotlib.use('Agg') for headless server environments
  - CSV input: reads structured experiment CSVs

Usage:
  python3 draw_results.py res/result.txt --save fig.png
  python3 draw_results.py --csv bench.csv --save fig.png
  python3 draw_results.py f1.txt f2.txt --labels "baseline,AJB"
"""

import argparse
import csv
import math
import sys
import os
from collections import defaultdict

# Agg backend for headless environments — must be set before pyplot import
# Graceful fallback when matplotlib is absent (data-dump-only mode)
_HAS_MPL = True
try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import numpy as np
except ImportError:
    _HAS_MPL = False
    plt = None
    np = None
    print("[AJB_WARN] matplotlib/numpy not installed — data-dump-only mode")


def _dump_state(label, **kwargs):
    """Breakpoint-style state dump: prints all kwargs as structured diagnostics.
    Useful for verifying intermediate computation during experiment runs."""
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    print(f"[AJB_BP] {label}: " + ", ".join(parts))


# ---------------------------------------------------------------------------
# Welford accumulator for outlier detection (matches parse_ajb_trace.py)
# ---------------------------------------------------------------------------
class WelfordAccumulator:
    """Numerically stable incremental mean/variance with merge support."""
    __slots__ = ('n', 'mean', '_m2', '_min', '_max')

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self._min = float('inf')
        self._max = float('-inf')

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    def merge(self, other):
        """Chan's parallel combine formula."""
        if other.n == 0:
            return
        if self.n == 0:
            self.n, self.mean, self._m2 = other.n, other.mean, other._m2
            self._min, self._max = other._min, other._max
            return
        combined_n = self.n + other.n
        delta = other.mean - self.mean
        self._m2 += other._m2 + delta * delta * self.n * other.n / combined_n
        self.mean = (self.mean * self.n + other.mean * other.n) / combined_n
        self.n = combined_n
        self._min = min(self._min, other._min)
        self._max = max(self._max, other._max)

    @property
    def stddev(self):
        return math.sqrt(self._m2 / (self.n - 1)) if self.n >= 2 else 0.0

    @property
    def cv(self):
        return self.stddev / abs(self.mean) if self.n >= 2 and self.mean != 0 else 0.0


# ---------------------------------------------------------------------------
# Legacy line-based reader (backward compat)
# ---------------------------------------------------------------------------
def read_data(filename):
    """Read comma-separated result file. Returns (X, Y) lists."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) >= 2:
                data.append(parts)
    if not data:
        print(f"[AJB_WARN] No data in {filename}")
        return [], []
    X = [int(row[0]) for row in data]
    Y = [float(row[-1]) for row in data]
    _dump_state("read_data", file=filename, rows=len(X),
                X_range=f"[{min(X)},{max(X)}]",
                Y_range=f"[{min(Y):.4f},{max(Y):.4f}]",
                Y_mean=f"{sum(Y)/len(Y):.4f}")
    return X, Y


# ---------------------------------------------------------------------------
# CSV reader for structured experiment results
# ---------------------------------------------------------------------------
def read_csv_data(csv_path):
    """Read experiment CSV. Expected columns: test_name, variant, timer_ms, ajb_total.
    Returns list of row dicts with numeric coercion."""
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ('timer_ms', 'ajb_total', 'seed'):
                if key in row:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            rows.append(row)
    print(f"[AJB_STATE] CSV loaded: {len(rows)} rows from {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Dual-Y-axis grouped bar chart with outlier detection
# ---------------------------------------------------------------------------
def draw_grouped_bar(csv_path, save_path, title=None):
    """Draw dual-Y-axis grouped bar chart from experiment CSV.

    Algorithm:
    1. Parse CSV, group by (test_name, variant)
    2. Welford-aggregate per-group means for timer_ms and ajb_total
    3. Compute cross-test outlier thresholds per variant (mean ± 2σ)
    4. Build bar positions with per-variant offsets
    5. Render left axis (timer_ms bars) and right axis (ajb_total markers)
    6. Annotate outlier bars with red borders + diamond markers
    """
    if not _HAS_MPL:
        print("[AJB_WARN] matplotlib absent — printing CSV summary only")
        rows = read_csv_data(csv_path)
        for r in rows[:20]:
            _dump_state("csv_row", **{k: v for k, v in r.items()
                        if k in ('test_name', 'variant', 'timer_ms', 'ajb_total')})
        return

    rows = read_csv_data(csv_path)

    has_timer = any(isinstance(r.get('timer_ms'), (int, float)) for r in rows)
    has_trace = any(isinstance(r.get('ajb_total'), (int, float)) for r in rows)

    if not has_timer and not has_trace:
        print("[AJB_ERROR] CSV has neither timer_ms nor ajb_total columns")
        sys.exit(1)

    # Discover groups preserving order
    test_names = list(dict.fromkeys(r.get('test_name', '') for r in rows))
    variants = list(dict.fromkeys(r.get('variant', '') for r in rows))
    print(f"[AJB_STATE] tests={test_names} variants={variants}")

    # Welford-aggregate per (test, variant) key
    timer_agg = {}
    trace_agg = {}
    for r in rows:
        key = (r.get('test_name', ''), r.get('variant', ''))
        if has_timer:
            if key not in timer_agg:
                timer_agg[key] = WelfordAccumulator()
            timer_agg[key].update(r['timer_ms'])
        if has_trace:
            if key not in trace_agg:
                trace_agg[key] = WelfordAccumulator()
            trace_agg[key].update(r['ajb_total'])

    # Debug: print aggregation state
    for key, acc in sorted(timer_agg.items()):
        print(f"[AJB_STATE] timer_agg[{key}]: n={acc.n} "
              f"mean={acc.mean:.3f} σ={acc.stddev:.3f} cv={acc.cv:.3f}")

    # Compute per-variant outlier thresholds across tests
    timer_thresh = {}
    for variant in variants:
        if has_timer:
            cross_test_acc = WelfordAccumulator()
            for t in test_names:
                key = (t, variant)
                if key in timer_agg:
                    cross_test_acc.update(timer_agg[key].mean)
            timer_thresh[variant] = (cross_test_acc.mean, cross_test_acc.stddev)
            print(f"[AJB_STATE] outlier_thresh[{variant}]: "
                  f"μ={cross_test_acc.mean:.3f} 2σ={2*cross_test_acc.stddev:.3f}")

    # ----- Plot construction -----
    n_tests = len(test_names)
    n_variants = len(variants)
    bar_width = 0.8 / max(n_variants, 1)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    variant_colors = {v: colors[i % len(colors)] for i, v in enumerate(variants)}

    fig, ax1 = plt.subplots(figsize=(max(10, n_tests * 2), 7))
    ax2 = ax1.twinx() if (has_timer and has_trace) else None

    x_pos = np.arange(n_tests)
    outlier_count = 0

    for vi, variant in enumerate(variants):
        offset = (vi - n_variants / 2 + 0.5) * bar_width
        positions = x_pos + offset

        if has_timer:
            heights = []
            edge_colors = []
            for t in test_names:
                key = (t, variant)
                val = timer_agg[key].mean if key in timer_agg else 0.0
                heights.append(val)
                # Outlier: >2σ from variant's cross-test mean
                thresh_m, thresh_s = timer_thresh.get(variant, (0, 0))
                is_outlier = thresh_s > 0 and abs(val - thresh_m) > 2 * thresh_s
                if is_outlier:
                    edge_colors.append('#D32F2F')
                    outlier_count += 1
                else:
                    edge_colors.append(variant_colors[variant])

            ax1.bar(positions, heights, bar_width * 0.9,
                    color=variant_colors[variant],
                    edgecolor=edge_colors,
                    linewidth=[3 if ec == '#D32F2F' else 1
                               for ec in edge_colors],
                    label=f'{variant} (timer_ms)', alpha=0.85)

            # Annotate outliers with red diamond + value label
            for pos, h, ec in zip(positions, heights, edge_colors):
                if ec == '#D32F2F':
                    ax1.plot(pos, h, 'D', color='#D32F2F', markersize=8,
                             zorder=5)
                    ax1.annotate(f'{h:.1f}', xy=(pos, h),
                                xytext=(0, 8), textcoords='offset points',
                                fontsize=7, color='#D32F2F',
                                ha='center', fontweight='bold')

        # Right axis: trace count markers
        if has_trace and ax2 is not None:
            trace_vals = [trace_agg.get((t, variant),
                          WelfordAccumulator()).mean for t in test_names]
            ax2.plot(positions, trace_vals, 's--',
                     color=variant_colors[variant], alpha=0.6,
                     markersize=6, label=f'{variant} (trace_count)')

    # Formatting
    ax1.set_xlabel('Test Name', fontsize=12)
    ax1.set_ylabel('timer_ms', fontsize=12, color='#333')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(test_names, rotation=30, ha='right', fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)

    if ax2 is not None:
        ax2.set_ylabel('trace_count (ajb_total)', fontsize=12, color='#666')

    plot_title = title or 'AJB vs Upstream Comparison'
    if outlier_count > 0:
        plot_title += f'  ({outlier_count} outlier{"s" if outlier_count > 1 else ""} >2σ)'
    ax1.set_title(plot_title, fontsize=14, fontweight='bold')

    # Combined legend from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2,
                   loc='upper left', fontsize=9)
    else:
        ax1.legend(loc='upper left', fontsize=9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[AJB_STATE] saved dual-axis comparison to {save_path}")
    else:
        print("[AJB_WARN] No --save path; Agg backend cannot show()")
    plt.close(fig)
    print(f"[AJB_STATE] plot complete: {n_tests} tests × {n_variants} variants, "
          f"{outlier_count} outliers marked")


# ---------------------------------------------------------------------------
# Legacy line plot (backward compat with old result.txt files)
# ---------------------------------------------------------------------------
def draw_legacy(args):
    """Original line-plot mode for .txt result files.
    Falls back to data-dump if matplotlib is unavailable."""
    if not _HAS_MPL:
        print("[AJB_STATE] matplotlib absent — dumping data only")
        for fname in args.files:
            if not os.path.exists(fname):
                print(f"[AJB_WARN] File not found: {fname}")
                continue
            X, Y = read_data(fname)
            for xi, yi in zip(X[:15], Y[:15]):
                print(f"  x={xi}  y={yi:.6f}")
            if len(X) > 15:
                print(f"  ... ({len(X) - 15} more rows)")
        return

    labels = (args.labels.split(",") if args.labels
              else [os.path.basename(f) for f in args.files])
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, filename in enumerate(args.files):
        if not os.path.exists(filename):
            print(f"[AJB_WARN] File not found: {filename}")
            continue
        X, Y = read_data(filename)
        if not X:
            continue
        label = labels[i] if i < len(labels) else f"series_{i}"
        ax.plot(X, Y, marker="o", markersize=3, label=label)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    if len(args.files) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"[AJB_STATE] saved line plot to {args.save}")
    else:
        print("[AJB_WARN] No --save path; Agg backend cannot show()")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="AJB result plotter (M961-M965)")
    parser.add_argument("files", nargs="*", help="Result file(s) (.txt legacy)")
    parser.add_argument("--csv", default=None,
                        help="Experiment CSV for grouped bar chart mode")
    parser.add_argument("--title", default="AJB Results", help="Plot title")
    parser.add_argument("--xlabel", default="Number of result tuples")
    parser.add_argument("--ylabel", default="Time (s)")
    parser.add_argument("--labels", default=None,
                        help="Comma-separated legend labels")
    parser.add_argument("--save", default=None,
                        help="Save to file instead of showing")
    parser.add_argument("--logx", action="store_true", help="Log-scale X axis")
    parser.add_argument("--logy", action="store_true", help="Log-scale Y axis")
    args = parser.parse_args()

    if args.csv:
        draw_grouped_bar(args.csv, args.save, args.title)
    elif args.files:
        draw_legacy(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
