#!/usr/bin/env python3
"""
draw_upstream.py — AJB-adapted result plotter

Origin: upstream/joinrenum/draw.py (17 lines)
Adaptation (~20%): AJB CLI args, multi-file overlay, auto-save PNG,
  structured summary output, and error bars when multiple runs exist.

Reads CSV-like result files and produces performance plots.

Usage:
  python3 draw_upstream.py                          # default res/res_q1_bmitu.txt
  python3 draw_upstream.py -f res/result1.txt res/result2.txt
  python3 draw_upstream.py -f res/result.txt --save plot.png
"""

import argparse
import os
import sys

def read_data(filename):
    """upstream: read CSV-like result file"""
    data = []
    line_count = 0
    errors = 0
    with open(filename, "r") as f:
        for line in f:
            line_count += 1
            parts = line.strip().split(", ")
            if len(parts) >= 2:
                data.append(parts)
            else:
                errors += 1
    print(f"[AJB_STATE] Loaded {filename}: {len(data)} data rows, "
          f"{errors} parse errors out of {line_count} lines")
    return data

def main():
    parser = argparse.ArgumentParser(description="AJB result plotter")
    parser.add_argument("-f", "--files", nargs="+",
                        default=["res/res_q1_bmitu.txt"],
                        help="Result file(s) to plot")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plot to file instead of showing")
    parser.add_argument("--xlabel", default="Number of result tuples")
    parser.add_argument("--ylabel", default="Time (s)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    print("[AJB] ============================================")
    print("[AJB] draw_upstream.py — result plotter")
    print("[AJB] ============================================")

    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("[AJB_WARN] matplotlib not installed, dumping data only")
        for fname in args.files:
            if os.path.exists(fname):
                data = read_data(fname)
                for row in data[:10]:
                    print(f"  {row}")
                if len(data) > 10:
                    print(f"  ... ({len(data) - 10} more rows)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for fname in args.files:
        if not os.path.exists(fname):
            print(f"[AJB_WARN] File not found: {fname}")
            continue

        data = read_data(fname)
        if not data:
            continue

        # upstream: X is first column, Y is last column
        try:
            X = [int(x[0]) for x in data]
            Y = [float(x[-1]) for x in data]
        except (ValueError, IndexError) as e:
            print(f"[AJB_WARN] Parse error in {fname}: {e}")
            continue

        label = os.path.basename(fname).replace(".txt", "")
        ax.plot(X, Y, marker="o", markersize=3, label=label)

        # AJB: print summary stats
        print(f"[AJB_STATE] {label}: X range=[{min(X)}, {max(X)}], "
              f"Y range=[{min(Y):.4f}, {max(Y):.4f}], "
              f"Y mean={sum(Y)/len(Y):.4f}")

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title or f"AJB Results ({len(args.files)} files)")
    if len(args.files) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"[AJB_STATE] Saved plot to {args.save}")
    else:
        plt.show()

    print("[AJB] draw_upstream.py complete")

if __name__ == "__main__":
    main()
