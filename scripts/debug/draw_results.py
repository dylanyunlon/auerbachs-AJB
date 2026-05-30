#!/usr/bin/env python3
"""
draw_results.py — AJB-adapted result plotter

Origin: upstream/joinrenum/draw.py (17 lines)
Adaptation (~20%): AJB CLI args, multiple file overlay, structured output,
  and save-to-file option for headless environments.

Usage:
  python3 draw_results.py res/result.txt
  python3 draw_results.py res/result.txt --save fig.png --title "Q1 AJB"
  python3 draw_results.py f1.txt f2.txt --labels "baseline,AJB"
"""

import argparse
import sys
import os

def read_data(filename):
    """Read comma-separated result file. Returns (X, Y) lists."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) >= 2:
                data.append(parts)
    X = [int(row[0]) for row in data]
    Y = [float(row[-1]) for row in data]
    # AJB: print data summary
    print(f"[AJB] {filename}: {len(X)} points, X=[{min(X)},{max(X)}], Y=[{min(Y):.4f},{max(Y):.4f}]")
    return X, Y

def main():
    parser = argparse.ArgumentParser(description="AJB result plotter")
    parser.add_argument("files", nargs="+", help="Result file(s)")
    parser.add_argument("--title", default="AJB Results", help="Plot title")
    parser.add_argument("--xlabel", default="Number of result tuples")
    parser.add_argument("--ylabel", default="Time (s)")
    parser.add_argument("--labels", default=None, help="Comma-separated legend labels")
    parser.add_argument("--save", default=None, help="Save to file instead of showing")
    parser.add_argument("--logx", action="store_true", help="Log-scale X axis")
    parser.add_argument("--logy", action="store_true", help="Log-scale Y axis")
    args = parser.parse_args()

    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("[AJB_ERROR] matplotlib not installed. pip install matplotlib")
        sys.exit(1)

    labels = args.labels.split(",") if args.labels else [os.path.basename(f) for f in args.files]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, filename in enumerate(args.files):
        if not os.path.exists(filename):
            print(f"[AJB_WARN] File not found: {filename}")
            continue
        X, Y = read_data(filename)
        label = labels[i] if i < len(labels) else f"series_{i}"
        ax.plot(X, Y, marker="o", markersize=3, label=label)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    if args.logx: ax.set_xscale("log")
    if args.logy: ax.set_yscale("log")
    if len(args.files) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"[AJB] Saved plot to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
