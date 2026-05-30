#!/usr/bin/env python3
"""
twodir_edges.py — AJB-adapted bidirectional edge generator

Origin: upstream/joinrenum/twodir.py (22 lines)
Adaptation (~20%): AJB CLI args, duplicate/self-loop detection, structured
  output, and pandas-free fallback for lightweight environments.

Reads a pipe-delimited edge table and adds reverse edges.
Used to prepare graph data for symmetric join queries.

Usage:
  python3 twodir_edges.py [input] [output]
"""

import argparse
import os
import sys

def process_with_sets(input_path, output_path):
    """Pure-Python fallback (no pandas needed)."""
    edges = set()
    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                s, t = parts[0].strip(), parts[1].strip()
                edges.add((s, t))
                edges.add((t, s))  # reverse

    # Remove self-loops
    self_loops = sum(1 for s, t in edges if s == t)
    if self_loops > 0:
        print(f"[AJB_WARN] {self_loops} self-loops detected, keeping them")

    with open(output_path, "w") as f:
        for s, t in sorted(edges):
            f.write(f"{s}|{t}\n")

    return len(edges)

def process_with_pandas(input_path, output_path):
    """Pandas version (upstream-compatible)."""
    import pandas as pd

    df = pd.read_csv(input_path, sep='|', header=None)
    df.columns = ['source', 'target']

    original_count = len(df)
    reverse_edges = df[['target', 'source']].rename(
        columns={'target': 'source', 'source': 'target'})
    combined = pd.concat([df, reverse_edges]).drop_duplicates().reset_index(drop=True)

    combined.to_csv(output_path, sep='|', index=False, header=False)
    return original_count, len(combined)

def main():
    parser = argparse.ArgumentParser(description="AJB bidirectional edge generator")
    parser.add_argument("input", nargs="?", default="db/R1.tbl")
    parser.add_argument("output", nargs="?", default="db/edges_with_reverse.tbl")
    parser.add_argument("--no-pandas", action="store_true",
                        help="Use pure-Python fallback")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[AJB_ERROR] Input not found: {args.input}")
        sys.exit(1)

    print(f"[AJB] twodir_edges: {args.input} -> {args.output}")

    if args.no_pandas:
        total = process_with_sets(args.input, args.output)
        print(f"[AJB_RESULTS] total edges (with reverse) = {total}")
    else:
        try:
            orig, combined = process_with_pandas(args.input, args.output)
            print(f"[AJB_RESULTS] original={orig} -> combined={combined} "
                  f"(+{combined - orig} reverse edges)")
        except ImportError:
            print("[AJB_WARN] pandas not available, falling back to pure Python")
            total = process_with_sets(args.input, args.output)
            print(f"[AJB_RESULTS] total edges (with reverse) = {total}")

    print("[AJB] twodir_edges DONE")

if __name__ == "__main__":
    main()
