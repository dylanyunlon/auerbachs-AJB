#!/usr/bin/env python3
"""
twodir_edges.py — AJB-adapted bidirectional edge generator with graph analysis

Origin: upstream/joinrenum/twodir.py (22 lines)
Algorithm changes (M976-M980):
  1. Union-Find with path compression + union by rank for connected
     component detection on the bidirectional edge set
  2. Graph density analysis: compute edge density = |E|/(|V|*(|V|-1)/2),
     classify as sparse/moderate/dense
  3. Kruskal's MST: sort edges, union-find merge, track component merges
     and total MST weight. Print each merge step.

Usage:
  python3 twodir_edges.py [input] [output]
"""

import argparse
import os
import sys


class UnionFind:
    """Union-Find with path compression and union by rank.

    Algorithm change #1: instead of just adding reverse edges,
    we track connected components during edge insertion.
    """
    __slots__ = ('parent', 'rank', 'n_components')

    def __init__(self, elements):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}
        self.n_components = len(elements)

    def find(self, x):
        # Path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Compress path
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, x, y):
        """Union by rank. Returns True if a merge happened (were different components)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True

    def component_sizes(self):
        """Return dict of root -> size."""
        sizes = {}
        for x in self.parent:
            r = self.find(x)
            sizes[r] = sizes.get(r, 0) + 1
        return sizes


def kruskal_mst(vertices, edges_with_weight):
    """Algorithm change #3: Kruskal's MST with diagnostic output.

    edges_with_weight: list of (weight, u, v)
    Returns (mst_weight, mst_edges, merge_log).
    """
    uf = UnionFind(vertices)
    sorted_edges = sorted(edges_with_weight)
    mst_edges = []
    mst_weight = 0
    merge_log = []

    print(f"[AJB_STATE][Kruskal] starting: {len(vertices)} vertices,"
          f" {len(sorted_edges)} candidate edges")

    for weight, u, v in sorted_edges:
        if uf.find(u) != uf.find(v):
            comp_u = uf.find(u)
            comp_v = uf.find(v)
            size_u = sum(1 for x in vertices if uf.find(x) == comp_u)
            size_v = sum(1 for x in vertices if uf.find(x) == comp_v)

            uf.union(u, v)
            mst_edges.append((u, v, weight))
            mst_weight += weight
            merge_log.append({
                "edge": (u, v),
                "weight": weight,
                "merged_components": (comp_u, comp_v),
                "sizes": (size_u, size_v),
                "remaining": uf.n_components,
            })

            print(f"[AJB_TRACE][Kruskal] merge ({u},{v}) w={weight}:"
                  f" comp({comp_u},size={size_u}) + comp({comp_v},size={size_v})"
                  f" → {uf.n_components} components left")

            if uf.n_components == 1:
                break

    print(f"[AJB_STATE][Kruskal] MST: {len(mst_edges)} edges,"
          f" total_weight={mst_weight}, components={uf.n_components}")
    return mst_weight, mst_edges, merge_log


def process_with_analysis(input_path, output_path):
    """Process edges with Union-Find component tracking and graph analysis."""
    raw_edges = []
    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                s, t = parts[0].strip(), parts[1].strip()
                raw_edges.append((s, t))

    # Build bidirectional edge set
    edge_set = set()
    for s, t in raw_edges:
        edge_set.add((s, t))
        edge_set.add((t, s))

    # Self-loop detection
    self_loops = sum(1 for s, t in edge_set if s == t)
    if self_loops > 0:
        print(f"[AJB_WARN] {self_loops} self-loops detected")

    # Extract vertices
    vertices = set()
    for s, t in edge_set:
        vertices.add(s)
        vertices.add(t)

    n_v = len(vertices)
    # Undirected edge count (divide by 2 since we have both directions)
    undirected_edges = set()
    for s, t in edge_set:
        if s != t:
            undirected_edges.add((min(s, t), max(s, t)))
    n_e = len(undirected_edges)

    # Algorithm change #1: Connected components via Union-Find
    uf = UnionFind(vertices)
    for s, t in undirected_edges:
        uf.union(s, t)

    comp_sizes = uf.component_sizes()
    print(f"[AJB_STATE][UnionFind] {uf.n_components} connected components")
    for root, size in sorted(comp_sizes.items(), key=lambda x: -x[1]):
        print(f"[AJB_TRACE][UnionFind]   component(root={root}): {size} vertices")

    # Algorithm change #2: Graph density analysis
    max_edges = n_v * (n_v - 1) // 2 if n_v > 1 else 1
    density = n_e / max_edges if max_edges > 0 else 0
    density_class = "dense" if density > 0.5 else ("moderate" if density > 0.1 else "sparse")
    print(f"[AJB_STATE][Density] |V|={n_v}, |E|={n_e},"
          f" density={density:.4f} ({density_class})")

    # Algorithm change #3: Kruskal MST (using edge index as weight proxy)
    weighted_edges = []
    for idx, (u, v) in enumerate(sorted(undirected_edges)):
        # Use hash-based weight so MST is deterministic but non-trivial
        w = (hash(u + v) % 1000) + 1
        weighted_edges.append((w, u, v))

    if n_v > 1 and n_e > 0:
        mst_weight, mst_edges, _ = kruskal_mst(vertices, weighted_edges)
    else:
        mst_weight = 0
        mst_edges = []
        print("[AJB_STATE][Kruskal] skipped: too few vertices/edges")

    # Write output
    with open(output_path, "w") as f:
        for s, t in sorted(edge_set):
            f.write(f"{s}|{t}\n")

    print(f"\n[AJB_RESULTS] input={len(raw_edges)} -> output={len(edge_set)} edges")
    print(f"[AJB_RESULTS] components={uf.n_components}, density={density:.4f},"
          f" mst_weight={mst_weight}")
    return len(edge_set)


def main():
    parser = argparse.ArgumentParser(
        description="AJB bidirectional edge generator + graph analysis")
    parser.add_argument("input", nargs="?", default="db/R1.tbl")
    parser.add_argument("output", nargs="?", default="db/edges_with_reverse.tbl")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[AJB_ERROR] Input not found: {args.input}")
        sys.exit(1)

    print(f"[AJB] twodir_edges: {args.input} -> {args.output}")
    process_with_analysis(args.input, args.output)
    print("[AJB] twodir_edges DONE")


if __name__ == "__main__":
    main()
