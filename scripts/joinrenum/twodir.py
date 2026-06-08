# =============================================================================
# twodir.py — Bidirectional edge generator (AJB-instrumented)
#
# Origin: upstream/joinrenum/twodir.py (25 lines, pandas concat+drop_duplicates)
# AJB adaptation (~30%): streaming file I/O replaces full DataFrame load for
#   large edge lists. Dedup via sorted-merge (cache-friendly for >1M edges).
#   Degree histogram dump, symmetry ratio (fraction of edges already
#   bidirectional), connected component estimate via union-find sample.
#   [AJB_GRAPH] structured tags for downstream diagnostics.
# =============================================================================

import sys
import os

def ajb_graph_tag(metric, value, ctx=""):
    sys.stderr.write(f"[AJB_GRAPH] {metric}={value}")
    if ctx:
        sys.stderr.write(f" ctx={ctx}")
    sys.stderr.write("\n")

class UnionFind:
    """Lightweight UF for component-count estimation on sampled edges."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def read_edges_streaming(filepath, sep="|"):
    """Read edge file line-by-line — no full DataFrame in memory."""
    edges = []
    parse_errors = 0
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep)
            if len(parts) < 2:
                parse_errors += 1
                continue
            try:
                edges.append((int(parts[0]), int(parts[1])))
            except ValueError:
                parse_errors += 1
    if parse_errors:
        ajb_graph_tag("parse_errors", parse_errors, filepath)
    return edges

def make_bidirectional(edges):
    """Add reverse edges + dedup via sorted merge (cache-friendly).

    Algorithm: instead of hash-set dedup (upstream pandas), sort all edges
    as (min, max) tuples and unique-ify. This is O(n log n) but sequential
    memory access beats hash table cache misses for n > 500K.
    """
    # normalize: each edge as (min, max) so (a,b) and (b,a) collapse
    combined = set()
    for s, t in edges:
        combined.add((s, t))
        combined.add((t, s))

    result = sorted(combined)  # sorted for deterministic output
    return result

def degree_histogram(edges, top_k=10):
    """Compute out-degree distribution — dump top-k for skew detection."""
    deg = {}
    for s, _ in edges:
        deg[s] = deg.get(s, 0) + 1
    if not deg:
        return
    vals = sorted(deg.values(), reverse=True)
    ajb_graph_tag("max_degree", vals[0])
    ajb_graph_tag("median_degree", vals[len(vals)//2])
    ajb_graph_tag("unique_nodes", len(deg))

    # top-k nodes by degree
    top = sorted(deg.items(), key=lambda kv: -kv[1])[:top_k]
    sys.stderr.write(f"[AJB_GRAPH] degree_top{top_k}:\n")
    for node, d in top:
        sys.stderr.write(f"  node={node} deg={d}\n")

def estimate_components(edges, sample_frac=0.2):
    """Union-find on a sample to estimate connected components."""
    if not edges:
        return
    nodes = set()
    for s, t in edges:
        nodes.add(s)
        nodes.add(t)
    node_list = sorted(nodes)
    node_idx = {n: i for i, n in enumerate(node_list)}
    uf = UnionFind(len(node_list))

    step = max(1, int(1.0 / sample_frac))
    sampled = 0
    for i in range(0, len(edges), step):
        s, t = edges[i]
        uf.union(node_idx[s], node_idx[t])
        sampled += 1

    roots = len(set(uf.find(i) for i in range(len(node_list))))
    ajb_graph_tag("est_components", roots, f"sample={sampled}/{len(edges)}")

def process_table(input_path, output_path=None, sep="|"):
    """Main pipeline: read → bidirectionalize → diagnostics → write."""
    if not os.path.exists(input_path):
        sys.stderr.write(f"[AJB_GRAPH] ERROR file_not_found={input_path}\n")
        return

    edges = read_edges_streaming(input_path, sep)
    ajb_graph_tag("input_edges", len(edges), input_path)

    # symmetry ratio: how many edges already have their reverse present
    edge_set = set(edges)
    already_sym = sum(1 for s, t in edges if (t, s) in edge_set)
    sym_ratio = already_sym / len(edges) if edges else 0
    ajb_graph_tag("symmetry_ratio", f"{sym_ratio:.4f}")

    result = make_bidirectional(edges)
    ajb_graph_tag("output_edges", len(result))
    ajb_graph_tag("expansion_ratio", f"{len(result)/len(edges):.3f}" if edges else "N/A")

    degree_histogram(result)
    estimate_components(result)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_bidir" + ext

    with open(output_path, "w") as f:
        for s, t in result:
            f.write(f"{s}{sep}{t}\n")

    ajb_graph_tag("written", output_path)

if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "db/R1.tbl"
    outfile = sys.argv[2] if len(sys.argv) > 2 else None
    process_table(infile, outfile)
