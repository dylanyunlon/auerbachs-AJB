#!/usr/bin/env python3
"""
ajb_gpu_topology_viz.py — M1131: GPU Interconnect Topology ASCII Visualizer

Reads ProbeInterconnect-style output (or nvidia-smi topo output) and
renders an ASCII art diagram of the GPU interconnect topology, labeling
each link with NVLink generation or PCIe bandwidth.

Usage:
    python3 ajb_gpu_topology_viz.py [topo_file]
    nvidia-smi topo -m | python3 ajb_gpu_topology_viz.py -

Algorithm notes:
    - Parses adjacency matrix from nvidia-smi or AJB ProbeInterconnect logs
    - Uses Kruskal's algorithm (union-find) to build a minimum spanning tree
      of the interconnect for the ASCII layout
    - Renders using box-drawing characters with link bandwidth annotations
"""

import sys
import re
from collections import defaultdict


# ============================================================================
# Union-Find for Kruskal MST
# ============================================================================

class UnionFind:
    """Weighted union-find with path compression for MST construction."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.component_size = [1] * n

    def find(self, x):
        # Path compression: make every node on the path point to root
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False  # already in same component
        # Union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.component_size[ra] += self.component_size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# ============================================================================
# Topology parser
# ============================================================================

LINK_PRIORITY = {
    'NV12': 0,   # NVLink 12 lanes (highest bandwidth)
    'NV8':  1,
    'NV6':  2,
    'NV4':  3,
    'NV2':  4,
    'NV1':  5,
    'SYS':  6,   # Cross-socket QPI/UPI
    'NODE': 7,   # Same NUMA node
    'PIX':  8,   # Same PCIe switch
    'PXB':  9,   # Cross PCIe bridge
    'PHB':  10,  # PCIe host bridge
    'SOC':  11,  # On-chip (SoC GPUs)
    'X':    99,  # Self
}


def parse_topo_matrix(lines):
    """Parse nvidia-smi topo -m or AJB ProbeInterconnect output."""
    gpus = []
    adjacency = {}  # (i, j) -> link_type

    # Find header line with GPU labels
    header = None
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # nvidia-smi format: "GPU0  GPU1  GPU2 ..." as header
        if re.match(r'^(GPU\d+\s+)+', line) or 'GPU0' in line:
            header = re.findall(r'GPU(\d+)', line)
            continue
        # AJB format: "[AJB_BP][Topo] GPU0->GPU1: NV4"
        m = re.match(r'.*GPU(\d+)\s*->\s*GPU(\d+)\s*:\s*(\w+)', line)
        if m:
            i, j, link = int(m.group(1)), int(m.group(2)), m.group(3)
            adjacency[(i, j)] = link
            if i not in gpus:
                gpus.append(i)
            if j not in gpus:
                gpus.append(j)
            continue
        # nvidia-smi matrix row: "GPU0  X  NV4  NV4  SYS"
        parts = line.split()
        if parts and parts[0].startswith('GPU'):
            gpu_id = int(re.findall(r'\d+', parts[0])[0])
            if gpu_id not in gpus:
                gpus.append(gpu_id)
            for col_idx, val in enumerate(parts[1:]):
                val = val.strip()
                if val in LINK_PRIORITY and val != 'X':
                    if header and col_idx < len(header):
                        other = int(header[col_idx])
                    else:
                        other = col_idx
                    adjacency[(gpu_id, other)] = val

    gpus.sort()
    return gpus, adjacency


def build_mst_edges(gpus, adjacency):
    """Kruskal's algorithm: build MST of GPU interconnect graph."""
    # Collect all edges sorted by priority (lower = better bandwidth)
    edges = []
    seen = set()
    for (i, j), link in adjacency.items():
        if i == j:
            continue
        edge_key = (min(i, j), max(i, j))
        if edge_key not in seen:
            seen.add(edge_key)
            priority = LINK_PRIORITY.get(link, 50)
            edges.append((priority, i, j, link))
    edges.sort()

    n = len(gpus)
    gpu_to_idx = {g: idx for idx, g in enumerate(gpus)}
    uf = UnionFind(n)
    mst = []

    for priority, i, j, link in edges:
        if i in gpu_to_idx and j in gpu_to_idx:
            if uf.union(gpu_to_idx[i], gpu_to_idx[j]):
                mst.append((i, j, link))
                if len(mst) == n - 1:
                    break

    return mst


# ============================================================================
# ASCII renderer
# ============================================================================

def render_topology(gpus, adjacency, mst_edges):
    """Render GPU topology as ASCII art."""
    lines = []
    lines.append("╔══════════════════════════════════════════════════════╗")
    lines.append("║        AJB GPU Interconnect Topology                ║")
    lines.append("╠══════════════════════════════════════════════════════╣")

    if not gpus:
        lines.append("║  No GPU topology data found.                       ║")
        lines.append("╚══════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    # Summary line
    n = len(gpus)
    num_nvlink = sum(1 for v in adjacency.values() if v.startswith('NV'))
    num_pcie = sum(1 for v in adjacency.values() if v in ('PIX', 'PXB', 'PHB', 'SYS', 'NODE'))
    lines.append(f"║  GPUs: {n}   NVLink edges: {num_nvlink}   "
                 f"PCIe edges: {num_pcie}".ljust(55) + "║")
    lines.append("╠══════════════════════════════════════════════════════╣")

    # Render each GPU as a box
    for g in gpus:
        box = f"  ┌─────────┐"
        lines.append("║" + box.ljust(55) + "║")
        label = f"  │  GPU {g}  │"
        lines.append("║" + label.ljust(55) + "║")
        box = f"  └────┬────┘"
        lines.append("║" + box.ljust(55) + "║")

        # Show connections from this GPU
        connections = []
        for (i, j), link in sorted(adjacency.items()):
            if i == g and i != j:
                bw_label = link
                if link.startswith('NV'):
                    lanes = link[2:]
                    bw_label = f"NVLink({lanes})"
                connections.append(f"GPU{j}:{bw_label}")

        if connections:
            conn_str = "       ├── " + ", ".join(connections[:3])
            lines.append("║" + conn_str[:55].ljust(55) + "║")
            if len(connections) > 3:
                conn_str2 = "       └── " + ", ".join(connections[3:6])
                lines.append("║" + conn_str2[:55].ljust(55) + "║")
        else:
            lines.append("║       └── (no peer connections)" + " " * 22 + "║")

    # MST summary
    lines.append("╠══════════════════════════════════════════════════════╣")
    lines.append("║  MST (minimum spanning tree of interconnect):       ║")
    for i, j, link in mst_edges:
        edge_str = f"    GPU{i} ──[{link}]── GPU{j}"
        lines.append("║" + edge_str.ljust(55) + "║")
    if not mst_edges:
        lines.append("║    (single GPU — no spanning tree)                  ║")

    lines.append("╚══════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1 and sys.argv[1] != '-':
        with open(sys.argv[1]) as f:
            input_lines = f.readlines()
    else:
        input_lines = sys.stdin.readlines()

    if not input_lines:
        # Generate synthetic topology for demo
        print("[AJB_BP][TopoViz] No input — generating 4-GPU demo topology")
        input_lines = [
            "GPU0 -> GPU1: NV4\n",
            "GPU0 -> GPU2: NV4\n",
            "GPU0 -> GPU3: SYS\n",
            "GPU1 -> GPU0: NV4\n",
            "GPU1 -> GPU2: SYS\n",
            "GPU1 -> GPU3: NV4\n",
            "GPU2 -> GPU0: NV4\n",
            "GPU2 -> GPU1: SYS\n",
            "GPU2 -> GPU3: NV4\n",
            "GPU3 -> GPU0: SYS\n",
            "GPU3 -> GPU1: NV4\n",
            "GPU3 -> GPU2: NV4\n",
        ]

    gpus, adjacency = parse_topo_matrix(input_lines)
    mst = build_mst_edges(gpus, adjacency)
    print(render_topology(gpus, adjacency, mst))

    # Summary stats
    print(f"\n[AJB_BP][TopoViz] Parsed {len(gpus)} GPUs, "
          f"{len(adjacency)} directed edges, {len(mst)} MST edges")


if __name__ == "__main__":
    main()
