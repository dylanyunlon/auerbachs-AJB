#!/usr/bin/env python3
"""
schemagen.py — AJB-adapted schema generator for joinrenum

Origin: upstream/joinrenum/schemagen.py (22 lines)
Algorithm changes (M971-M975):
  1. Hypergraph analysis: treat schema as hypergraph, compute fractional
     edge cover via LP (scipy.optimize.linprog as fallback for glpk)
  2. Join width estimation: greedy upper bound on Generalized Hypertree
     Width (GHW) via iterative bag construction
  3. Diagnostic output: print LP tableau, variable values, bag contents
     at each step of the greedy decomposition

Usage:
  python3 schemagen.py
  python3 schemagen.py --relations 6 --variables 12 --factor 3 --outdir db/
"""

import argparse
import os
import sys
from random import randint, seed as set_seed


class HypergraphAnalyzer:
    """Algorithm change #1: Hypergraph fractional edge cover computation.

    Given a hypergraph H = (V, E) where E is a set of hyperedges,
    compute the minimum fractional edge cover:
      min sum(w_e) for e in E
      s.t. for each v in V: sum(w_e for e containing v) >= 1
           w_e >= 0

    This is exactly a linear program. We use scipy.optimize.linprog
    (or fall back to a simple greedy fractional assignment).
    """

    def __init__(self, vertices, hyperedges):
        self.V = list(vertices)
        self.E = list(hyperedges)  # each is a frozenset of vertices
        self.v_index = {v: i for i, v in enumerate(self.V)}

    def fractional_edge_cover(self):
        """Solve the LP for minimum fractional edge cover number.

        Returns (optimal_value, weights_dict).
        """
        n_v = len(self.V)
        n_e = len(self.E)

        if n_v == 0 or n_e == 0:
            print("[AJB_STATE][HypergraphLP] empty hypergraph, cover=0")
            return 0.0, {}

        # Build constraint matrix A_ub: for each vertex v,
        # -sum(w_e for e containing v) <= -1  (negate for linprog convention)
        try:
            from scipy.optimize import linprog

            c = [1.0] * n_e  # minimize sum of weights
            A_ub = []
            b_ub = []
            for v in self.V:
                row = [0.0] * n_e
                for j, edge in enumerate(self.E):
                    if v in edge:
                        row[j] = -1.0
                A_ub.append(row)
                b_ub.append(-1.0)

            bounds = [(0, None)] * n_e

            print(f"[AJB_STATE][HypergraphLP] solving: {n_e} vars (edges),"
                  f" {n_v} constraints (vertices)")
            for i, v in enumerate(self.V):
                nonzero = [j for j in range(n_e) if A_ub[i][j] != 0]
                print(f"[AJB_TRACE][HypergraphLP]   {v}: covered by edges"
                      f" {[list(self.E[j]) for j in nonzero]}")

            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            if result.success:
                weights = {}
                for j, edge in enumerate(self.E):
                    if result.x[j] > 1e-9:
                        weights[str(sorted(edge))] = round(result.x[j], 6)
                opt = round(result.fun, 6)
                print(f"[AJB_STATE][HypergraphLP] optimal cover = {opt}")
                for k, w in weights.items():
                    print(f"[AJB_TRACE][HypergraphLP]   edge {k}: weight={w}")
                return opt, weights
            else:
                print(f"[AJB_WARN][HypergraphLP] LP failed: {result.message}")
                return self._greedy_fractional_cover()

        except ImportError:
            print("[AJB_WARN] scipy not available, using greedy fractional cover")
            return self._greedy_fractional_cover()

    def _greedy_fractional_cover(self):
        """Greedy fallback: assign weight 1/|e| to each edge for uncovered vertices."""
        covered = {v: 0.0 for v in self.V}
        weights = {j: 0.0 for j in range(len(self.E))}

        for v in self.V:
            if covered[v] >= 1.0:
                continue
            # Find edge containing v with maximum overlap with uncovered
            best_j = -1
            best_score = -1
            for j, edge in enumerate(self.E):
                if v in edge:
                    score = sum(1 for u in edge if covered[u] < 1.0)
                    if score > best_score:
                        best_score = score
                        best_j = j
            if best_j >= 0:
                need = 1.0 - covered[v]
                weights[best_j] += need
                for u in self.E[best_j]:
                    covered[u] += need
                print(f"[AJB_TRACE][GreedyCover] vertex {v}: assign"
                      f" {need:.3f} to edge {sorted(self.E[best_j])}")

        total = sum(weights.values())
        result = {str(sorted(self.E[j])): round(w, 6)
                  for j, w in weights.items() if w > 1e-9}
        print(f"[AJB_STATE][GreedyCover] total cover weight = {round(total, 6)}")
        return total, result

    def greedy_ghw_upper_bound(self):
        """Algorithm change #2: Greedy upper bound on Generalized Hypertree Width.

        Build a tree decomposition greedily:
        1. Start with uncovered edges = all edges
        2. At each step, form a bag that covers the maximum uncovered edges
        3. The width is max |bag| over all bags in the decomposition
        """
        uncovered = set(range(len(self.E)))
        bags = []
        width = 0

        print(f"[AJB_STATE][GHW] starting greedy decomposition:"
              f" {len(self.E)} edges, {len(self.V)} vertices")

        step = 0
        while uncovered:
            step += 1
            # Greedy: pick vertex subset (bag) that covers most uncovered edges
            # Strategy: union of the most-connected uncovered edge's vertices
            best_bag = set()
            best_covered = set()

            for j in uncovered:
                candidate_bag = set(self.E[j])
                # Extend bag with overlapping uncovered edges
                covers = set()
                for k in uncovered:
                    if self.E[k].issubset(candidate_bag):
                        covers.add(k)
                    elif self.E[k] & candidate_bag:
                        # Extend bag to include this edge too
                        extended = candidate_bag | self.E[k]
                        ext_covers = {m for m in uncovered if self.E[m].issubset(extended)}
                        if len(ext_covers) > len(covers):
                            candidate_bag = extended
                            covers = ext_covers

                if len(covers) > len(best_covered):
                    best_bag = candidate_bag
                    best_covered = covers

            if not best_covered:
                # Fallback: just pick one uncovered edge
                j = min(uncovered)
                best_bag = set(self.E[j])
                best_covered = {j}

            bags.append(sorted(best_bag))
            bag_width = len(best_bag)
            if bag_width > width:
                width = bag_width
            uncovered -= best_covered

            print(f"[AJB_TRACE][GHW] step {step}: bag={sorted(best_bag)}"
                  f" (size={bag_width}), covers {len(best_covered)} edges,"
                  f" remaining={len(uncovered)}")

        print(f"[AJB_STATE][GHW] decomposition: {len(bags)} bags,"
              f" width={width}, ghw_upper_bound={width}")
        return width, bags


def gen_schema(relation_num, variable_num, factor, outdir, rand_seed=None):
    if rand_seed is not None:
        set_seed(rand_seed)

    os.makedirs(outdir, exist_ok=True)

    relations_path = os.path.join(outdir, "relations.txt")
    numlines_path  = os.path.join(outdir, "numlines.txt")
    filenames_path = os.path.join(outdir, "filenames.txt")

    schema = []

    with open(relations_path, "w") as f:
        for i in range(relation_num):
            variables = [f"V{j+1}" for j in range(variable_num)
                         if randint(1, factor) == 1]
            if not variables:
                variables = [f"V{randint(1, variable_num)}"]
            schema.append((f"R{i+1}", variables))
            f.write(f"R{i+1}({','.join(variables)})\n")

    with open(numlines_path, "w") as f:
        for i in range(relation_num):
            f.write(f"R{i+1} 0\n")

    with open(filenames_path, "w") as f:
        for i in range(relation_num):
            f.write(f"R{i+1} {outdir}/R{i+1}.tbl\n")

    print(f"[AJB_STATE] Schema generated in {outdir}/")
    print(f"  relations = {relation_num}")
    print(f"  variables = {variable_num}")
    print(f"  factor    = {factor} (1/{factor} probability per variable)")
    for name, vars_list in schema:
        print(f"  {name}({', '.join(vars_list)})  [{len(vars_list)} attrs]")

    all_vars = set()
    for _, vars_list in schema:
        all_vars.update(vars_list)
    print(f"  total unique variables = {len(all_vars)}")

    # Algorithm change #1 + #2: Hypergraph analysis
    print(f"\n[AJB_BP] === Hypergraph Analysis ===")
    hyperedges = [frozenset(vars_list) for _, vars_list in schema]
    analyzer = HypergraphAnalyzer(sorted(all_vars), hyperedges)

    cover_val, cover_weights = analyzer.fractional_edge_cover()
    ghw, bags = analyzer.greedy_ghw_upper_bound()

    print(f"\n[AJB_STATE] Summary:")
    print(f"  fractional_edge_cover = {cover_val}")
    print(f"  ghw_upper_bound       = {ghw}")
    print(f"  n_bags                = {len(bags)}")

    return schema


def main():
    parser = argparse.ArgumentParser(description="AJB schema generator + hypergraph analysis")
    parser.add_argument("--relations", type=int, default=4)
    parser.add_argument("--variables", type=int, default=8)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--outdir", default="db/")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    gen_schema(args.relations, args.variables, args.factor, args.outdir, args.seed)
    print("[AJB] schemagen DONE")


if __name__ == "__main__":
    main()
