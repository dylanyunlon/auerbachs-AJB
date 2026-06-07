#!/usr/bin/env python3
"""
ajb_join_optimizer.py — Join ordering algorithms for AJB query optimization

Claude #19 (M1001-M1005): Three join ordering strategies.

1. DPccp: Dynamic Programming over Connected Complement Pairs
2. IKKBZHeuristic: Rank-based greedy join ordering  
3. GreedyJoinOrdering: Cost-based greedy pair selection
"""

import itertools
import math
from collections import defaultdict


class JoinGraph:
    """Representation of a join query as a graph."""
    
    def __init__(self):
        self.relations = {}  # name -> cardinality
        self.edges = []      # (r1, r2, selectivity)
    
    def add_relation(self, name, cardinality):
        self.relations[name] = cardinality
        print(f"[AJB_TRACE][JoinGraph] add_relation({name}, card={cardinality})")
    
    def add_join(self, r1, r2, selectivity):
        self.edges.append((r1, r2, selectivity))
        print(f"[AJB_TRACE][JoinGraph] add_join({r1}⋈{r2}, sel={selectivity})")
    
    def neighbors(self, rel_set):
        """Get relations adjacent to a set of relations."""
        nbrs = set()
        for r1, r2, _ in self.edges:
            if r1 in rel_set and r2 not in rel_set:
                nbrs.add(r2)
            elif r2 in rel_set and r1 not in rel_set:
                nbrs.add(r1)
        return nbrs
    
    def is_connected(self, rel_set):
        """Check if a subset of relations is connected."""
        if len(rel_set) <= 1:
            return True
        visited = set()
        stack = [next(iter(rel_set))]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for r1, r2, _ in self.edges:
                if r1 == node and r2 in rel_set and r2 not in visited:
                    stack.append(r2)
                elif r2 == node and r1 in rel_set and r1 not in visited:
                    stack.append(r1)
        return visited == rel_set
    
    def join_cost(self, s1, s2, card1, card2):
        """Estimate cost of joining two relation sets."""
        sel = 1.0
        for r1, r2, s in self.edges:
            if (r1 in s1 and r2 in s2) or (r1 in s2 and r2 in s1):
                sel *= s
        return card1 * card2 * sel


def dp_ccp(graph):
    """DPccp: DP over Connected Complement Pairs.
    
    For each connected subset S of relations, find optimal join tree.
    Enumerate by splitting S into connected complement pairs (S1, S2).
    """
    rels = list(graph.relations.keys())
    n = len(rels)
    rel_idx = {r: i for i, r in enumerate(rels)}
    
    # DP table: bitmask -> (cost, cardinality, plan_string)
    dp = {}
    
    # Base cases: single relations
    for r in rels:
        mask = 1 << rel_idx[r]
        dp[mask] = (0, graph.relations[r], r)
        print(f"[AJB_TRACE][DPccp] base: {{{r}}} cost=0 card={graph.relations[r]}")
    
    # Enumerate subsets by size
    for size in range(2, n + 1):
        for subset in itertools.combinations(range(n), size):
            mask = sum(1 << i for i in subset)
            rel_set = frozenset(rels[i] for i in subset)
            
            if not graph.is_connected(rel_set):
                continue
            
            best = None
            # Try all splits into two non-empty connected parts
            submask = (mask - 1) & mask
            while submask > 0:
                complement = mask ^ submask
                if submask < complement:  # avoid duplicates
                    if submask in dp and complement in dp:
                        s1 = frozenset(rels[i] for i in range(n) if submask & (1 << i))
                        s2 = frozenset(rels[i] for i in range(n) if complement & (1 << i))
                        
                        if graph.is_connected(s1) and graph.is_connected(s2):
                            c1, card1, p1 = dp[submask]
                            c2, card2, p2 = dp[complement]
                            join_card = graph.join_cost(s1, s2, card1, card2)
                            total_cost = c1 + c2 + card1 + card2
                            
                            if best is None or total_cost < best[0]:
                                best = (total_cost, join_card, f"({p1}⋈{p2})")
                
                submask = (submask - 1) & mask
            
            if best:
                dp[mask] = best
                print(f"[AJB_TRACE][DPccp] {rel_set}: cost={best[0]:.0f}"
                      f" card={best[1]:.0f} plan={best[2]}")
    
    full_mask = (1 << n) - 1
    if full_mask in dp:
        cost, card, plan = dp[full_mask]
        print(f"[AJB_STATE][DPccp] optimal: cost={cost:.0f} plan={plan}")
        return cost, plan
    return None, None


def ikkbz_heuristic(graph):
    """IKKBZ heuristic: rank-based greedy join ordering.
    
    Rank = T(R) / (T(R) - 1) where T(R) is the "benefit" of joining R.
    At each step, join the relation with lowest rank.
    """
    remaining = set(graph.relations.keys())
    plan = None
    current_card = 0
    total_cost = 0
    joined = set()
    
    # Start with smallest relation
    start = min(remaining, key=lambda r: graph.relations[r])
    plan = start
    current_card = graph.relations[start]
    joined.add(start)
    remaining.remove(start)
    print(f"[AJB_TRACE][IKKBZ] start: {start} card={current_card}")
    
    step = 0
    while remaining:
        step += 1
        best_r = None
        best_rank = float('inf')
        best_card = 0
        
        for r in remaining:
            # Only consider adjacent relations
            if not graph.neighbors(joined) & {r}:
                continue
            
            join_card = graph.join_cost(joined, {r}, current_card, graph.relations[r])
            benefit = join_card / current_card if current_card > 0 else float('inf')
            rank = benefit / (benefit - 1) if benefit > 1 else float('inf')
            
            print(f"[AJB_TRACE][IKKBZ]   candidate {r}: join_card={join_card:.0f}"
                  f" benefit={benefit:.2f} rank={rank:.2f}")
            
            if rank < best_rank:
                best_rank = rank
                best_r = r
                best_card = join_card
        
        if best_r is None:
            # No adjacent relation found, pick closest
            best_r = min(remaining, key=lambda r: graph.relations[r])
            best_card = current_card * graph.relations[best_r]
        
        joined.add(best_r)
        remaining.remove(best_r)
        total_cost += current_card + graph.relations[best_r]
        current_card = best_card
        plan = f"({plan}⋈{best_r})"
        print(f"[AJB_STATE][IKKBZ] step {step}: join {best_r},"
              f" cost_so_far={total_cost:.0f} card={current_card:.0f}")
    
    print(f"[AJB_STATE][IKKBZ] final: cost={total_cost:.0f} plan={plan}")
    return total_cost, plan


def greedy_join_ordering(graph):
    """Greedy: at each step, join the pair with smallest intermediate result."""
    active = {}
    for r, c in graph.relations.items():
        active[frozenset([r])] = (c, r)
    
    total_cost = 0
    step = 0
    
    while len(active) > 1:
        step += 1
        best = None
        
        for s1, s2 in itertools.combinations(active.keys(), 2):
            # Check if s1 and s2 share an edge
            has_edge = False
            for r1, r2, _ in graph.edges:
                if (r1 in s1 and r2 in s2) or (r1 in s2 and r2 in s1):
                    has_edge = True
                    break
            if not has_edge:
                continue
            
            card1, p1 = active[s1]
            card2, p2 = active[s2]
            join_card = graph.join_cost(s1, s2, card1, card2)
            cost = card1 + card2
            
            if best is None or cost < best[0]:
                best = (cost, s1, s2, join_card, f"({p1}⋈{p2})")
        
        if best is None:
            break
        
        cost, s1, s2, new_card, new_plan = best
        total_cost += cost
        merged = s1 | s2
        del active[s1]
        del active[s2]
        active[merged] = (new_card, new_plan)
        
        print(f"[AJB_TRACE][Greedy] step {step}: {new_plan}"
              f" cost={cost:.0f} card={new_card:.0f}")
    
    final = list(active.values())[0]
    print(f"[AJB_STATE][Greedy] final: cost={total_cost:.0f} plan={final[1]}")
    return total_cost, final[1]


def demo():
    print("=" * 64)
    print("  AJB Join Optimizer Demo")
    print("=" * 64)
    
    g = JoinGraph()
    g.add_relation("R", 1000)
    g.add_relation("S", 500)
    g.add_relation("T", 200)
    g.add_relation("U", 800)
    g.add_join("R", "S", 0.01)
    g.add_join("S", "T", 0.05)
    g.add_join("T", "U", 0.02)
    g.add_join("R", "U", 0.03)
    
    print("\n--- DPccp ---")
    c1, p1 = dp_ccp(g)
    
    print("\n--- IKKBZ ---")
    c2, p2 = ikkbz_heuristic(g)
    
    print("\n--- Greedy ---")
    c3, p3 = greedy_join_ordering(g)
    
    print(f"\nComparison: DPccp={c1:.0f} IKKBZ={c2:.0f} Greedy={c3:.0f}")


if __name__ == "__main__":
    demo()
