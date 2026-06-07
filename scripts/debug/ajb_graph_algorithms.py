"""
超图分析中的三种图算法实现
用于 Join 查询的超图结构分析

1. Tarjan's Strongly Connected Components (有向图)
2. Minimum Degree Ordering (树分解的启发式方法)
3. Chordal Graph Recognition (Maximum Cardinality Search)

每个算法都包含详细的执行过程打印。
"""

from collections import defaultdict
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
#  ANSI 颜色常量 (终端输出美化)
# ═══════════════════════════════════════════════════════════════════════════════

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def header(title: str) -> None:
    """打印算法标题横幅"""
    width = 72
    print(f"\n{BOLD}{CYAN}{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}{RESET}\n")


def step_print(msg: str) -> None:
    print(f"  {DIM}│{RESET} {msg}")


def section(msg: str) -> None:
    print(f"\n  {BOLD}{YELLOW}▸ {msg}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  算法 1: Tarjan's Strongly Connected Components
# ═══════════════════════════════════════════════════════════════════════════════


class TarjanSCC:
    """
    Tarjan 强连通分量算法

    在 Join 查询优化中,超图的有向关联结构可以通过 SCC 分析来检测
    循环依赖,从而判断查询是否为 acyclic (无环) 的。
    """

    def __init__(self, vertices: list, edges: dict[str, list[str]]):
        self.vertices = vertices
        self.adj = defaultdict(list)
        for u, neighbors in edges.items():
            for v in neighbors:
                self.adj[u].append(v)

        self.index_counter = 0
        self.index = {}          # 节点的 DFS 序号
        self.lowlink = {}        # 节点的 low-link 值
        self.on_stack = {}       # 节点是否在栈上
        self.stack = []          # Tarjan 栈
        self.sccs: list[list[str]] = []  # 结果: 所有 SCC

    def _strongconnect(self, v: str) -> None:
        self.index[v] = self.index_counter
        self.lowlink[v] = self.index_counter
        self.index_counter += 1
        self.stack.append(v)
        self.on_stack[v] = True

        step_print(
            f"访问 {BOLD}{v}{RESET}  "
            f"index={self.index[v]}  lowlink={self.lowlink[v]}  "
            f"栈={self.stack[:]}"
        )

        for w in self.adj[v]:
            if w not in self.index:
                step_print(f"  → 树边 {v}→{w},递归进入 {w}")
                self._strongconnect(w)
                old = self.lowlink[v]
                self.lowlink[v] = min(self.lowlink[v], self.lowlink[w])
                if old != self.lowlink[v]:
                    step_print(
                        f"  ↻ 回溯后更新: lowlink[{v}] = "
                        f"min({old}, lowlink[{w}]={self.lowlink[w]}) = "
                        f"{GREEN}{self.lowlink[v]}{RESET}"
                    )
            elif self.on_stack.get(w, False):
                old = self.lowlink[v]
                self.lowlink[v] = min(self.lowlink[v], self.index[w])
                step_print(
                    f"  → 回边 {v}→{w} (在栈上): lowlink[{v}] = "
                    f"min({old}, index[{w}]={self.index[w]}) = "
                    f"{GREEN}{self.lowlink[v]}{RESET}"
                )
            else:
                step_print(f"  → 交叉边 {v}→{w} (已完成,忽略)")

        # 如果 v 是 SCC 的根节点
        if self.lowlink[v] == self.index[v]:
            component = []
            while True:
                w = self.stack.pop()
                self.on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            self.sccs.append(component)
            step_print(
                f"  {RED}★ 弹出 SCC (根={v}): {component}  "
                f"栈={self.stack[:]}{RESET}"
            )

    def run(self) -> list[list[str]]:
        section("开始 Tarjan SCC 算法")
        for v in self.vertices:
            if v not in self.index:
                step_print(f"\n从未访问节点 {BOLD}{v}{RESET} 开始 DFS")
                self._strongconnect(v)
        return self.sccs


def demo_tarjan():
    """
    示例: 一个表示 join 依赖关系的有向图

    考虑查询: R(A,B) ⋈ S(B,C) ⋈ T(C,A) ⋈ U(D,E)
    属性之间的函数依赖可以构成有向图,其中 SCC 揭示了循环依赖。
    """
    header("算法 1: Tarjan's Strongly Connected Components")

    print(f"  {BOLD}场景:{RESET} Join 查询的属性依赖图")
    print(f"  查询: R(A,B) ⋈ S(B,C) ⋈ T(C,A) ⋈ U(D,E)")
    print(f"  有向边表示函数依赖: A→B, B→C, C→A, D→E\n")
    print(f"  有向图结构:")
    print(f"    A → B → C → A  (形成环)")
    print(f"    D → E           (无环)")
    print(f"    C → D           (连接两个部分)\n")

    vertices = ["A", "B", "C", "D", "E"]
    edges = {
        "A": ["B"],
        "B": ["C"],
        "C": ["A", "D"],
        "D": ["E"],
    }

    tarjan = TarjanSCC(vertices, edges)
    sccs = tarjan.run()

    section("结果汇总")
    print(f"\n  共发现 {BOLD}{len(sccs)}{RESET} 个强连通分量:\n")
    for i, scc in enumerate(sccs):
        trivial = "  (平凡分量)" if len(scc) == 1 else f"  {RED}(存在循环依赖!){RESET}"
        print(f"    SCC {i + 1}: {scc}{trivial}")

    has_cycle = any(len(scc) > 1 for scc in sccs)
    print()
    if has_cycle:
        print(
            f"  {RED}⚠  检测到循环依赖 — "
            f"该查询的超图不是无环的 (not α-acyclic){RESET}"
        )
    else:
        print(f"  {GREEN}✓  无循环依赖 — 超图是无环的{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  算法 2: Minimum Degree Ordering (树分解启发式)
# ═══════════════════════════════════════════════════════════════════════════════


class MinDegreeOrdering:
    """
    最小度数排序 (Minimum Degree Heuristic)

    在树分解中,消除排序的质量直接影响 tree-width。
    最小度数启发式每次消除当前度数最小的节点,
    并将其邻居两两连接 (fill-in),模拟高斯消元过程。
    """

    def __init__(self, vertices: list[str], edges: list[tuple[str, str]]):
        self.adj: dict[str, set[str]] = defaultdict(set)
        for u, v in edges:
            self.adj[u].add(v)
            self.adj[v].add(u)
        # 确保所有节点都在 adj 中 (即使是孤立节点)
        for v in vertices:
            if v not in self.adj:
                self.adj[v] = set()
        self.remaining = set(vertices)

    def run(self) -> list[str]:
        section("开始最小度数消除")
        ordering = []
        step_number = 0

        while self.remaining:
            step_number += 1

            # 找度数最小的节点 (平局时选字典序最小)
            min_deg = float("inf")
            chosen = None
            for v in sorted(self.remaining):
                deg = len(self.adj[v] & self.remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v

            # 当前节点在剩余图中的邻居
            neighbors = sorted(self.adj[chosen] & self.remaining - {chosen})

            step_print(
                f"\n{BOLD}步骤 {step_number}{RESET}: "
                f"选择节点 {MAGENTA}{chosen}{RESET}  "
                f"(度数={min_deg}, 邻居={neighbors})"
            )

            # 计算 fill-in 边: 邻居之间缺失的边
            fill_in_edges = []
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    u, v = neighbors[i], neighbors[j]
                    if v not in self.adj[u]:
                        fill_in_edges.append((u, v))
                        self.adj[u].add(v)
                        self.adj[v].add(u)

            if fill_in_edges:
                step_print(
                    f"  {YELLOW}Fill-in 边 (+{len(fill_in_edges)}): "
                    f"{fill_in_edges}{RESET}"
                )
            else:
                step_print(f"  {GREEN}无 fill-in 边{RESET}")

            # 从图中消除该节点
            self.remaining.remove(chosen)
            ordering.append(chosen)

            # 显示剩余图状态
            remaining_edges = []
            seen = set()
            for v in sorted(self.remaining):
                for w in sorted(self.adj[v] & self.remaining):
                    edge = (min(v, w), max(v, w))
                    if edge not in seen:
                        seen.add(edge)
                        remaining_edges.append(edge)
            step_print(f"  剩余节点: {sorted(self.remaining)}")
            step_print(f"  剩余边:   {remaining_edges}")

        return ordering


def demo_min_degree():
    """
    示例: 超图的 primal graph (原始图)

    查询: R(A,B,C) ⋈ S(B,C,D) ⋈ T(C,D,E) ⋈ U(A,E)
    原始图: 同一关系内的属性两两相连
    """
    header("算法 2: Minimum Degree Ordering (树分解)")

    print(f"  {BOLD}场景:{RESET} Join 查询超图的原始图 (primal graph)")
    print(f"  查询: R(A,B,C) ⋈ S(B,C,D) ⋈ T(C,D,E) ⋈ U(A,E)")
    print(f"  每个关系内的属性两两连接:\n")
    print(f"    R: A-B, A-C, B-C")
    print(f"    S: B-C, B-D, C-D")
    print(f"    T: C-D, C-E, D-E")
    print(f"    U: A-E\n")

    vertices = ["A", "B", "C", "D", "E"]
    edges = [
        # R(A,B,C)
        ("A", "B"), ("A", "C"), ("B", "C"),
        # S(B,C,D)
        ("B", "D"), ("C", "D"),  # B-C 已有
        # T(C,D,E)
        ("C", "E"), ("D", "E"),  # C-D 已有
        # U(A,E)
        ("A", "E"),
    ]

    # 去重
    unique_edges = list(set((min(u, v), max(u, v)) for u, v in edges))
    print(f"  去重后共 {len(unique_edges)} 条边: {sorted(unique_edges)}\n")

    mdo = MinDegreeOrdering(vertices, unique_edges)
    ordering = mdo.run()

    section("结果汇总")
    print(f"\n  消除排序 (elimination ordering): {ordering}")
    print(f"  树分解时从此排序构建消除树,tree-width 取决于消除过程中")
    print(f"  产生的最大团的大小 - 1。\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  算法 3: Chordal Graph Recognition (Maximum Cardinality Search)
# ═══════════════════════════════════════════════════════════════════════════════


class ChordalRecognition:
    """
    弦图识别 (Maximum Cardinality Search)

    在数据库理论中,一个超图是 α-acyclic 当且仅当其
    primal graph 在适当的超边约束下是弦图。
    MCS 产生一个排序,如果该排序是 perfect elimination ordering (PEO),
    则图是弦图。
    """

    def __init__(self, vertices: list[str], edges: list[tuple[str, str]]):
        self.vertices = set(vertices)
        self.adj: dict[str, set[str]] = defaultdict(set)
        for u, v in edges:
            self.adj[u].add(v)
            self.adj[v].add(u)
        for v in vertices:
            if v not in self.adj:
                self.adj[v] = set()

    def maximum_cardinality_search(self) -> list[str]:
        """执行 MCS,返回排序 (逆序为 PEO 候选)"""
        section("执行 Maximum Cardinality Search (MCS)")

        unnumbered = set(self.vertices)
        ordering = []       # 按选择顺序记录
        # weight[v] = v 有多少已编号的邻居
        weight = {v: 0 for v in self.vertices}

        for i in range(len(self.vertices), 0, -1):
            # 选择 weight 最大的未编号节点 (平局选字典序最小)
            chosen = max(
                sorted(unnumbered),
                key=lambda v: weight[v],
            )

            labeled_neighbors = [
                n for n in sorted(self.adj[chosen])
                if n not in unnumbered
            ]

            step_print(
                f"编号 σ({i}) = {MAGENTA}{chosen}{RESET}  "
                f"已选邻居数={BOLD}{weight[chosen]}{RESET}  "
                f"已选邻居={labeled_neighbors}"
            )

            ordering.append(chosen)
            unnumbered.remove(chosen)

            # 更新未编号邻居的 weight
            for w in self.adj[chosen]:
                if w in unnumbered:
                    weight[w] += 1

        return ordering

    def verify_peo(self, ordering: list[str]) -> bool:
        """
        验证 MCS 输出的排序是否为 Perfect Elimination Ordering.

        对于排序中的每个节点 v,检查:
        v 在排序中后面的邻居 (即"右邻居") 是否构成团。
        """
        section("验证 Perfect Elimination Ordering (PEO)")

        position = {v: i for i, v in enumerate(ordering)}
        is_peo = True

        for idx, v in enumerate(ordering):
            # v 的"右邻居": 排序中位于 v 之后的邻居
            right_neighbors = [
                w for w in self.adj[v]
                if position[w] > idx
            ]
            right_neighbors.sort(key=lambda w: position[w])

            if len(right_neighbors) <= 1:
                step_print(
                    f"节点 {MAGENTA}{v}{RESET}: "
                    f"右邻居={right_neighbors}  "
                    f"{GREEN}✓ (≤1个右邻居,自动满足){RESET}"
                )
                continue

            # 检查右邻居是否两两相连 (构成团)
            clique_ok = True
            missing = []
            for i in range(len(right_neighbors)):
                for j in range(i + 1, len(right_neighbors)):
                    u, w = right_neighbors[i], right_neighbors[j]
                    if w not in self.adj[u]:
                        clique_ok = False
                        missing.append((u, w))

            if clique_ok:
                step_print(
                    f"节点 {MAGENTA}{v}{RESET}: "
                    f"右邻居={right_neighbors}  "
                    f"{GREEN}✓ 构成团{RESET}"
                )
            else:
                step_print(
                    f"节点 {MAGENTA}{v}{RESET}: "
                    f"右邻居={right_neighbors}  "
                    f"{RED}✗ 不构成团! 缺少边: {missing}{RESET}"
                )
                is_peo = False

        return is_peo


def demo_chordal():
    """
    示例 1: 弦图 (对应 α-acyclic 的超图)
    示例 2: 非弦图 (含无弦环)
    """
    header("算法 3: Chordal Graph Recognition (MCS)")

    # ——— 示例 1: 弦图 ———
    print(f"  {BOLD}示例 A: 弦图{RESET}")
    print(f"  查询: R(A,B) ⋈ S(B,C) ⋈ T(A,B,C)")
    print(f"  原始图: A-B, B-C, A-C (三角形,天然弦图)\n")

    vertices_1 = ["A", "B", "C"]
    edges_1 = [("A", "B"), ("B", "C"), ("A", "C")]

    cr1 = ChordalRecognition(vertices_1, edges_1)
    ordering_1 = cr1.maximum_cardinality_search()
    is_chordal_1 = cr1.verify_peo(ordering_1)

    section("示例 A 结论")
    print(f"\n  MCS 排序: {ordering_1}")
    if is_chordal_1:
        print(f"  {GREEN}✓ 图是弦图 — PEO = {ordering_1}{RESET}")
        print(f"  {GREEN}  → 对应的超图是 α-acyclic{RESET}")
    else:
        print(f"  {RED}✗ 图不是弦图{RESET}")

    # ——— 示例 2: 非弦图 ———
    print(f"\n{'─' * 60}\n")
    print(f"  {BOLD}示例 B: 非弦图 (含4-环无弦){RESET}")
    print(f"  查询: R(A,B) ⋈ S(B,C) ⋈ T(C,D) ⋈ U(D,A)")
    print(f"  原始图: A-B-C-D-A (4-环,缺少对角线)\n")

    vertices_2 = ["A", "B", "C", "D"]
    edges_2 = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]

    cr2 = ChordalRecognition(vertices_2, edges_2)
    ordering_2 = cr2.maximum_cardinality_search()
    is_chordal_2 = cr2.verify_peo(ordering_2)

    section("示例 B 结论")
    print(f"\n  MCS 排序: {ordering_2}")
    if is_chordal_2:
        print(f"  {GREEN}✓ 图是弦图 — PEO = {ordering_2}{RESET}")
    else:
        print(f"  {RED}✗ 图不是弦图 — 存在无弦环{RESET}")
        print(f"  {RED}  → 对应的超图不是 α-acyclic{RESET}")

    # ——— 示例 3: 更复杂的弦图 ———
    print(f"\n{'─' * 60}\n")
    print(f"  {BOLD}示例 C: 较复杂的弦图{RESET}")
    print(f"  查询: R(A,B,C) ⋈ S(C,D,E) ⋈ T(A,C)")
    print(f"  原始图包含弦,所有 ≥4 的环都有弦\n")

    vertices_3 = ["A", "B", "C", "D", "E"]
    edges_3 = [
        ("A", "B"), ("A", "C"), ("B", "C"),   # R 的团
        ("C", "D"), ("C", "E"), ("D", "E"),   # S 的团
    ]

    cr3 = ChordalRecognition(vertices_3, edges_3)
    ordering_3 = cr3.maximum_cardinality_search()
    is_chordal_3 = cr3.verify_peo(ordering_3)

    section("示例 C 结论")
    print(f"\n  MCS 排序: {ordering_3}")
    if is_chordal_3:
        print(f"  {GREEN}✓ 图是弦图 — PEO = {ordering_3}{RESET}")
        print(f"  {GREEN}  → 对应的超图是 α-acyclic,可用 Yannakakis 算法高效求解{RESET}")
    else:
        print(f"  {RED}✗ 图不是弦图{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  综合分析 Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def demo_pipeline():
    """
    综合示例: 对同一个 join 查询依次运行三个算法,
    展示完整的超图分析流程。
    """
    header("综合分析 Pipeline: Join 查询超图分析")

    query = "R(A,B,C) ⋈ S(B,C,D) ⋈ T(C,D,E) ⋈ U(A,E)"
    print(f"  {BOLD}目标查询:{RESET} {query}\n")
    print(f"  分析流程:")
    print(f"    1) 构建属性依赖的有向图 → Tarjan SCC 检测循环")
    print(f"    2) 构建原始图 → Minimum Degree 求消除排序")
    print(f"    3) 构建原始图 → MCS 检测弦图性质\n")

    # --- Step 1: Tarjan ---
    print(f"\n{'━' * 60}")
    print(f"  {BOLD}Phase 1: 依赖图的强连通分量{RESET}")
    print(f"  (假设函数依赖: A→B, B→C, C→D, D→E, E→A)")

    dep_vertices = ["A", "B", "C", "D", "E"]
    dep_edges = {
        "A": ["B"],
        "B": ["C"],
        "C": ["D"],
        "D": ["E"],
        "E": ["A"],
    }

    tarjan = TarjanSCC(dep_vertices, dep_edges)
    sccs = tarjan.run()
    section("Phase 1 结论")
    for i, scc in enumerate(sccs):
        print(f"    SCC {i + 1}: {scc}")
    if any(len(s) > 1 for s in sccs):
        print(f"  {RED}  → 存在循环依赖{RESET}")

    # --- Step 2: Min Degree ---
    print(f"\n{'━' * 60}")
    print(f"  {BOLD}Phase 2: 原始图的最小度数消除{RESET}")

    primal_vertices = ["A", "B", "C", "D", "E"]
    primal_edges = [
        ("A", "B"), ("A", "C"), ("B", "C"),  # R
        ("B", "D"), ("C", "D"),               # S (B-C 已有)
        ("C", "E"), ("D", "E"),               # T (C-D 已有)
        ("A", "E"),                           # U
    ]
    unique = list(set((min(u, v), max(u, v)) for u, v in primal_edges))

    mdo = MinDegreeOrdering(primal_vertices, unique)
    elim_order = mdo.run()
    section("Phase 2 结论")
    print(f"    消除排序: {elim_order}")

    # --- Step 3: Chordal ---
    print(f"\n{'━' * 60}")
    print(f"  {BOLD}Phase 3: 弦图检测 (MCS){RESET}")

    cr = ChordalRecognition(primal_vertices, unique)
    mcs_order = cr.maximum_cardinality_search()
    is_chordal = cr.verify_peo(mcs_order)

    section("Phase 3 结论")
    print(f"    MCS 排序: {mcs_order}")
    if is_chordal:
        print(f"    {GREEN}✓ 弦图 — 查询超图 α-acyclic{RESET}")
    else:
        print(f"    {RED}✗ 非弦图 — 查询超图非 α-acyclic{RESET}")

    # --- 综合结论 ---
    section("综合分析结论")
    print(f"\n  查询: {query}")
    print(f"  1. 强连通分量: {len(sccs)} 个 SCC")
    print(f"  2. 消除排序:   {elim_order}")
    print(f"  3. 弦图性质:   {'是' if is_chordal else '否'}")
    acyclic = is_chordal
    if acyclic:
        print(f"\n  {GREEN}{BOLD}结论: 该查询可以通过 Yannakakis 算法在")
        print(f"  线性时间内完成 full reducer,无需笛卡尔积。{RESET}\n")
    else:
        print(f"\n  {YELLOW}{BOLD}结论: 该查询超图不满足 α-acyclicity,")
        print(f"  可能需要 GHD / fractional hypertree decomposition")
        print(f"  等更高级的分解方法。{RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  主程序入口
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    demo_tarjan()
    demo_min_degree()
    demo_chordal()
    demo_pipeline()
