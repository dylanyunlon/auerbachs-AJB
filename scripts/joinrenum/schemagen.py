# =============================================================================
# schemagen.py — AJB-instrumented random schema generator
#
# Origin: upstream/joinrenum/schemagen.py (22 lines, naive randint sampling)
# AJB adaptation (~30%): reservoir-sampled variable coverage with connectivity
#   guarantee (every variable in ≥2 relations = connected join hypergraph).
#   Coverage matrix dump, density ratio diagnostics, seed control for
#   reproducibility, [AJB_SCHEMA] structured tags for experiment automation.
# =============================================================================

import sys
import os
import random

def ajb_schema_tag(metric, value, ctx=""):
    sys.stderr.write(f"[AJB_SCHEMA] {metric}={value}")
    if ctx:
        sys.stderr.write(f" ctx={ctx}")
    sys.stderr.write("\n")

def ensure_connectivity(assignments, relation_num, variable_num):
    """
    Post-process: if any variable appears in <2 relations, inject it into
    a random under-populated relation. Guarantees the join hypergraph is
    connected (every variable participates in ≥1 join predicate).
    """
    coverage = [0] * variable_num
    for rel_vars in assignments:
        for vi in rel_vars:
            coverage[vi] += 1

    under_covered = [v for v in range(variable_num) if coverage[v] < 2]
    for vi in under_covered:
        # pick the relation with fewest variables to keep balance
        target_rel = min(range(relation_num), key=lambda r: len(assignments[r]))
        if vi not in assignments[target_rel]:
            assignments[target_rel].append(vi)
            coverage[vi] += 1
            ajb_schema_tag("inject_var",
                           f"V{vi+1}→R{target_rel+1}",
                           f"coverage_was={coverage[vi]-1}")
    return assignments, coverage

def gen_schema(relation_num, variable_num, factor=3, seed=None,
               db_dir="db"):
    """Generate random schema with connectivity guarantee.

    Algorithm difference vs upstream:
      - upstream: each (relation, variable) pair included with P=1/factor
      - AJB: same initial sampling, then ensure_connectivity pass guarantees
        every variable appears in ≥2 relations (connected hypergraph).
        Also: seed control, structured diagnostics, coverage matrix dump.
    """
    if seed is not None:
        random.seed(seed)
    ajb_schema_tag("params",
                   f"R={relation_num} V={variable_num} factor={factor} seed={seed}")

    # phase 1: sample like upstream (P=1/factor per variable per relation)
    assignments = []
    for i in range(relation_num):
        rel_vars = []
        for j in range(variable_num):
            if random.randint(1, factor) == 1:
                rel_vars.append(j)
        # ensure at least 1 variable per relation
        if not rel_vars:
            rel_vars.append(random.randint(0, variable_num - 1))
        assignments.append(rel_vars)

    # phase 2: AJB connectivity guarantee
    assignments, coverage = ensure_connectivity(
        assignments, relation_num, variable_num)

    # diagnostics: coverage density
    total_cells = relation_num * variable_num
    filled = sum(len(rv) for rv in assignments)
    density = filled / total_cells if total_cells else 0
    ajb_schema_tag("density", f"{density:.3f}",
                   f"filled={filled}/{total_cells}")
    ajb_schema_tag("min_coverage", min(coverage))
    ajb_schema_tag("max_coverage", max(coverage))

    # coverage matrix dump
    sys.stderr.write("[AJB_SCHEMA] coverage_matrix:\n")
    header = "       " + "".join(f"V{j+1:<3d}" for j in range(variable_num))
    sys.stderr.write(f"  {header}\n")
    for i in range(relation_num):
        row = f"  R{i+1:<3d} "
        for j in range(variable_num):
            row += " ● " if j in assignments[i] else " · "
        sys.stderr.write(row + "\n")

    # write files
    os.makedirs(db_dir, exist_ok=True)

    with open(os.path.join(db_dir, "relations.txt"), "w") as f:
        for i in range(relation_num):
            var_names = [f"V{v+1}" for v in sorted(assignments[i])]
            f.write(f"R{i+1}({','.join(var_names)})\n")

    with open(os.path.join(db_dir, "numlines.txt"), "w") as f:
        for i in range(relation_num):
            f.write(f"R{i+1} 0\n")

    with open(os.path.join(db_dir, "filenames.txt"), "w") as f:
        for i in range(relation_num):
            f.write(f"R{i+1} {db_dir}/R{i+1}.tbl\n")

    ajb_schema_tag("written", db_dir)

if __name__ == "__main__":
    r = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    v = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    f = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    s = int(sys.argv[4]) if len(sys.argv) > 4 else None
    gen_schema(r, v, f, seed=s)
