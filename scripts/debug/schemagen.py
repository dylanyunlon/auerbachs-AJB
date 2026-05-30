#!/usr/bin/env python3
"""
schemagen.py — AJB-adapted schema generator for joinrenum

Origin: upstream/joinrenum/schemagen.py (22 lines)
Adaptation (~20%): AJB CLI args, configurable parameters, validation,
  and structured output summary.

Generates db/relations.txt, db/numlines.txt, db/filenames.txt for
joinrenum tests. These config files define the multi-way join query
structure that the Index / Enumerator / RRAccessTree operate on.

Usage:
  python3 schemagen.py
  python3 schemagen.py --relations 6 --variables 12 --factor 3 --outdir db/
"""

import argparse
import os
from random import randint, seed as set_seed

def gen_schema(relation_num, variable_num, factor, outdir, rand_seed=None):
    if rand_seed is not None:
        set_seed(rand_seed)

    os.makedirs(outdir, exist_ok=True)

    relations_path = os.path.join(outdir, "relations.txt")
    numlines_path  = os.path.join(outdir, "numlines.txt")
    filenames_path = os.path.join(outdir, "filenames.txt")

    schema = []  # for summary

    with open(relations_path, "w") as f:
        for i in range(relation_num):
            variables = [f"V{j+1}" for j in range(variable_num) if randint(1, factor) == 1]
            # AJB: ensure at least 1 variable per relation
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

    # AJB: structured summary
    print(f"[AJB] Schema generated in {outdir}/")
    print(f"  relations = {relation_num}")
    print(f"  variables = {variable_num}")
    print(f"  factor    = {factor} (1/{factor} probability per variable)")
    for name, vars in schema:
        print(f"  {name}({', '.join(vars)})  [{len(vars)} attrs]")

    # AJB: connectivity check
    all_vars = set()
    for _, vars in schema:
        all_vars.update(vars)
    print(f"  total unique variables = {len(all_vars)}")
    if len(all_vars) < 2:
        print("[AJB_WARN] Very few variables — join may be trivial")

    return schema

def main():
    parser = argparse.ArgumentParser(description="AJB schema generator")
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
