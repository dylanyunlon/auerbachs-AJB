#!/usr/bin/env python3
"""
analyze_and_fill_paper.py — Convert experiment CSVs to paper table values.

Reads experiment_data/results/sort_comparison_*.csv and join_comparison_*.csv,
computes AJB vs upstream speedups, and outputs LaTeX table snippets + summary.

Usage:
  python3 scripts/analyze_and_fill_paper.py [experiment_data/results/]
"""
import csv, glob, os, sys
from collections import defaultdict

def load_csvs(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                row['_source'] = os.path.basename(f)
                rows.append(row)
    return rows

def analyze_sort(data_dir):
    rows = load_csvs(os.path.join(data_dir, 'sort_comparison_*.csv'))
    if not rows:
        print("[WARN] No sort_comparison CSVs found"); return {}
    groups = defaultdict(dict)
    for r in rows:
        key = (r.get('gpu_name',''), r.get('num_elements',''), r.get('distribution',''))
        method = r.get('method','')
        try: ms = float(r.get('sort_ms', 0))
        except: ms = 0
        if ms > 0: groups[key][method] = ms

    results = {}
    for key, methods in groups.items():
        ajb_ms = methods.get('ajb', 0)
        up_ms = methods.get('upstream', 0)
        if ajb_ms > 0 and up_ms > 0:
            results[key] = {
                'ajb_ms': ajb_ms, 'upstream_ms': up_ms,
                'speedup': up_ms / ajb_ms,
                'gpu': key[0], 'n': key[1], 'dist': key[2],
            }
    return results

def analyze_join(data_dir):
    rows = load_csvs(os.path.join(data_dir, 'join_comparison_*.csv'))
    if not rows:
        print("[WARN] No join_comparison CSVs found"); return {}
    groups = defaultdict(dict)
    for r in rows:
        key = (r.get('gpu_config',''), r.get('r_elements',''),
               r.get('s_elements',''), r.get('distribution',''))
        method = r.get('method','')
        try: ms = float(r.get('join_ms', 0))
        except: ms = 0
        if ms > 0: groups[key][method] = ms

    results = {}
    for key, methods in groups.items():
        ajb_ms = methods.get('ajb', 0)
        up_ms = methods.get('upstream', 0)
        if ajb_ms > 0 and up_ms > 0:
            results[key] = {
                'ajb_ms': ajb_ms, 'upstream_ms': up_ms,
                'speedup': up_ms / ajb_ms,
                'gpus': key[0], 'r': key[1], 's': key[2], 'dist': key[3],
            }
    return results

def generate_latex_table1(sort_results):
    if not sort_results: return "% No sort data yet\n"
    lines = ["% Table 1: Sort Performance (AJB vs Upstream)",
             "\\begin{tabular}{llrrr}", "\\toprule",
             "GPU & Distribution & Upstream (ms) & AJB (ms) & Speedup \\\\",
             "\\midrule"]
    for key in sorted(sort_results.keys()):
        r = sort_results[key]
        lines.append(f"  {r['gpu']} & {r['dist']} & {r['upstream_ms']:.1f} & "
                     f"{r['ajb_ms']:.1f} & {r['speedup']:.2f}$\\times$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)

def generate_latex_table2(join_results):
    if not join_results: return "% No join data yet\n"
    lines = ["% Table 2: Join Performance (AJB vs Upstream)",
             "\\begin{tabular}{llrrr}", "\\toprule",
             "GPUs & |R|:|S| & Upstream (ms) & AJB (ms) & Speedup \\\\",
             "\\midrule"]
    for key in sorted(join_results.keys()):
        r = join_results[key]
        lines.append(f"  {r['gpus']} & {r['r']}:{r['s']} & {r['upstream_ms']:.1f} & "
                     f"{r['ajb_ms']:.1f} & {r['speedup']:.2f}$\\times$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'experiment_data/results'
    print(f"[analyze] Loading from {data_dir}/")
    sort_results = analyze_sort(data_dir)
    join_results = analyze_join(data_dir)
    print(f"[analyze] Sort: {len(sort_results)} entries, Join: {len(join_results)} entries")

    table1 = generate_latex_table1(sort_results)
    table2 = generate_latex_table2(join_results)
    print("\n=== Table 1 (Sort) ===\n" + table1)
    print("\n=== Table 2 (Join) ===\n" + table2)

    # Summary stats
    sort_sp = [r['speedup'] for r in sort_results.values()]
    join_sp = [r['speedup'] for r in join_results.values()]
    os.makedirs('experiment_data/paper', exist_ok=True)
    with open('experiment_data/paper/table1_sort.tex', 'w') as f: f.write(table1)
    with open('experiment_data/paper/table2_join.tex', 'w') as f: f.write(table2)
    with open('experiment_data/paper/summary.txt', 'w') as f:
        if sort_sp:
            f.write(f"sort_min_speedup: {min(sort_sp):.4f}\n")
            f.write(f"sort_max_speedup: {max(sort_sp):.4f}\n")
            f.write(f"sort_mean_speedup: {sum(sort_sp)/len(sort_sp):.4f}\n")
        if join_sp:
            f.write(f"join_min_speedup: {min(join_sp):.4f}\n")
            f.write(f"join_max_speedup: {max(join_sp):.4f}\n")
            f.write(f"join_mean_speedup: {sum(join_sp)/len(join_sp):.4f}\n")
    print("[analyze] Written to experiment_data/paper/")

if __name__ == '__main__':
    main()
