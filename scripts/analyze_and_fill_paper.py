#!/usr/bin/env python3
"""
AJB实验数据分析 + LaTeX表格自动填充

用法: python3 scripts/analyze_and_fill_paper.py experiment_data/<timestamp>/summary.csv

流程:
  1. 读取summary.csv (ags1服务器自动产出)
  2. 计算Table 1 (ICL balance score) 和 Table 2 (wallclock time) 的数据
  3. 对比4种方法: AJB vs uniform-cadence vs pin-local vs eager
  4. 生成LaTeX表格片段, 可直接粘贴到paper/ajb_reconstructed.tex
  5. 检查AJB是否超越SOTA baseline, 打印诊断
"""
import sys, csv, os
from collections import defaultdict
import math

def welford_finalize(n, mean, M2):
    """Welford online variance: return (mean, stddev)"""
    if n < 2: return (mean, 0.0)
    return (mean, math.sqrt(M2 / (n - 1)))

def analyze_results(csv_path):
    """Parse summary.csv and compute paper table data"""
    if not os.path.exists(csv_path):
        print(f"[AJB_BP] File not found: {csv_path}")
        return

    # Parse CSV
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[AJB_STATE] Loaded {len(rows)} result rows from {csv_path}")

    # Group by (method, distribution, kx, input_size)
    grouped = defaultdict(list)
    for row in rows:
        method = row.get("method", "unknown")
        dist = row.get("distribution", "unknown")
        kx = row.get("kx", "0")
        n = row.get("input_size", "0")
        key = (method, dist, kx, n)
        grouped[key].append(row)

    # ---- Table 1: ICL Balance Score ----
    print("\n[AJB_STATE] === Table 1: ICL Balance Score (normalized, higher=better) ===")
    distributions = ["uniform", "zipfian", "foreign_key", "many_to_many"]
    methods = ["ajb", "uniform_cadence", "pin_local", "eager"]
    method_labels = {"ajb": "AJB", "uniform_cadence": "Uniform-cadence",
                     "pin_local": "Pin-local", "eager": "Eager repartition"}

    print(f"{'Method':<20} " + " ".join(f"{d:<14}" for d in distributions) + " Avg")
    print("-" * 90)
    for method in methods:
        scores = []
        for dist in distributions:
            # Find results for this method+dist (any kx/n)
            matching = [v for (m, d, _, _), v in grouped.items()
                       if m == method and d == dist]
            if matching:
                # Use balance_score from CSV if available
                vals = []
                for group in matching:
                    for row in group:
                        bs = row.get("balance_score") or row.get("load_balance_gini")
                        if bs:
                            try: vals.append(float(bs))
                            except: pass
                avg = sum(vals) / len(vals) if vals else 0.0
                scores.append(avg)
            else:
                scores.append(0.0)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"{method_labels.get(method, method):<20} " +
              " ".join(f"{s:<14.1f}" for s in scores) + f" {avg_score:.1f}")

    # ---- Table 2: Wallclock Time ----
    print("\n[AJB_STATE] === Table 2: End-to-end join time (seconds) ===")
    input_sizes = sorted(set(n for _, _, _, n in grouped.keys()))
    kx_values = sorted(set(kx for _, _, kx, _ in grouped.keys()))

    header = "Method"
    for n in input_sizes[:3]:
        for kx in kx_values[:2]:
            header += f"  {n}/{kx}"
    print(header)
    print("-" * 100)

    for method in methods:
        line = f"{method_labels.get(method, method):<20}"
        for n in input_sizes[:3]:
            for kx in kx_values[:2]:
                key = (method, "uniform", kx, n)
                if key in grouped:
                    times = []
                    for row in grouped[key]:
                        t = row.get("total_time_sec") or row.get("join_time")
                        if t:
                            try: times.append(float(t))
                            except: pass
                    if times:
                        mean = sum(times) / len(times)
                        stddev = math.sqrt(sum((x - mean)**2 for x in times) / max(len(times)-1, 1))
                        line += f"  {mean:.2f}±{stddev:.3f}"
                    else:
                        line += "  N/A"
                else:
                    line += "  N/A"
        print(line)

    # ---- AJB vs SOTA比较 ----
    print("\n[AJB_STATE] === AJB vs Baselines ===")
    for (method, dist, kx, n), rows_list in sorted(grouped.items()):
        if method != "ajb": continue
        # Find corresponding uniform-cadence
        uc_key = ("uniform_cadence", dist, kx, n)
        if uc_key in grouped:
            ajb_times = [float(r.get("total_time_sec", "0")) for r in rows_list if r.get("total_time_sec")]
            uc_times = [float(r.get("total_time_sec", "0")) for r in grouped[uc_key] if r.get("total_time_sec")]
            if ajb_times and uc_times:
                speedup = (sum(uc_times)/len(uc_times)) / (sum(ajb_times)/len(ajb_times))
                print(f"  {dist} kx={kx} n={n}: AJB {speedup:.2f}x vs uniform-cadence")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_and_fill_paper.py <summary.csv>")
        sys.exit(1)
    analyze_results(sys.argv[1])
