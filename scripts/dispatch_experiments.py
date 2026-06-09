#!/usr/bin/env python3
"""
dispatch_experiments.py — 实验数据 → 论文数据的自动化流水线

流程:
  1. 从 experiment_data/ 读取最新实验CSV
  2. 计算 Table 1 (ICL balance scores) 和 Table 2 (wallclock times)
  3. 计算 AJB vs baselines 的 speedup / volume reduction
  4. 输出替换指令: 哪些tex行需要更新为实测数据
  5. 生成子Claude的任务prompt

用法:
  python3 scripts/dispatch_experiments.py [results_dir]
  python3 scripts/dispatch_experiments.py experiment_data/20260609_143000
"""
import os, sys, glob, csv, json, statistics

def find_latest_results():
    """找到最新的实验结果目录"""
    dirs = sorted(glob.glob("experiment_data/2026*"))
    if dirs:
        return dirs[-1]
    # 回退到任何有summary.csv的目录
    for d in sorted(glob.glob("experiment_data/*/summary.csv")):
        return os.path.dirname(d)
    return None

def load_csvs(results_dir):
    """加载所有实验CSV, 返回按method×distribution×kx×input_size分组的数据"""
    data = {}
    for csvfile in sorted(glob.glob(f"{results_dir}/*.csv")):
        base = os.path.basename(csvfile)
        if base == "summary.csv":
            continue
        try:
            with open(csvfile) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    data[base] = rows
        except Exception as e:
            print(f"  skip {base}: {e}")
    return data

def extract_metrics(data):
    """从CSV行提取关键指标: wallclock, balance_score, bytes_transferred"""
    results = []
    for filename, rows in data.items():
        parts = filename.replace(".csv", "").split("_")
        method = parts[0]
        
        for row in rows:
            entry = {
                "method": method,
                "file": filename,
                "wallclock_s": float(row.get("total_time_s", row.get("wallclock_s", 0))),
                "balance_score": float(row.get("balance_score", row.get("partition_balance", 0))),
                "bytes_cross_tier": float(row.get("bytes_cross_tier", row.get("slow_tier_bytes", 0))),
                "output_cardinality": int(row.get("output_cardinality", row.get("num_matches", 0))),
                "distribution": row.get("distribution", "unknown"),
                "kx": int(row.get("kx", row.get("transfer_period", 0))),
                "input_size": int(row.get("input_size", row.get("num_elements", 0))),
            }
            results.append(entry)
    return results

def compute_paper_tables(results):
    """计算论文Table 1和Table 2的数据"""
    
    # ---- Table 1: ICL balance scores ----
    # 按method×distribution分组, 取balance_score均值
    table1 = {}
    for r in results:
        key = (r["method"], r["distribution"])
        if key not in table1:
            table1[key] = []
        table1[key].append(r["balance_score"])
    
    print("\n=== TABLE 1: Balance Scores (ICL) ===")
    print(f"{'Method':<20} {'Uniform':>10} {'Zipfian':>10} {'FK':>10} {'M2M':>10} {'Avg':>10}")
    print("-" * 70)
    
    methods_order = ["ajb", "uniform_cadence", "pin_local", "eager"]
    dists_order = ["uniform", "zipfian", "foreign_key", "many_to_many"]
    
    table1_tex = {}
    for method in methods_order:
        vals = []
        for dist in dists_order:
            scores = table1.get((method, dist), [0])
            mean_score = statistics.mean(scores) if scores else 0
            vals.append(mean_score)
        avg = statistics.mean(vals) if vals else 0
        vals.append(avg)
        table1_tex[method] = vals
        
        print(f"{method:<20} {vals[0]:>10.1f} {vals[1]:>10.1f} {vals[2]:>10.1f} {vals[3]:>10.1f} {vals[4]:>10.1f}")
    
    # ---- Table 2: Wallclock times ----
    # 按method×kx×input_size分组
    table2 = {}
    for r in results:
        key = (r["method"], r["kx"], r["input_size"])
        if key not in table2:
            table2[key] = []
        table2[key].append(r["wallclock_s"])
    
    print("\n=== TABLE 2: Wallclock Times (seconds) ===")
    sizes = [1000000000, 7000000000, 13000000000]
    kxs = [16, 256]
    
    print(f"{'Method':<20}", end="")
    for s in sizes:
        label = f"{s//1000000000}B"
        for kx in kxs:
            print(f"  K={kx:>3}", end="")
    print()
    print("-" * 90)
    
    table2_tex = {}
    for method in methods_order:
        print(f"{method:<20}", end="")
        row_vals = []
        for s in sizes:
            for kx in kxs:
                times = table2.get((method, kx, s), [0])
                mean_t = statistics.mean(times) if times else 0
                std_t = statistics.stdev(times) if len(times) > 1 else 0
                print(f"  {mean_t:>5.2f}±{std_t:.3f}", end="")
                row_vals.append((mean_t, std_t))
        table2_tex[method] = row_vals
        print()
    
    # ---- Speedup 计算 ----
    print("\n=== SPEEDUPS ===")
    
    # AJB vs eager (bytes reduction)
    ajb_bytes = [r["bytes_cross_tier"] for r in results if r["method"] == "ajb" and r["bytes_cross_tier"] > 0]
    eager_bytes = [r["bytes_cross_tier"] for r in results if r["method"] == "eager" and r["bytes_cross_tier"] > 0]
    uc_bytes = [r["bytes_cross_tier"] for r in results if r["method"] == "uniform_cadence" and r["bytes_cross_tier"] > 0]
    
    if ajb_bytes and eager_bytes:
        ratio_eager = statistics.mean(eager_bytes) / statistics.mean(ajb_bytes)
        print(f"  AJB vs Eager (byte volume): {ratio_eager:.1f}x reduction")
    
    if ajb_bytes and uc_bytes:
        ratio_uc = statistics.mean(uc_bytes) / statistics.mean(ajb_bytes)
        print(f"  AJB vs Uniform-cadence (byte volume): {ratio_uc:.1f}x reduction")
    
    # Wallclock speedups at 13B
    for kx in [16, 256]:
        ajb_times = table2.get(("ajb", kx, 13000000000), [])
        eager_times = table2.get(("eager", kx, 13000000000), [])
        uc_times = table2.get(("uniform_cadence", kx, 13000000000), [])
        
        if ajb_times and eager_times:
            speedup = statistics.mean(eager_times) / statistics.mean(ajb_times)
            print(f"  AJB vs Eager wallclock (13B, K={kx}): {speedup:.2f}x")
        if ajb_times and uc_times:
            speedup = statistics.mean(uc_times) / statistics.mean(ajb_times)
            print(f"  AJB vs UC wallclock (13B, K={kx}): {speedup:.2f}x")
    
    return table1_tex, table2_tex

def generate_tex_patch(table1, table2):
    """生成需要替换到tex中的具体数值"""
    print("\n=== TEX REPLACEMENT COMMANDS ===")
    print("以下数据需要替换到 paper/ajb_reconstructed.tex 中:")
    
    # Table 1 行
    method_labels = {
        "ajb": "AJB",
        "uniform_cadence": "Uniform-cadence",
        "pin_local": "Pin-local",
        "eager": "Eager repartition"
    }
    
    print("\n--- Table 1 (\\label{tab:icl}) ---")
    for method, vals in table1.items():
        label = method_labels.get(method, method)
        tex_row = f"{label:<20} & {vals[0]:.1f} & {vals[1]:.1f} & {vals[2]:.1f} & {vals[3]:.1f} & {vals[4]:.1f} \\\\\\"
        print(tex_row)
    
    print("\n--- Table 2 (\\label{tab:wallclock}) ---")
    for method, vals in table2.items():
        label = method_labels.get(method, method)
        parts = []
        for mean_t, std_t in vals:
            parts.append(f"${mean_t:.2f} \\pm {std_t:.3f}$")
        tex_row = f"{label} & " + " & ".join(parts) + " \\\\\\"
        print(tex_row)

def generate_subclaude_prompt(results_dir, table1, table2):
    """生成子Claude分析任务的prompt"""
    prompt = f"""你是AJB项目的第17位Claude。前一位Claude已经在ags1服务器上运行了GPU实验。

实验数据在: {results_dir}/

你的任务:
1. git pull 获取最新实验数据
2. 运行 python3 scripts/dispatch_experiments.py 查看汇总
3. 用实测数据替换 paper/ajb_reconstructed.tex 中Table 1和Table 2的占位数字
4. 更新Abstract和Conclusion中的speedup数字
5. git commit + push

关键数字需要替换:
- "170x fewer bytes" → 用实测 AJB vs eager 的比值
- "2x fewer than uniform-cadence" → 用实测比值  
- "1.3-2.1x end-to-end speedups" → 用实测13B wallclock比值
- Table 1 的 balance scores
- Table 2 的 wallclock times

注意:
- 不要改算法代码,只改tex中的数据
- 如果实测数据比占位数字差,如实报告
- git commit作者: dylanyunlon <dogechat@163.com>
- 直接push到main,不开分支
"""
    
    prompt_file = f"{results_dir}/subclaude_prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(prompt)
    print(f"\n[AJB] Sub-Claude prompt saved: {prompt_file}")

# ---- Main ----
if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_results()
    
    if not results_dir:
        print("[AJB] No experiment results found.")
        print("[AJB] Run experiments first: bash scripts/ags1_deploy_and_run.sh")
        sys.exit(1)
    
    print(f"[AJB] Analyzing: {results_dir}")
    
    data = load_csvs(results_dir)
    if not data:
        print(f"[AJB] No CSV files in {results_dir}")
        sys.exit(1)
    
    print(f"[AJB] Loaded {len(data)} CSV files")
    
    results = extract_metrics(data)
    table1, table2 = compute_paper_tables(results)
    generate_tex_patch(table1, table2)
    generate_subclaude_prompt(results_dir, table1, table2)
