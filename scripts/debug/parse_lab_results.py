#!/usr/bin/env python3
"""
parse_lab_results.py — Parse experiment_data/ logs into paper-ready tables

Sub-Claudes pull the repo, run this script, and get structured data
for filling into ajb_reconstructed.tex.

Usage:
  python3 scripts/debug/parse_lab_results.py [experiment_data/results/]
"""
import os, sys, re, csv, json
from pathlib import Path
from collections import defaultdict

def parse_ajb_trace(filepath):
    """Extract [AJB_*] tagged lines from a log file."""
    tags = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            m = re.match(r'\[AJB_(\w+)\]\s*(.*)', line.strip())
            if m:
                tags[m.group(1)].append(m.group(2))
    return dict(tags)

def parse_timer_lines(lines):
    """Extract timing data from [AJB_TIMER] lines."""
    timers = {}
    for line in lines:
        m = re.match(r'(\w+)\s+(\d+\.?\d*)\s*(?:ms|s)', line)
        if m:
            timers[m.group(1)] = float(m.group(2))
        m2 = re.match(r'<<<\s+(\w+)\s+(\d+\.?\d*)\s+s', line)
        if m2:
            timers[m2.group(1)] = float(m2.group(2)) * 1000  # to ms
    return timers

def parse_results_csv(filepath):
    """Parse a CSV results file."""
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def summarize_rq_data(result_dir):
    """Build summary tables for each RQ."""
    result_dir = Path(result_dir)
    summary = {}

    # Joinrenum CPU results
    for f in sorted(result_dir.glob("joinrenum_cpu_*.csv")):
        rows = parse_results_csv(f)
        summary['joinrenum_cpu'] = rows
        print(f"\n=== Joinrenum CPU ({f.name}) ===")
        for r in rows:
            print(f"  {r['test_name']:30s} {r['status']:12s} {r.get('wall_time_ms','?'):>8s}ms")

    # RQ results from trace logs
    for rq in ['rq1_drift', 'rq2_cadence', 'rq3_volume', 'rq4_scale', 'rq5_skew', 'rq6_renum']:
        files = sorted(result_dir.glob(f"{rq}_*.csv"))
        if files:
            print(f"\n=== {rq.upper()} ({files[-1].name}) ===")
            with open(files[-1]) as f:
                print(f.read()[:2000])

    # Parse individual test logs for detailed AJB trace data
    log_dir = result_dir.parent / "logs"
    if log_dir.exists():
        for f in sorted(log_dir.glob("test_*_*.txt"))[:5]:
            tags = parse_ajb_trace(f)
            if tags:
                print(f"\n=== {f.stem} AJB tags ===")
                for tag, lines in tags.items():
                    print(f"  [{tag}]: {len(lines)} entries")
                    for l in lines[:3]:
                        print(f"    {l[:120]}")

    return summary

if __name__ == "__main__":
    result_dir = sys.argv[1] if len(sys.argv) > 1 else "experiment_data/results"
    if not os.path.isdir(result_dir):
        print(f"No results directory found: {result_dir}")
        print("Run lab_experiment_runner.sh on the lab machine first, then git pull.")
        sys.exit(1)
    summarize_rq_data(result_dir)
