#!/usr/bin/env python3
"""
parse_ajb_trace.py — AJB trace log parser and analyzer

Parses the structured [AJB_TIMER], [AJB_STATE], [AJB_RESULTS], [AJB_MEM]
output from AJB-instrumented binaries and produces a summary report.

Usage:
  ./ajb_benchmark 2>&1 | python3 parse_ajb_trace.py
  python3 parse_ajb_trace.py trace.log
  python3 parse_ajb_trace.py trace.log --json > summary.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict

def parse_trace(lines):
    """Parse AJB trace lines into structured sections."""
    timers = []
    states = []
    results = []
    memory = []
    errors = []
    warnings = []

    timer_re = re.compile(r'\[AJB_TIMER\]\s*<<<\s*(.+?)\s+([\d.]+)\s*s')
    mem_re = re.compile(r'\[AJB_MEM\]\s*(\S+)\s+.*?RSS=(\d+)\s*kB')
    state_re = re.compile(r'\[AJB_STATE\]\s*(.*)')
    result_re = re.compile(r'\[AJB_RESULTS?\]\s*(.*)')
    error_re = re.compile(r'\[AJB_ERROR\]\s*(.*)')
    warn_re = re.compile(r'\[AJB_WARN\]\s*(.*)')

    for line in lines:
        line = line.strip()

        m = timer_re.search(line)
        if m:
            timers.append({"phase": m.group(1), "seconds": float(m.group(2))})
            continue

        m = mem_re.search(line)
        if m:
            memory.append({"label": m.group(1), "rss_kb": int(m.group(2))})
            continue

        m = state_re.search(line)
        if m:
            states.append(m.group(1))
            continue

        m = result_re.search(line)
        if m:
            results.append(m.group(1))
            continue

        m = error_re.search(line)
        if m:
            errors.append(m.group(1))

        m = warn_re.search(line)
        if m:
            warnings.append(m.group(1))

    return {
        "timers": timers,
        "memory": memory,
        "states": states,
        "results": results,
        "errors": errors,
        "warnings": warnings,
    }

def print_summary(parsed):
    """Print human-readable summary."""
    print("=" * 60)
    print(" AJB Trace Summary")
    print("=" * 60)

    if parsed["timers"]:
        print("\nTiming breakdown:")
        total = 0
        for t in parsed["timers"]:
            pct = ""
            total += t["seconds"]
        for t in parsed["timers"]:
            pct = f" ({100*t['seconds']/total:.1f}%)" if total > 0 else ""
            print(f"  {t['phase']:40s}  {t['seconds']:10.6f} s{pct}")
        print(f"  {'TOTAL':40s}  {total:10.6f} s")

    if parsed["memory"]:
        print("\nMemory snapshots:")
        for m in parsed["memory"]:
            print(f"  {m['label']:20s}  RSS={m['rss_kb']} kB ({m['rss_kb']/1024:.1f} MB)")

    if parsed["results"]:
        print("\nResults:")
        for r in parsed["results"]:
            print(f"  {r}")

    if parsed["warnings"]:
        print(f"\nWarnings ({len(parsed['warnings'])}):")
        for w in parsed["warnings"]:
            print(f"  ⚠ {w}")

    if parsed["errors"]:
        print(f"\nErrors ({len(parsed['errors'])}):")
        for e in parsed["errors"]:
            print(f"  ✗ {e}")

    if not parsed["errors"]:
        print("\n✓ No errors detected")

def main():
    parser = argparse.ArgumentParser(description="AJB trace parser")
    parser.add_argument("file", nargs="?", default="-", help="Trace log file (- for stdin)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    if args.file == "-":
        lines = sys.stdin.readlines()
    else:
        with open(args.file) as f:
            lines = f.readlines()

    parsed = parse_trace(lines)

    if args.json:
        json.dump(parsed, sys.stdout, indent=2)
        print()
    else:
        print_summary(parsed)

if __name__ == "__main__":
    main()
