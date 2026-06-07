#!/usr/bin/env python3
"""
ajb_experiment_runner.py — M1031-M1040: full experiment pipeline for AJB

Scans src/joinrenum/tests/ for test_*.cpp, compiles and runs each,
collects [AJB_BP]/[AJB_STATE] output, aggregates traces via
parse_ajb_trace.py, checks diff rates against upstream, and produces
experiment_results_m1031.csv.

Algorithmic requirements:
  - Welford online aggregation for per-test timing statistics
  - Heap-based top-k selection (heapq) instead of sorted() for slowest tests
"""

import csv
import glob
import heapq
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TEST_DIR = os.path.join(PROJECT_ROOT, "src", "joinrenum", "tests")
UPSTREAM_DIR = os.path.join(PROJECT_ROOT, "upstream")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build_experiment_m1031")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "experiment_results_m1031.csv")
TRACE_SCRIPT = os.path.join(SCRIPT_DIR, "parse_ajb_trace.py")


# ---------------------------------------------------------------------------
# Welford accumulator — used for timing aggregation across test runs
# ---------------------------------------------------------------------------
class WelfordTiming:
    """Welford's online algorithm for incremental mean/variance.

    Each test may be compiled and run once, but if re-run or if multiple
    timer events exist within a single test, we aggregate them here in
    O(1) space per update. The parallel merge via Chan's formula lets us
    combine partial aggregations from different test categories.
    """
    __slots__ = ("n", "mean", "_m2", "_min", "_max")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")

    def update(self, x):
        """Incorporate one timing observation using Welford's recurrence."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    def merge(self, other):
        """Chan's parallel formula: combine two partial aggregations in O(1)."""
        if other.n == 0:
            return
        if self.n == 0:
            self.n, self.mean, self._m2 = other.n, other.mean, other._m2
            self._min, self._max = other._min, other._max
            return
        combined_n = self.n + other.n
        delta = other.mean - self.mean
        self._m2 += other._m2 + delta * delta * self.n * other.n / combined_n
        self.mean = (self.mean * self.n + other.mean * other.n) / combined_n
        self.n = combined_n
        self._min = min(self._min, other._min)
        self._max = max(self._max, other._max)

    @property
    def variance(self):
        return self._m2 / (self.n - 1) if self.n >= 2 else 0.0

    @property
    def stddev(self):
        return math.sqrt(self.variance)

    def snapshot(self):
        d = {"count": self.n, "mean": self.mean, "stddev": self.stddev}
        if self.n > 0:
            d["min"] = self._min
            d["max"] = self._max
        return d


def _heap_top_k(items, k, key_fn):
    """Select top-k largest items using a min-heap of size k.

    This is O(n log k) instead of O(n log n) from sorted(), which matters
    when the test suite grows large. We maintain a min-heap of k elements;
    each new candidate replaces the heap root only if it's larger.

    Args:
        items: iterable of elements
        k: number of top elements to return
        key_fn: callable returning a comparable value for each item
    Returns:
        list of top-k items sorted descending by key_fn
    """
    if k <= 0:
        return []
    heap = []  # min-heap of (key, index, item) tuples
    for idx, item in enumerate(items):
        entry = (key_fn(item), idx, item)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    # Extract and reverse so largest comes first
    result = []
    while heap:
        _, _, item = heapq.heappop(heap)
        result.append(item)
    result.reverse()
    return result


# ---------------------------------------------------------------------------
# ExperimentRunner — compile + run test_*.cpp, capture AJB output
# ---------------------------------------------------------------------------
class ExperimentRunner:
    """Scans test directory for test_*.cpp, compiles each with g++, runs
    the binary, and collects structured [AJB_BP]/[AJB_STATE] output lines."""

    COMPILE_CMD = (
        "g++ -O2 -std=c++17 -DAJB_DEBUG "
        "-I{include_dir} {src} -o {out} -lglpk 2>&1"
    )
    AJB_TAG_RE = re.compile(r"\[AJB_(BP|STATE|TIMER|TRACE|RESULTS?|MEM|WARN|ERROR)\]")

    def __init__(self, test_dir=TEST_DIR, build_dir=BUILD_DIR):
        self.test_dir = test_dir
        self.build_dir = build_dir
        self.include_dir = os.path.join(PROJECT_ROOT, "src", "joinrenum")
        os.makedirs(self.build_dir, exist_ok=True)
        self.results = []  # list of dicts per test
        self._timing_agg = WelfordTiming()  # global timing aggregator

    def discover_tests(self):
        """Find all test_*.cpp files in the test directory."""
        pattern = os.path.join(self.test_dir, "test_*.cpp")
        return sorted(glob.glob(pattern))

    def _compile(self, src_path, bin_name):
        """Compile a single test source. Returns (success, stderr_log)."""
        out_path = os.path.join(self.build_dir, bin_name)
        cmd = self.COMPILE_CMD.format(
            include_dir=self.include_dir, src=src_path, out=out_path
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return proc.returncode == 0, proc.stdout + proc.stderr, out_path

    def _run(self, bin_path, test_name):
        """Run a compiled test binary, return (exit_code, combined_output, elapsed_s)."""
        args = []
        if "unordered_map" in test_name:
            args = ["500000"]
        elif "index" in test_name and "full" not in test_name and "upstream" not in test_name:
            args = ["100000"]
        elif "run_bpt" in test_name:
            args = ["5"]

        cwd = os.path.join(PROJECT_ROOT, "src", "joinrenum")
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [bin_path] + args,
                capture_output=True, text=True, timeout=60, cwd=cwd
            )
            elapsed = time.monotonic() - t0
            combined = proc.stdout + proc.stderr
            return proc.returncode, combined, elapsed
        except subprocess.TimeoutExpired:
            return -1, "[AJB_ERROR] timeout after 60s", time.monotonic() - t0

    def _extract_ajb_lines(self, output):
        """Extract lines containing AJB structured tags, return counts by type."""
        counts = defaultdict(int)
        ajb_lines = []
        for line in output.splitlines():
            m = self.AJB_TAG_RE.search(line)
            if m:
                counts[m.group(1)] += 1
                ajb_lines.append(line)
        return dict(counts), ajb_lines

    def run_all(self):
        """Compile and run every discovered test, collecting results."""
        sources = self.discover_tests()
        print(f"[ExperimentRunner] discovered {len(sources)} test sources", file=sys.stderr)

        for src in sources:
            name = os.path.splitext(os.path.basename(src))[0]
            print(f"  [{name}] compiling...", end="", file=sys.stderr, flush=True)

            ok, compile_log, bin_path = self._compile(src, name)
            if not ok:
                print(f" SKIP (compile failed)", file=sys.stderr)
                self.results.append({
                    "test_name": name, "status": "COMPILE_FAIL",
                    "elapsed_s": 0, "ajb_counts": {}, "ajb_lines": [],
                    "output": compile_log,
                })
                continue

            print(f" running...", end="", file=sys.stderr, flush=True)
            exit_code, output, elapsed = self._run(bin_path, name)
            counts, ajb_lines = self._extract_ajb_lines(output)
            status = "PASS" if exit_code == 0 else "FAIL"

            # Welford-aggregate the wall-clock timing for this test
            self._timing_agg.update(elapsed)

            self.results.append({
                "test_name": name,
                "status": status,
                "elapsed_s": elapsed,
                "ajb_counts": counts,
                "ajb_lines": ajb_lines,
                "output": output,
            })
            bp = counts.get("BP", 0)
            st = counts.get("STATE", 0)
            print(f" {status} ({elapsed:.3f}s, BP={bp}, STATE={st})", file=sys.stderr)

        print(f"[ExperimentRunner] timing stats: {self._timing_agg.snapshot()}", file=sys.stderr)
        return self.results

    def top_k_slowest(self, k=5):
        """Use heap-based selection to find the k slowest tests."""
        return _heap_top_k(self.results, k, key_fn=lambda r: r["elapsed_s"])


# ---------------------------------------------------------------------------
# BenchmarkSuite — invoke parse_ajb_trace.py per test, aggregate to JSON
# ---------------------------------------------------------------------------
class BenchmarkSuite:
    """For each test result with AJB output, pipes through parse_ajb_trace.py
    in --json mode and aggregates all trace summaries into a single JSON."""

    def __init__(self, runner_results, trace_script=TRACE_SCRIPT):
        self.results = runner_results
        self.trace_script = trace_script
        self.aggregated = {}

    def analyze(self):
        """Run parse_ajb_trace.py --json on each test's output."""
        if not os.path.isfile(self.trace_script):
            print(f"[BenchmarkSuite] WARN: {self.trace_script} not found, skipping trace analysis",
                  file=sys.stderr)
            return self.aggregated

        for r in self.results:
            if r["status"] == "COMPILE_FAIL" or not r["ajb_lines"]:
                continue
            name = r["test_name"]
            try:
                proc = subprocess.run(
                    [sys.executable, self.trace_script, "--json"],
                    input=r["output"], capture_output=True, text=True, timeout=30
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    parsed = json.loads(proc.stdout)
                    self.aggregated[name] = {
                        "timer_count": len(parsed.get("timers_raw", [])),
                        "hot_phases": parsed.get("hot_phases", []),
                        "state_count": len(parsed.get("states", [])),
                        "bp_labels": list(parsed.get("breakpoints", {}).keys()),
                        "trace_events": len(parsed.get("trace_events", [])),
                        "errors": parsed.get("errors", []),
                    }
            except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
                print(f"  [{name}] trace analysis error: {exc}", file=sys.stderr)

        print(f"[BenchmarkSuite] analyzed {len(self.aggregated)}/{len(self.results)} tests",
              file=sys.stderr)
        return self.aggregated


# ---------------------------------------------------------------------------
# DiffRateChecker — compare upstream/ vs src/ same-name files
# ---------------------------------------------------------------------------
class DiffRateChecker:
    """For each file present in both upstream/ and src/, compute the
    diff rate (fraction of changed lines). Warn if any file's diff rate
    is below 20%, meaning the adaptation is minimal."""

    DIFF_THRESHOLD = 0.20  # warn below this

    def __init__(self, upstream_dir=UPSTREAM_DIR, src_dir=SRC_DIR):
        self.upstream_dir = upstream_dir
        self.src_dir = src_dir
        self.file_diffs = []

    def _find_common_files(self):
        """Walk upstream/ and find files with a same-name counterpart under src/."""
        pairs = []
        for root, _dirs, files in os.walk(self.upstream_dir):
            for fname in files:
                if not fname.endswith((".hpp", ".cpp", ".h", ".py")):
                    continue
                up_path = os.path.join(root, fname)
                # Search for same filename under src/
                for sroot, _sdirs, sfiles in os.walk(self.src_dir):
                    if fname in sfiles:
                        src_path = os.path.join(sroot, fname)
                        pairs.append((up_path, src_path, fname))
                        break
        return pairs

    @staticmethod
    def _diff_rate(path_a, path_b):
        """Compute diff rate as fraction of lines that differ.
        Uses unified diff line count: changed / max(total_a, total_b)."""
        try:
            with open(path_a) as fa:
                lines_a = fa.readlines()
            with open(path_b) as fb:
                lines_b = fb.readlines()
        except (OSError, UnicodeDecodeError):
            return None

        max_lines = max(len(lines_a), len(lines_b))
        if max_lines == 0:
            return 0.0

        # Count differing lines by simple alignment comparison
        common_len = min(len(lines_a), len(lines_b))
        changed = sum(1 for i in range(common_len) if lines_a[i] != lines_b[i])
        changed += abs(len(lines_a) - len(lines_b))
        return changed / max_lines

    def check(self):
        """Run diff-rate analysis on all common files. Return list of results."""
        pairs = self._find_common_files()
        warnings = []

        for up_path, src_path, fname in pairs:
            rate = self._diff_rate(up_path, src_path)
            if rate is None:
                continue
            entry = {
                "file": fname,
                "upstream": up_path,
                "src": src_path,
                "diff_rate": round(rate, 4),
                "warn": rate < self.DIFF_THRESHOLD,
            }
            self.file_diffs.append(entry)
            if entry["warn"]:
                warnings.append(entry)
                print(f"[DiffRateChecker] WARNING: {fname} diff_rate={rate:.1%} "
                      f"(below {self.DIFF_THRESHOLD:.0%} threshold)", file=sys.stderr)

        print(f"[DiffRateChecker] checked {len(self.file_diffs)} file pairs, "
              f"{len(warnings)} below threshold", file=sys.stderr)
        return self.file_diffs


# ---------------------------------------------------------------------------
# CSV writer — produce experiment_results_m1031.csv
# ---------------------------------------------------------------------------
def write_csv(runner, bench_data, diff_data, path=RESULTS_CSV):
    """Write final CSV combining test results, trace analysis, and diff info."""
    fieldnames = [
        "test_name", "status", "elapsed_s", "elapsed_mean", "elapsed_stddev",
        "bp_count", "state_count", "timer_count", "trace_events",
        "hot_phases", "errors", "diff_files_below_threshold",
    ]
    # Compute a Welford aggregate across all test timings for the summary row
    agg = WelfordTiming()
    for r in runner.results:
        agg.update(r["elapsed_s"])

    low_diff_files = [d["file"] for d in diff_data if d.get("warn")]

    rows = []
    for r in runner.results:
        name = r["test_name"]
        bench = bench_data.get(name, {})
        rows.append({
            "test_name": name,
            "status": r["status"],
            "elapsed_s": f"{r['elapsed_s']:.4f}",
            "elapsed_mean": f"{agg.mean:.4f}",
            "elapsed_stddev": f"{agg.stddev:.4f}",
            "bp_count": r["ajb_counts"].get("BP", 0),
            "state_count": r["ajb_counts"].get("STATE", 0),
            "timer_count": bench.get("timer_count", 0),
            "trace_events": bench.get("trace_events", 0),
            "hot_phases": ";".join(bench.get("hot_phases", [])),
            "errors": ";".join(bench.get("errors", [])),
            "diff_files_below_threshold": ";".join(low_diff_files) if rows == [] else "",
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[CSV] wrote {len(rows)} rows to {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 64, file=sys.stderr)
    print(" AJB Experiment Runner — M1031-M1040", file=sys.stderr)
    print(f" Project root: {PROJECT_ROOT}", file=sys.stderr)
    print("=" * 64, file=sys.stderr)

    # Phase 1: compile and run all tests
    runner = ExperimentRunner()
    runner.run_all()

    # Phase 2: trace analysis via parse_ajb_trace.py
    suite = BenchmarkSuite(runner.results)
    bench_data = suite.analyze()

    # Phase 3: diff-rate check upstream vs src
    checker = DiffRateChecker()
    diff_data = checker.check()

    # Phase 4: write CSV
    write_csv(runner, bench_data, diff_data)

    # Phase 5: report top-k slowest tests (heap-based)
    top_k = runner.top_k_slowest(k=5)
    print(f"\n{'=' * 64}", file=sys.stderr)
    print(f" Top-{len(top_k)} slowest tests (heap-selected):", file=sys.stderr)
    for i, r in enumerate(top_k, 1):
        print(f"  {i}. {r['test_name']:40s} {r['elapsed_s']:.4f}s  [{r['status']}]",
              file=sys.stderr)

    # Summary
    passed = sum(1 for r in runner.results if r["status"] == "PASS")
    failed = sum(1 for r in runner.results if r["status"] == "FAIL")
    skipped = sum(1 for r in runner.results if r["status"] == "COMPILE_FAIL")
    total_ajb = sum(sum(r["ajb_counts"].values()) for r in runner.results)
    timing = runner._timing_agg.snapshot()

    print(f"\n{'=' * 64}")
    print(f" M1031-M1040 Experiment Results")
    print(f"  Tests: {passed} PASS, {failed} FAIL, {skipped} SKIP")
    print(f"  AJB events collected: {total_ajb}")
    print(f"  Welford timing: mean={timing['mean']:.4f}s stddev={timing['stddev']:.4f}s")
    print(f"  Trace analyses: {len(bench_data)}")
    print(f"  Diff-rate files checked: {len(diff_data)}")
    low = sum(1 for d in diff_data if d.get("warn"))
    if low:
        print(f"  WARNING: {low} files below {DiffRateChecker.DIFF_THRESHOLD:.0%} diff threshold")
    print(f"  CSV output: {RESULTS_CSV}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
