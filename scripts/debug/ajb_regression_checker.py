#!/usr/bin/env python3
"""
ajb_regression_checker.py — AJB regression testing framework (M1051-M1070)

BaselineCapture: compile and run test_*.cpp binaries, collect stderr rolling hash.
RegressionRunner: re-run and compare against baseline, report changed tests.
Algorithm: heapq for top-3 largest output changes, SequenceMatcher for trace diff rate.
"""

import os
import sys
import glob
import json
import hashlib
import heapq
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "joinrenum", "tests")
DEFAULT_BUILD_DIR = "/tmp/ajb_regression_build"
DEFAULT_BASELINE_PATH = "/tmp/ajb_regression_baseline.json"
CXX = os.environ.get("CXX", "g++")
CXXFLAGS = os.environ.get("CXXFLAGS", "-std=c++17 -O2 -w").split()
TIMEOUT_SEC = 60


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    stderr_hash: str
    stderr_text: str
    returncode: int
    elapsed_sec: float


@dataclass
class DiffRecord:
    name: str
    old_hash: str
    new_hash: str
    diff_ratio: float          # 1.0 = completely different
    stderr_old: str = ""
    stderr_new: str = ""


# ---------------------------------------------------------------------------
# BaselineCapture
# ---------------------------------------------------------------------------
class BaselineCapture:
    """Compile each test_*.cpp, run it, and store a rolling SHA-256 of stderr."""

    def __init__(self, src_dir: str = DEFAULT_SRC_DIR, build_dir: str = DEFAULT_BUILD_DIR):
        self.src_dir = os.path.abspath(src_dir)
        self.build_dir = os.path.abspath(build_dir)
        os.makedirs(self.build_dir, exist_ok=True)

    def _find_tests(self) -> List[str]:
        pattern = os.path.join(self.src_dir, "test_*.cpp")
        return sorted(glob.glob(pattern))

    def _compile(self, src: str) -> Optional[str]:
        stem = Path(src).stem
        out = os.path.join(self.build_dir, stem)
        cmd = [CXX] + CXXFLAGS + ["-o", out, src, "-lm"]
        ret = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_SEC)
        if ret.returncode != 0:
            sys.stderr.write(f"[WARN] compile failed: {stem}\n{ret.stderr.decode(errors='replace')[:500]}\n")
            return None
        return out

    @staticmethod
    def _rolling_hash(text: str) -> str:
        h = hashlib.sha256()
        for line in text.splitlines(keepends=True):
            h.update(line.encode("utf-8", errors="replace"))
        return h.hexdigest()

    def _run_one(self, binary: str) -> TestResult:
        name = Path(binary).stem
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [binary],
                capture_output=True,
                timeout=TIMEOUT_SEC,
            )
            elapsed = time.monotonic() - t0
            stderr_text = proc.stderr.decode("utf-8", errors="replace")
            return TestResult(
                name=name,
                stderr_hash=self._rolling_hash(stderr_text),
                stderr_text=stderr_text,
                returncode=proc.returncode,
                elapsed_sec=round(elapsed, 4),
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            return TestResult(
                name=name,
                stderr_hash="TIMEOUT",
                stderr_text="",
                returncode=-1,
                elapsed_sec=round(elapsed, 4),
            )

    def capture(self) -> Dict[str, TestResult]:
        results: Dict[str, TestResult] = {}
        sources = self._find_tests()
        if not sources:
            sys.stderr.write(f"[WARN] no test_*.cpp found in {self.src_dir}\n")
            return results
        for src in sources:
            binary = self._compile(src)
            if binary is None:
                continue
            result = self._run_one(binary)
            results[result.name] = result
            sys.stderr.write(f"  [baseline] {result.name}: hash={result.stderr_hash[:16]}… "
                             f"rc={result.returncode} t={result.elapsed_sec}s\n")
        return results

    @staticmethod
    def save(results: Dict[str, TestResult], path: str = DEFAULT_BASELINE_PATH):
        payload = {k: asdict(v) for k, v in results.items()}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        sys.stderr.write(f"[baseline] saved {len(results)} entries -> {path}\n")

    @staticmethod
    def load(path: str = DEFAULT_BASELINE_PATH) -> Dict[str, TestResult]:
        with open(path) as f:
            raw = json.load(f)
        return {k: TestResult(**v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# RegressionRunner
# ---------------------------------------------------------------------------
class RegressionRunner:
    """Re-run tests, compare against baseline, report regressions."""

    def __init__(self, baseline: Dict[str, TestResult],
                 src_dir: str = DEFAULT_SRC_DIR,
                 build_dir: str = DEFAULT_BUILD_DIR):
        self.baseline = baseline
        self.capturer = BaselineCapture(src_dir, build_dir)

    def run(self) -> List[DiffRecord]:
        current = self.capturer.capture()
        diffs: List[DiffRecord] = []
        all_names = sorted(set(self.baseline.keys()) | set(current.keys()))

        for name in all_names:
            old = self.baseline.get(name)
            new = current.get(name)

            if old is None:
                diffs.append(DiffRecord(
                    name=name, old_hash="<missing>",
                    new_hash=new.stderr_hash if new else "<missing>",
                    diff_ratio=1.0,
                    stderr_new=new.stderr_text if new else "",
                ))
                continue
            if new is None:
                diffs.append(DiffRecord(
                    name=name, old_hash=old.stderr_hash,
                    new_hash="<missing>", diff_ratio=1.0,
                    stderr_old=old.stderr_text,
                ))
                continue

            if old.stderr_hash == new.stderr_hash:
                continue

            ratio = 1.0 - SequenceMatcher(
                None, old.stderr_text, new.stderr_text
            ).ratio()

            diffs.append(DiffRecord(
                name=name,
                old_hash=old.stderr_hash,
                new_hash=new.stderr_hash,
                diff_ratio=round(ratio, 6),
                stderr_old=old.stderr_text,
                stderr_new=new.stderr_text,
            ))

        return diffs

    @staticmethod
    def top_k_changes(diffs: List[DiffRecord], k: int = 3) -> List[DiffRecord]:
        if len(diffs) <= k:
            return sorted(diffs, key=lambda d: d.diff_ratio, reverse=True)
        return heapq.nlargest(k, diffs, key=lambda d: d.diff_ratio)

    @staticmethod
    def print_report(diffs: List[DiffRecord], top_k: int = 3):
        if not diffs:
            print("[regression] all tests match baseline — no regressions detected.")
            return

        print(f"[regression] {len(diffs)} test(s) changed vs baseline:\n")
        for d in diffs:
            flag = "NEW" if d.old_hash == "<missing>" else (
                   "GONE" if d.new_hash == "<missing>" else "CHANGED")
            print(f"  [{flag}] {d.name}  diff_ratio={d.diff_ratio:.4f}")
            print(f"         old_hash={d.old_hash[:16]}…  new_hash={d.new_hash[:16]}…")

        top = RegressionRunner.top_k_changes(diffs, top_k)
        print(f"\n[regression] top-{len(top)} largest trace changes:")
        for i, d in enumerate(top, 1):
            print(f"  #{i}  {d.name}  diff_ratio={d.diff_ratio:.4f}")
            if d.stderr_old and d.stderr_new:
                sm = SequenceMatcher(None,
                                     d.stderr_old.splitlines(),
                                     d.stderr_new.splitlines())
                opcodes = sm.get_opcodes()
                changed_lines = sum(
                    max(j2 - j1, i2 - i1)
                    for tag, i1, i2, j1, j2 in opcodes
                    if tag != "equal"
                )
                total_lines = max(
                    len(d.stderr_old.splitlines()),
                    len(d.stderr_new.splitlines()),
                    1,
                )
                print(f"       line-level diff: {changed_lines}/{total_lines} lines changed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="AJB regression checker: capture baseline or detect regressions")
    parser.add_argument("action", choices=["baseline", "check"],
                        help="'baseline' to capture, 'check' to compare")
    parser.add_argument("--src-dir", default=DEFAULT_SRC_DIR,
                        help="directory containing test_*.cpp")
    parser.add_argument("--build-dir", default=DEFAULT_BUILD_DIR,
                        help="temp build directory for binaries")
    parser.add_argument("--baseline-file", default=DEFAULT_BASELINE_PATH,
                        help="path to baseline JSON file")
    parser.add_argument("--top-k", type=int, default=3,
                        help="number of top regressions to detail")
    args = parser.parse_args()

    if args.action == "baseline":
        cap = BaselineCapture(args.src_dir, args.build_dir)
        results = cap.capture()
        if not results:
            sys.exit("[ERROR] no tests compiled successfully")
        BaselineCapture.save(results, args.baseline_file)
        print(f"[baseline] captured {len(results)} test(s) -> {args.baseline_file}")

    elif args.action == "check":
        if not os.path.exists(args.baseline_file):
            sys.exit(f"[ERROR] baseline not found: {args.baseline_file}\n"
                     f"  run '{sys.argv[0]} baseline' first")
        baseline = BaselineCapture.load(args.baseline_file)
        runner = RegressionRunner(baseline, args.src_dir, args.build_dir)
        diffs = runner.run()
        runner.print_report(diffs, args.top_k)
        if diffs:
            sys.exit(1)


if __name__ == "__main__":
    main()
