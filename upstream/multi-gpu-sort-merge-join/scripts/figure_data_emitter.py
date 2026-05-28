"""figure_data_emitter.py

Aggregate the multi-GPU sort-merge-join benchmark CSV output into the
per-method / per-seed / mean+/-std JSON schema used by the project's
reference figure-data files (see data.zip: gradient_norm_24k_data.json,
ppl_vs_time_1B_30k_data.json, ...).

This module is ADDITIVE. It does not modify the C++ benchmark, the
existing experiment_specifications.py, run_experiments.py, or
plot_experiments.py. It only *reads* the CSV those produce.

Ground-truth field mapping (verified by grep, not invented):

  X-axis candidates  -> columns emitted by join_benchmark.cu / the
                        experiment_specifications input_columns:
                          num_elements          (r_num_elements / s_num_elements)
                          num_threads
                          chunk_count           (chunk_size)
                          random_seed           (the seed we aggregate over)
  Series ("method")  -> join_algorithm (or sort_algorithm), the categorical
                        we draw one curve per, mirroring "methods" in the demo.
  Y-axis candidates  -> the benchmark's TimeDurations tags, emitted as CSV:
                          sort_phase, merge_phase, join_phase, total
                        plus num_matches (output cardinality) when present.

Provenance is recorded honestly: metadata.source is the CSV path + the
current git SHA, NOT a paper-figure PNG. These are measured numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import pandas


# Y metrics we know the benchmark can emit (TimeDurations tags + cardinality).
# We only emit those actually present as columns in the given CSV.
KNOWN_Y_METRICS: Sequence[str] = (
    "sort_phase",
    "merge_phase",
    "join_phase",
    "total",
    "total_cpu_merge_duration",
    "cpu_merge_duration",
    "num_matches",
)

# Columns that are never a Y metric / never a per-curve series.
SEED_COLUMN = "random_seed"


def _git_sha(repo_dir: pathlib.Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Population-consistent sample mean and std (ddof=1 when n>1).

    Matches the demo files, whose 'std' is per-x-point dispersion across
    seeds. With a single seed std is 0.0 (not NaN) so downstream plotting
    of error bands never breaks.
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = sum(values) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (mean, math.sqrt(var))


def aggregate(
    csv_path: pathlib.Path,
    x_column: str,
    series_column: str,
    y_metric: str,
    reported_final: Optional[Dict[str, float]] = None,
) -> dict:
    """Build the figure-data dict for one (x, series, y) choice.

    The resulting structure mirrors gradient_norm_24k_data.json:

        {
          "metadata": {...},
          "x_axis": "<x_column>",
          "steps": [sorted unique x values],
          "methods": {
             "<series value>": {
                 "seed_0": [...y over x...],
                 ...,
                 "mean": [...over x...],
                 "std":  [...over x...],
                 "reported_final": <float or null>
             }
          }
        }
    """
    df = pandas.read_csv(csv_path, header=0)

    for needed in (x_column, series_column, y_metric):
        if needed not in df.columns:
            raise KeyError(
                f"column {needed!r} not in CSV columns {list(df.columns)}; "
                f"emit it from the benchmark or pick another."
            )

    has_seed = SEED_COLUMN in df.columns

    # Common x grid across all series so curves align (demo files share one
    # 'steps' axis). Sorted unique x values.
    steps: List[float] = sorted(df[x_column].dropna().unique().tolist())

    methods: Dict[str, dict] = {}
    for series_value, sdf in df.groupby(series_column):
        seed_values = (
            sorted(sdf[SEED_COLUMN].dropna().unique().tolist()) if has_seed else [0]
        )

        # For each seed, a y-vector indexed by the common x grid.
        per_seed: Dict[str, List[Optional[float]]] = {}
        for i, seed in enumerate(seed_values):
            if has_seed:
                seed_df = sdf[sdf[SEED_COLUMN] == seed]
            else:
                seed_df = sdf
            # x -> y (mean within an (x,seed) cell if duplicated runs exist)
            lut = seed_df.groupby(x_column)[y_metric].mean().to_dict()
            per_seed[f"seed_{i}"] = [lut.get(x, None) for x in steps]

        # mean/std across seeds at each x point, skipping missing cells.
        mean_curve: List[Optional[float]] = []
        std_curve: List[Optional[float]] = []
        for xi in range(len(steps)):
            col = [
                per_seed[k][xi]
                for k in per_seed
                if per_seed[k][xi] is not None
            ]
            if col:
                m, s = _mean_std(col)
                mean_curve.append(m)
                std_curve.append(s)
            else:
                mean_curve.append(None)
                std_curve.append(None)

        entry = dict(per_seed)
        entry["mean"] = mean_curve
        entry["std"] = std_curve
        entry["reported_final"] = (
            (reported_final or {}).get(str(series_value), None)
        )
        methods[str(series_value)] = entry

    repo_dir = csv_path.resolve().parents[1]
    return {
        "metadata": {
            "panel": f"{y_metric} vs {x_column}",
            "source": f"{csv_path.name}@{_git_sha(repo_dir)}",
            "x_axis": x_column,
            "series": series_column,
            "y_metric": y_metric,
            "n_seeds": len(
                df[SEED_COLUMN].dropna().unique()
            ) if has_seed else 1,
            "n_methods": len(methods),
            "total_points": int(df.shape[0]),
        },
        "steps": steps,
        "methods": methods,
    }


def discover(csv_path: pathlib.Path) -> dict:
    """Report which X axes, series, and Y metrics are available in a CSV.

    Lets a caller (or a human) see the real options before emitting,
    instead of guessing column names.
    """
    df = pandas.read_csv(csv_path, header=0)
    cols = list(df.columns)
    y_present = [c for c in KNOWN_Y_METRICS if c in cols]
    numeric = df.select_dtypes("number").columns.tolist()
    x_candidates = [
        c for c in numeric if c not in y_present and c != SEED_COLUMN
    ]
    series_candidates = [c for c in cols if c not in numeric]
    return {
        "columns": cols,
        "x_candidates": x_candidates,
        "series_candidates": series_candidates,
        "y_metrics_present": y_present,
        "has_seed": SEED_COLUMN in cols,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate benchmark CSV into figure-data JSON "
        "(demo schema)."
    )
    p.add_argument("csv", type=pathlib.Path, help="benchmark output CSV")
    p.add_argument("--discover", action="store_true",
                   help="print available x/series/y columns and exit")
    p.add_argument("-x", "--x-column", default="num_elements")
    p.add_argument("-s", "--series-column", default="join_algorithm")
    p.add_argument("-y", "--y-metric", default="join_phase")
    p.add_argument("-o", "--output", type=pathlib.Path,
                   help="output JSON path (default: stdout)")
    args = p.parse_args()

    if args.discover:
        print(json.dumps(discover(args.csv), indent=2))
        return

    data = aggregate(
        args.csv, args.x_column, args.series_column, args.y_metric
    )
    text = json.dumps(data, indent=1)
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output} "
              f"({len(data['methods'])} methods, {len(data['steps'])} x-points)")
    else:
        print(text)


if __name__ == "__main__":
    main()
