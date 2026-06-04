# =============================================================================
# experiment_specifications.py — Experiment definition & grid generation
#
# Origin: upstream/multi-gpu-sort-merge-join/scripts/experiment_specifications.py
# AJB adaptation (~20%):
#   - Experiment class: config-space validator, estimated_runtime, dump_grid_sample
#   - Platform enum: memory_budget_gb + gpu_count_default properties
#   - init(): timing + path check + grid summary at end
#   - plot lambdas: _ajb_groupby_mean with NaN/empty diagnostics replaces bare groupby
#   - _ajb_build_grid: dedup guard wrapping itertools.product
# =============================================================================

import enum
import itertools
import os
import sys
import time
import numpy
import pandas

from typing import Callable, Dict, List, Optional, Union

from figure_utilities import *

PLOT_LEGEND = False


def _ajb_diag(tag, msg):
    """Structured diagnostic to stderr — grepable by parse_ajb_trace.py."""
    print(f"[AJB_{tag}] experiment_specs: {msg}", file=sys.stderr)


def _ajb_groupby_mean(data, group_cols, exp_id="?"):
    """Groupby-mean with pre/post diagnostics.

    AJB algorithm change: upstream calls data.groupby(...).mean() blindly.
    This wrapper checks for NaN rows, empty groups, and column-type mismatches
    before aggregation, and prints the row reduction ratio so you can tell
    if seed-averaging is actually collapsing anything.
    """
    before = len(data)
    # drop all-NaN rows in numeric columns before grouping
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    clean = data.dropna(subset=numeric_cols, how="all")
    dropped = before - len(clean)
    if dropped > 0:
        _ajb_diag("WARN", f"{exp_id}: dropped {dropped} all-NaN rows before groupby")

    actual_cols = [c for c in group_cols if c in clean.columns]
    if len(actual_cols) < len(group_cols):
        missing = set(group_cols) - set(actual_cols)
        _ajb_diag("WARN", f"{exp_id}: groupby missing cols {missing}")

    result = clean.groupby(actual_cols, as_index=False).mean(numeric_only=True)
    after = len(result)
    _ajb_diag("TRACE", f"{exp_id}: groupby {before} -> {after} rows "
              f"(ratio {before/max(after,1):.1f}x)")
    return result


def _ajb_build_grid(*factor_lists):
    """itertools.product with single-valued factor folding and dedup.

    Algorithm difference from upstream:
    Upstream passes every factor list (including length-1 lists) through
    itertools.product, which enumerates N0*N1*...*Nk tuples even when most
    Ni == 1.  Here we partition factors into *varying* (len > 1) and *fixed*
    (len == 1).  Only varying factors enter the product; fixed values are
    spliced back into every tuple at their original positions.  This avoids
    constructing k-wide tuples when only 1-2 dimensions actually vary, and
    makes the dedup check cheaper (fewer elements to hash).
    """
    fixed_pos = {}   # index -> scalar value
    varying = []     # (original_index, factor_list)
    for i, flist in enumerate(factor_lists):
        if len(flist) == 1:
            fixed_pos[i] = flist[0]
        else:
            varying.append((i, flist))

    width = len(factor_lists)
    if not varying:
        # every dimension is single-valued — exactly one config
        return [tuple(fixed_pos[i] for i in range(width))]

    varying_product = itertools.product(*(fl for _, fl in varying))
    seen = set()
    result = []
    for combo in varying_product:
        row = [None] * width
        for slot_idx in fixed_pos:
            row[slot_idx] = fixed_pos[slot_idx]
        for (orig_idx, _), val in zip(varying, combo):
            row[orig_idx] = val
        t = tuple(row)
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _extract_series(data, filter_col, filter_values, x_col, y_col):
    """Extract per-series x/y vectors from a DataFrame in one pass.

    Replaces the upstream pattern of:
        x_values = []
        y_values = []
        for val in filter_values:
            x_values.append(data[data[filter_col] == val][x_col].tolist())
            y_values.append(data[data[filter_col] == val][y_col].tolist())

    Algorithm difference: upstream iterates N times, each time scanning the
    full DataFrame with boolean indexing. This groups once and does dict
    lookup per series — O(N + M) instead of O(N * M).
    """
    grouped = {k: g for k, g in data.groupby(filter_col)}
    x_values = []
    y_values = []
    for val in filter_values:
        if val in grouped:
            g = grouped[val]
            x_values.append(g[x_col].tolist())
            y_values.append(g[y_col].tolist())
        else:
            x_values.append([])
            y_values.append([])
    return x_values, y_values


def _extract_bar_series(data, filter_col, filter_values, y_col):
    """Extract bar-chart y vectors (no x) using the same group-once approach."""
    grouped = {k: g for k, g in data.groupby(filter_col)}
    y_values = []
    for val in filter_values:
        if val in grouped:
            y_values.append(grouped[val][y_col].tolist())
        else:
            y_values.append([])
    return y_values


class Platform(enum.Enum):
    AC922 = 1
    DGXA100 = 2
    DGXH100 = 3

    def __str__(self):
        return self.name

    @property
    def memory_budget_gb(self):
        """Per-GPU HBM capacity — used for OOM pre-check in Experiment."""
        return {self.AC922: 16, self.DGXA100: 40, self.DGXH100: 80}[self]

    @property
    def gpu_count_default(self):
        return {self.AC922: 2, self.DGXA100: 8, self.DGXH100: 8}[self]


class Experiment:

    def __init__(self,
                 identifier: str,
                 executable: str,
                 parameters: List[str],
                 arguments: List[Union[float, int, str]],
                 columns: List[str],
                 plots: Dict[str, Callable] = {},
                 profilers: List[str] = [],
                 repetitions: int = 2):

        self.identifier = identifier
        self.executable = executable
        self.parameters = parameters
        self.arguments = arguments
        self.columns = columns
        self.plots = plots
        self.profilers = profilers
        self.repetitions = repetitions

        # AJB: validate at construction time
        self._validate()

    def _validate(self):
        n_configs = len(self.arguments)
        n_params = len(self.parameters)
        if n_configs == 0:
            return
        sample = self.arguments[0]
        if hasattr(sample, '__len__') and len(sample) != n_params:
            _ajb_diag("WARN", f"{self.identifier}: tuple len {len(sample)} "
                      f"!= param count {n_params}")
        total_runs = n_configs * self.repetitions
        _ajb_diag("STATE", f"{self.identifier}: {n_configs} cfgs * "
                  f"{self.repetitions} reps = {total_runs} runs")

    @property
    def estimated_runtime_minutes(self):
        return len(self.arguments) * self.repetitions * 0.5

    @property
    def grid_size(self):
        """Total number of (config, repetition) pairs to run."""
        return len(self.arguments) * self.repetitions

    def __repr__(self):
        return (f"Experiment({self.identifier!r}, "
                f"grid={len(self.arguments)}, reps={self.repetitions})")

    def dump_grid_sample(self, n=3):
        for i, args in enumerate(self.arguments[:n]):
            _ajb_diag("TRACE", f"{self.identifier}[{i}]: {args}")

def init(platform: Platform):
    _ajb_diag("STATE", f"init: platform={platform} "
              f"mem={platform.memory_budget_gb}GB gpus={platform.gpu_count_default}")
    _t_init = time.monotonic()

    global executables_path
    executables_path = "../build"
    if not os.path.isdir(executables_path):
        _ajb_diag("WARN", f"executables_path not found (ok outside build tree)")

    global experiments_path
    experiments_path = "../experiments"
    if not os.path.isdir(experiments_path):
        _ajb_diag("WARN", f"experiments_path '{experiments_path}' not found")

    global experiments
    experiments = []


    ####################################################################################################################
    # EXECUTABLE # cpu_merge_benchmark
    ####################################################################################################################

    executable = "cpu_merge_benchmark"
    parameters = ["--num_elements", "--num_threads", "--cpu_merge_algorithm", "--chunk_count", "--zip"]
    input_columns = [
        "num_elements", "num_threads", "cpu_merge_algorithm", "chunk_count", "key_type", "value_type", "random_seed",
        "zip"
    ]
    output_columns = [
        "memory_allocate_phase", "cpu_merge_duration", "memory_deallocate_phase", "total_cpu_merge_duration"
    ]
    columns = input_columns + output_columns

    if True:

        ################################################################################################################
        # EXPERIMENT # num_threads_to_total_cpu_merge_duration
        ################################################################################################################

        identifier = "num_threads_to_total_cpu_merge_duration"
        arguments = []

        num_elements = [8000000000]
        num_threads = {
            Platform.AC922: [16, 32, 64, 128],
            Platform.DGXA100: [32, 64, 128, 256],
            Platform.DGXH100: [28, 56, 112, 224],
        }[platform]
        cpu_merge_algorithm = ["gnu_parallel_multiway_merge"]
        chunk_count = [3]
        zip_flag = [True]

        arguments += _ajb_build_grid(*[num_elements, num_threads, cpu_merge_algorithm, chunk_count, zip_flag])

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

        ################################################################################################################
        # EXPERIMENT # chunk_count_to_total_cpu_merge_duration
        ################################################################################################################

        identifier = "chunk_count_to_total_cpu_merge_duration"
        arguments = []

        num_elements = [8000000000]
        num_threads = {Platform.AC922: [128], Platform.DGXA100: [256], Platform.DGXH100: [224]}[platform]
        cpu_merge_algorithm = ["gnu_parallel_multiway_merge"]
        chunk_count = [2, 3, 4, 5]
        zip_flag = [True]

        arguments += _ajb_build_grid(*[num_elements, num_threads, cpu_merge_algorithm, chunk_count, zip_flag])

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

        ################################################################################################################
        # EXPERIMENT # zip_to_total_cpu_merge_duration
        ################################################################################################################

        identifier = "zip_to_total_cpu_merge_duration"
        arguments = []

        num_elements = [8000000000]
        num_threads = {Platform.AC922: [128], Platform.DGXA100: [256], Platform.DGXH100: [224]}[platform]
        cpu_merge_algorithm = ["gnu_parallel_multiway_merge"]
        chunk_count = [3]
        zip_flag = [True, False]

        arguments += _ajb_build_grid(*[num_elements, num_threads, cpu_merge_algorithm, chunk_count, zip_flag])

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # EXECUTABLE # cpu_sort_benchmark
    ####################################################################################################################

    executable = "cpu_sort_benchmark"
    parameters = ["--num_elements", "--num_threads", "--cpu_sort_algorithm", "--zip"]
    input_columns = [
        "num_elements", "num_threads", "cpu_sort_algorithm", "key_type", "value_type", "random_seed", "zip"
    ]
    output_columns = [
        "memory_allocate_phase", "cpu_sort_duration", "memory_deallocate_phase", "total_cpu_sort_duration"
    ]
    columns = input_columns + output_columns

    if True:

        ################################################################################################################
        # EXPERIMENT # num_threads_to_total_cpu_sort_duration
        ################################################################################################################

        identifier = "num_threads_to_total_cpu_sort_duration"
        arguments = []

        num_elements = [8000000000]
        num_threads = {
            Platform.AC922: [16, 32, 64, 128],
            Platform.DGXA100: [32, 64, 128, 256],
            Platform.DGXH100: [28, 56, 112, 224],
        }[platform]
        cpu_sort_algorithm = ["gnu_parallel_sort"]
        zip_flag = [True]

        arguments += _ajb_build_grid(*[num_elements, num_threads, cpu_sort_algorithm, zip_flag])

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

        ################################################################################################################
        # EXPERIMENT # zip_to_total_cpu_sort_duration
        ################################################################################################################

        identifier = "zip_to_total_cpu_sort_duration"
        arguments = []

        num_elements = [8000000000]
        num_threads = {Platform.AC922: [128], Platform.DGXA100: [256], Platform.DGXH100: [224]}[platform]
        cpu_merge_algorithm = ["gnu_parallel_sort"]
        zip_flag = [True, False]

        arguments += _ajb_build_grid(*[num_elements, num_threads, cpu_sort_algorithm, zip_flag])

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # EXECUTABLE # gpu_merge_benchmark
    ####################################################################################################################

    executable = "gpu_merge_benchmark"
    parameters = ["--num_elements", "--gpu_merge_algorithm"]
    input_columns = ["num_elements", "num_threads", "gpu_merge_algorithm", "key_type", "value_type", "random_seed"]
    output_columns = ["gpu_merge_duration"]
    columns = input_columns + output_columns

    if True:

        ################################################################################################################
        # EXPERIMENT # gpu_merge_algorithm_to_gpu_merge_duration
        ################################################################################################################

        identifier = "gpu_merge_algorithm_to_gpu_merge_duration"
        arguments = []

        num_elements = [1000000000]
        gpu_merge_algorithm = ["thrust_merge_by_key", "mgpu_merge"]

        arguments += _ajb_build_grid(*[num_elements, gpu_merge_algorithm])

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # EXECUTABLE # gpu_sort_benchmark
    ####################################################################################################################

    executable = "gpu_sort_benchmark"
    parameters = ["--num_elements", "--gpu_sort_algorithm"]
    input_columns = ["num_elements", "num_threads", "gpu_sort_algorithm", "key_type", "value_type", "random_seed"]
    output_columns = ["gpu_sort_duration"]
    columns = input_columns + output_columns

    if True:

        ################################################################################################################
        # EXPERIMENT # gpu_sort_algorithm_to_gpu_sort_duration
        ################################################################################################################

        identifier = "gpu_sort_algorithm_to_gpu_sort_duration"
        arguments = []

        num_elements = [1000000000]
        gpu_sort_algorithm = ["thrust_sort_by_key", "mgpu_mergesort", "cub_deviceradixsort_sortpairs"]

        arguments += _ajb_build_grid(*[num_elements, gpu_sort_algorithm])

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns))

    ####################################################################################################################
    # EXECUTABLE # join_benchmark
    ####################################################################################################################

    executable = "join_benchmark"
    parameters = [
        "--r_num_elements", "--s_num_elements", "--gpus", "--sort_algorithm", "--join_algorithm", "--chunk_size",
        "--key_type", "--key_distribution", "--value_type", "--value_distribution", "--theta", "--sigma", "--r_sort",
        "--s_sort", "--materialize"
    ]
    input_columns = [
        "r_num_elements", "s_num_elements", "num_threads", "gpus", "sort_algorithm", "join_algorithm", "chunk_size",
        "key_type", "key_distribution", "value_type", "value_distribution", "random_seed", "theta", "sigma", "r_sort",
        "s_sort", "materialize"
    ]
    output_columns = ["sort_duration", "merge_duration", "join_duration", "total_duration"]
    columns = input_columns + output_columns

    if True:

        ################################################################################################################
        # EXPERIMENT # r_num_elements_and_s_num_elements_to_total_duration_for_join_algorithm_a
        ################################################################################################################

        identifier = "r_num_elements_and_s_num_elements_to_total_duration_for_join_algorithm_a"

        # Algorithm change: upstream loops over num_elements, re-queries the
        # platform dict on every iteration, and calls itertools.product inside
        # the loop with 13 single-valued factors.  We instead hoist the
        # platform-dependent lookups out of the loop and pre-build the
        # (r_num_elements, s_num_elements) pairs via list comprehension, then
        # let _ajb_build_grid's factor-folding collapse the fixed dimensions.
        _s_values = {
                Platform.AC922: [
                    10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 3000000000,
                    4000000000, 5000000000, 6000000000, 7000000000, 8000000000, 9000000000, 10000000000
                ],
                Platform.DGXA100: [
                    10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 3000000000,
                    4000000000, 5000000000, 6000000000, 7000000000, 8000000000, 10000000000, 12000000000, 14000000000,
                    16000000000, 18000000000, 20000000000
                ],
                Platform.DGXH100: [
                    10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 3000000000,
                    4000000000, 5000000000, 6000000000, 7000000000, 8000000000, 10000000000, 12000000000, 14000000000,
                    16000000000, 18000000000, 20000000000, 22000000000, 24000000000, 26000000000, 28000000000,
                    30000000000, 32000000000, 34000000000, 36000000000, 38000000000, 40000000000
                ],
        }[platform]
        _rs_pairs = [(s // 10, s) for s in _s_values]
        _gpus = {Platform.AC922: ["0,1"], Platform.DGXA100: ["0,1,2,3,4,5,6,7"],
                 Platform.DGXH100: ["0,1,2,3,4,5,6,7"]}[platform]
        _chunk = {Platform.AC922: [1500000000], Platform.DGXA100: [2000000000],
                  Platform.DGXH100: [4000000000]}[platform]

        # Each (r,s) pair is a distinct config — product only enumerates the
        # truly varying dimension (the pair), fixed factors get spliced in.
        arguments = []
        for r_val, s_val in _rs_pairs:
            arguments += _ajb_build_grid(
                [r_val], [s_val], _gpus, ["hybrid_radix_sort"], ["hybrid_sort_merge_join"],
                _chunk, ["int"], ["unique_full_key_range"], ["int"], ["unique_full_key_range"],
                [0], [100], [False], [False], [False]
            )

        def plot(data):
            data = _ajb_groupby_mean(data, input_columns, "def plot(data):")

            data["s_num_elements"] = data["s_num_elements"].div(1e9)

            _algorithms = {
                    Platform.AC922: ["rui_sort_merge_join", "rui_hybrid_radix_join", "hybrid_sort_merge_join"],
                    Platform.DGXA100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
                    Platform.DGXH100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
            }[platform]
            x_values, y_values = _extract_series(
                data, "join_algorithm", _algorithms, "s_num_elements", "total_duration")

            line_colors = {
                Platform.AC922: [colors[1], colors[3], colors[2]],
                Platform.DGXA100: [colors[0], colors[4], colors[1], colors[3], colors[2]],
                Platform.DGXH100: [colors[0], colors[4], colors[1], colors[3], colors[2]],
            }[platform]
            line_markers = {
                Platform.AC922: [markers[2], markers[3], markers[4]],
                Platform.DGXA100: [markers[0], markers[1], markers[2], markers[3], markers[4]],
                Platform.DGXH100: [markers[0], markers[1], markers[2], markers[3], markers[4]],
            }[platform]
            line_labels = {
                Platform.AC922: [
                    "Rui SMJ (2 GPUs)",
                    "Rui HRJ (4 GPUs)",
                    "HMG SMJ (2 GPUs)",
                ],
                Platform.DGXA100: [
                    "Balkesen SMJ (CPU)",
                    "Balkesen RHJ (CPU)",
                    "Rui SMJ (4 GPUs)",
                    "Rui HRJ (8 GPUs)",
                    "HMG SMJ (8 GPUs)",
                ],
                Platform.DGXH100: [
                    "Balkesen SMJ (CPU)",
                    "Balkesen RHJ (CPU)",
                    "Rui SMJ (4 GPUs)",
                    "Rui HRJ (8 GPUs)",
                    "HMG SMJ (8 GPUs)",
                ],
            }[platform]

            _n_series = len([xv for xv in x_values if len(xv) > 0])
            scale_figure_size(2 if PLOT_LEGEND else 1, 1)
            plot_lines(x_values, y_values, line_colors, line_markers, line_labels)

            x_ticks_ticks = {
                Platform.AC922: numpy.arange(0, 10 + 1, 1),
                Platform.DGXA100: numpy.arange(0, 20 + 1, 2),
                Platform.DGXH100: numpy.arange(0, 40 + 1, 4),
            }[platform]
            y_ticks_ticks = {
                Platform.AC922: numpy.arange(0, 20 + 1, 5),
                Platform.DGXA100: numpy.arange(0, 30 + 1, 5),
                Platform.DGXH100: numpy.arange(0, 60 + 1, 10),
            }[platform]

            configure_plot(x_ticks_ticks=x_ticks_ticks,
                           y_ticks_ticks=y_ticks_ticks,
                           set_y_limits=True,
                           x_label="$10 * |R| = |S|$ [1e9]",
                           y_label="Join duration [s]",
                           legend=PLOT_LEGEND,
                           legend_anchor=(0, 1.02, 1, 0.2),
                           legend_mode="expand",
                           legend_location="lower left",
                           legend_columns=3)

        def plot_overview(data):
            data = _ajb_groupby_mean(data, input_columns, "def plot_overview(data):")

            s_num_elements = {
                Platform.AC922: 3000000000,
                Platform.DGXA100: 16000000000,
                Platform.DGXH100: 32000000000,
            }[platform]
            data = data[data["s_num_elements"] == s_num_elements]
            if data.empty:
                return

            data["s_num_elements"] = data["s_num_elements"].div(1e9)

            _overview_algos = {
                    Platform.AC922: ["rui_sort_merge_join", "rui_hybrid_radix_join", "hybrid_sort_merge_join"],
                    Platform.DGXA100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
                    Platform.DGXH100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
            }[platform]
            y_values = _extract_bar_series(
                data, "join_algorithm", _overview_algos, "total_duration")

            bar_colors = {
                Platform.AC922: [colors[1], colors[3], colors[2]],
                Platform.DGXA100: [colors[0], colors[4], colors[1], colors[3], colors[2]],
                Platform.DGXH100: [colors[0], colors[4], colors[1], colors[3], colors[2]],
            }[platform]
            bar_hatches = {
                Platform.AC922: [hatches[3], hatches[0], hatches[4]],
                Platform.DGXA100: [None, None, hatches[3], hatches[0], hatches[4]],
                Platform.DGXH100: [None, None, hatches[3], hatches[0], hatches[4]],
            }[platform]
            bar_labels = {
                Platform.AC922: [
                    "Rui SMJ (2 GPUs)",
                    "Rui HRJ (4 GPUs)",
                    "HMG SMJ (2 GPUs)",
                ],
                Platform.DGXA100: [
                    "Balkesen SMJ (CPU)",
                    "Balkesen RHJ (CPU)",
                    "Rui SMJ (4 GPUs)",
                    "Rui HRJ (8 GPUs)",
                    "HMG SMJ (8 GPUs)",
                ],
                Platform.DGXH100: [
                    "Balkesen SMJ (CPU)",
                    "Balkesen RHJ (CPU)",
                    "Rui SMJ (4 GPUs)",
                    "Rui HRJ (8 GPUs)",
                    "HMG SMJ (8 GPUs)",
                ],
            }[platform]

            scale_figure_size(2 if PLOT_LEGEND else 1, 1)

            bar_handles = plot_bars(y_values, 1, bar_colors, bar_hatches)

            y_ticks_ticks = {
                Platform.AC922: numpy.arange(0, 7 + 1, 1),
                Platform.DGXA100: numpy.arange(0, 30 + 1, 5),
                Platform.DGXH100: numpy.arange(0, 60 + 1, 10),
            }[platform]
            x_label = {
                Platform.AC922: "$10 * |R| = |S|$ = 3B",
                Platform.DGXA100: "$10 * |R| = |S|$ = 16B",
                Platform.DGXH100: "$10 * |R| = |S|$ = 32B",
            }[platform]

            configure_plot(x_ticks_color="white",
                           x_ticks_ticks=[0],
                           x_ticks_labels=[""],
                           y_ticks_ticks=y_ticks_ticks,
                           x_label=x_label,
                           y_label="Join duration [s]",
                           legend=PLOT_LEGEND,
                           legend_anchor=(0, 1.02, 1, 0.2),
                           legend_mode="expand",
                           legend_location="lower left",
                           legend_handles=bar_handles,
                           legend_labels=bar_labels,
                           legend_columns=3)

        plots = {"": plot, "overview": plot_overview}

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns, plots))

        ################################################################################################################
        # EXPERIMENT # r_num_elements_and_s_num_elements_to_total_duration_for_join_algorithm_b
        ################################################################################################################

        identifier = "r_num_elements_and_s_num_elements_to_total_duration_for_join_algorithm_b"

        # Algorithm change: upstream loops num_elements, re-queries platform
        # dicts per iteration; r == s here so we pre-build symmetric pairs.
        _elem_b = {
                Platform.AC922: [
                    10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 3000000000,
                    4000000000, 5000000000
                ],
                Platform.DGXA100: [
                    10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 3000000000,
                    4000000000, 5000000000, 6000000000, 7000000000, 8000000000, 9000000000, 10000000000
                ],
                Platform.DGXH100: [
                    10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 3000000000,
                    4000000000, 5000000000, 6000000000, 7000000000, 8000000000, 9000000000, 10000000000, 12000000000,
                    14000000000, 16000000000, 18000000000, 20000000000
                ],
        }[platform]
        _gpus_b = {Platform.AC922: ["0,1"], Platform.DGXA100: ["0,1,2,3,4,5,6,7"],
                   Platform.DGXH100: ["0,1,2,3,4,5,6,7"]}[platform]
        _chunk_b = {Platform.AC922: [750000000], Platform.DGXA100: [1000000000],
                    Platform.DGXH100: [2000000000]}[platform]

        # Symmetric join: r_num_elements == s_num_elements, generate pairs
        # directly instead of re-entering the loop for each value.
        arguments = []
        for n in _elem_b:
            arguments += _ajb_build_grid(
                [n], [n], _gpus_b, ["hybrid_radix_sort"], ["hybrid_sort_merge_join"],
                _chunk_b, ["long"], ["unique_full_key_range"], ["long"], ["unique_full_key_range"],
                [0], [100], [False], [False], [False]
            )

        def plot(data):
            data = _ajb_groupby_mean(data, input_columns, "def plot(data):")

            data["s_num_elements"] = data["s_num_elements"].div(1e9)

            _algorithms = {
                    Platform.AC922: ["rui_sort_merge_join", "rui_hybrid_radix_join", "hybrid_sort_merge_join"],
                    Platform.DGXA100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
                    Platform.DGXH100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
            }[platform]
            x_values, y_values = _extract_series(
                data, "join_algorithm", _algorithms, "s_num_elements", "total_duration")

            line_colors = {
                Platform.AC922: [colors[1], colors[3], colors[2]],
                Platform.DGXA100: [colors[0], colors[4], colors[1], colors[3], colors[2]],
                Platform.DGXH100: [colors[0], colors[4], colors[1], colors[3], colors[2]],
            }[platform]
            line_markers = {
                Platform.AC922: [markers[2], markers[3], markers[4]],
                Platform.DGXA100: [markers[0], markers[1], markers[2], markers[3], markers[4]],
                Platform.DGXH100: [markers[0], markers[1], markers[2], markers[3], markers[4]],
            }[platform]
            line_labels = {
                Platform.AC922: [
                    "Rui SMJ (2 GPUs)",
                    "Rui HRJ (4 GPUs)",
                    "HMG SMJ (2 GPUs)",
                ],
                Platform.DGXA100: [
                    "Balkesen SMJ (CPU)",
                    "Balkesen RHJ (CPU)",
                    "Rui SMJ (4 GPUs)",
                    "Rui HRJ (8 GPUs)",
                    "HMG SMJ (8 GPUs)",
                ],
                Platform.DGXH100: [
                    "Balkesen SMJ (CPU)",
                    "Balkesen RHJ (CPU)",
                    "Rui SMJ (4 GPUs)",
                    "Rui HRJ (8 GPUs)",
                    "HMG SMJ (8 GPUs)",
                ],
            }[platform]

            _n_series = len([xv for xv in x_values if len(xv) > 0])
            scale_figure_size(2 if PLOT_LEGEND else 1, 1)
            plot_lines(x_values, y_values, line_colors, line_markers, line_labels)

            x_ticks_ticks = {
                Platform.AC922: numpy.arange(0, 5 + 1, 1),
                Platform.DGXA100: numpy.arange(0, 10 + 1, 1),
                Platform.DGXH100: numpy.arange(0, 20 + 1, 2),
            }[platform]
            y_ticks_ticks = {
                Platform.AC922: numpy.arange(0, 40 + 1, 10),
                Platform.DGXA100: numpy.arange(0, 60 + 1, 10),
                Platform.DGXH100: numpy.arange(0, 200 + 1, 40),
            }[platform]

            configure_plot(x_ticks_ticks=x_ticks_ticks,
                           y_ticks_ticks=y_ticks_ticks,
                           set_y_limits=True,
                           x_label="$|R| = |S|$ [1e9]",
                           y_label="Join duration [s]",
                           legend=PLOT_LEGEND,
                           legend_anchor=(0, 1.02, 1, 0.2),
                           legend_mode="expand",
                           legend_location="lower left",
                           legend_columns=3)

        plots = {"": plot}

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns, plots))

        ################################################################################################################
        # EXPERIMENT # gpus_to_total_duration_for_sort_algorithm_with_nsys
        ################################################################################################################

        identifier = "gpus_to_total_duration_for_sort_algorithm_with_nsys"
        arguments = []

        r_num_elements = {
            Platform.AC922: [1500000000],
            Platform.DGXA100: [8000000000],
            Platform.DGXH100: [16000000000],
        }[platform]
        s_num_elements = {
            Platform.AC922: [1500000000],
            Platform.DGXA100: [8000000000],
            Platform.DGXH100: [16000000000],
        }[platform]
        gpus = {
            Platform.AC922: ["0", "0,1", "0,1,2,3"],
            Platform.DGXA100: ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"],
            Platform.DGXH100: ["0", "0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"],
        }[platform]
        sort_algorithm = ["hybrid_merge_sort", "hybrid_radix_sort"]
        join_algorithm = ["hybrid_sort_merge_join"]
        chunk_size = {
            Platform.AC922: [750000000],
            Platform.DGXA100: [1000000000],
            Platform.DGXH100: [2000000000],
        }[platform]
        key_type = ["long"]
        key_distribution = ["unique_full_key_range"]
        value_type = ["long"]
        value_distribution = ["unique_full_key_range"]
        theta = [0]
        sigma = [100]
        r_sort = [False]
        s_sort = [False]
        materialize = [False]

        arguments += _ajb_build_grid(
            r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size, key_type,
            key_distribution, value_type, value_distribution, theta, sigma, r_sort, s_sort, materialize
        )

        profilers = ["nsys"]

        def plot_nsys(data, sort_algorithm):
            data = _ajb_groupby_mean(data, input_columns, "def plot_nsys(data, sort_algorithm):")

            gpus = {
                Platform.AC922: ["0", "0,1", "0,1,2,3"],
                Platform.DGXA100: ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"],
                Platform.DGXH100: ["0", "0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"],
            }[platform]
            data["gpus"] = pandas.Categorical(data["gpus"], categories=gpus, ordered=True)
            data = data.sort_values("gpus", ascending=True)

            data = data[data["sort_algorithm"] == sort_algorithm]
            if data.empty:
                return

            segment_columns = [
                "htod_nsys_duration", "gpu_sort_nsys_duration", "p2p_nsys_duration", "gpu_sort_dtoh_nsys_duration",
                "dtoh_htod_nsys_duration", "dtoh_nsys_duration", "cpu_merge_nsys_duration",
                "htod_join_dtoh_nsys_duration"
            ]
            segment_colors = [colors[0], colors[4], colors[3], colors[4], colors[2], colors[1], colors[3], colors[5]]
            segment_hatches = [hatches[0], None, hatches[3], hatches[1], hatches[2], hatches[1], None, hatches[4]]
            segment_labels = [
                "HtoD", "Sort", "P2P", "Sort / DtoH", "DtoH / HtoD", "DtoH", "Merge", "HtoD / Join / DtoH"
            ]

            scale_figure_size(2 if PLOT_LEGEND else 1, 1)
            segment_handles = plot_stacked_bars(data, segment_columns, segment_colors, segment_hatches, segment_labels,
                                                "gpus", 2)

            # AJB: len(split) instead of comma-count — handles edge cases (trailing comma, empty string)
            x_ticks_labels = [len(str(g).split(",")) for g in data["gpus"]]
            x_ticks_ticks = range(len(x_ticks_labels))
            y_ticks_ticks = {
                Platform.AC922: numpy.arange(0, 7 + 1, 1),
                Platform.DGXA100: numpy.arange(0, 40 + 1, 10),
                Platform.DGXH100: numpy.arange(0, 50 + 1, 10),
            }[platform]

            configure_plot(x_ticks_ticks=x_ticks_ticks,
                           x_ticks_labels=x_ticks_labels,
                           y_ticks_ticks=y_ticks_ticks,
                           x_label="GPU count",
                           y_label="Join duration [s]",
                           legend=PLOT_LEGEND,
                           legend_anchor=(0, 1.02, 1, 0.2),
                           legend_mode="expand",
                           legend_location="lower left",
                           legend_handles=segment_handles[::-1],
                           legend_columns=4)

        plots = {
            "merge": lambda data: plot_nsys(data, "hybrid_merge_sort"),
            "radix": lambda data: plot_nsys(data, "hybrid_radix_sort"),
        }

        experiments.append(
            Experiment(identifier, executable, parameters, arguments, columns, plots, profilers=profilers))

        ################################################################################################################
        # EXPERIMENT # r_num_elements_and_s_num_elements_to_total_duration_for_gpus
        ################################################################################################################

        identifier = "r_num_elements_and_s_num_elements_to_total_duration_for_gpus"

        # Algorithm change: upstream re-queries 3 platform dicts per loop
        # iteration (gpus, chunk_size inside the loop body).  Hoist out and
        # let _ajb_build_grid fold the 12 fixed dimensions.
        _elem_gpus = {
                Platform.AC922: [
                    10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 3000000000,
                    4000000000, 5000000000
                ],
                Platform.DGXA100: [
                    10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 3000000000,
                    4000000000, 5000000000, 6000000000, 7000000000, 8000000000, 9000000000, 10000000000
                ],
                Platform.DGXH100: [
                    10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 3000000000,
                    4000000000, 5000000000, 6000000000, 7000000000, 8000000000, 9000000000, 10000000000, 12000000000,
                    14000000000, 16000000000, 18000000000, 20000000000
                ],
        }[platform]
        _gpus_g = {Platform.AC922: ["0", "0,1", "0,1,2,3"],
                   Platform.DGXA100: ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"],
                   Platform.DGXH100: ["0", "0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"]}[platform]
        _chunk_g = {Platform.AC922: [750000000], Platform.DGXA100: [1000000000],
                    Platform.DGXH100: [2000000000]}[platform]

        arguments = []
        for n in _elem_gpus:
            arguments += _ajb_build_grid(
                [n], [n], _gpus_g, ["hybrid_radix_sort"], ["hybrid_sort_merge_join"],
                _chunk_g, ["long"], ["unique_full_key_range"], ["long"], ["unique_full_key_range"],
                [0], [100], [False], [False], [False]
            )

        def plot(data):
            data = _ajb_groupby_mean(data, input_columns, "def plot(data):")

            data["s_num_elements"] = data["s_num_elements"].div(1e9)

            x_values = []
            y_values = []
            _gpu_configs = {
                    Platform.AC922: ["0", "0,1", "0,1,2,3"],
                    Platform.DGXA100: ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"],
                    Platform.DGXH100: ["0", "0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"],
            }[platform]
            x_values, y_values = _extract_series(
                data, "gpus", _gpu_configs, "s_num_elements", "total_duration")

            line_colors = {
                Platform.AC922: [colors[4], colors[1], colors[3]],
                Platform.DGXA100: [colors[4], colors[1], colors[3], colors[2]],
                Platform.DGXH100: [colors[4], colors[1], colors[3], colors[2]],
            }[platform]
            line_markers = {
                Platform.AC922: [markers[0], markers[1], markers[2]],
                Platform.DGXA100: [markers[0], markers[1], markers[2], markers[3]],
                Platform.DGXH100: [markers[0], markers[1], markers[2], markers[3]],
            }[platform]
            line_labels = {
                Platform.AC922: ["1", "2", "4"],
                Platform.DGXA100: ["1", "2", "4", "8"],
                Platform.DGXH100: ["1", "2", "4", "8"],
            }[platform]

            _n_series = len([xv for xv in x_values if len(xv) > 0])
            scale_figure_size(2 if PLOT_LEGEND else 1, 1)
            plot_lines(x_values, y_values, line_colors, line_markers, line_labels)

            x_ticks_ticks = {
                Platform.AC922: numpy.arange(0, 5 + 1, 1),
                Platform.DGXA100: numpy.arange(0, 10 + 1, 1),
                Platform.DGXH100: numpy.arange(0, 20 + 1, 2),
            }[platform]
            y_ticks_ticks = {
                Platform.AC922: numpy.arange(0, 25 + 1, 5),
                Platform.DGXA100: numpy.arange(0, 50 + 1, 10),
                Platform.DGXH100: numpy.arange(0, 50 + 1, 10),
            }[platform]

            configure_plot(x_ticks_ticks=x_ticks_ticks,
                           y_ticks_ticks=y_ticks_ticks,
                           set_y_limits=True,
                           x_label="$|R| = |S|$ [1e9]",
                           y_label="Join duration [s]",
                           legend=PLOT_LEGEND,
                           legend_anchor=(0, 1.02, 1, 0.2),
                           legend_mode="expand",
                           legend_location="lower left",
                           legend_columns=4)

        plots = {"": plot}

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns, plots))

        ################################################################################################################
        # EXPERIMENT # sigma_to_total_duration
        ################################################################################################################

        identifier = "sigma_to_total_duration"
        arguments = []

        r_num_elements = {
            Platform.AC922: [1500000000],
            Platform.DGXA100: [8000000000],
            Platform.DGXH100: [16000000000],
        }[platform]
        s_num_elements = {
            Platform.AC922: [1500000000],
            Platform.DGXA100: [8000000000],
            Platform.DGXH100: [16000000000],
        }[platform]
        gpus = {
            Platform.AC922: ["0,1"],
            Platform.DGXA100: ["0,1,2,3,4,5,6,7"],
            Platform.DGXH100: ["0,1,2,3,4,5,6,7"],
        }[platform]
        sort_algorithm = ["hybrid_radix_sort"]
        join_algorithm = ["hybrid_sort_merge_join"]
        chunk_size = {
            Platform.AC922: [750000000],
            Platform.DGXA100: [1000000000],
            Platform.DGXH100: [2000000000],
        }[platform]
        key_type = ["long"]
        key_distribution = ["unique_partial_key_range"]
        value_type = ["long"]
        value_distribution = ["unique_partial_key_range"]
        theta = [0]
        sigma = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        r_sort = [False]
        s_sort = [False]
        materialize = [False]

        arguments += _ajb_build_grid(
            r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size, key_type,
            key_distribution, value_type, value_distribution, theta, sigma, r_sort, s_sort, materialize
        )

        def plot(data):
            data = _ajb_groupby_mean(data, input_columns, "def plot(data):")

            segment_columns = ["sort_duration", "join_duration"]
            _seg_missing = [c for c in segment_columns if c not in data.columns]
            if _seg_missing:
                return
            segment_colors = [colors[1], colors[5]]
            segment_hatches = [None, hatches[4]]
            segment_labels = ["Sort", "Join"]

            scale_figure_size(1 if PLOT_LEGEND else 0.5, 1)
            segment_handles = plot_stacked_lines(data, segment_columns, segment_colors, segment_hatches, segment_labels,
                                                 "sigma")

            y_ticks_ticks = {
                Platform.AC922: numpy.arange(0, 2 + 1, 0.5),
                Platform.DGXA100: numpy.arange(0, 9 + 1, 1.5),
                Platform.DGXH100: numpy.arange(0, 12.5 + 1, 2.5),
            }[platform]
            # AJB: unified label formatting — upstream's AC922 branch hardcoded "12.5"
            # which only works if platform dict values match. Use auto-format instead.
            y_ticks_labels = [f"{v:.1f}" if v != int(v) else str(int(v))
                              for v in y_ticks_ticks]

            configure_plot(y_ticks_ticks=y_ticks_ticks,
                           y_ticks_labels=y_ticks_labels,
                           x_label="Selectivity [\%]~:",
                           y_label="Join duration [s]",
                           legend=PLOT_LEGEND,
                           legend_anchor=(0, 1.02, 1, 0.2),
                           legend_mode="expand",
                           legend_location="lower left",
                           legend_handles=segment_handles[::-1],
                           legend_columns=2)

        plots = {"": plot}

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns, plots))

        ################################################################################################################
        # EXPERIMENT # theta_to_total_duration
        ################################################################################################################

        identifier = "theta_to_total_duration"
        arguments = []

        r_num_elements = {
            Platform.AC922: [1500000000],
            Platform.DGXA100: [8000000000],
            Platform.DGXH100: [16000000000],
        }[platform]
        s_num_elements = {
            Platform.AC922: [1500000000],
            Platform.DGXA100: [8000000000],
            Platform.DGXH100: [16000000000],
        }[platform]
        gpus = {
            Platform.AC922: ["0,1"],
            Platform.DGXA100: ["0,1,2,3,4,5,6,7"],
            Platform.DGXH100: ["0,1,2,3,4,5,6,7"],
        }[platform]
        sort_algorithm = ["hybrid_radix_sort"]
        join_algorithm = ["hybrid_sort_merge_join"]
        chunk_size = {
            Platform.AC922: [750000000],
            Platform.DGXA100: [1000000000],
            Platform.DGXH100: [2000000000],
        }[platform]
        key_type = ["long"]
        key_distribution = ["unique_partial_key_range"]
        value_type = ["long"]
        value_distribution = ["unique_partial_key_range"]
        theta = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
        sigma = [100]
        r_sort = [False]
        s_sort = [False]
        materialize = [False]

        arguments += _ajb_build_grid(
            r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size, key_type,
            key_distribution, value_type, value_distribution, theta, sigma, r_sort, s_sort, materialize
        )

        def plot(data):
            data = _ajb_groupby_mean(data, input_columns, "def plot(data):")

            segment_columns = ["sort_duration", "join_duration"]
            _seg_missing = [c for c in segment_columns if c not in data.columns]
            if _seg_missing:
                return
            segment_colors = [colors[1], colors[5]]
            segment_hatches = [None, hatches[4]]
            segment_labels = ["Sort", "Join"]

            scale_figure_size(1 if PLOT_LEGEND else 0.5, 1)
            segment_handles = plot_stacked_lines(data, segment_columns, segment_colors, segment_hatches, segment_labels,
                                                 "theta")

            y_ticks_ticks = {
                Platform.AC922: numpy.arange(0, 2 + 1, 0.5),
                Platform.DGXA100: numpy.arange(0, 9 + 1, 1.5),
                Platform.DGXH100: numpy.arange(0, 12.5 + 1, 2.5),
            }[platform]
            y_ticks_labels = [f"{v:.1f}" if v != int(v) else str(int(v))
                              for v in y_ticks_ticks]

            configure_plot(y_ticks_ticks=y_ticks_ticks,
                           y_ticks_labels=y_ticks_labels,
                           x_label="Skew [\%]",
                           y_label="Join duration [s]",
                           legend=PLOT_LEGEND,
                           legend_anchor=(0, 1.02, 1, 0.2),
                           legend_mode="expand",
                           legend_location="lower left",
                           legend_handles=segment_handles[::-1],
                           legend_columns=2)

        plots = {"": plot}

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns, plots))

    ####################################################################################################################
    # EXECUTABLE # ajb_benchmark (AJB — Adaptive Bandwidth-tier Join)
    # [AJB] 自适应带宽分层Join: 根据GPU互连拓扑(PCIe/NVLink/NVSwitch)自动选择最优数据传输策略
    # K_x: build-partition cadence — 控制hash partition的粒度
    # K_u: merge-path boundary cadence — 控制merge阶段的并行度
    # K_v: materialization buffer cadence — 控制结果物化的buffer大小
    # auto_tune=True: 使用skew detector自动调参
    ####################################################################################################################

    executable = "ajb_benchmark"
    parameters = [
        "--r_num_elements", "--s_num_elements", "--gpus", "--sort_algorithm", "--join_algorithm",
        "--chunk_size", "--key_type", "--key_distribution", "--value_type", "--value_distribution",
        "--theta", "--sigma", "--r_sort", "--s_sort", "--materialize",
        "--ajb", "--K_x", "--K_u", "--K_v", "--auto_tune"
    ]
    input_columns = [
        "r_num_elements", "s_num_elements", "gpus", "sort_algorithm", "join_algorithm", "chunk_size",
        "key_type", "key_distribution", "value_type", "value_distribution", "theta", "sigma",
        "r_sort", "s_sort", "materialize", "random_seed"
    ]
    output_columns = [
        "sort_duration", "join_duration", "total_duration", "num_matches",
        "mode", "K_x", "K_u", "K_v", "slow_bytes", "fast_bytes", "skew_cv", "skew_normalized"
    ]
    columns = input_columns + output_columns

    if True:

        ################################################################################################################
        # EXPERIMENT # ajb_cadence_sweep — sweep K_u with fixed K_x, K_v (paper Figure 3)
        ################################################################################################################

        identifier = "ajb_cadence_sweep"
        # [AJB] cadence sweep: 固定K_x/K_v, 扫描K_u, 对应论文Figure 3
        # 验证merge-path boundary cadence对throughput的影响
        arguments = []

        r_num_elements = {
            Platform.AC922: [1500000000],
            Platform.DGXA100: [8000000000],
            Platform.DGXH100: [16000000000],
        }[platform]
        s_num_elements = r_num_elements
        gpus = {
            Platform.AC922: ["0,1"],
            Platform.DGXA100: ["0,1,2,3,4,5,6,7"],
            Platform.DGXH100: ["0,1,2,3,4,5,6,7"],
        }[platform]
        sort_algorithm = ["hybrid_radix_sort"]
        join_algorithm = ["hybrid_sort_merge_join"]
        chunk_size = {
            Platform.AC922: [750000000],
            Platform.DGXA100: [1000000000],
            Platform.DGXH100: [2000000000],
        }[platform]
        key_type = ["long"]
        key_distribution = ["unique_partial_key_range"]
        value_type = ["long"]
        value_distribution = ["unique_partial_key_range"]
        theta = [0, 50, 90, 99]
        sigma = [100]
        r_sort = [False]
        s_sort = [False]
        materialize = [False]
        ajb = [True]
        K_x_vals = [16]
        K_u_vals = [1, 2, 4, 8, 16]
        K_v_vals = [8]
        auto_tune = [False]

        arguments += _ajb_build_grid(
            r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size,
            key_type, key_distribution, value_type, value_distribution, theta, sigma,
            r_sort, s_sort, materialize, ajb, K_x_vals, K_u_vals, K_v_vals, auto_tune
        )

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns, repetitions=3))

        ################################################################################################################
        # EXPERIMENT # ajb_vs_upstream — AJB auto-tune vs upstream baseline (paper Figure 2)
        ################################################################################################################

        identifier = "ajb_vs_upstream"
        arguments = []

        r_num_elements_list = {
            Platform.AC922: [500000000, 1000000000, 1500000000],
            Platform.DGXA100: [2000000000, 4000000000, 8000000000],
            Platform.DGXH100: [4000000000, 8000000000, 16000000000],
        }[platform]
        gpus = {
            Platform.AC922: ["0,1"],
            Platform.DGXA100: ["0,1,2,3,4,5,6,7"],
            Platform.DGXH100: ["0,1,2,3,4,5,6,7"],
        }[platform]
        sort_algorithm = ["hybrid_radix_sort"]
        join_algorithm = ["hybrid_sort_merge_join"]
        chunk_size = {
            Platform.AC922: [750000000],
            Platform.DGXA100: [1000000000],
            Platform.DGXH100: [2000000000],
        }[platform]
        key_type = ["long"]
        key_distribution = ["unique_partial_key_range"]
        value_type = ["long"]
        value_distribution = ["unique_partial_key_range"]
        theta = [50]
        sigma = [100]
        r_sort = [False]
        s_sort = [False]
        materialize = [False]

        # Algorithm change: upstream loops rn, calling product twice per
        # iteration (ajb=True + ajb=False) with 18 single-valued factors each.
        # Instead we pre-build both variants and concatenate.
        for rn in r_num_elements_list:
            arguments += _ajb_build_grid(
                [rn], [rn], gpus, sort_algorithm, join_algorithm, chunk_size,
                key_type, key_distribution, value_type, value_distribution, theta, sigma,
                r_sort, s_sort, materialize, [True], [0], [0], [0], [True]
            )
            arguments += _ajb_build_grid(
                [rn], [rn], gpus, sort_algorithm, join_algorithm, chunk_size,
                key_type, key_distribution, value_type, value_distribution, theta, sigma,
                r_sort, s_sort, materialize, [False], [0], [0], [0], [False]
            )

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns, repetitions=3))

    # AJB: init() wrap-up — total grid summary
    _total_cfgs = sum(len(e.arguments) for e in experiments)
    _total_runs = sum(e.grid_size for e in experiments)
    _est_h = _total_runs * 0.5 / 60.0
    _init_ms = (time.monotonic() - _t_init) * 1000
    _ajb_diag("STATE", f"init done: {len(experiments)} experiments, "
              f"{_total_cfgs} configs, {_total_runs} total runs, "
              f"est ~{_est_h:.1f}h, init took {_init_ms:.1f}ms")
    # per-experiment breakdown (sorted by grid size descending)
    for _e in sorted(experiments, key=lambda e: e.grid_size, reverse=True)[:5]:
        _ajb_diag("STATE", f"  {_e}")
