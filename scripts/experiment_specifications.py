import enum
import itertools
import numpy
import pandas

from typing import Callable, Dict, List, Union

from figure_utilities import *

PLOT_LEGEND = False


class Platform(enum.Enum):
    AC922 = 1
    DGXA100 = 2
    DGXH100 = 3

    def __str__(self):
        return self.name


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


def init(platform: Platform):
    global executables_path
    executables_path = "../build"

    global experiments_path
    experiments_path = "../experiments"

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

        arguments += list(itertools.product(*[num_elements, num_threads, cpu_merge_algorithm, chunk_count, zip_flag]))

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

        arguments += list(itertools.product(*[num_elements, num_threads, cpu_merge_algorithm, chunk_count, zip_flag]))

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

        arguments += list(itertools.product(*[num_elements, num_threads, cpu_merge_algorithm, chunk_count, zip_flag]))

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

        arguments += list(itertools.product(*[num_elements, num_threads, cpu_sort_algorithm, zip_flag]))

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

        arguments += list(itertools.product(*[num_elements, num_threads, cpu_sort_algorithm, zip_flag]))

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

        arguments += list(itertools.product(*[num_elements, gpu_merge_algorithm]))

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

        arguments += list(itertools.product(*[num_elements, gpu_sort_algorithm]))

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
        arguments = []

        for num_elements in {
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
        }[platform]:
            r_num_elements = [num_elements // 10]
            s_num_elements = [num_elements]
            gpus = {
                Platform.AC922: ["0,1"],
                Platform.DGXA100: ["0,1,2,3,4,5,6,7"],
                Platform.DGXH100: ["0,1,2,3,4,5,6,7"]
            }[platform]
            sort_algorithm = ["hybrid_radix_sort"]
            join_algorithm = ["hybrid_sort_merge_join"]
            chunk_size = {
                Platform.AC922: [1500000000],
                Platform.DGXA100: [2000000000],
                Platform.DGXH100: [4000000000]
            }[platform]
            key_type = ["int"]
            key_distribution = ["unique_full_key_range"]
            value_type = ["int"]
            value_distribution = ["unique_full_key_range"]
            theta = [0]
            sigma = [100]
            r_sort = [False]
            s_sort = [False]
            materialize = [False]

            arguments += list(
                itertools.product(*[
                    r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size, key_type,
                    key_distribution, value_type, value_distribution, theta, sigma, r_sort, s_sort, materialize
                ]))

        def plot(data):
            data = data.groupby(input_columns, as_index=False).mean()

            data["s_num_elements"] = data["s_num_elements"].div(1e9)

            x_values = []
            y_values = []
            for algorithm in {
                    Platform.AC922: ["rui_sort_merge_join", "rui_hybrid_radix_join", "hybrid_sort_merge_join"],
                    Platform.DGXA100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
                    Platform.DGXH100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
            }[platform]:
                x_values.append(data[data["join_algorithm"] == algorithm]["s_num_elements"].tolist())
                y_values.append(data[data["join_algorithm"] == algorithm]["total_duration"].tolist())

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
            data = data.groupby(input_columns, as_index=False).mean()

            s_num_elements = {
                Platform.AC922: 3000000000,
                Platform.DGXA100: 16000000000,
                Platform.DGXH100: 32000000000,
            }[platform]
            data = data[data["s_num_elements"] == s_num_elements]

            data["s_num_elements"] = data["s_num_elements"].div(1e9)

            y_values = []
            for algorithm in {
                    Platform.AC922: ["rui_sort_merge_join", "rui_hybrid_radix_join", "hybrid_sort_merge_join"],
                    Platform.DGXA100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
                    Platform.DGXH100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
            }[platform]:
                y_values.append(data[data["join_algorithm"] == algorithm]["total_duration"].tolist())

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
        arguments = []

        for num_elements in {
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
        }[platform]:
            r_num_elements = [num_elements]
            s_num_elements = [num_elements]
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
            key_distribution = ["unique_full_key_range"]
            value_type = ["long"]
            value_distribution = ["unique_full_key_range"]
            theta = [0]
            sigma = [100]
            r_sort = [False]
            s_sort = [False]
            materialize = [False]

            arguments += list(
                itertools.product(*[
                    r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size, key_type,
                    key_distribution, value_type, value_distribution, theta, sigma, r_sort, s_sort, materialize
                ]))

        def plot(data):
            data = data.groupby(input_columns, as_index=False).mean()

            data["s_num_elements"] = data["s_num_elements"].div(1e9)

            x_values = []
            y_values = []
            for algorithm in {
                    Platform.AC922: ["rui_sort_merge_join", "rui_hybrid_radix_join", "hybrid_sort_merge_join"],
                    Platform.DGXA100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
                    Platform.DGXH100: [
                        "balkesen_sort_merge_join", "balkesen_radix_hash_join", "rui_sort_merge_join",
                        "rui_hybrid_radix_join", "hybrid_sort_merge_join"
                    ],
            }[platform]:
                x_values.append(data[data["join_algorithm"] == algorithm]["s_num_elements"].tolist())
                y_values.append(data[data["join_algorithm"] == algorithm]["total_duration"].tolist())

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

        arguments += list(
            itertools.product(*[
                r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size, key_type,
                key_distribution, value_type, value_distribution, theta, sigma, r_sort, s_sort, materialize
            ]))

        profilers = ["nsys"]

        def plot_nsys(data, sort_algorithm):
            data = data.groupby(input_columns, as_index=False).mean()

            gpus = {
                Platform.AC922: ["0", "0,1", "0,1,2,3"],
                Platform.DGXA100: ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"],
                Platform.DGXH100: ["0", "0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"],
            }[platform]
            data["gpus"] = pandas.Categorical(data["gpus"], categories=gpus, ordered=True)
            data = data.sort_values("gpus", ascending=True)

            data = data[data["sort_algorithm"] == sort_algorithm]

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

            x_ticks_labels = [gpus.count(",") + 1 for gpus in data["gpus"]]
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
        arguments = []

        for num_elements in {
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
        }[platform]:
            r_num_elements = [num_elements]
            s_num_elements = [num_elements]
            gpus = {
                Platform.AC922: ["0", "0,1", "0,1,2,3"],
                Platform.DGXA100: ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"],
                Platform.DGXH100: ["0", "0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"],
            }[platform]
            sort_algorithm = ["hybrid_radix_sort"]
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

            arguments += list(
                itertools.product(*[
                    r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size, key_type,
                    key_distribution, value_type, value_distribution, theta, sigma, r_sort, s_sort, materialize
                ]))

        def plot(data):
            data = data.groupby(input_columns, as_index=False).mean()

            data["s_num_elements"] = data["s_num_elements"].div(1e9)

            x_values = []
            y_values = []
            for gpus in {
                    Platform.AC922: ["0", "0,1", "0,1,2,3"],
                    Platform.DGXA100: ["0", "0,2", "0,2,4,6", "0,1,2,3,4,5,6,7"],
                    Platform.DGXH100: ["0", "0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"],
            }[platform]:
                x_values.append(data[data["gpus"] == gpus]["s_num_elements"].tolist())
                y_values.append(data[data["gpus"] == gpus]["total_duration"].tolist())

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

        arguments += list(
            itertools.product(*[
                r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size, key_type,
                key_distribution, value_type, value_distribution, theta, sigma, r_sort, s_sort, materialize
            ]))

        def plot(data):
            data = data.groupby(input_columns, as_index=False).mean()

            segment_columns = ["sort_duration", "join_duration"]
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
            y_ticks_labels = y_ticks_ticks
            if platform == Platform.AC922:
                y_ticks_labels = [str(y_ticks_label) for y_ticks_label in y_ticks_labels]
                y_ticks_labels[-1] = "12.5"

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

        arguments += list(
            itertools.product(*[
                r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size, key_type,
                key_distribution, value_type, value_distribution, theta, sigma, r_sort, s_sort, materialize
            ]))

        def plot(data):
            data = data.groupby(input_columns, as_index=False).mean()

            segment_columns = ["sort_duration", "join_duration"]
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
            y_ticks_labels = y_ticks_ticks
            if platform == Platform.AC922:
                y_ticks_labels = [str(y_ticks_label) for y_ticks_label in y_ticks_labels]
                y_ticks_labels[-1] = "12.5"

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

        arguments += list(
            itertools.product(*[
                r_num_elements, s_num_elements, gpus, sort_algorithm, join_algorithm, chunk_size,
                key_type, key_distribution, value_type, value_distribution, theta, sigma,
                r_sort, s_sort, materialize, ajb, K_x_vals, K_u_vals, K_v_vals, auto_tune
            ]))

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

        for rn in r_num_elements_list:
            # AJB auto-tune
            arguments += list(
                itertools.product(*[
                    [rn], [rn], gpus, sort_algorithm, join_algorithm, chunk_size,
                    key_type, key_distribution, value_type, value_distribution, theta, sigma,
                    r_sort, s_sort, materialize, [True], [0], [0], [0], [True]
                ]))
            # upstream baseline
            arguments += list(
                itertools.product(*[
                    [rn], [rn], gpus, sort_algorithm, join_algorithm, chunk_size,
                    key_type, key_distribution, value_type, value_distribution, theta, sigma,
                    r_sort, s_sort, materialize, [False], [0], [0], [0], [False]
                ]))

        experiments.append(Experiment(identifier, executable, parameters, arguments, columns, repetitions=3))
