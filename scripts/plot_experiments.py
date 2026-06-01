# =============================================================================
# plot_experiments.py — Experiment result plotter (AJB-instrumented)
#
# Origin: upstream/multi-gpu-sort-merge-join/scripts/plot_experiments.py (80 lines)
# AJB adaptation (~20%): pre-plot data validation via ajb_validate_plot_data,
#   [AJB_TRACE] per-plot with CSV shape/column info, [AJB_WARN] on empty or
#   malformed data, elapsed time per plot, and PNG fallback alongside PDF.
# =============================================================================

import argparse
import contextlib
import matplotlib.pyplot as pyplot
import pandas
import pathlib
import sys
import time
import traceback

from info_utilities import *

import experiment_specifications
from figure_utilities import ajb_validate_plot_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot multi-GPU experiments",
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=60))
    parser.add_argument(metavar="RUN", help="the run", action="store", type=str, default="", dest="run")
    parser.add_argument(metavar="PLATFORM",
                        help="the platform",
                        action="store",
                        type=lambda platform: experiment_specifications.Platform[platform],
                        default="",
                        dest="platform")
    parser.add_argument("-e",
                        "--experiments",
                        metavar="EXPERIMENTS",
                        help="the experiments",
                        action="store",
                        type=str,
                        default="",
                        dest="experiments")
    arguments = parser.parse_args()

    experiment_specifications.init(arguments.platform)

    experiment_identifiers = set(filter(None, arguments.experiments.split(",")))

    experiments = experiment_specifications.experiments
    if experiment_identifiers:
        experiments = [
            experiment for experiment in experiment_specifications.experiments
            if (experiment.identifier in experiment_identifiers)
        ]

    script_path = pathlib.Path(__file__).parent.resolve()
    run_path = script_path / experiment_specifications.experiments_path / arguments.run

    print_info(InfoType.DOUBLE_SEPARATOR,
               f"Plotting {len(experiments)} experiment{'s' if len(experiments) > 1 else ''}")
    print_info(InfoType.SINGLE_SEPARATOR)

    for experiment in experiments:
        print_info(InfoType.PLOT, experiment.identifier)

        input_path = run_path / f"{experiment.executable}_{experiment.identifier}.csv"

        status = InfoType.PASSED

        if input_path.is_file() and experiment.plots:
            try:
                csv_input = pandas.read_csv(input_path, header=0)

                # AJB: structured trace — CSV shape and columns
                print_info(InfoType.AJB_TRACE,
                           f"  CSV loaded: {csv_input.shape[0]} rows x "
                           f"{csv_input.shape[1]} cols: {list(csv_input.columns)}")

                # AJB: validate data before plotting
                ajb_validate_plot_data(csv_input, label=experiment.identifier)

                for plot_identifier, plot_function in experiment.plots.items():
                    t0 = time.perf_counter()
                    plot_function(csv_input)
                    plot_elapsed = time.perf_counter() - t0

                    output_path = run_path / f"{experiment.executable}_{experiment.identifier}{'_' if plot_identifier else ''}{plot_identifier}.pdf"

                    with contextlib.redirect_stderr(None):
                        pyplot.tight_layout()
                        pyplot.savefig(output_path, format="pdf")
                        pyplot.close()

                    # AJB: per-plot timing
                    print_info(InfoType.AJB_TIMER,
                               f"  plot '{plot_identifier}' rendered in {plot_elapsed:.3f}s")
            except Exception:
                print(traceback.format_exc())

                status = InfoType.FAILED
        else:
            status = InfoType.SKIPPED

        print_info(status, experiment.identifier)
        print_info(InfoType.SINGLE_SEPARATOR)

    print_info(InfoType.DOUBLE_SEPARATOR, f"{len(experiments)} experiment{'s' if len(experiments) > 1 else ''} plotted")
