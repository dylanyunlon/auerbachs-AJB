# =============================================================================
# plot_experiments.py — Experiment result plotter (AJB-instrumented)
#
# Origin: upstream/multi-gpu-sort-merge-join/scripts/plot_experiments.py (80 lines)
# AJB adaptation (~20%):
#   - batch progress counter with ETA (upstream has no progress reporting)
#   - per-plot memory-delta tracking via tracemalloc when available
#   - PNG fallback alongside PDF so headless CI gets raster output
#   - aggregated pass/fail/skip summary at the end with per-plot timings
#   - data shape + column dump before each plot (replaces blind read_csv)
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

# AJB: optional memory tracking — tracemalloc is stdlib but has overhead,
# so we only import it when AJB_TRACE_MEM is set in the environment
_trace_mem = False
try:
    import os as _os
    if _os.environ.get("AJB_TRACE_MEM"):
        import tracemalloc
        tracemalloc.start()
        _trace_mem = True
except Exception:
    pass


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

    total_exp = len(experiments)
    print_info(InfoType.DOUBLE_SEPARATOR,
               f"Plotting {total_exp} experiment{'s' if total_exp > 1 else ''}")
    print_info(InfoType.SINGLE_SEPARATOR)

    # AJB: batch-level accumulators for the summary
    t_batch_start = time.monotonic()
    pass_ct, fail_ct, skip_ct = 0, 0, 0
    plot_timings = []

    for exp_idx, experiment in enumerate(experiments):
        print_info(InfoType.PLOT, experiment.identifier)
        # AJB: progress fraction — upstream prints nothing between start and end
        print_info(InfoType.AJB_TRACE,
                   f"  [{exp_idx+1}/{total_exp}] {experiment.identifier}")

        input_path = run_path / f"{experiment.executable}_{experiment.identifier}.csv"

        status = InfoType.PASSED

        if input_path.is_file() and experiment.plots:
            try:
                csv_input = pandas.read_csv(input_path, header=0)

                # AJB: dump data shape + first few columns so you know what the
                # plot function is receiving without opening the CSV by hand
                n_rows, n_cols = csv_input.shape
                col_preview = list(csv_input.columns[:8])
                if n_cols > 8:
                    col_preview.append(f"...+{n_cols - 8} more")
                print_info(InfoType.AJB_TRACE,
                           f"  CSV: {n_rows}×{n_cols} cols={col_preview}")

                ajb_validate_plot_data(csv_input, label=experiment.identifier)

                # AJB: memory snapshot before the plot batch if tracking is on
                mem_before = tracemalloc.get_traced_memory()[0] if _trace_mem else 0

                for plot_identifier, plot_function in experiment.plots.items():
                    t0 = time.perf_counter()
                    plot_function(csv_input)
                    elapsed = time.perf_counter() - t0
                    plot_timings.append((experiment.identifier, plot_identifier, elapsed))

                    output_path = run_path / f"{experiment.executable}_{experiment.identifier}{'_' if plot_identifier else ''}{plot_identifier}.pdf"

                    with contextlib.redirect_stderr(None):
                        pyplot.tight_layout()
                        pyplot.savefig(output_path, format="pdf")
                        # AJB: also emit PNG for headless/CI preview
                        png_path = output_path.with_suffix(".png")
                        try:
                            pyplot.savefig(png_path, format="png", dpi=120)
                        except Exception:
                            pass
                        pyplot.close()

                    print_info(InfoType.AJB_TIMER,
                               f"  plot '{plot_identifier}' {elapsed:.3f}s")

                # AJB: memory delta
                if _trace_mem:
                    mem_after = tracemalloc.get_traced_memory()[0]
                    delta_mb = (mem_after - mem_before) / (1024 * 1024)
                    print_info(InfoType.AJB_TRACE,
                               f"  mem delta: {delta_mb:+.1f} MB")

            except Exception:
                print(traceback.format_exc())
                status = InfoType.FAILED
        else:
            status = InfoType.SKIPPED

        if status == InfoType.PASSED:
            pass_ct += 1
        elif status == InfoType.FAILED:
            fail_ct += 1
        else:
            skip_ct += 1

        print_info(status, experiment.identifier)
        print_info(InfoType.SINGLE_SEPARATOR)

    # AJB: batch summary with timing breakdown — upstream only prints a count
    batch_elapsed = time.monotonic() - t_batch_start
    print_info(InfoType.AJB_TRACE,
               f"SUMMARY: {pass_ct} passed, {fail_ct} failed, "
               f"{skip_ct} skipped in {batch_elapsed:.1f}s")
    if plot_timings:
        slowest = max(plot_timings, key=lambda t: t[2])
        print_info(InfoType.AJB_TRACE,
                   f"  slowest: {slowest[0]}/{slowest[1]} = {slowest[2]:.3f}s")
    print_info(InfoType.DOUBLE_SEPARATOR, f"{total_exp} experiment{'s' if total_exp > 1 else ''} plotted")
