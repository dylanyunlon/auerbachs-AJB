# =============================================================================
# run_experiments.py — Multi-GPU experiment runner (AJB-instrumented)
#
# Origin: upstream/multi-gpu-sort-merge-join/scripts/run_experiments.py (112 lines)
# AJB adaptation (~20%): per-experiment [AJB_TRACE] start/end tags, stderr
#   capture to .stderr.log alongside CSV, per-repetition progress reporting,
#   structured failure dump with the failing command + return code, and
#   summary statistics (total time, pass/fail count) for parse_ajb_trace.py.
# =============================================================================

import argparse
import pathlib
import pytictoc
import subprocess
import sys
import time

from info_utilities import *

import experiment_specifications

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run multi-GPU experiments",
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=60))
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
    executable_path = script_path / experiment_specifications.executables_path
    experiment_path = script_path / experiment_specifications.experiments_path / time.strftime("%Y_%m_%d_%H_%M_%S")

    experiment_path.mkdir(parents=True, exist_ok=True)

    print_info(InfoType.DOUBLE_SEPARATOR, f"Running {len(experiments)} experiment{'s' if len(experiments) > 1 else ''}")
    print_info(InfoType.SINGLE_SEPARATOR)

    # AJB: track pass/fail counts for summary
    pass_count = 0
    fail_count = 0

    timer = pytictoc.TicToc()
    elapsed_times = []

    for experiment in experiments:
        print_info(InfoType.RUN, experiment.identifier)
        # AJB: structured trace for experiment start
        print_info(InfoType.AJB_TRACE,
                   f"experiment={experiment.identifier} "
                   f"configs={len(experiment.arguments)} "
                   f"reps={experiment.repetitions}")

        is_success = True
        timer.tic()

        output_path = experiment_path / f"{experiment.executable}_{experiment.identifier}.csv"
        # AJB: capture stderr to separate log for AJB trace parsing
        stderr_path = experiment_path / f"{experiment.executable}_{experiment.identifier}.stderr.log"

        with output_path.open("a") as output_file:
            quoted_columns = [f"\"{column}\"" for column in experiment.columns]

            output_file.write(f"{','.join(quoted_columns)}\n")

        for index, arguments in enumerate(experiment.arguments):
            # AJB: 用列表构建命令代替字符串拼接——更安全, 无shell injection
            cmd_parts = ["numactl", "-m", "0",
                         str(executable_path / experiment.executable)]

            for (parameter, argument) in zip(experiment.parameters, arguments):
                if isinstance(argument, bool):
                    if argument:
                        cmd_parts.append(parameter)
                else:
                    cmd_parts.extend([parameter, str(argument)])

            command = " ".join(cmd_parts)

            for repetition in range(experiment.repetitions):
                # AJB: per-repetition进度 + ETA估算
                done_reps = index * experiment.repetitions + repetition
                total_reps = len(experiment.arguments) * experiment.repetitions
                if elapsed_times and done_reps > 0:
                    avg_per_rep = sum(elapsed_times) / max(len(elapsed_times), 1) / max(experiment.repetitions, 1)
                    remaining = (total_reps - done_reps) * avg_per_rep
                    eta_str = f" ETA={remaining:.0f}s"
                else:
                    eta_str = ""
                print_info(InfoType.AJB_TRACE,
                           f"  config {index+1}/{len(experiment.arguments)} "
                           f"rep {repetition+1}/{experiment.repetitions}{eta_str}")

                output = subprocess.run(command,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,   # AJB: capture stderr
                                        universal_newlines=True,
                                        shell=True)

                if output.returncode == 0:
                    with output_path.open("a") as output_file:
                        output_file.write(output.stdout)
                    # AJB: append stderr (contains [AJB_*] tags) to log
                    if output.stderr:
                        with stderr_path.open("a") as sf:
                            sf.write(f"--- config={index} rep={repetition} ---\n")
                            sf.write(output.stderr)
                else:
                    # AJB: structured failure dump
                    print_info(InfoType.AJB_FAIL,
                               f"rc={output.returncode} cmd={command}")
                    if output.stderr:
                        for line in output.stderr.strip().split('\n')[-5:]:
                            print_info(InfoType.AJB_FAIL, f"  stderr: {line}")
                    print(f"[ERROR] {command}: subprocess.run failed.")
                    is_success = False

            for profiler in experiment.profilers:
                if profiler == "nsys":
                    profiler_output_path = experiment_path / f"{experiment.executable}_{experiment.identifier}_{index}.nsys-rep"

                    profiler_command = f"nsys profile -o {profiler_output_path} {command}"

                    output = subprocess.run(profiler_command,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL,
                                            universal_newlines=True,
                                            shell=True)

                    if output.returncode != 0:
                        print(f"[ERROR] {profiler_command}: subprocess.run failed.")
                        is_success = False

        elapsed_times.append(round(timer.tocvalue(), 2))

        if is_success:
            pass_count += 1
        else:
            fail_count += 1

        print_info(InfoType.PASSED if is_success else InfoType.FAILED,
                   f"{experiment.identifier} ({elapsed_times[-1]}s)")
        print_info(InfoType.SINGLE_SEPARATOR)

    # AJB: structured summary for parse_ajb_trace.py
    total_time = sum(elapsed_times)
    print_info(InfoType.AJB_TIMER, f"total_experiment_time={total_time:.2f}s")
    print_info(InfoType.AJB_TRACE,
               f"SUMMARY: {pass_count} passed, {fail_count} failed, "
               f"{len(experiments)} total in {total_time:.1f}s")

    print_info(InfoType.DOUBLE_SEPARATOR,
               f"{len(experiments)} experiment{'s' if len(experiments) > 1 else ''} ran ({total_time}s)")
