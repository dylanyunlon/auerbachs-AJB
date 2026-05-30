import argparse
import pathlib
import pytictoc
import subprocess
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

    timer = pytictoc.TicToc()
    elapsed_times = []

    for experiment in experiments:
        print_info(InfoType.RUN, experiment.identifier)

        is_success = True
        timer.tic()

        output_path = experiment_path / f"{experiment.executable}_{experiment.identifier}.csv"

        with output_path.open("a") as output_file:
            quoted_columns = [f"\"{column}\"" for column in experiment.columns]

            output_file.write(f"{','.join(quoted_columns)}\n")

        for index, arguments in enumerate(experiment.arguments):
            command = f"numactl -m 0 {executable_path / experiment.executable}"

            for (parameter, argument) in zip(experiment.parameters, arguments):
                if isinstance(argument, bool):
                    command += f" {parameter}" if argument else f""
                else:
                    command += f" {parameter} {argument}"

            for repetition in range(experiment.repetitions):
                output = subprocess.run(command,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL,
                                        universal_newlines=True,
                                        shell=True)

                if output.returncode == 0:
                    with output_path.open("a") as output_file:
                        output_file.write(output.stdout)
                else:
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

        print_info(InfoType.PASSED if is_success else InfoType.FAILED,
                   f"{experiment.identifier} ({elapsed_times[-1]}s)")
        print_info(InfoType.SINGLE_SEPARATOR)

    print_info(InfoType.DOUBLE_SEPARATOR,
               f"{len(experiments)} experiment{'s' if len(experiments) > 1 else ''} ran ({sum(elapsed_times)}s)")
