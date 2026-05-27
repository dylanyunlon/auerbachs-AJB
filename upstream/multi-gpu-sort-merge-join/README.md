# Efficiently Joining Large Relations on Multi-GPU Systems

This repository contains the source code for our VLDB '25 paper.

Growing data volumes present a mounting challenge to relational joins. GPUs have gained widespread adoption as database accelerators for operators such as joins due to their high instruction throughput and memory bandwidth. Most published GPU-accelerated joins are single-GPU algorithms that do not leverage modern multi-GPU platforms effectively. The few proposed multi-GPU algorithms either fail to exploit the high-speed P2P interconnects between the GPUs or to handle large out-of-core data natively. In this paper, we present a heterogeneous multi-GPU sort-merge join that overcomes both limitations. It is composed of a merge- or radix partitioning-based P2P-enabled multi-GPU sort phase, a parallel CPU-based multiway merge phase, and a hybrid join phase that combines a CPU merge path partition with a binary search-based multi-GPU join strategy. We evaluate our novel multi-GPU join on two platforms with fast NVLink- and NVSwitch-based P2P interconnects. We show that our join outperforms state-of-the-art CPU and GPU baselines regardless of the workload. It outperforms parallel CPU sort-merge and radix-hash joins up to 15.2x and 5.5x, respectively. Compared to non-P2P-enabled multi-GPU joins, it achieves speedups of 8.7x (sort-merge) and 2.5x (hybrid-radix). We measure that our join's hybrid join phase with overlapped copy and compute operations contributes as little as 22% to its end-to-end runtime. If the input relations are pre-sorted, it is up to 14.4x faster than the hybrid-radix join. Our join scales well with the number of GPUs and benefits from data skew with as much as 12% shorter join durations.

## Table of Contents
Directory | Description
----------|------------
src/ | Sources of the multi-GPU join algorithm and benchmarks.
scripts/ | Scripts to run and plot the benchmarks.

## Useful Commands
### Initializing the Project
`git submodule update --init --recursive`

### Building the Project
`./build.sh`

### Running the Benchmarks
`python3 scripts/run_experiments.py PLATFORM`

This creates an `experiments` folder and places the benchmark results for the `PLATFORM` into a subfolder, named after the current date/time (e.g., `2025_02_25_23_59_59`).

### Plotting the Graphs
`python3 scripts/plot_experiments.py RUN PLATFORM`

This creates `.pdf` plots in the `RUN` folder (e.g., `2025_02_25_23_59_59`).
