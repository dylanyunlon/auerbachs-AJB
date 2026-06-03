# Auerbachs — Adaptive Hash Join on Mixed-Bandwidth GPU Interconnects

> *"Uns ist ganz kannibalisch wohl, als wie fünfhundert Säuen!"*
> — Auerbachs Keller, Faust I

Data partitions from different bandwidth tiers (NVLink vs PCIe) converge
for adaptive hash joins on A6000 ×2 + H100 ×1 + CPU.

## Upstream

| Directory | Origin | Role |
|-----------|--------|------|
| `upstream/multi-gpu-sort-merge-join` | [hpides/multi-gpu-sort-merge-join](https://github.com/hpides/multi-gpu-sort-merge-join) | Multi-GPU sort-merge join (VLDB '25) |
| `upstream/joinrenum` | [Chen-Py/JoinREnum](https://github.com/Chen-Py/JoinREnum) | Random-order enumeration for joins |

## Quick Start

### Prerequisites

- CUDA toolkit ≥ 11.7 (sm_70/80/90)
- CMake ≥ 3.18
- OpenMP
- libglpk-dev (for joinrenum tests)
- Python 3.8+ with matplotlib, pandas (for plotting/analysis)

### Initializing the Project

```bash
git submodule update --init --recursive
```

### Building

```bash
./build.sh          # Release build
./build.sh Debug    # Debug build with CUDA -G
```

For verbose output: `AJB_VERBOSE=1 ./build.sh`

Build artifacts land in `build/`. The build script prints `[AJB_BUILD]`
markers with elapsed times so you can spot configure vs compile bottlenecks.

### Running GPU Benchmarks

```bash
python3 scripts/run_experiments.py PLATFORM
```

Results go into `experiments/<timestamp>/`. Each experiment emits
`[AJB_TRACE]` markers to stderr; capture with:

```bash
python3 scripts/run_experiments.py dgx 2> run.stderr.log
grep '\[AJB_TIMER\]' run.stderr.log     # phase timings
grep '\[AJB_WARN\]'  run.stderr.log     # anomalies
```

### Running JoinREnum CPU Tests

```bash
# All _full tests (upstream logic + AJB breakpoint diagnostics)
bash scripts/debug/build_and_test.sh

# Single test with AddressSanitizer
bash scripts/debug/debug_enumerator.sh

# Perf + flamegraph profiling
bash scripts/debug/run_perf_profile.sh build/tests/ajb_test_index_full

# All _upstream algorithm-variant tests
bash scripts/debug/run_upstream_tests.sh --list    # see targets
bash scripts/debug/run_upstream_tests.sh test_count_oracle_upstream
```

### Plotting

```bash
python3 scripts/plot_experiments.py RUN PLATFORM
```

Produces `.pdf` plots in the `RUN` folder. Pre-plot data validation
via `ajb_validate_plot_data` catches NaN/Inf/negative values before
matplotlib sees them.

## Debug Output Tags

All AJB instrumentation uses structured stderr tags. Filter by prefix:

```
[AJB]            Test name + verdict (PASSED / FAILED)
[AJB_TRACE]      Step-by-step execution trace (printf breakpoints)
[AJB_STATE]      Data structure dumps (bucket contents, tree shape)
[AJB_MEM]        Memory snapshots (RSS, VSize, peak, delta)
[AJB_TIMER]      Phase timings (build / query / sort / join)
[AJB_BP]         Named breakpoints for interactive debugging
[AJB_WARN]       Non-fatal warnings (skew, fallback paths)
[AJB_FAIL]       Fatal errors / assertion failures
[AJB_BUILD]      Build system markers (configure / compile / link)
[AJB_FORMAT]     Code formatter progress
```

Parse all tags at once:

```bash
./some_test 2>&1 | python3 scripts/debug/parse_ajb_trace.py
```

## Directory Layout

```
src/
  ajb_join/           AJB-specific modules (scheduler, detector, etc.)
  common/             GPU utilities (allocators, profiling, data gen)
  hybrid_sort/        Radix + merge sort kernels
  merge_join/         Multi-GPU merge join kernels
  joinrenum/          JoinREnum headers + tests/ + tools/ + db/
  *_benchmark.cu      Benchmark drivers (GPU + CPU)
scripts/
  debug/              Build, test, profile, and trace scripts
  *.py                Experiment run + plot + figure pipeline
paper/                NeurIPS 2026 LaTeX source
upstream/             Unmodified upstream sources for diffing
```

## Formatting

```bash
./format.sh    # clang-format (C/CUDA) + yapf (Python)
```
