# PLAN — Figure-Data Emitter for Multi-GPU Sort-Merge Join

## Objective

Produce publishable, plottable result data from the join benchmark in the
same per-method / per-seed / mean±std schema as the reference figure-data
files committed in `data.zip` (e.g. `gradient_norm_24k_data.json`,
`ppl_vs_time_1B_30k_data.json`), but with **honest provenance** — numbers
measured by our benchmark, not digitized from a paper figure.

## Scope decisions

- **Additive only.** No edits to the CUDA benchmark, the existing
  `experiment_specifications.py`, `run_experiments.py`, or
  `plot_experiments.py`. We only *read* the CSV they emit.
- **Real fields only.** Every X / series / Y dimension maps to a symbol
  that exists in the codebase (verified by grep), not invented:
  - X candidates: `num_elements`, `num_threads`, `chunk_count`
    (from `experiment_specifications.input_columns`)
  - Series ("method"): `join_algorithm` / `sort_algorithm`
    (from `join_benchmark.cu` settings emit, lines 123-131)
  - Y metrics: `sort_phase`, `merge_phase`, `join_phase`, `total`
    (the `TimeDurations` tags emitted at `join_benchmark.cu` lines 132-136)
    plus `num_matches` (output cardinality)

## Schema target (matches demo files)

```
{ "metadata": {...}, "steps": [x...],
  "methods": { "<series>": { "seed_i": [...], "mean": [...],
                             "std": [...], "reported_final": <float|null> } } }
```

## Tasks

1. Map real measurement symbols (DONE — grep-verified).
2. Build `scripts/figure_data_emitter.py` with `aggregate()`, `discover()`,
   and a CLI (DONE).
3. Validate on real CPU against a schema-faithful CSV; assert leaf-key
   parity with the demo file (DONE).
4. Critique (user-bug angle + system angle) and record caveats (DONE).
5. **Open / hardware-dependent:** run the real CUDA benchmark on the
   A6000×2 + H100 server, pipe to CSV, emit the actual figure JSON,
   diff against demo format. Requires GPU hardware not present in the
   build sandbox.

## Known caveats (carried forward)

- Benchmark prints durations wrapped in `termcolor` ANSI codes; piping
  stdout directly to CSV will embed escape codes. Recommend guarding color
  with `isatty()` at the C++ side (NOT done here — would touch the join
  kernel; deferred to a deliberate decision).
- `num_matches` is a count, not a duration — do not co-plot on a time Y axis.
- `aggregate()` loads the full CSV into memory; fine for summary CSVs,
  not for raw per-tuple dumps.
