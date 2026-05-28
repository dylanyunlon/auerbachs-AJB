# PROGRESS — Figure-Data Emitter

Author: dylanyunlon <dogechat@163.com>
Base commit: 34e9e99 (add data output design format example)

## Completed

- [x] Mapped the reference JSON schema from `data.zip`
  (leaf keys: `seed_i`, `mean`, `std`, `reported_final`; shared `steps` X axis).
- [x] Grep-verified the real measurement surface of the benchmark:
  - settings struct + CSV emit in `src/join_benchmark.cu` (lines 28-34, 123-136)
  - `TimeDurations` singleton in `src/common/profile_utilities.cuh`
  - `input_columns` / `output_columns` in `scripts/experiment_specifications.py`
- [x] Implemented `scripts/figure_data_emitter.py` (additive, no source edits):
  - `discover(csv)` — lists real x / series / y options in a CSV
  - `aggregate(csv, x, series, y)` — emits demo-schema JSON
  - CLI with `--discover`, `-x/-s/-y`, `-o`
- [x] Validated on real CPU: schema-identical to
  `gradient_norm_24k_data.json` (six leaf keys, aligned steps,
  equal-length seed/mean/std curves).
- [x] `git status` confirms a single new file; no source modified,
  so no upstream diff to reconcile and no regression surface in existing code.

## Verification snapshot

```
leaf keys match demo: True
seed curve len == steps len: True
mean len == steps len: True
metadata.source: <csv_name>@<git_sha>   # honest provenance, not a PNG
```

## Not done (requires hardware)

- [ ] Run the real CUDA benchmark on A6000×2 + H100, capture CSV,
  emit actual figure JSON, diff vs demo format.
  Blocked: no GPU in the build sandbox. Needs the production server.

## Critiques recorded

- User-angle: single-seed std → 0.0 (not NaN); ragged x-grids → `None`
  (no misalignment); column drift → explicit `KeyError`; `num_matches`
  must not be co-plotted with durations.
- System-angle: upstream `termcolor` ANSI codes contaminate piped CSV
  (flagged, not patched — deliberate, would touch the join kernel);
  full-CSV in-memory load; emitter is CPU-only and never touches the GPU.
