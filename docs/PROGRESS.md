# PROGRESS — AJB (Adaptive Join on Mixed-Bandwidth Interconnects)

Author: dylanyunlon <dogechat@163.com>

## Development Timeline — Claude Session Tracker

Each Claude session operates on a milestone range (Mxxx-Mxxx).
The system grows incrementally: each Claude picks up where the last left off.

```
Session   Milestone     Commit                          Scope
─────────────────────────────────────────────────────────────────────────────
Claude-1  M001-M025     fd08dc3  paper                  AJB title, abstract, Sections 1-2
Claude-2  M026-M050     faee0df  paper                  Section 3 Correctness & Balance
Claude-3  M051-M075     e589759  paper                  Section 4 Experimental Design
Claude-4  M076-M100     16cc8ef  paper                  Section 5 RQ1-RQ4, Figures, Tables
Claude-5  (skipped)     —                               —
Claude-6  M126-M150     f612f0d  paper                  Sections 6-7, final assembly
Claude-7  M001-M150     9a96584  code                   Full upstream port + AJB integration
Claude-8  M151-M175     d27c9fb  code                   JoinREnum test/tool port + debug
─────────────────────────────────────────────────────────────────────────────
```

## What's Done (through Claude-8, M175)

### Paper (complete draft)
- [x] Sections 1-7 fully written in `paper/ajb_reconstructed.tex`
- [x] NeurIPS 2026 style, self-contained compile
- [x] Figures 2-5, Tables 1-2 (data-driven, pending real GPU runs)

### Code — Upstream Integration (complete)
- [x] `upstream/multi-gpu-sort-merge-join` → `src/common/`, `src/hybrid_sort/`, `src/merge_join/`
- [x] `upstream/joinrenum` → `src/joinrenum/` (all 18 headers + 2 existing tests)
- [x] AJB patches: `debug_utilities.cuh` (+156 lines), `profile_utilities.cuh` (+42 lines)
- [x] AJB patches: `Index.hpp` (+48 lines — sample/randomAccess for SkewDetector)

### Code — AJB Modules (complete, needs GPU testing)
- [x] `src/ajb_join/tier_transfer_scheduler.cuh` (406 lines) — K_x/K_u/K_v cadence
- [x] `src/ajb_join/ajb_merge_join.cuh` (382 lines) — adaptive merge pipeline
- [x] `src/ajb_join/ajb_hybrid_sort.cuh` (285 lines) — WSD chunk schedule
- [x] `src/ajb_join/skew_detector.cuh` (214 lines) — key distribution analysis
- [x] `src/ajb_benchmark.cu` (548 lines) — unified benchmark driver
- [x] `src/joinrenum/ajb_renum_adapter.hpp` (341 lines) — REnum↔GPU bridge

### Code — Test & Debug Infrastructure (Claude-8, NEW)
- [x] 7 test harnesses in `src/joinrenum/tests/` (812 lines total)
- [x] 4 tool programs in `src/joinrenum/tools/` (278 lines total)
- [x] 8 debug scripts in `scripts/debug/` (689 lines total)
- [x] Unified [AJB_*] tag system + `parse_ajb_trace.py` parser
- [x] CMakeLists.txt: `ajb_joinrenum_tests` convenience target

### Code — Experiment Infrastructure
- [x] `scripts/experiment_specifications.py` — AJB experiments (cadence_sweep, vs_upstream, skew_sensitivity)
- [x] `scripts/figure_data_emitter.py` — CSV → JSON schema converter

## What's Next — Planned Milestones

```
Future    Milestone     Owner       Scope
─────────────────────────────────────────────────────────────────────────────
Claude-9  M176-M200     next        GPU build validation: CMake on CUDA host,
                                    fix any compilation errors, verify all
                                    targets build cleanly on sm_70/80/90

Claude-10 M201-M225     next        Benchmark dry-run: run ajb_benchmark on
                                    2-GPU setup, capture first CSV output,
                                    validate figure_data_emitter pipeline

Claude-11 M226-M250     next        Experiment execution: run full experiment
                                    suite (cadence_sweep, vs_upstream,
                                    skew_sensitivity), collect result CSVs

Claude-12 M251-M275     next        Data analysis: generate figure JSONs,
                                    produce final plots, fill in paper
                                    tables with real measured numbers

Claude-13 M276-M300     next        Paper revision: update Sections 5-6 with
                                    real data, revise claims against actual
                                    speedup numbers, camera-ready polish

Claude-14 M301-M325     next        Robustness: edge cases (1-GPU, 8-GPU,
                                    extreme skew θ=0.99), error handling,
                                    graceful degradation tests
─────────────────────────────────────────────────────────────────────────────
```

## File Inventory (post Claude-8)

```
src/                         13,976 lines  (72 source files)
  ajb_join/                   1,287 lines  (4 AJB-specific modules)
  common/                     2,xxx lines  (16 utilities, 2 AJB-patched)
  hybrid_sort/                x,xxx lines  (12 sort kernels, upstream)
  merge_join/                   xxx lines  (4 join kernels, upstream)
  joinrenum/                  7,xxx lines  (18 headers + 7 tests + 4 tools + 4 harnesses)
  benchmarks (*.cu)           1,xxx lines  (7 benchmark drivers)

scripts/                      2,509 lines  (14 files)
  debug/                        689 lines  (8 new debug scripts)
  experiment/analysis           1,820 lines (6 upstream + AJB extensions)

paper/                          tex + sty  (NeurIPS 2026 format)
docs/                          PLAN.md + PROGRESS.md
```

## Known Blockers

1. **No GPU in sandbox** — all CUDA code is compile-verified by structure
   but not runtime-tested. First GPU run is Claude-10 scope.
2. **GLPK dependency** — joinrenum tests need `libglpk-dev`. CMake
   gracefully skips if not found.
3. **ANSI color in CSV** — upstream `termcolor` contaminates piped output.
   Flagged but not patched (would touch upstream kernels).
