// =============================================================================
// ajb_benchmark.cu — AJB (Adaptive Join on Mixed-Bandwidth) Benchmark Driver
//
// This is the AJB equivalent of the upstream join_benchmark.cu.
// It integrates all AJB components:
//   - TierTransferScheduler (bandwidth-tier-aware transfer scheduling)
//   - AJBMergeJoin (adaptive merge join pipeline)
//   - AJBHybridSort (WSD chunk-size schedule + tier-aware assignment)
//   - SkewDetector (key-distribution analysis for cadence auto-tuning)
//   - Extended debug_utilities (breakpoint macros, state dumps)
//
// ~80% of the CLI, relation generation, and output format is retained from
// upstream join_benchmark.cu. ~20% is new: AJB-specific options, the
// scheduler integration, and the debug/trace output.
//
// Build:  nvcc -o ajb_benchmark src/ajb_benchmark.cu [cuda flags]
// Or CPU-only test:  g++ -x c++ -DCPU_ONLY_TEST -o ajb_test src/ajb_benchmark.cu
// =============================================================================

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

// Conditional includes for CUDA vs CPU-only testing
#ifndef CPU_ONLY_TEST
  #include <csv.hpp>
  #include <cxxopts.hpp>
  #include <tabulate/table.hpp>
  #include <termcolor/termcolor.hpp>
#endif

#include "common/config_utilities.cuh"
#include "common/data_generator.cuh"
#include "common/debug_utilities.cuh"
#include "common/device_allocator.cuh"
#include "common/host_allocator.cuh"
#include "common/math_utilities.cuh"
#include "common/options_limits.cuh"
#include "common/parallel_algorithms.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/relation_generator.cuh"
#include "common/stream_pool.cuh"

// Upstream sort/join (retained for the non-AJB baseline)
#include "hybrid_sort/hybrid_sort.cuh"
#include "merge_join/merge_join.cuh"

// AJB-specific modules
#include "ajb_join/tier_transfer_scheduler.cuh"
#include "ajb_join/ajb_merge_join.cuh"
#include "ajb_join/ajb_hybrid_sort.cuh"
#include "ajb_join/skew_detector.cuh"

// =============================================================================
// Welford online statistics accumulator for benchmark runs
// =============================================================================
struct WelfordAccumulator {
  size_t count_ = 0;
  double mean_ = 0.0;
  double M2_ = 0.0;

  void Update(double x) {
    ++count_;
    double delta = x - mean_;
    mean_ += delta / static_cast<double>(count_);
    double delta2 = x - mean_;
    M2_ += delta * delta2;
  }

  double Mean() const { return mean_; }
  double Variance() const { return count_ < 2 ? 0.0 : M2_ / (count_ - 1); }
  double Stddev() const { return std::sqrt(Variance()); }

  void DumpState(const char* label) const {
    fprintf(stderr, "[AJB_STATE] WelfordAccumulator<%s> n=%zu mean=%.6f var=%.6f stddev=%.6f\n",
            label, count_, Mean(), Variance(), Stddev());
  }
};

// =============================================================================
// Sort verification: check sorted order, report first N inversions
// =============================================================================
template <typename T>
size_t VerifySortedOrder(const T* data, size_t n, size_t max_report = 3) {
  size_t inversion_count = 0;
  for (size_t i = 1; i < n; ++i) {
    if (data[i] < data[i - 1]) {
      if (inversion_count < max_report) {
        fprintf(stderr, "[AJB_STATE] SortInversion at pos %zu: data[%zu]=%lu > data[%zu]=%lu\n",
                i, i - 1, static_cast<unsigned long>(data[i - 1]),
                i, static_cast<unsigned long>(data[i]));
      }
      ++inversion_count;
    }
  }
  if (inversion_count == 0) {
    fprintf(stderr, "[AJB_STATE] SortVerification PASSED: %zu elements in sorted order\n", n);
  } else {
    fprintf(stderr, "[AJB_STATE] SortVerification FAILED: %zu inversions in %zu elements\n",
            inversion_count, n);
  }
  return inversion_count;
}

// =============================================================================
// Gini coefficient for load-balance measurement
// =============================================================================
inline double ComputeGiniCoefficient(const std::vector<size_t>& chunks) {
  if (chunks.empty()) return 0.0;
  size_t n = chunks.size();
  double sum = 0.0;
  for (auto c : chunks) sum += static_cast<double>(c);
  if (sum == 0.0) return 0.0;
  double abs_diff_sum = 0.0;
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      abs_diff_sum += std::abs(static_cast<double>(chunks[i]) - static_cast<double>(chunks[j]));
  return abs_diff_sum / (2.0 * n * sum);
}

// =============================================================================
// Extended Settings (upstream Settings + AJB fields)
// =============================================================================
struct AJBSettings {
  // --- Upstream fields (unchanged) ---
  size_t r_num_elements;
  size_t s_num_elements;
  size_t num_threads;
  std::vector<int> gpus;
  std::string sort_algorithm;
  std::string join_algorithm;
  size_t chunk_size;
  std::string key_type;
  std::string key_distribution;
  std::string value_type;
  std::string value_distribution;
  uint32_t random_seed;
  uint32_t theta;
  uint32_t sigma;
  bool r_sort;
  bool s_sort;
  bool save;
  bool materialize;
  bool print;

  // --- AJB-specific fields ---
  bool use_ajb;              // false = upstream baseline, true = AJB adaptive
  size_t K_x;                // manual transfer period for build partitions
  size_t K_u;                // manual transfer period for merge-path boundaries
  size_t K_v;                // manual transfer period for materialization buffers
  bool auto_tune_cadence;    // auto-detect skew and set K_x/K_u/K_v
  size_t debug_interval;     // print debug snapshots every N chunk groups
  size_t num_runs;            // number of benchmark repetitions for Welford stats
  double warmup_ms;           // warmup phase duration in milliseconds
  std::string events_csv;    // path to export transfer events CSV
  std::string timing_csv;    // path to export timing breakdown CSV

  void DebugPrint() const {
    printf("\n");
    printf("================================================================\n");
    printf("[AJBSettings] Configuration Summary\n");
    printf("================================================================\n");
    printf("  R elements:     %zu\n", r_num_elements);
    printf("  S elements:     %zu\n", s_num_elements);
    printf("  Threads:        %zu\n", num_threads);
    printf("  GPUs:           [");
    for (size_t i = 0; i < gpus.size(); ++i)
      printf("%d%s", gpus[i], i + 1 < gpus.size() ? ", " : "");
    printf("]\n");
    printf("  Sort algo:      %s\n", sort_algorithm.c_str());
    printf("  Join algo:      %s\n", join_algorithm.c_str());
    printf("  Chunk size:     %zu\n", chunk_size);
    printf("  Key dist:       %s\n", key_distribution.c_str());
    printf("  Seed:           %u\n", random_seed);
    printf("  --- AJB ---\n");
    printf("  Mode:           %s\n", use_ajb ? "AJB-ADAPTIVE" : "BASELINE");
    printf("  Cadence:        K_x=%zu K_u=%zu K_v=%zu%s\n",
           K_x, K_u, K_v, auto_tune_cadence ? " (auto)" : " (manual)");
    printf("  Debug interval: %zu\n", debug_interval);
    printf("  Num runs:       %zu\n", num_runs);
    printf("  Warmup ms:      %.2f\n", warmup_ms);
    if (!events_csv.empty())
      printf("  Events CSV:     %s\n", events_csv.c_str());
    if (!timing_csv.empty())
      printf("  Timing CSV:     %s\n", timing_csv.c_str());
    printf("================================================================\n\n");
  }
};

// =============================================================================
// AJB Benchmark Runner
// =============================================================================
template <typename T, typename V>
void RunAJBBenchmark(AJBSettings& settings) {
  AJBTimer overall_timer("AJBBenchmark");

  // --- Step 1: Environment setup ---
  AJB_BREAKPOINT("Starting benchmark with R=%zu S=%zu GPUs=%zu",
                 settings.r_num_elements, settings.s_num_elements,
                 settings.gpus.size());
  AJBReportMemory("pre-setup");

  #ifndef CPU_ONLY_TEST
  ConfigureMultiProcess(settings.num_threads);
  ConfigurePeerAccess(settings.gpus);
  #endif

  // --- Step 2: Generate relations ---
  printf("\n[AJB] Generating relations...\n");
  PinnedVector<T> r_keys(settings.r_num_elements);
  PinnedVector<V> r_values(settings.r_num_elements);
  PinnedVector<T> s_keys(settings.s_num_elements);
  PinnedVector<V> s_values(settings.s_num_elements);

  // For CPU-only testing, generate simple test data
  #ifdef CPU_ONLY_TEST
  {
    AJBTimer gen_timer("data_generation");
    std::mt19937 rng(settings.random_seed);
    std::uniform_int_distribution<T> key_dist(0, settings.r_num_elements / 2);
    for (size_t i = 0; i < settings.r_num_elements; ++i) {
      r_keys[i] = key_dist(rng);
      r_values[i] = static_cast<V>(i);
    }
    for (size_t i = 0; i < settings.s_num_elements; ++i) {
      s_keys[i] = key_dist(rng);
      s_values[i] = static_cast<V>(i);
    }
    printf("  Generated %zu R-tuples and %zu S-tuples (CPU test mode)\n",
           settings.r_num_elements, settings.s_num_elements);
  }
  size_t expected_matches = 0;  // unknown in test mode
  #else
  size_t expected_matches;
  {
    AJBTimer gen_timer("data_generation");
    expected_matches = RelationGenerator::ComputeDistributions<T, V>(
        {r_keys.data(), r_values.data(), settings.r_num_elements},
        {s_keys.data(), s_values.data(), settings.s_num_elements},
        settings.num_threads, settings.key_distribution,
        settings.value_distribution, settings.random_seed,
        settings.theta, settings.sigma,
        settings.r_sort, settings.s_sort);
  }
  #endif

  AJB_BREAKPOINT("Relations generated, expected_matches=%zu", expected_matches);
  AJBDumpArray("R_keys", r_keys.data(), r_keys.size(), 8, 4);
  AJBDumpArray("S_keys", s_keys.data(), s_keys.size(), 8, 4);
  AJBReportMemory("post-generation");

  // --- Step 3: Skew detection ---
  fprintf(stderr, "[AJB_STATE] SkewDetection BEGIN sample_size=%zu\n",
          std::min<size_t>(10000, settings.r_num_elements));
  double effective_skew = 0.0;
  {
    AJBTimer skew_timer("skew_detection");
    auto [skew_r, skew_s] = DetectJoinSkew(
        r_keys.data(), r_keys.size(),
        s_keys.data(), s_keys.size(),
        std::min<size_t>(10000, settings.r_num_elements));
    effective_skew = std::max(skew_r.normalized, skew_s.normalized);
  }
  fprintf(stderr, "[AJB_STATE] SkewDetection END effective_skew=%.6f\n", effective_skew);

  // --- Step 4: Interconnect probing & cadence configuration ---
  fprintf(stderr, "[AJB_STATE] TopologyProbe BEGIN gpus=%zu\n", settings.gpus.size());
  std::vector<DeviceLink> topology;
  TransferCadence cadence;

  {
    AJBTimer topo_timer("topology_probe");
    topology = ProbeInterconnect(settings.gpus);
  }
  fprintf(stderr, "[AJB_STATE] TopologyProbe END links=%zu\n", topology.size());

  if (settings.auto_tune_cadence) {
    size_t est_chunk_groups = settings.r_num_elements /
        (settings.chunk_size > 0 ? settings.chunk_size : settings.r_num_elements);
    cadence = AutoTuneCadence(topology, est_chunk_groups, effective_skew);
  } else {
    cadence.K_x = settings.K_x;
    cadence.K_u = settings.K_u;
    cadence.K_v = settings.K_v;
    cadence.DebugPrint("ManualCadence");
  }

  // --- Step 5: Initialize scheduler ---
  fprintf(stderr, "[AJB_STATE] SchedulerInit BEGIN cadence K_x=%zu K_u=%zu K_v=%zu\n",
          cadence.K_x, cadence.K_u, cadence.K_v);
  TierTransferScheduler scheduler;
  scheduler.Initialize(settings.gpus, topology, cadence);
  fprintf(stderr, "[AJB_STATE] SchedulerInit END\n");

  // --- Step 6: Allocate device resources ---
  std::vector<HostAllocator> host_allocators(settings.gpus.size());
  std::vector<DeviceAllocator> device_allocators(settings.gpus.size());
  std::vector<StreamPool> stream_pools(settings.gpus.size());

  // --- Step 7: Sort phase ---
  if (settings.join_algorithm == "hybrid_sort_merge_join" ||
      settings.join_algorithm == "ajb_sort_merge_join") {

    const size_t temporary_num = std::max(settings.r_num_elements,
                                          settings.s_num_elements);
    PinnedVector<T> tmp_keys(temporary_num);
    PinnedVector<V> tmp_values(temporary_num);

    std::vector<SortChunkSnapshot> sort_snapshots;

    // --- Step 7: Sort phase with event-based pipeline overlap ---
    fprintf(stderr, "[AJB_STATE] SortPhase BEGIN r_elements=%zu s_elements=%zu gpus=%zu\n",
            settings.r_num_elements, settings.s_num_elements, settings.gpus.size());

    // Welford accumulators for multi-run timing statistics
    WelfordAccumulator welford_sort, welford_merge, welford_join, welford_total;

    if (settings.sort_algorithm == "gnu_parallel_sort") {
      AJBTimer sort_timer("gnu_parallel_sort");
      TimeScope ts("sort_phase");

      fprintf(stderr, "[AJB_STATE] gnu_parallel_sort: starting R-sort (%zu elements)\n",
              settings.r_num_elements);
      if (!settings.r_sort) {
        printf("[AJB] Sorting R with gnu_parallel_sort...\n");
        ParallelSortPairs<T, V>(r_keys, r_values);
      }

      fprintf(stderr, "[AJB_STATE] gnu_parallel_sort: starting S-sort (%zu elements)\n",
              settings.s_num_elements);
      if (!settings.s_sort) {
        printf("[AJB] Sorting S with gnu_parallel_sort...\n");
        ParallelSortPairs<T, V>(s_keys, s_values);
      }
    } else if (settings.use_ajb) {
      // AJB adaptive hybrid sort with WSD schedule
      if (settings.sort_algorithm == "hybrid_merge_sort" ||
          settings.sort_algorithm == "hybrid_radix_sort") {
        printf("[AJB] Using AJBHybridSort with WSD schedule\n");

        if (!settings.r_sort) {
          if (settings.sort_algorithm == "hybrid_merge_sort") {
            AJBHybridSort<T, V, HybridSortKernel::kMerge>(
                r_keys, r_values, tmp_keys, tmp_values,
                settings.gpus, host_allocators, device_allocators,
                stream_pools, settings.r_num_elements, settings.chunk_size,
                topology, &sort_snapshots, settings.debug_interval);
          } else {
            AJBHybridSort<T, V, HybridSortKernel::kRadix>(
                r_keys, r_values, tmp_keys, tmp_values,
                settings.gpus, host_allocators, device_allocators,
                stream_pools, settings.r_num_elements, settings.chunk_size,
                topology, &sort_snapshots, settings.debug_interval);
          }
        }
        if (!settings.s_sort) {
          if (settings.sort_algorithm == "hybrid_merge_sort") {
            AJBHybridSort<T, V, HybridSortKernel::kMerge>(
                s_keys, s_values, tmp_keys, tmp_values,
                settings.gpus, host_allocators, device_allocators,
                stream_pools, settings.s_num_elements, settings.chunk_size,
                topology, &sort_snapshots, settings.debug_interval);
          } else {
            AJBHybridSort<T, V, HybridSortKernel::kRadix>(
                s_keys, s_values, tmp_keys, tmp_values,
                settings.gpus, host_allocators, device_allocators,
                stream_pools, settings.s_num_elements, settings.chunk_size,
                topology, &sort_snapshots, settings.debug_interval);
          }
        }
      }
    } else {
      // Upstream baseline sort (unchanged)
      if (settings.sort_algorithm == "hybrid_merge_sort") {
        if (!settings.r_sort) {
          HybridSort<T, V, HybridSortKernel::kMerge>(
              r_keys, r_values, tmp_keys, tmp_values,
              settings.gpus, host_allocators, device_allocators,
              stream_pools, settings.r_num_elements, settings.chunk_size);
        }
        if (!settings.s_sort) {
          HybridSort<T, V, HybridSortKernel::kMerge>(
              s_keys, s_values, tmp_keys, tmp_values,
              settings.gpus, host_allocators, device_allocators,
              stream_pools, settings.s_num_elements, settings.chunk_size);
        }
      } else if (settings.sort_algorithm == "hybrid_radix_sort") {
        if (!settings.r_sort) {
          HybridSort<T, V, HybridSortKernel::kRadix>(
              r_keys, r_values, tmp_keys, tmp_values,
              settings.gpus, host_allocators, device_allocators,
              stream_pools, settings.r_num_elements, settings.chunk_size);
        }
        if (!settings.s_sort) {
          HybridSort<T, V, HybridSortKernel::kRadix>(
              s_keys, s_values, tmp_keys, tmp_values,
              settings.gpus, host_allocators, device_allocators,
              stream_pools, settings.s_num_elements, settings.chunk_size);
        }
      }
    }

    AJB_BREAKPOINT("Sort phase complete");
    AJBDumpArray("R_keys_sorted", r_keys.data(), r_keys.size(), 8, 4);

    // Sort correctness verification — check sorted order and report inversions
    fprintf(stderr, "[AJB_STATE] SortVerify BEGIN: checking R_keys (%zu elements)\n",
            r_keys.size());
    size_t r_inversions = VerifySortedOrder(r_keys.data(), r_keys.size(), 3);
    fprintf(stderr, "[AJB_STATE] SortVerify BEGIN: checking S_keys (%zu elements)\n",
            s_keys.size());
    size_t s_inversions = VerifySortedOrder(s_keys.data(), s_keys.size(), 3);
    fprintf(stderr, "[AJB_STATE] SortPhase END r_inversions=%zu s_inversions=%zu\n",
            r_inversions, s_inversions);
    (void)r_inversions; (void)s_inversions;

    // --- Step 8: Join phase ---
    fprintf(stderr, "[AJB_STATE] JoinPhase BEGIN mode=%s materialize=%d\n",
            settings.use_ajb ? "AJB-ADAPTIVE" : "BASELINE", settings.materialize);
    JoinResult<T> join_result;
    std::vector<ChunkGroupSnapshot<T>> join_snapshots;

    if (settings.use_ajb) {
      printf("\n[AJB] Running AJBMergeJoin (adaptive)...\n");
      join_result = AJBMergeJoin(
          r_keys, r_values, s_keys, s_values,
          settings.gpus, device_allocators, stream_pools,
          scheduler, settings.materialize,
          &join_snapshots, settings.debug_interval);
    } else {
      printf("\n[AJB] Running upstream MergeJoin (baseline)...\n");
      join_result = MergeJoin(
          r_keys, r_values, s_keys, s_values,
          settings.gpus, device_allocators, stream_pools,
          settings.materialize);
    }

    fprintf(stderr, "[AJB_STATE] JoinPhase END matches=%zu\n", join_result.count_);

    AJB_BREAKPOINT("Join phase complete, matches=%zu", join_result.count_);

    // Compute per-GPU chunk sizes and load-balance Gini coefficient
    std::vector<size_t> per_gpu_chunks(settings.gpus.size());
    size_t total_elements = settings.r_num_elements + settings.s_num_elements;
    for (size_t g = 0; g < settings.gpus.size(); ++g) {
      per_gpu_chunks[g] = total_elements / settings.gpus.size()
                        + (g < total_elements % settings.gpus.size() ? 1 : 0);
    }
    double load_gini = ComputeGiniCoefficient(per_gpu_chunks);

    fprintf(stderr, "[AJB_STATE] LoadBalance gini=%.6f per_gpu_chunk_0=%zu num_gpus=%zu\n",
            load_gini, per_gpu_chunks.empty() ? 0 : per_gpu_chunks[0],
            settings.gpus.size());

    // Update Welford accumulators with this run's timings
    double sort_dur = TimeDurations::Get().GetDuration("sort_phase");
    double merge_dur = TimeDurations::Get().GetDuration("merge_phase");
    double join_dur = TimeDurations::Get().GetDuration("join_phase");
    double total_dur = TimeDurations::Get().GetTotalDuration();
    welford_sort.Update(sort_dur);
    welford_merge.Update(merge_dur);
    welford_join.Update(join_dur);
    welford_total.Update(total_dur);

    welford_sort.DumpState("sort_phase");
    welford_merge.DumpState("merge_phase");
    welford_join.DumpState("join_phase");
    welford_total.DumpState("total");

    // --- Step 9: Output results (upstream CSV format + AJB extras) ---
    std::cout << settings.r_num_elements << ","
              << settings.s_num_elements << ","
              << settings.num_threads << ",\"";
    for (size_t i = 0; i < settings.gpus.size(); ++i) {
      std::cout << settings.gpus[i]
                << (i + 1 < settings.gpus.size() ? "," : "");
    }
    std::cout << "\",\"" << settings.sort_algorithm
              << "\",\"" << settings.join_algorithm
              << "\"," << settings.chunk_size
              << ",\"" << settings.key_type
              << "\",\"" << settings.key_distribution
              << "\",\"" << settings.value_type
              << "\",\"" << settings.value_distribution
              << "\"," << settings.random_seed
              << "," << settings.theta
              << "," << settings.sigma
              << "," << settings.r_sort
              << "," << settings.s_sort
              << "," << settings.materialize
              << "," << std::fixed << std::setprecision(9)
              << TimeDurations::Get().GetDuration("sort_phase")
              << "," << TimeDurations::Get().GetDuration("merge_phase")
              << "," << TimeDurations::Get().GetDuration("join_phase")
              << "," << TimeDurations::Get().GetTotalDuration()
              // AJB extra columns
              << "," << (settings.use_ajb ? "ajb" : "baseline")
              << "," << cadence.K_x
              << "," << cadence.K_u
              << "," << cadence.K_v
              << "," << scheduler.GetSlowTierBytes()
              << "," << scheduler.GetFastTierBytes()
              << "," << effective_skew
              // Per-GPU and Welford stats columns
              << "," << (per_gpu_chunks.empty() ? 0 : per_gpu_chunks[0])
              << "," << std::fixed << std::setprecision(6) << load_gini
              << "," << std::fixed << std::setprecision(3) << settings.warmup_ms
              << "," << std::fixed << std::setprecision(9) << welford_sort.Mean()
              << "," << welford_sort.Stddev()
              << "," << welford_total.Mean()
              << "," << welford_total.Stddev()
              << std::endl;

    // --- Step 10: Correctness check ---
    fprintf(stderr, "[AJB_STATE] CorrectnessCheck BEGIN expected=%zu got=%zu\n",
            expected_matches, join_result.count_);
    #ifndef CPU_ONLY_TEST
    if (join_result.count_ != expected_matches) {
      printf("\n[ERROR] Match count mismatch: got %zu, expected %zu\n",
             join_result.count_, expected_matches);
      fprintf(stderr, "[AJB_STATE] CorrectnessCheck FAILED\n");
    } else {
      printf("\n[OK] Match count verified: %zu\n", join_result.count_);
      fprintf(stderr, "[AJB_STATE] CorrectnessCheck PASSED\n");
    }
    #else
    printf("\n[INFO] CPU test mode — match count: %zu (no reference)\n",
           join_result.count_);
    fprintf(stderr, "[AJB_STATE] CorrectnessCheck SKIPPED (CPU_ONLY_TEST)\n");
    #endif

    // --- Step 11: Export diagnostics ---
    if (!settings.events_csv.empty()) {
      scheduler.ExportEventsCSV(settings.events_csv);
    }

    if (!settings.timing_csv.empty()) {
      TimeDurations::Get().ExportCSV(settings.timing_csv);
    }

    // Print full timing breakdown
    TimeDurations::Get().PrintAllDurations("AJBBenchmark");
    AJBReportMemory("post-benchmark");
    AJB_PRINT_COUNTS();

    // Final Welford summary to stderr
    fprintf(stderr, "[AJB_STATE] BenchmarkEnd total_runs=%zu\n", welford_total.count_);
    fprintf(stderr, "[AJB_BP] sort_mean=%.6f sort_stddev=%.6f merge_mean=%.6f merge_stddev=%.6f\n",
            welford_sort.Mean(), welford_sort.Stddev(),
            welford_merge.Mean(), welford_merge.Stddev());
    fprintf(stderr, "[AJB_BP] join_mean=%.6f join_stddev=%.6f total_mean=%.6f total_stddev=%.6f\n",
            welford_join.Mean(), welford_join.Stddev(),
            welford_total.Mean(), welford_total.Stddev());
    fprintf(stderr, "[AJB_STATE] Throughput: %.2f Mtuples/sec (based on total mean)\n",
            welford_total.Mean() > 0.0
              ? (settings.r_num_elements + settings.s_num_elements) / welford_total.Mean() / 1e6
              : 0.0);
  }
}

// =============================================================================
// CPU-only test main (for development without GPU hardware)
// =============================================================================
#ifdef CPU_ONLY_TEST
int main(int argc, char* argv[]) {
  printf("=== AJB Benchmark — CPU-Only Test Mode ===\n\n");

  AJBSettings s;
  s.r_num_elements = (argc > 1) ? std::stoul(argv[1]) : 10000;
  s.s_num_elements = (argc > 2) ? std::stoul(argv[2]) : 10000;
  s.num_threads = 4;
  s.gpus = {0, 1};  // simulated
  s.sort_algorithm = "gnu_parallel_sort";
  s.join_algorithm = "ajb_sort_merge_join";
  s.chunk_size = 0;
  s.key_type = "int";
  s.key_distribution = "uniform";
  s.value_type = "int";
  s.value_distribution = "uniform";
  s.random_seed = 42;
  s.theta = 0;
  s.sigma = 0;
  s.r_sort = false;
  s.s_sort = false;
  s.save = false;
  s.materialize = false;
  s.print = false;

  // AJB settings
  s.use_ajb = true;
  s.K_x = 256;
  s.K_u = 16;
  s.K_v = 1;
  s.auto_tune_cadence = true;
  s.debug_interval = 1;
  s.num_runs = 3;
  s.warmup_ms = 50.0;
  s.events_csv = "ajb_events.csv";
  s.timing_csv = "ajb_timing.csv";

  s.DebugPrint();

  RunAJBBenchmark<uint32_t, uint32_t>(s);

  return 0;
}
#endif

// =============================================================================
// Full CUDA main (with cxxopts CLI, matches upstream interface + AJB extras)
// =============================================================================
#ifndef CPU_ONLY_TEST
int main(int argc, char* argv[]) {
  cxxopts::Options options("ajb_benchmark",
      "AJB: Adaptive Hash Join on Mixed-Bandwidth GPU Interconnects");

  options.set_width(250);

  // Upstream options (unchanged)
  options.add_options()
    ("r_num_elements", "elements in R " + OptionsLimits::GetNumElementsLimits(),
     cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumElements()))
    ("s_num_elements", "elements in S " + OptionsLimits::GetNumElementsLimits(),
     cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumElements()))
    ("num_threads", "threads " + OptionsLimits::GetNumThreadsLimits(),
     cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumThreads()))
    ("gpus", "visible GPUs " + OptionsLimits::GetGpusLimits(),
     cxxopts::value<std::vector<int>>()->default_value(OptionsLimits::GetDefaultGpus()))
    ("sort_algorithm", "sort algo " + OptionsLimits::GetSortAlgorithmLimits(),
     cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultSortAlgorithm()))
    ("join_algorithm", "join algo",
     cxxopts::value<std::string>()->default_value("ajb_sort_merge_join"))
    ("chunk_size", "chunk size " + OptionsLimits::GetChunkSizeLimits(),
     cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultChunkSize()))
    ("key_type", "key type " + OptionsLimits::GetTypeLimits(),
     cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()))
    ("key_distribution", "key dist " + OptionsLimits::GetJoinDistributionLimits(),
     cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultJoinDistribution()))
    ("value_type", "value type", cxxopts::value<std::string>()->default_value("int"))
    ("value_distribution", "value dist", cxxopts::value<std::string>()->default_value("unique"))
    ("random_seed", "random seed", cxxopts::value<uint32_t>()->default_value("42"))
    ("theta", "theta", cxxopts::value<uint32_t>()->default_value("0"))
    ("sigma", "sigma", cxxopts::value<uint32_t>()->default_value("0"))
    ("r_sort", "R pre-sorted", cxxopts::value<bool>()->default_value("false"))
    ("s_sort", "S pre-sorted", cxxopts::value<bool>()->default_value("false"))
    ("save", "save input", cxxopts::value<bool>()->default_value("false"))
    ("materialize", "materialize output", cxxopts::value<bool>()->default_value("false"))
    ("print", "print output", cxxopts::value<bool>()->default_value("false"))
    // AJB-specific options
    ("ajb", "use AJB adaptive mode", cxxopts::value<bool>()->default_value("true"))
    ("K_x", "build partition transfer period", cxxopts::value<size_t>()->default_value("256"))
    ("K_u", "merge-path boundary transfer period", cxxopts::value<size_t>()->default_value("16"))
    ("K_v", "materialization buffer transfer period", cxxopts::value<size_t>()->default_value("1"))
    ("auto_tune", "auto-tune cadence from skew", cxxopts::value<bool>()->default_value("true"))
    ("debug_interval", "debug print every N chunks", cxxopts::value<size_t>()->default_value("10"))
    ("num_runs", "number of benchmark repetitions for Welford stats", cxxopts::value<size_t>()->default_value("1"))
    ("warmup_ms", "warmup phase duration in ms", cxxopts::value<double>()->default_value("0.0"))
    ("events_csv", "export transfer events CSV", cxxopts::value<std::string>()->default_value(""))
    ("timing_csv", "export timing CSV", cxxopts::value<std::string>()->default_value(""))
    ("help", "show help", cxxopts::value<bool>()->default_value("false"));

  auto result = options.parse(argc, argv);

  if (result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  AJBSettings s;
  s.r_num_elements = result["r_num_elements"].as<size_t>();
  s.s_num_elements = result["s_num_elements"].as<size_t>();
  s.num_threads = result["num_threads"].as<size_t>();
  s.gpus = result["gpus"].as<std::vector<int>>();
  s.sort_algorithm = result["sort_algorithm"].as<std::string>();
  s.join_algorithm = result["join_algorithm"].as<std::string>();
  s.chunk_size = result["chunk_size"].as<size_t>();
  s.key_type = result["key_type"].as<std::string>();
  s.key_distribution = result["key_distribution"].as<std::string>();
  s.value_type = result["value_type"].as<std::string>();
  s.value_distribution = result["value_distribution"].as<std::string>();
  s.random_seed = result["random_seed"].as<uint32_t>();
  s.theta = result["theta"].as<uint32_t>();
  s.sigma = result["sigma"].as<uint32_t>();
  s.r_sort = result["r_sort"].as<bool>();
  s.s_sort = result["s_sort"].as<bool>();
  s.save = result["save"].as<bool>();
  s.materialize = result["materialize"].as<bool>();
  s.print = result["print"].as<bool>();
  // AJB
  s.use_ajb = result["ajb"].as<bool>();
  s.K_x = result["K_x"].as<size_t>();
  s.K_u = result["K_u"].as<size_t>();
  s.K_v = result["K_v"].as<size_t>();
  s.auto_tune_cadence = result["auto_tune"].as<bool>();
  s.debug_interval = result["debug_interval"].as<size_t>();
  s.num_runs = result["num_runs"].as<size_t>();
  s.warmup_ms = result["warmup_ms"].as<double>();
  s.events_csv = result["events_csv"].as<std::string>();
  s.timing_csv = result["timing_csv"].as<std::string>();

  s.DebugPrint();

  if (s.key_type == "int" && s.value_type == "int") {
    RunAJBBenchmark<uint32_t, uint32_t>(s);
  } else if (s.key_type == "long" && s.value_type == "long") {
    RunAJBBenchmark<uint64_t, uint64_t>(s);
  }

  return 0;
}
#endif
