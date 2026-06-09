// =============================================================================
// ajb_renum_adapter.hpp — AJB adapter for joinrenum's RandOrderEnum
// =============================================================================
// Bridges the random-enumeration join from joinrenum into AJB's
// tier-aware transfer scheduling. Key responsibilities:
//   1. Use RandOrderEnum::sample() to draw join-result samples
//   2. Feed samples into skew detection (CV + tanh normalization)
//   3. Map detected skew → cadence parameters (K_x, K_u, K_v)
//   4. Produce debug dumps at every stage for experiment tracing
//
// This is the "20% new code" on top of the upstream joinrenum library.
// =============================================================================
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

// --- joinrenum headers (order matters due to upstream include deps) ----------
// NOTE: REnum.hpp and Enumerator.hpp both pull Index.hpp which has no
// include guards. We include only REnum.hpp here; if Enumerator is
// needed, include it separately with care.
#include "REnum.hpp"

// --- AJB headers ------------------------------------------------------------
#ifdef AJB_WITH_CUDA
#include "../ajb_join/skew_detector.cuh"
#include "../ajb_join/tier_transfer_scheduler.cuh"
#include "../common/debug_utilities.cuh"
#include "../common/profile_utilities.cuh"
#else
// CPU-only stubs for the symbols we need
#include <cstdint>
#include <iostream>
struct SkewEstimate {
  float cv;
  float normalized;
  bool is_high_skew;
};
#endif

namespace ajb {

// ============================================================================
// SampleAccumulator — collects join-result samples and computes key frequency
// ============================================================================
struct SampleAccumulator {
  std::unordered_map<int, size_t> key_freq;     // key → count
  std::vector<int>                all_keys;      // flat list of all sampled keys
  size_t                          total_tuples;  // total tuples sampled
  size_t                          num_rounds;    // how many sample() calls

  SampleAccumulator() : total_tuples(0), num_rounds(0) {}

  void AddSample(const std::vector<int>& tuple) {
    for (int k : tuple) {
      key_freq[k]++;
      all_keys.push_back(k);
    }
    total_tuples++;
    num_rounds++;
  }

  // Coefficient of variation of key frequencies
  float ComputeCV() const {
    if (key_freq.empty()) return 0.0f;

    double sum = 0.0;
    for (auto& kv : key_freq) sum += kv.second;
    double mean = sum / key_freq.size();
    if (mean < 1e-9) return 0.0f;

    double var = 0.0;
    for (auto& kv : key_freq) {
      double d = kv.second - mean;
      var += d * d;
    }
    var /= key_freq.size();

    return static_cast<float>(std::sqrt(var) / mean);
  }

  // tanh-normalized skew ∈ [0,1]
  SkewEstimate EstimateSkew(float threshold = 0.6f) const {
    float cv = ComputeCV();
    float norm = std::tanh(cv);
    return {cv, norm, norm > threshold};
  }

  void PrintHistogram(size_t top_n = 20) const {
    // Sort by frequency descending
    std::vector<std::pair<int, size_t>> sorted_freq(key_freq.begin(), key_freq.end());
    std::sort(sorted_freq.begin(), sorted_freq.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    size_t max_freq = sorted_freq.empty() ? 1 : sorted_freq[0].second;
    size_t show = std::min(top_n, sorted_freq.size());

    printf("[AJB-REnum] key frequency histogram (top %zu of %zu distinct keys):\n",
           show, key_freq.size());
    for (size_t i = 0; i < show; i++) {
      int bar_len = static_cast<int>(40.0 * sorted_freq[i].second / max_freq);
      printf("  key=%6d  freq=%6zu  ", sorted_freq[i].first, sorted_freq[i].second);
      for (int b = 0; b < bar_len; b++) printf("█");
      printf("\n");
    }
    if (sorted_freq.size() > show) {
      printf("  ... and %zu more keys\n", sorted_freq.size() - show);
    }
  }
};

// ============================================================================
// REnumSkewProbe — samples from RandOrderEnum and estimates join skew
// ============================================================================
struct REnumSkewProbe {
  size_t num_samples;     // how many sample() calls to make
  float  skew_threshold;  // threshold for "high skew"

  REnumSkewProbe(size_t n = 1000, float thresh = 0.6f)
    : num_samples(n), skew_threshold(thresh) {}

  // Run sampling and return skew estimate + accumulator.
  // Uses Algorithm L (Li 1994) reservoir sampling when num_samples is large:
  //   - Exponential jump distances avoid checking every element
  //   - O(k(1 + log(n/k))) expected comparisons vs O(n) for Algorithm R
  //   - Collision rate tracked via EMA; buffer doubles when EMA > 0.95
  std::pair<SkewEstimate, SampleAccumulator> Probe(RandOrderEnum& renum) const {
    SampleAccumulator acc;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Collision tracking (hash-based duplicate detection)
    std::unordered_set<size_t> seen_hashes;
    size_t collisions = 0;
    double collision_ema = 0.0;
    const double ema_alpha = 0.01;  // smoothing factor

    // Adaptive reservoir: Algorithm L for large sample counts
    // W = exp(log(U(0,1))/k) where k is reservoir size
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    size_t reservoir_k = std::min(num_samples, (size_t)10000);
    std::vector<std::vector<int>> reservoir(reservoir_k);
    size_t total_attempted = 0;
    size_t empty_returns = 0;

    // Phase 1: fill reservoir
    for (size_t i = 0; i < reservoir_k; i++) {
      std::vector<int> tuple = renum.sample();
      total_attempted++;
      if (!tuple.empty()) {
        reservoir[i] = tuple;
        acc.AddSample(tuple);

        // Collision check
        size_t h = 0;
        for (int v : tuple) h ^= std::hash<int>{}(v) * 2654435761ULL;
        if (seen_hashes.count(h)) {
          collisions++;
        }
        seen_hashes.insert(h);
        collision_ema = ema_alpha * (seen_hashes.count(h) > 1 ? 1.0 : 0.0)
                        + (1.0 - ema_alpha) * collision_ema;
      } else {
        empty_returns++;
      }
    }

    // Phase 2: Algorithm L — exponential jumps for remaining samples
    if (num_samples > reservoir_k) {
      // W = exp(log(U) / k)
      double W = std::exp(std::log(uni(rng)) / reservoir_k);
      size_t i = reservoir_k;
      size_t jump_count = 0;

      while (i < num_samples) {
        // Skip distance: geometric with parameter W
        double skip = std::floor(std::log(uni(rng)) / std::log(1.0 - W));
        i += static_cast<size_t>(skip) + 1;
        if (i >= num_samples) break;

        // Replace random element in reservoir
        std::vector<int> tuple = renum.sample();
        total_attempted++;
        if (!tuple.empty()) {
          size_t replace_idx = rng() % reservoir_k;
          reservoir[replace_idx] = tuple;
          acc.AddSample(tuple);
          jump_count++;

          // Update collision EMA
          size_t h = 0;
          for (int v : tuple) h ^= std::hash<int>{}(v) * 2654435761ULL;
          double is_collision = seen_hashes.count(h) ? 1.0 : 0.0;
          collision_ema = ema_alpha * is_collision + (1.0 - ema_alpha) * collision_ema;
          seen_hashes.insert(h);

          // Adaptive: if collision rate too high, double the sampling effort
          if (collision_ema > 0.95 && reservoir_k < 100000) {
            fprintf(stderr, "[AJB_STATE][ReservoirAdapt] collision_ema=%.4f > 0.95, "
                    "doubling reservoir from %zu\n", collision_ema, reservoir_k);
            reservoir_k *= 2;
            reservoir.resize(reservoir_k);
            W = std::exp(std::log(uni(rng)) / reservoir_k);
          }
        }

        // Update W
        W *= std::exp(std::log(uni(rng)) / reservoir_k);

        // [AJB_STATE] periodic collision diagnostic
        if (total_attempted % 1000 == 0 && total_attempted > 0) {
          fprintf(stderr, "[AJB_STATE][adapter_collision_ema] attempted=%zu "
                  "collision_ema=%.4f collisions=%zu reservoir_k=%zu jumps=%zu\n",
                  total_attempted, collision_ema, collisions, reservoir_k, jump_count);
        }
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    SkewEstimate est = acc.EstimateSkew(skew_threshold);

    printf("[AJB-REnum] probe complete: %zu requested, %zu attempted in %.4fs\n",
           num_samples, total_attempted, elapsed);
    printf("[AJB-REnum]   total_tuples=%zu  distinct_keys=%zu  empty_returns=%zu\n",
           acc.total_tuples, acc.key_freq.size(), empty_returns);
    printf("[AJB-REnum]   CV=%.4f  normalized=%.4f  is_high_skew=%s\n",
           est.cv, est.normalized, est.is_high_skew ? "YES" : "NO");
    printf("[AJB-REnum]   collision_ema=%.4f  collisions=%zu  reservoir_k=%zu\n",
           collision_ema, collisions, reservoir_k);

    return {est, acc};
  }
};

// ============================================================================
// CadenceFromSkew — map skew estimate → cadence parameters
// ============================================================================
// This implements the mapping from paper Section 3 (Theorem 1):
//   - Low skew  → large K_u (merge-path boundaries transfer infrequently)
//   - High skew → small K_u (need frequent boundary rebalancing)
//   - K_x, K_v scale with topology (handled by TierTransferScheduler)
// ============================================================================
struct CadenceRecommendation {
  int K_x;         // build-partition cadence
  int K_u;         // merge-path boundary cadence
  int K_v;         // materialization buffer cadence
  float skew_cv;   // for logging
  float skew_norm; // for logging
  std::string rationale;
};

inline CadenceRecommendation CadenceFromSkew(
    const SkewEstimate& est,
    int num_gpus = 2,
    int base_K = 8)
{
  CadenceRecommendation rec;
  rec.skew_cv   = est.cv;
  rec.skew_norm = est.normalized;

  // K_x: build partitions — transfer every K_x chunk-groups
  // Relatively insensitive to skew; scales with GPU count
  rec.K_x = base_K * num_gpus;

  // K_u: merge-path boundaries — skew-sensitive (Theorem 1)
  // High skew → small K_u (frequent rebalancing)
  // Low skew  → large K_u (boundaries stay valid longer)
  if (est.is_high_skew) {
    rec.K_u = std::max(1, base_K / 4);
    rec.rationale = "high skew → aggressive boundary refresh";
  } else if (est.normalized > 0.3f) {
    rec.K_u = std::max(2, base_K / 2);
    rec.rationale = "moderate skew → balanced boundary refresh";
  } else {
    rec.K_u = base_K;
    rec.rationale = "low skew → relaxed boundary refresh";
  }

  // K_v: materialization buffers — moderate sensitivity
  // High skew can cause output-buffer hotspots
  rec.K_v = est.is_high_skew ? std::max(2, base_K / 2) : base_K;

  printf("[AJB-REnum] cadence recommendation:\n");
  printf("  K_x=%d  K_u=%d  K_v=%d\n", rec.K_x, rec.K_u, rec.K_v);
  printf("  rationale: %s\n", rec.rationale.c_str());

  return rec;
}

// ============================================================================
// REnumDiagnostics — full diagnostic dump for experiment CSV
// ============================================================================
struct REnumDiagnostics {
  double  probe_time_sec;
  size_t  num_samples;
  size_t  total_tuples;
  size_t  distinct_keys;
  float   cv;
  float   normalized_skew;
  bool    is_high_skew;
  int     K_x, K_u, K_v;

  void PrintSummary() const {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║       AJB × REnum Diagnostics            ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  probe_time     = %.4f s                 \n", probe_time_sec);
    printf("║  num_samples    = %zu                    \n", num_samples);
    printf("║  total_tuples   = %zu                    \n", total_tuples);
    printf("║  distinct_keys  = %zu                    \n", distinct_keys);
    printf("║  CV             = %.4f                   \n", cv);
    printf("║  norm_skew      = %.4f                   \n", normalized_skew);
    printf("║  is_high_skew   = %s                     \n", is_high_skew ? "YES" : "NO");
    printf("║  K_x=%d  K_u=%d  K_v=%d                  \n", K_x, K_u, K_v);
    printf("╚══════════════════════════════════════════╝\n");
  }

  void ExportCSV(const std::string& path) const {
    std::ofstream f(path);
    f << "probe_time_sec,num_samples,total_tuples,distinct_keys,cv,normalized_skew,is_high_skew,K_x,K_u,K_v\n";
    f << probe_time_sec << "," << num_samples << "," << total_tuples << ","
      << distinct_keys << "," << cv << "," << normalized_skew << ","
      << (is_high_skew ? 1 : 0) << "," << K_x << "," << K_u << "," << K_v << "\n";
    f.close();
    printf("[AJB-REnum] diagnostics exported to %s\n", path.c_str());
  }
};

// ============================================================================
// RunREnumProbeAndRecommend — top-level entry point
// ============================================================================
// Usage:
//   RandOrderEnum renum("db/filenames.txt", "db/numlines.txt", "db/relations.txt",
//                        relationNames, relationVars);
//   auto diag = ajb::RunREnumProbeAndRecommend(renum, 2000, 2);
//   // diag.K_x, diag.K_u, diag.K_v are ready for TierTransferScheduler
// ============================================================================
inline REnumDiagnostics RunREnumProbeAndRecommend(
    RandOrderEnum& renum,
    size_t num_samples = 1000,
    int num_gpus = 2,
    int base_K = 8,
    float skew_threshold = 0.6f,
    const std::string& csv_path = "")
{
  printf("\n[AJB-REnum] === Starting REnum skew probe ===\n");

  REnumSkewProbe probe(num_samples, skew_threshold);
  auto t0 = std::chrono::high_resolution_clock::now();
  auto [est, acc] = probe.Probe(renum);
  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();

  // Show histogram in debug mode
#ifdef AJB_TRACE_DECISIONS
  acc.PrintHistogram(30);
#endif

  CadenceRecommendation rec = CadenceFromSkew(est, num_gpus, base_K);

  REnumDiagnostics diag;
  diag.probe_time_sec   = elapsed;
  diag.num_samples      = num_samples;
  diag.total_tuples     = acc.total_tuples;
  diag.distinct_keys    = acc.key_freq.size();
  diag.cv               = est.cv;
  diag.normalized_skew  = est.normalized;
  diag.is_high_skew     = est.is_high_skew;
  diag.K_x              = rec.K_x;
  diag.K_u              = rec.K_u;
  diag.K_v              = rec.K_v;

  diag.PrintSummary();

  if (!csv_path.empty()) {
    diag.ExportCSV(csv_path);
  }

  printf("[AJB-REnum] === Probe complete ===\n\n");
  return diag;
}

// ============================================================================
// BatchREnumProbe — run multiple probes with different sample sizes
// ============================================================================
// Useful for sensitivity analysis: how stable is the skew estimate
// as we increase the number of samples?
// ============================================================================
inline std::vector<REnumDiagnostics> BatchREnumProbe(
    RandOrderEnum& renum,
    const std::vector<size_t>& sample_sizes = {100, 500, 1000, 5000},
    int num_gpus = 2,
    int base_K = 8,
    float skew_threshold = 0.6f)
{
  printf("[AJB-REnum] === Batch probe: %zu configurations ===\n", sample_sizes.size());

  std::vector<REnumDiagnostics> results;
  results.reserve(sample_sizes.size());

  for (size_t ns : sample_sizes) {
    printf("\n--- sample_size=%zu ---\n", ns);
    auto diag = RunREnumProbeAndRecommend(renum, ns, num_gpus, base_K, skew_threshold);
    results.push_back(diag);
  }

  // Stability report
  printf("\n[AJB-REnum] === Batch stability report ===\n");
  printf("  %-10s  %-8s  %-8s  %-6s  %-6s  %-6s\n",
         "samples", "CV", "norm", "K_x", "K_u", "K_v");
  for (auto& d : results) {
    printf("  %-10zu  %-8.4f  %-8.4f  %-6d  %-6d  %-6d\n",
           d.num_samples, d.cv, d.normalized_skew, d.K_x, d.K_u, d.K_v);
  }

  return results;
}

} // namespace ajb
