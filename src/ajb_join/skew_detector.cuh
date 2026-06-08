#pragma once
// =============================================================================
// skew_detector.cuh — AJB Key-Distribution Skew Detector
//
// Estimates the skewness of the join key distribution using lightweight
// random sampling. The skew estimate feeds into AutoTuneCadence():
//   - High skew => lower K_u (more frequent boundary transfers for correctness)
//   - Low skew  => higher K_u (save bandwidth, boundaries stay valid longer)
//
// Adapted from upstream/joinrenum sampling logic (~80% algorithmic structure
// from Index.hpp / REnum.hpp, ~20% new: histogram-based skew metric + debug).
//
// The detector runs on CPU before the GPU join pipeline starts, so it adds
// minimal overhead to the critical path.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "common/pinned_vector.cuh"

// ---------------------------------------------------------------------------
// Skew metric: coefficient of variation of the key frequency histogram
//
//   skew = std(bucket_counts) / mean(bucket_counts)
//
// A uniform distribution gives skew ~= 0; Zipfian gives skew >> 1.
// We normalize to [0, 1] via tanh for the cadence tuner.
// ---------------------------------------------------------------------------
struct SkewEstimate {
  double raw_cv;           // coefficient of variation (unbounded)
  double normalized;       // tanh(raw_cv), in [0, 1)
  double chi_sq;           // chi-squared goodness-of-fit statistic
  double chi_sq_pvalue;    // approximate p-value (Wilson-Hilferty)
  double ks_distance;      // Kolmogorov-Smirnov max CDF deviation
  size_t sample_size;
  size_t num_buckets;
  size_t max_bucket_count;
  size_t min_bucket_count;

  void DebugPrint(const char* label = "SkewEstimate") const {
    fprintf(stderr, "[AJB_BP][%s] cv=%.4f norm=%.4f chi2=%.2f(p=%.4f) ks=%.4f "
            "n=%zu buckets=%zu max=%zu min=%zu\n",
            label, raw_cv, normalized, chi_sq, chi_sq_pvalue, ks_distance,
            sample_size, num_buckets, max_bucket_count, min_bucket_count);
    // frequency histogram top-5 buckets (diagnostic)
  }};

// ---------------------------------------------------------------------------
// Lightweight key-frequency histogram
// ---------------------------------------------------------------------------
template <typename T>
struct KeyHistogram {
  std::vector<size_t> counts;
  T min_key;
  T max_key;
  size_t num_buckets;

  KeyHistogram(size_t n_buckets, T lo, T hi)
      : counts(n_buckets, 0), min_key(lo), max_key(hi), num_buckets(n_buckets) {}

  void Insert(T key) {
    if (max_key <= min_key) return;
    double normalized = static_cast<double>(key - min_key) / (max_key - min_key);
    size_t bucket = static_cast<size_t>(normalized * (num_buckets - 1));
    bucket = std::min(bucket, num_buckets - 1);
    counts[bucket]++;
  }

  void DebugPrint(size_t max_display = 20) const {
    printf("  [KeyHistogram] %zu buckets, range=[%s, %s]\n",
           num_buckets,
           std::to_string(min_key).c_str(),
           std::to_string(max_key).c_str());
    
    size_t total = 0;
    size_t max_c = 0;
    for (auto c : counts) {
      total += c;
      max_c = std::max(max_c, c);
    }

    // ASCII bar chart
    size_t step = std::max<size_t>(1, num_buckets / max_display);
    for (size_t i = 0; i < num_buckets && i < max_display * step; i += step) {
      size_t sum = 0;
      for (size_t j = i; j < std::min(i + step, num_buckets); ++j) {
        sum += counts[j];
      }
      size_t bar_len = (max_c > 0) ? (sum * 40) / (max_c * step) : 0;
      printf("  B[%3zu]: %6zu |", i / step, sum);
      for (size_t b = 0; b < bar_len; ++b) printf("#");
      printf("\n");
    }
    printf("  Total samples: %zu\n", total);
  }
};

// ---------------------------------------------------------------------------
// Detect skew via random sampling
// ---------------------------------------------------------------------------
template <typename T>
SkewEstimate DetectSkew(const T* keys, size_t n,
                        size_t sample_size = 10000,
                        size_t num_buckets = 64,
                        uint32_t seed = 42) {
  printf("\n[DetectSkew] Analyzing %zu keys (sampling %zu)...\n", n, sample_size);

  if (n == 0) {
    SkewEstimate est{0.0, 0.0, 0, num_buckets, 0, 0};
    est.DebugPrint("EmptyInput");
    return est;
  }

  // --- Sample random keys ---
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> dist(0, n - 1);

  sample_size = std::min(sample_size, n);
  std::vector<T> samples(sample_size);
  for (size_t i = 0; i < sample_size; ++i) {
    samples[i] = keys[dist(rng)];
  }

  // --- Find range ---
  T min_key = *std::min_element(samples.begin(), samples.end());
  T max_key = *std::max_element(samples.begin(), samples.end());

  printf("  Key range: [%s, %s]\n",
         std::to_string(min_key).c_str(),
         std::to_string(max_key).c_str());

  if (min_key == max_key) {
    // All sampled keys are the same — extreme skew
    SkewEstimate est{100.0, 1.0, sample_size, num_buckets, sample_size, 0};
    est.DebugPrint("AllSameKey");
    return est;
  }

  // --- Freedman-Diaconis adaptive bucket count ---
  // IQR-based: bin_width = 2 * IQR * n^(-1/3)
  {
    std::vector<T> sorted_samples(samples);
    std::sort(sorted_samples.begin(), sorted_samples.end());
    size_t q1_idx = sorted_samples.size() / 4;
    size_t q3_idx = 3 * sorted_samples.size() / 4;
    double iqr = static_cast<double>(sorted_samples[q3_idx]) -
                 static_cast<double>(sorted_samples[q1_idx]);
    if (iqr > 0) {
      double bin_width = 2.0 * iqr * std::pow(static_cast<double>(sample_size), -1.0/3.0);
      double range = static_cast<double>(max_key) - static_cast<double>(min_key);
      size_t fd_buckets = static_cast<size_t>(std::ceil(range / bin_width));
      // clamp to [8, 512]
      fd_buckets = std::max<size_t>(8, std::min<size_t>(512, fd_buckets));
      fprintf(stderr, "[AJB_BP][Skew] Freedman-Diaconis: IQR=%.1f bin_w=%.1f -> %zu buckets (was %zu)\n",
              iqr, bin_width, fd_buckets, num_buckets);
      num_buckets = fd_buckets;
    }
  }

  // --- Build histogram ---
  KeyHistogram<T> hist(num_buckets, min_key, max_key);
  for (const auto& key : samples) {
    hist.Insert(key);
  }

  #ifdef AJB_TRACE_SKEW
  hist.DebugPrint();
  #endif

  // --- Compute coefficient of variation ---
  double mean = static_cast<double>(sample_size) / num_buckets;
  double sum_sq_diff = 0.0;
  size_t max_count = 0, min_count = sample_size;

  for (size_t i = 0; i < num_buckets; ++i) {
    double diff = static_cast<double>(hist.counts[i]) - mean;
    sum_sq_diff += diff * diff;
    max_count = std::max(max_count, hist.counts[i]);
    min_count = std::min(min_count, hist.counts[i]);
  }

  double variance = sum_sq_diff / num_buckets;
  double stddev = std::sqrt(variance);
  double cv = (mean > 0) ? stddev / mean : 0.0;
  double normalized = std::tanh(cv);

  // --- Chi-squared goodness-of-fit (H0: uniform distribution) ---
  double chi_sq = 0.0;
  double expected = static_cast<double>(sample_size) / num_buckets;
  for (size_t i = 0; i < num_buckets; ++i) {
    double diff_cs = static_cast<double>(hist.counts[i]) - expected;
    chi_sq += (diff_cs * diff_cs) / expected;
  }
  // Wilson-Hilferty approximation for chi-sq p-value
  double df = static_cast<double>(num_buckets - 1);
  double wh_z = std::pow(chi_sq / df, 1.0/3.0) - (1.0 - 2.0/(9.0*df));
  double wh_denom = std::sqrt(2.0 / (9.0 * df));
  double chi_sq_pvalue = 0.5 * std::erfc(wh_z / (wh_denom * std::sqrt(2.0)));

  // --- Kolmogorov-Smirnov distance (empirical vs uniform CDF) ---
  double ks_distance = 0.0;
  {
    double cum_observed = 0.0;
    for (size_t i = 0; i < num_buckets; ++i) {
      cum_observed += static_cast<double>(hist.counts[i]) / sample_size;
      double cum_expected = static_cast<double>(i + 1) / num_buckets;
      double deviation = std::fabs(cum_observed - cum_expected);
      if (deviation > ks_distance) ks_distance = deviation;
    }
  }

  // [AJB_BP] diagnostic: top-5 bucket frequencies
  {
    std::vector<std::pair<size_t,size_t>> freq_idx(num_buckets);
    for (size_t i = 0; i < num_buckets; ++i) freq_idx[i] = {hist.counts[i], i};
    std::partial_sort(freq_idx.begin(),
                      freq_idx.begin() + std::min<size_t>(5, num_buckets),
                      freq_idx.end(),
                      [](auto&a, auto&b){ return a.first > b.first; });
    fprintf(stderr, "[AJB_BP][Skew] top-5 buckets:");
    for (size_t k = 0; k < std::min<size_t>(5, num_buckets); ++k)
      fprintf(stderr, " [%zu]=%zu", freq_idx[k].second, freq_idx[k].first);
    fprintf(stderr, "\n");
  }

  SkewEstimate est{cv, normalized, chi_sq, chi_sq_pvalue, ks_distance,
                   sample_size, num_buckets, max_count, min_count};
  est.DebugPrint("Computed");

  // --- Print diagnostic ---
  // Combined judgment: use both chi-sq rejection and KS distance
  bool chi_rejects_uniform = (chi_sq_pvalue < 0.05);
  bool ks_rejects_uniform  = (ks_distance > 1.36 / std::sqrt(static_cast<double>(num_buckets)));
  fprintf(stderr, "[AJB_BP][Skew] chi2 %s uniform (p=%.4f), KS %s uniform (D=%.4f, crit=%.4f)\n",
          chi_rejects_uniform ? "REJECTS" : "accepts",
          chi_sq_pvalue,
          ks_rejects_uniform ? "REJECTS" : "accepts",
          ks_distance,
          1.36 / std::sqrt(static_cast<double>(num_buckets)));

  if (!chi_rejects_uniform && !ks_rejects_uniform) {
    printf("  -> Distribution is NEARLY UNIFORM (both tests accept H0)\n");
    printf("     Implication: K_u can be large, boundaries stay valid\n");
  } else if (chi_rejects_uniform && ks_rejects_uniform) {
    printf("  -> Distribution is HIGHLY SKEWED (both tests reject H0)\n");
    printf("     Implication: K_u must be small, frequent boundary syncs needed\n");
  } else {
    printf("  -> Distribution has MODERATE SKEW (tests disagree)\n");
    printf("     Implication: K_u should be moderate for correctness\n");
  }
  printf("\n");

  return est;
}

// ---------------------------------------------------------------------------
// Detect skew for both R and S relations (the join has two inputs)
// ---------------------------------------------------------------------------
template <typename T>
std::pair<SkewEstimate, SkewEstimate> DetectJoinSkew(
    const T* keys_r, size_t n_r,
    const T* keys_s, size_t n_s,
    size_t sample_size = 10000) {
  
  printf("================================================================\n");
  printf("[DetectJoinSkew] R-relation:\n");
  printf("================================================================\n");
  auto skew_r = DetectSkew(keys_r, n_r, sample_size);

  printf("================================================================\n");
  printf("[DetectJoinSkew] S-relation:\n");
  printf("================================================================\n");
  auto skew_s = DetectSkew(keys_s, n_s, sample_size);

  // The effective skew for cadence tuning is the max of both
  double effective = std::max(skew_r.normalized, skew_s.normalized);
  printf("[DetectJoinSkew] Effective skew for cadence tuning: %.4f "
         "(max of R=%.4f, S=%.4f)\n\n",
         effective, skew_r.normalized, skew_s.normalized);

  return {skew_r, skew_s};
}
