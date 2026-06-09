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

// ---------------------------------------------------------------------------
// HyperLogLog cardinality estimator — O(1) space per register
// Estimates the number of distinct keys in a relation without sorting.
// Used for join selectivity prediction: if |R ∩ S| / |R| is small,
// the join is highly selective and we can use smaller output buffers.
//
// Algorithm: hash each key, use first p bits as register index,
// count leading zeros of remaining bits. The harmonic mean of
// 2^(max_leading_zeros) across registers estimates cardinality.
// ---------------------------------------------------------------------------
template <typename T>
struct HyperLogLogEstimator {
  static constexpr int kPrecision = 14;  // 2^14 = 16384 registers
  static constexpr size_t kNumRegisters = 1u << kPrecision;
  std::vector<uint8_t> registers;

  HyperLogLogEstimator() : registers(kNumRegisters, 0) {}

  // FNV-1a hash for 64-bit keys
  static uint64_t Hash(T key) {
    uint64_t h = 14695981039346656037ULL;  // FNV offset basis
    uint64_t k = static_cast<uint64_t>(key);
    for (int i = 0; i < 8; ++i) {
      h ^= (k >> (i * 8)) & 0xFF;
      h *= 1099511628211ULL;  // FNV prime
    }
    return h;
  }

  void Insert(T key) {
    uint64_t h = Hash(key);
    size_t idx = h >> (64 - kPrecision);  // top p bits as register index
    uint64_t w = (h << kPrecision) | (1ULL << (kPrecision - 1));  // remaining bits
    // Count leading zeros + 1
    uint8_t rho = 1;
    while ((w & (1ULL << 63)) == 0 && rho < 64) {
      rho++;
      w <<= 1;
    }
    if (rho > registers[idx]) registers[idx] = rho;
  }

  double Estimate() const {
    // Harmonic mean of 2^(-M[j])
    double sum = 0.0;
    int zeros = 0;
    for (size_t j = 0; j < kNumRegisters; ++j) {
      sum += std::pow(2.0, -static_cast<double>(registers[j]));
      if (registers[j] == 0) zeros++;
    }
    double alpha_m;
    // Bias correction factor
    if (kNumRegisters == 16)    alpha_m = 0.673;
    else if (kNumRegisters == 32)  alpha_m = 0.697;
    else if (kNumRegisters == 64)  alpha_m = 0.709;
    else alpha_m = 0.7213 / (1.0 + 1.079 / kNumRegisters);

    double estimate = alpha_m * kNumRegisters * kNumRegisters / sum;

    // Small range correction (linear counting)
    if (estimate <= 2.5 * kNumRegisters && zeros > 0) {
      estimate = kNumRegisters * std::log(static_cast<double>(kNumRegisters) / zeros);
    }
    return estimate;
  }

  void DumpState(const char* label) const {
    double est = Estimate();
    // Register utilization
    int used = 0;
    int max_reg = 0;
    for (size_t j = 0; j < kNumRegisters; ++j) {
      if (registers[j] > 0) used++;
      if (registers[j] > max_reg) max_reg = registers[j];
    }
    fprintf(stderr, "[AJB_STATE][HLL][%s] estimated_cardinality=%.0f "
            "registers_used=%d/%zu max_rho=%d\n",
            label, est, used, kNumRegisters, max_reg);
  }
};

// Estimate join selectivity using HLL on both relations
template <typename T>
double EstimateJoinSelectivity(const T* keys_r, size_t n_r,
                                const T* keys_s, size_t n_s,
                                size_t sample_limit = 500000) {
  HyperLogLogEstimator<T> hll_r, hll_s, hll_union;

  size_t r_sample = std::min(n_r, sample_limit);
  size_t s_sample = std::min(n_s, sample_limit);

  // Sample R keys
  std::mt19937 rng(12345);
  for (size_t i = 0; i < r_sample; ++i) {
    size_t idx = (n_r <= sample_limit) ? i : (rng() % n_r);
    hll_r.Insert(keys_r[idx]);
    hll_union.Insert(keys_r[idx]);
  }
  // Sample S keys
  for (size_t i = 0; i < s_sample; ++i) {
    size_t idx = (n_s <= sample_limit) ? i : (rng() % n_s);
    hll_s.Insert(keys_s[idx]);
    hll_union.Insert(keys_s[idx]);
  }

  double card_r = hll_r.Estimate();
  double card_s = hll_s.Estimate();
  double card_union = hll_union.Estimate();

  // |R ∩ S| ≈ |R| + |S| - |R ∪ S| (inclusion-exclusion)
  double card_intersection = std::max(0.0, card_r + card_s - card_union);

  // Selectivity = |R ⋈ S| / (|R| × |S|)
  // Under uniform distribution: selectivity ≈ |R ∩ S| / max(|R|, |S|)
  double selectivity = (card_union > 0) ? card_intersection / card_union : 0.0;

  hll_r.DumpState("R-keys");
  hll_s.DumpState("S-keys");
  hll_union.DumpState("union");
  fprintf(stderr, "[AJB_STATE][JoinSelectivity] card_R=%.0f card_S=%.0f "
          "card_union=%.0f card_intersect=%.0f selectivity=%.6f\n",
          card_r, card_s, card_union, card_intersection, selectivity);

  return selectivity;
}

// ---------------------------------------------------------------------------
// Two-Sample Kolmogorov-Smirnov test: compare key distributions between
// two relations (or two GPU partitions) to detect distribution drift.
// Returns the KS distance and approximate p-value.
// Used by TierTransferScheduler to decide if repartitioning is needed.
// ---------------------------------------------------------------------------
struct TwoSampleKSResult {
  double ks_distance;     // max |F_1(x) - F_2(x)|
  double p_value;         // approximate p-value (Smirnov formula)
  size_t n1, n2;          // sample sizes
  bool reject_h0;         // reject H0: same distribution? (at alpha=0.05)

  void DebugPrint(const char* label) const {
    fprintf(stderr, "[AJB_BP][KS2Sample][%s] D=%.6f p=%.6f n1=%zu n2=%zu -> %s\n",
            label, ks_distance, p_value, n1, n2,
            reject_h0 ? "DIFFERENT distributions" : "same distribution");
  }
};

template <typename T>
TwoSampleKSResult TwoSampleKSTest(const T* data1, size_t n1,
                                    const T* data2, size_t n2,
                                    double alpha = 0.05,
                                    size_t max_samples = 100000) {
  TwoSampleKSResult result;
  result.n1 = std::min(n1, max_samples);
  result.n2 = std::min(n2, max_samples);

  // Reservoir sample if data too large
  std::vector<double> s1(result.n1), s2(result.n2);
  std::mt19937 rng(42);

  if (n1 <= max_samples) {
    for (size_t i = 0; i < n1; ++i) s1[i] = static_cast<double>(data1[i]);
  } else {
    // Algorithm R reservoir sampling
    for (size_t i = 0; i < max_samples; ++i) s1[i] = static_cast<double>(data1[i]);
    for (size_t i = max_samples; i < n1; ++i) {
      size_t j = rng() % (i + 1);
      if (j < max_samples) s1[j] = static_cast<double>(data1[i]);
    }
  }

  if (n2 <= max_samples) {
    for (size_t i = 0; i < n2; ++i) s2[i] = static_cast<double>(data2[i]);
  } else {
    for (size_t i = 0; i < max_samples; ++i) s2[i] = static_cast<double>(data2[i]);
    for (size_t i = max_samples; i < n2; ++i) {
      size_t j = rng() % (i + 1);
      if (j < max_samples) s2[j] = static_cast<double>(data2[i]);
    }
  }

  // Sort both samples
  std::sort(s1.begin(), s1.end());
  std::sort(s2.begin(), s2.end());

  // Merge-walk to compute KS statistic
  double max_diff = 0.0;
  size_t i = 0, j = 0;
  while (i < result.n1 && j < result.n2) {
    double cdf1 = static_cast<double>(i + 1) / result.n1;
    double cdf2 = static_cast<double>(j + 1) / result.n2;
    double diff = std::fabs(cdf1 - cdf2);
    if (diff > max_diff) max_diff = diff;

    if (s1[i] <= s2[j]) ++i; else ++j;
  }
  // Handle remaining elements
  while (i < result.n1) {
    double cdf1 = static_cast<double>(i + 1) / result.n1;
    double diff = std::fabs(cdf1 - 1.0);
    if (diff > max_diff) max_diff = diff;
    ++i;
  }
  while (j < result.n2) {
    double cdf2 = static_cast<double>(j + 1) / result.n2;
    double diff = std::fabs(0.0 - cdf2);
    if (diff > max_diff) max_diff = diff;
    ++j;
  }

  result.ks_distance = max_diff;

  // Approximate p-value using asymptotic Smirnov distribution
  double n_eff = static_cast<double>(result.n1 * result.n2)
                 / (result.n1 + result.n2);
  double lambda = (std::sqrt(n_eff) + 0.12 + 0.11 / std::sqrt(n_eff)) * max_diff;

  // Kolmogorov-Smirnov asymptotic p-value: P(D > d) ≈ 2 Σ (-1)^{k-1} exp(-2k²λ²)
  double p = 0.0;
  for (int k = 1; k <= 20; ++k) {
    double sign = (k % 2 == 1) ? 1.0 : -1.0;
    double term = sign * std::exp(-2.0 * k * k * lambda * lambda);
    p += term;
  }
  result.p_value = std::max(0.0, std::min(1.0, 2.0 * p));
  result.reject_h0 = (result.p_value < alpha);

  return result;
}

// ---------------------------------------------------------------------------
// Entropy-based adaptive bucket count: choose the number of histogram buckets
// that maximizes the information content (Shannon entropy) of the histogram.
// This replaces Freedman-Diaconis when the distribution is multimodal.
// ---------------------------------------------------------------------------
template <typename T>
size_t EntropyOptimalBuckets(const T* data, size_t n,
                              size_t min_buckets = 8,
                              size_t max_buckets = 512) {
  if (n < 2 * min_buckets) return min_buckets;

  // Find data range
  T min_val = data[0], max_val = data[0];
  for (size_t i = 1; i < n; ++i) {
    if (data[i] < min_val) min_val = data[i];
    if (data[i] > max_val) max_val = data[i];
  }
  if (max_val <= min_val) return min_buckets;

  double best_entropy = -1.0;
  size_t best_k = min_buckets;

  // Search over candidate bucket counts (powers of 2 for speed)
  for (size_t k = min_buckets; k <= max_buckets; k *= 2) {
    std::vector<size_t> counts(k, 0);
    double range = static_cast<double>(max_val - min_val);

    for (size_t i = 0; i < n; ++i) {
      double norm = static_cast<double>(data[i] - min_val) / range;
      size_t bucket = static_cast<size_t>(norm * (k - 1));
      bucket = std::min(bucket, k - 1);
      counts[bucket]++;
    }

    // Shannon entropy: H = -Σ p_i log2(p_i)
    double H = 0.0;
    for (size_t b = 0; b < k; ++b) {
      if (counts[b] == 0) continue;
      double p = static_cast<double>(counts[b]) / n;
      H -= p * std::log2(p);
    }

    // Penalize for too many empty buckets (Bayesian Information Criterion style)
    size_t nonempty = 0;
    for (size_t b = 0; b < k; ++b) if (counts[b] > 0) nonempty++;
    double penalty = 0.5 * static_cast<double>(nonempty) * std::log2(n) / n;
    double score = H - penalty;

    if (score > best_entropy) {
      best_entropy = score;
      best_k = k;
    }
  }

  fprintf(stderr, "[AJB_BP][EntropyBuckets] n=%zu optimal_k=%zu entropy=%.4f\n",
          n, best_k, best_entropy);
  return best_k;
}
