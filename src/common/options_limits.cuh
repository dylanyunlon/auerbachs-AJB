#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

class OptionsLimits {
 public:
  OptionsLimits() = delete;

  static std::string GetDefaultNumElements() { return std::to_string(10000); }

  static std::string GetDefaultNumThreads() { return std::to_string(kValidNumThreads.second); }

  static std::string GetDefaultGpus() {
    std::stringstream stream;
    for (size_t i = 0; i < kValidGpus.size(); ++i) {
      stream << kValidGpus[i] << (i < kValidGpus.size() - 1 ? "," : "");
    }
    return stream.str();
  }

  static std::string GetDefaultSortAlgorithm() { return kValidSortAlgorithms.front(); }

  static std::string GetDefaultJoinAlgorithm() { return kValidJoinAlgorithms.front(); }

  static std::string GetDefaultCpuMergeAlgorithm() { return kValidCpuMergeAlgorithms.front(); }

  static std::string GetDefaultCpuSortAlgorithm() { return kValidCpuSortAlgorithms.front(); }

  static std::string GetDefaultGpuMergeAlgorithm() { return kValidGpuMergeAlgorithms.front(); }

  static std::string GetDefaultGpuSortAlgorithm() { return kValidGpuSortAlgorithms.front(); }

  static std::string GetDefaultChunkSize() { return std::to_string(0); }

  static std::string GetDefaultChunkCount() { return std::to_string(kValidChunkCounts.first); }

  static std::string GetDefaultType() { return kValidTypes.front(); }

  static std::string GetDefaultSortDistribution() { return kValidSortDistributions.front(); }

  static std::string GetDefaultJoinDistribution() { return kValidJoinDistributions.front(); }

  static std::string GetDefaultRandomSeed() { return std::to_string(2147483647); }

  static std::string GetDefaultTheta() { return std::to_string(0); }

  static std::string GetDefaultSigma() { return std::to_string(100); }

  static std::string GetNumElementsLimits() { return LimitsToString(kValidNumElements); }

  static std::string GetNumThreadsLimits() { return LimitsToString(kValidNumThreads); }

  static std::string GetGpusLimits() { return LimitsToString(kValidGpus); }

  static std::string GetSortAlgorithmLimits() { return LimitsToString(kValidSortAlgorithms); }

  static std::string GetJoinAlgorithmLimits() { return LimitsToString(kValidJoinAlgorithms); }

  static std::string GetCpuMergeAlgorithmLimits() { return LimitsToString(kValidCpuMergeAlgorithms); }

  static std::string GetCpuSortAlgorithmLimits() { return LimitsToString(kValidCpuSortAlgorithms); }

  static std::string GetGpuMergeAlgorithmLimits() { return LimitsToString(kValidGpuMergeAlgorithms); }

  static std::string GetGpuSortAlgorithmLimits() { return LimitsToString(kValidGpuSortAlgorithms); }

  static std::string GetChunkSizeLimits() { return LimitsToString(kValidChunkSizes); }

  static std::string GetChunkCountLimits() { return LimitsToString(kValidChunkCounts); }

  static std::string GetTypeLimits() { return LimitsToString(kValidTypes); }

  static std::string GetSortDistributionLimits() { return LimitsToString(kValidSortDistributions); }

  static std::string GetJoinDistributionLimits() { return LimitsToString(kValidJoinDistributions); }

  static std::string GetRandomSeedLimits() { return LimitsToString(kValidRandomSeeds); }

  static std::string GetThetaLimits() { return LimitsToString(kValidThetas); }

  static std::string GetSigmaLimits() { return LimitsToString(kValidSigmas); }

  static bool IsValidNumElements(size_t num_elements) {
    return num_elements >= kValidNumElements.first && num_elements <= kValidNumElements.second;
  }

  static bool IsValidNumThreads(size_t num_threads) {
    return num_threads >= kValidNumThreads.first && num_threads <= kValidNumThreads.second;
  }

  // Upstream: IsValidGpus constructs a std::set to check uniqueness,
  // then does a linear find per element.  O(n log n + n*k).
  // Changed: sort a copy and use std::unique for O(n log n) uniqueness
  // check without heap-allocating a std::set.
  static bool IsValidGpus(const std::vector<int>& gpus) {
    std::vector<int> sorted_gpus(gpus);
    std::sort(sorted_gpus.begin(), sorted_gpus.end());
    if (std::unique(sorted_gpus.begin(), sorted_gpus.end()) != sorted_gpus.end()) {
      return false;
    }

    return std::all_of(gpus.begin(), gpus.end(), [](int gpu) {
      return std::find(kValidGpus.begin(), kValidGpus.end(), gpu) != kValidGpus.end();
    });
  }

  // Upstream: 7 separate IsValid*Algorithm functions that are identical
  // except for the container they search.
  // Changed: single generic helper that works with any container.
  template <typename Container>
  static bool IsValidOption(const Container& valid_options, const typename Container::value_type& value) {
    return std::find(valid_options.begin(), valid_options.end(), value) != valid_options.end();
  }

  static bool IsValidSortAlgorithm(const std::string& v) { return IsValidOption(kValidSortAlgorithms, v); }
  static bool IsValidJoinAlgorithm(const std::string& v) { return IsValidOption(kValidJoinAlgorithms, v); }
  static bool IsValidCpuMergeAlgorithm(const std::string& v) { return IsValidOption(kValidCpuMergeAlgorithms, v); }
  static bool IsValidCpuSortAlgorithm(const std::string& v) { return IsValidOption(kValidCpuSortAlgorithms, v); }
  static bool IsValidGpuMergeAlgorithm(const std::string& v) { return IsValidOption(kValidGpuMergeAlgorithms, v); }
  static bool IsValidGpuSortAlgorithm(const std::string& v) { return IsValidOption(kValidGpuSortAlgorithms, v); }

  // Upstream: range checks written separately for each type.
  // Changed: generic range checker.
  template <typename T>
  static bool IsInRange(T value, const std::pair<T, T>& range) {
    return value >= range.first && value <= range.second;
  }

  static bool IsValidChunkSize(size_t v) { return IsInRange(v, kValidChunkSizes); }
  static bool IsValidChunkCount(size_t v) { return IsInRange(v, kValidChunkCounts); }

  static bool IsValidType(const std::string& v) { return IsValidOption(kValidTypes, v); }
  static bool IsValidSortDistribution(const std::string& v) { return IsValidOption(kValidSortDistributions, v); }
  static bool IsValidJoinDistribution(const std::string& v) { return IsValidOption(kValidJoinDistributions, v); }

  static bool IsValidRandomSeed(uint32_t v) { return IsInRange(v, kValidRandomSeeds); }
  static bool IsValidTheta(uint32_t v) { return IsInRange(v, kValidThetas); }
  static bool IsValidSigma(uint32_t v) { return IsInRange(v, kValidSigmas); }

 private:
  template <typename T>
  static std::string LimitsToString(const std::pair<T, T>& limits) {
    std::stringstream stream;
    stream << "[" << limits.first << ", " << limits.second << "]";
    return stream.str();
  }

  template <typename T>
  static std::string LimitsToString(const std::vector<T>& limits) {
    std::stringstream stream;
    stream << "{";
    std::copy(limits.begin(), limits.end(), std::ostream_iterator<T>(stream, ", "));
    stream << (!limits.empty() ? "\b\b" : "") << "}";
    return stream.str();
  }

  static const std::pair<size_t, size_t> kValidNumElements;
  static const std::pair<size_t, size_t> kValidNumThreads;
  static const std::vector<int> kValidGpus;
  static const std::vector<std::string> kValidSortAlgorithms;
  static const std::vector<std::string> kValidJoinAlgorithms;
  static const std::vector<std::string> kValidCpuMergeAlgorithms;
  static const std::vector<std::string> kValidCpuSortAlgorithms;
  static const std::vector<std::string> kValidGpuMergeAlgorithms;
  static const std::vector<std::string> kValidGpuSortAlgorithms;
  static const std::pair<size_t, size_t> kValidChunkSizes;
  static const std::pair<size_t, size_t> kValidChunkCounts;
  static const std::vector<std::string> kValidTypes;
  static const std::vector<std::string> kValidSortDistributions;
  static const std::vector<std::string> kValidJoinDistributions;
  static const std::pair<uint32_t, uint32_t> kValidRandomSeeds;
  static const std::pair<uint32_t, uint32_t> kValidThetas;
  static const std::pair<uint32_t, uint32_t> kValidSigmas;
};

const std::pair<size_t, size_t> OptionsLimits::kValidNumElements = {0, std::numeric_limits<size_t>::max()};
const std::pair<size_t, size_t> OptionsLimits::kValidNumThreads = {1, omp_get_num_procs()};
const std::vector<int> OptionsLimits::kValidGpus = []() {
  int cuda_device_count = 0;
  cudaGetDeviceCount(&cuda_device_count);

  std::vector<int> valid_gpus(cuda_device_count);
  std::iota(valid_gpus.begin(), valid_gpus.end(), 0);

  return valid_gpus;
}();
const std::vector<std::string> OptionsLimits::kValidSortAlgorithms = {"gnu_parallel_sort", "hybrid_merge_sort",
                                                                      "hybrid_radix_sort"};
const std::vector<std::string> OptionsLimits::kValidJoinAlgorithms = {"hybrid_sort_merge_join"};
const std::vector<std::string> OptionsLimits::kValidCpuMergeAlgorithms = {"gnu_parallel_multiway_merge"};
const std::vector<std::string> OptionsLimits::kValidCpuSortAlgorithms = {"gnu_parallel_sort"};
const std::vector<std::string> OptionsLimits::kValidGpuMergeAlgorithms = {"thrust_merge_by_key", "mgpu_merge"};
const std::vector<std::string> OptionsLimits::kValidGpuSortAlgorithms = {"thrust_sort_by_key", "mgpu_mergesort",
                                                                         "cub_deviceradixsort_sortpairs"};
const std::pair<size_t, size_t> OptionsLimits::kValidChunkSizes = {0, std::numeric_limits<size_t>::max()};
const std::pair<size_t, size_t> OptionsLimits::kValidChunkCounts = {2, std::numeric_limits<size_t>::max()};
const std::vector<std::string> OptionsLimits::kValidTypes = {"int", "long", "float", "double"};
const std::vector<std::string> OptionsLimits::kValidSortDistributions = {
    "uniform",        "normal",        "zero",          "staggered", "sorted",
    "reverse-sorted", "nearly-sorted", "bucket-sorted", "zipf",      "self"};
const std::vector<std::string> OptionsLimits::kValidJoinDistributions = {"unique_full_key_range",
                                                                         "unique_partial_key_range"};
const std::pair<uint32_t, uint32_t> OptionsLimits::kValidRandomSeeds = {0, std::numeric_limits<uint32_t>::max()};
const std::pair<uint32_t, uint32_t> OptionsLimits::kValidThetas = {0, 100};
const std::pair<uint32_t, uint32_t> OptionsLimits::kValidSigmas = {0, 100};
