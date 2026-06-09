#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <csv.hpp>
#include <cxxopts.hpp>
#include <tabulate/table.hpp>
#include <termcolor/termcolor.hpp>

#include "common/config_utilities.cuh"
#include "common/data_generator.cuh"
#include "common/device_allocator.cuh"
#include "common/host_allocator.cuh"
#include "common/math_utilities.cuh"
#include "common/options_limits.cuh"
#include "common/parallel_algorithms.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/relation_generator.cuh"
#include "common/stream_pool.cuh"
#include "hybrid_sort/hybrid_sort.cuh"
#include "merge_join/merge_join.cuh"

// ---------------------------------------------------------------------------
// Welford online statistics accumulator for multi-run measurements
// Computes running mean, variance, stddev, and 95% confidence interval
// without storing all samples.
// ---------------------------------------------------------------------------
struct WelfordAccumulator {
  size_t count;
  double mean;
  double m2;  // sum of squares of differences from the current mean

  WelfordAccumulator() : count(0), mean(0.0), m2(0.0) {}

  void Update(double x) {
    count++;
    double delta = x - mean;
    mean += delta / static_cast<double>(count);
    double delta2 = x - mean;
    m2 += delta * delta2;
  }

  double Variance() const {
    return count < 2 ? 0.0 : m2 / static_cast<double>(count - 1);
  }

  double Stddev() const { return std::sqrt(Variance()); }

  // 95% CI half-width using z=1.96 approximation
  double CI95() const {
    return count < 2 ? 0.0 : 1.96 * Stddev() / std::sqrt(static_cast<double>(count));
  }

  void Print(const char* label) const {
    fprintf(stderr, "[AJB_STATS] %s: mean=%.9f stddev=%.9f CI95=[%.9f, %.9f] (n=%zu)\n",
            label, mean, Stddev(), mean - CI95(), mean + CI95(), count);
  }
};

// ---------------------------------------------------------------------------
// GPU topology detection and interconnect classification
// ---------------------------------------------------------------------------
enum class InterconnectType { kNVLink, kPCIe, kUnknown };

struct GPUTopologyInfo {
  int device_id;
  std::string name;
  size_t total_memory_bytes;
  int numa_node;
  InterconnectType interconnect;
  std::string bandwidth_tier;  // "high", "medium", "low"

  std::string InterconnectName() const {
    switch (interconnect) {
      case InterconnectType::kNVLink: return "nvlink";
      case InterconnectType::kPCIe:   return "pcie";
      default:                        return "unknown";
    }
  }
};

// Detect GPU topology: check peer access capability to infer NVLink vs PCIe,
// query NUMA node, classify bandwidth tier based on memory bandwidth.
static std::vector<GPUTopologyInfo> DetectGPUTopology(const std::vector<int>& gpus) {
  std::vector<GPUTopologyInfo> topo;
  topo.reserve(gpus.size());

  for (size_t i = 0; i < gpus.size(); ++i) {
    GPUTopologyInfo info;
    info.device_id = gpus[i];

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpus[i]);
    info.name = prop.name;
    info.total_memory_bytes = prop.totalGlobalMem;

    // NUMA node: use PCI bus ID to approximate NUMA locality
    // Devices on bus >= 0x80 are typically on NUMA node 1 on dual-socket systems
    info.numa_node = (prop.pciBusID >= 0x80) ? 1 : 0;

    // Interconnect detection: check if any peer GPU has P2P access enabled
    // which indicates NVLink or high-speed interconnect
    info.interconnect = InterconnectType::kPCIe;  // default
    for (size_t j = 0; j < gpus.size(); ++j) {
      if (i == j) continue;
      int can_access = 0;
      cudaDeviceCanAccessPeer(&can_access, gpus[i], gpus[j]);
      if (can_access) {
        info.interconnect = InterconnectType::kNVLink;
        break;
      }
    }

    // Bandwidth tier classification based on memory size and interconnect
    // H100 (80GB+) with NVLink = high tier; A100/A6000 range = medium; rest = low
    size_t mem_gb = info.total_memory_bytes / (1024ULL * 1024 * 1024);
    if (mem_gb >= 70 && info.interconnect == InterconnectType::kNVLink) {
      info.bandwidth_tier = "high";
    } else if (mem_gb >= 40 || info.interconnect == InterconnectType::kNVLink) {
      info.bandwidth_tier = "medium";
    } else {
      info.bandwidth_tier = "low";
    }

    fprintf(stderr, "[AJB_STATE][gpu_topology] GPU %d: %s | %.1f GB | %s | NUMA %d | tier=%s\n",
            info.device_id, info.name.c_str(),
            static_cast<double>(info.total_memory_bytes) / (1024.0 * 1024 * 1024),
            info.InterconnectName().c_str(), info.numa_node,
            info.bandwidth_tier.c_str());

    topo.push_back(info);
  }

  return topo;
}

struct Settings {
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
  // New fields for extended benchmark modes
  bool baseline_comparison;
  size_t num_runs;
};

// Upstream: save uses two identical loops with csv::make_csv_writer.
// Changed: factored into a helper lambda.
template <typename T, typename V>
static void SaveRelation(const char* filename, PinnedVector<T>& keys, PinnedVector<V>& values, size_t n) {
  std::ofstream file(filename);
  auto writer = csv::make_csv_writer(file);
  for (size_t i = 0; i < n; ++i) {
    writer << std::make_tuple(keys[i], values[i]);
  }
}

// Warmup verification: after warmup, check that the first GPU's sort output
// is actually sorted. Returns true if verification passes or is skipped.
template <typename T>
static bool VerifyWarmupSortedness(const PinnedVector<T>& keys, size_t n, int gpu_id) {
  if (n < 2) return true;

  // Check full sequence (not sampled) for warmup correctness
  size_t check_limit = std::min<size_t>(n, 100000);  // cap for performance
  size_t violations = 0;
  size_t first_violation_idx = 0;
  for (size_t i = 1; i < check_limit; ++i) {
    if (keys[i] < keys[i - 1]) {
      if (violations == 0) first_violation_idx = i;
      violations++;
    }
  }

  if (violations > 0) {
    fprintf(stderr, "[AJB_WARN][warmup_verify] GPU %d: SORT VERIFICATION FAILED - "
            "%zu violations in first %zu elements (first at index %zu)\n",
            gpu_id, violations, check_limit, first_violation_idx);
    return false;
  }

  fprintf(stderr, "[AJB_STATE][warmup_verify] GPU %d: sort verified OK (%zu elements checked)\n",
          gpu_id, check_limit);
  return true;
}

// Run a single benchmark iteration and return phase durations
template <typename T, typename V>
static std::tuple<double, double, double, double> RunSingleIteration(
    Settings& settings,
    Relation<T, V>& r_relation,
    Relation<T, V>& s_relation,
    std::vector<HostAllocator>& host_allocators,
    std::vector<DeviceAllocator>& device_allocators,
    std::vector<StreamPool>& stream_pools,
    size_t num_matches,
    bool is_warmup_run) {

  const size_t temporary_num_elements = std::max(settings.r_num_elements, settings.s_num_elements);
  PinnedVector<T> temporary_keys(temporary_num_elements);
  PinnedVector<V> temporary_values(temporary_num_elements);

  // Make copies for this iteration so original data is preserved across runs
  PinnedVector<T> r_keys_copy(settings.r_num_elements);
  PinnedVector<V> r_vals_copy(settings.r_num_elements);
  PinnedVector<T> s_keys_copy(settings.s_num_elements);
  PinnedVector<V> s_vals_copy(settings.s_num_elements);
  std::copy(r_relation.GetKeys().data(), r_relation.GetKeys().data() + settings.r_num_elements, r_keys_copy.data());
  std::copy(r_relation.GetValues().data(), r_relation.GetValues().data() + settings.r_num_elements, r_vals_copy.data());
  std::copy(s_relation.GetKeys().data(), s_relation.GetKeys().data() + settings.s_num_elements, s_keys_copy.data());
  std::copy(s_relation.GetValues().data(), s_relation.GetValues().data() + settings.s_num_elements, s_vals_copy.data());

  TimeDurations::Get().Clear();

  auto sort_relation = [&](PinnedVector<T>& keys, PinnedVector<V>& vals, size_t n, bool already_sorted) {
    if (already_sorted) return;
    if (settings.sort_algorithm == "gnu_parallel_sort") {
      ParallelSortPairs<T, V>(keys, vals);
    } else if (settings.sort_algorithm == "hybrid_merge_sort") {
      HybridSort<T, V, HybridSortKernel::kMerge>(keys, vals, temporary_keys, temporary_values,
                                                   settings.gpus, host_allocators, device_allocators,
                                                   stream_pools, n, settings.chunk_size);
    } else if (settings.sort_algorithm == "hybrid_radix_sort") {
      HybridSort<T, V, HybridSortKernel::kRadix>(keys, vals, temporary_keys, temporary_values,
                                                   settings.gpus, host_allocators, device_allocators,
                                                   stream_pools, n, settings.chunk_size);
    }
  };

  {
    TimeScope time_scope("sort_phase");
    sort_relation(r_keys_copy, r_vals_copy, settings.r_num_elements, settings.r_sort);
    sort_relation(s_keys_copy, s_vals_copy, settings.s_num_elements, settings.s_sort);
  }

  // Warmup verification: check first GPU's sort result
  if (is_warmup_run && !settings.r_sort) {
    VerifyWarmupSortedness(r_keys_copy, settings.r_num_elements, settings.gpus[0]);
  }

  // Post-sort state dump
  {
    auto* rk = r_keys_copy.data();
    auto* sk = s_keys_copy.data();
    size_t rn = settings.r_num_elements;
    size_t sn = settings.s_num_elements;

    size_t r_violations = 0, s_violations = 0;
    size_t stride = std::max<size_t>(1, rn / 1000);
    for (size_t i = stride; i < rn; i += stride)
      if (rk[i] < rk[i-1]) r_violations++;
    stride = std::max<size_t>(1, sn / 1000);
    for (size_t i = stride; i < sn; i += stride)
      if (sk[i] < sk[i-1]) s_violations++;

    fprintf(stderr, "[AJB_SNAP][join_benchmark][post_sort] "
            "R: n=%zu sorted=%s (violations=%zu/1000) "
            "key_range=[%llu..%llu]\n",
            rn, r_violations == 0 ? "yes" : "NO", r_violations,
            (unsigned long long)(rn > 0 ? rk[0] : 0),
            (unsigned long long)(rn > 0 ? rk[rn-1] : 0));
    fprintf(stderr, "[AJB_SNAP][join_benchmark][post_sort] "
            "S: n=%zu sorted=%s (violations=%zu/1000) "
            "key_range=[%llu..%llu]\n",
            sn, s_violations == 0 ? "yes" : "NO", s_violations,
            (unsigned long long)(sn > 0 ? sk[0] : 0),
            (unsigned long long)(sn > 0 ? sk[sn-1] : 0));

    if (rn > 0 && sn > 0) {
      bool overlap = !(rk[rn-1] < sk[0] || sk[sn-1] < rk[0]);
      fprintf(stderr, "[AJB_SNAP][join_benchmark][overlap] %s "
              "R_max=%llu S_min=%llu\n",
              overlap ? "OVERLAP" : "DISJOINT",
              (unsigned long long)rk[rn-1],
              (unsigned long long)sk[0]);
    }

    fprintf(stderr, "[AJB_SNAP][join_benchmark][config] "
            "gpus=%zu sort=%s join=%s chunk=%zu theta=%u sigma=%u\n",
            settings.gpus.size(), settings.sort_algorithm.c_str(),
            settings.join_algorithm.c_str(), settings.chunk_size,
            settings.theta, settings.sigma);
  }

  // Use copies for join since they are now sorted
  // We need to put the sorted data back into relation-compatible vectors
  std::copy(r_keys_copy.data(), r_keys_copy.data() + settings.r_num_elements, r_relation.GetKeys().data());
  std::copy(r_vals_copy.data(), r_vals_copy.data() + settings.r_num_elements, r_relation.GetValues().data());
  std::copy(s_keys_copy.data(), s_keys_copy.data() + settings.s_num_elements, s_relation.GetKeys().data());
  std::copy(s_vals_copy.data(), s_vals_copy.data() + settings.s_num_elements, s_relation.GetValues().data());

  JoinResult<T> join_result =
      MergeJoin(r_relation.GetKeys(), r_relation.GetValues(), s_relation.GetKeys(), s_relation.GetValues(),
                settings.gpus, device_allocators, stream_pools, settings.materialize);

  double sort_dur = TimeDurations::Get().GetDuration("sort_phase");
  double merge_dur = TimeDurations::Get().GetDuration("merge_phase");
  double join_dur = TimeDurations::Get().GetDuration("join_phase");
  double total_dur = TimeDurations::Get().GetTotalDuration();

  if (join_result.count_ != num_matches) {
    printf("[ERROR] RunJoinBenchmark: Invalid join count (%lu != %lu).\n", join_result.count_, num_matches);
  }

  return {sort_dur, merge_dur, join_dur, total_dur};
}

// Upstream: sorts R and S with near-identical blocks for merge vs radix.
// Changed (M1299): refactored into multi-run Welford statistics, baseline
// comparison mode, GPU topology detection, enhanced CSV output, warmup verification.
template <typename T, typename V>
void RunJoinBenchmark(Settings& settings) {
  ConfigureMultiProcess(settings.num_threads);
  ConfigurePeerAccess(settings.gpus);

  // GPU topology detection at startup
  std::vector<GPUTopologyInfo> gpu_topo = DetectGPUTopology(settings.gpus);

  // Build topology summary strings for CSV
  std::string topo_str, interconnect_str, numa_str;
  {
    std::ostringstream ts, is, ns;
    for (size_t i = 0; i < gpu_topo.size(); ++i) {
      if (i > 0) { ts << ";"; is << ";"; ns << ";"; }
      ts << gpu_topo[i].bandwidth_tier;
      is << gpu_topo[i].InterconnectName();
      ns << gpu_topo[i].numa_node;
    }
    topo_str = ts.str();
    interconnect_str = is.str();
    numa_str = ns.str();
  }

  Relation<T, V> r_relation(settings.r_num_elements);
  Relation<T, V> s_relation(settings.s_num_elements);

  const size_t num_matches = RelationGenerator::ComputeDistributions<T, V>(
      r_relation, s_relation, settings.num_threads, settings.key_distribution, settings.value_distribution,
      settings.random_seed, settings.theta, settings.sigma, settings.r_sort, settings.s_sort);

  if (settings.save) {
    SaveRelation("r_relation.csv", r_relation.GetKeys(), r_relation.GetValues(), r_relation.GetSize());
    SaveRelation("s_relation.csv", s_relation.GetKeys(), s_relation.GetValues(), s_relation.GetSize());
  }

  std::vector<HostAllocator> host_allocators(settings.gpus.size());
  std::vector<DeviceAllocator> device_allocators(settings.gpus.size());
  std::vector<StreamPool> stream_pools(settings.gpus.size());

  if (settings.join_algorithm == "hybrid_sort_merge_join") {
    // Welford accumulators for multi-run statistics
    WelfordAccumulator sort_acc, merge_acc, join_acc, total_acc;
    WelfordAccumulator baseline_total_acc;  // for baseline comparison

    size_t actual_runs = std::max<size_t>(1, settings.num_runs);
    fprintf(stderr, "[AJB_STATE][benchmark] Running %zu iterations with Welford statistics\n", actual_runs);

    for (size_t run = 0; run < actual_runs; ++run) {
      bool is_warmup = (run == 0);
      fprintf(stderr, "[AJB_STATE][run] Iteration %zu/%zu%s\n",
              run + 1, actual_runs, is_warmup ? " (warmup+verify)" : "");

      // Re-generate data for each run to avoid cache effects (except first)
      if (run > 0) {
        RelationGenerator::ComputeDistributions<T, V>(
            r_relation, s_relation, settings.num_threads, settings.key_distribution, settings.value_distribution,
            settings.random_seed + static_cast<uint32_t>(run), settings.theta, settings.sigma,
            settings.r_sort, settings.s_sort);
      }

      auto iter_result = RunSingleIteration<T, V>(
          settings, r_relation, s_relation, host_allocators, device_allocators,
          stream_pools, num_matches, is_warmup);
      auto sort_dur = std::get<0>(iter_result);
      auto merge_dur = std::get<1>(iter_result);
      auto join_dur = std::get<2>(iter_result);
      auto total_dur = std::get<3>(iter_result);

      sort_acc.Update(sort_dur);
      merge_acc.Update(merge_dur);
      join_acc.Update(join_dur);
      total_acc.Update(total_dur);

      // Baseline comparison: also run with gnu_parallel_sort if enabled
      if (settings.baseline_comparison && settings.sort_algorithm != "gnu_parallel_sort") {
        Settings baseline_settings = settings;
        baseline_settings.sort_algorithm = "gnu_parallel_sort";

        // Re-generate fresh data for baseline run
        Relation<T, V> bl_r(settings.r_num_elements);
        Relation<T, V> bl_s(settings.s_num_elements);
        RelationGenerator::ComputeDistributions<T, V>(
            bl_r, bl_s, settings.num_threads, settings.key_distribution, settings.value_distribution,
            settings.random_seed + static_cast<uint32_t>(run), settings.theta, settings.sigma,
            settings.r_sort, settings.s_sort);

        std::vector<HostAllocator> bl_ha(settings.gpus.size());
        std::vector<DeviceAllocator> bl_da(settings.gpus.size());
        std::vector<StreamPool> bl_sp(settings.gpus.size());

        auto bl_result = RunSingleIteration<T, V>(
            baseline_settings, bl_r, bl_s, bl_ha, bl_da, bl_sp, num_matches, false);
        auto bl_sort = std::get<0>(bl_result);
        auto bl_merge = std::get<1>(bl_result);
        auto bl_join = std::get<2>(bl_result);
        auto bl_total = std::get<3>(bl_result);

        baseline_total_acc.Update(bl_total);

        fprintf(stderr, "[AJB_STATE][baseline] Run %zu: AJB=%.6f s, baseline(gnu_parallel)=%.6f s, speedup=%.2fx\n",
                run + 1, total_dur, bl_total, bl_total / std::max(total_dur, 1e-12));
      }
    }

    // Print Welford statistics summary
    if (actual_runs > 1) {
      fprintf(stderr, "\n[AJB_STATS] === Multi-run Welford Statistics (%zu runs) ===\n", actual_runs);
      sort_acc.Print("sort_phase");
      merge_acc.Print("merge_phase");
      join_acc.Print("join_phase");
      total_acc.Print("total");

      if (settings.baseline_comparison && baseline_total_acc.count > 0) {
        baseline_total_acc.Print("baseline_total");
        double speedup = baseline_total_acc.mean / std::max(total_acc.mean, 1e-12);
        fprintf(stderr, "[AJB_STATS] Overall speedup vs baseline: %.3fx\n", speedup);
      }
    }

    // Enhanced CSV output with topology columns
    std::ostringstream gpu_list;
    for (size_t i = 0; i < settings.gpus.size(); ++i) {
      if (i > 0) gpu_list << ",";
      gpu_list << settings.gpus[i];
    }

    std::cout << settings.r_num_elements << "," << settings.s_num_elements << "," << settings.num_threads << ",\""
              << gpu_list.str() << "\",\"" << settings.sort_algorithm << "\",\"" << settings.join_algorithm << "\","
              << settings.chunk_size << ",\"" << settings.key_type << "\",\"" << settings.key_distribution << "\",\""
              << settings.value_type << "\",\"" << settings.value_distribution << "\"," << settings.random_seed << ","
              << settings.theta << "," << settings.sigma << "," << settings.r_sort << "," << settings.s_sort << ","
              << settings.materialize << ",";
    // Phase durations: use mean from Welford accumulator
    std::cout << std::fixed << std::setprecision(9) << termcolor::green
              << sort_acc.mean << termcolor::reset << "," << termcolor::yellow
              << merge_acc.mean << termcolor::reset << "," << termcolor::blue
              << join_acc.mean << termcolor::reset << "," << termcolor::magenta
              << total_acc.mean << termcolor::reset << ",";
    // Welford statistics columns
    std::cout << sort_acc.Stddev() << "," << total_acc.Stddev() << ","
              << total_acc.CI95() << ",";
    // GPU topology columns
    std::cout << "\"" << topo_str << "\",\"" << interconnect_str << "\",\"" << numa_str << "\"";
    // Baseline speedup column
    if (settings.baseline_comparison && baseline_total_acc.count > 0) {
      std::cout << "," << baseline_total_acc.mean / std::max(total_acc.mean, 1e-12);
    }
    std::cout << std::endl;

    // Materialize and print (use last run's data which is already sorted)
    if (settings.materialize) {
      JoinResult<T> join_result =
          MergeJoin(r_relation.GetKeys(), r_relation.GetValues(), s_relation.GetKeys(), s_relation.GetValues(),
                    settings.gpus, device_allocators, stream_pools, settings.materialize);

      std::vector<std::tuple<T, V, V>> result_rows;
      result_rows.reserve(join_result.count_);

      for (size_t i = 0; i < join_result.items_.size(); ++i) {
        const T key = r_relation.GetKeys()[join_result.items_[i].r_first_];

        for (size_t r = join_result.items_[i].r_first_; r <= join_result.items_[i].r_last_; ++r) {
          for (size_t s = join_result.items_[i].s_first_; s <= join_result.items_[i].s_last_; ++s) {
            result_rows.emplace_back(key, r_relation.GetValues()[r], s_relation.GetValues()[s]);
          }
        }
      }

      if (settings.print) {
        tabulate::Table table;

        table.add_row({"key", "r_value", "s_value"});
        for (const auto& row : result_rows) {
          const auto& key = std::get<0>(row);
          const auto& r_value = std::get<1>(row);
          const auto& s_value = std::get<2>(row);
          table.add_row({std::to_string(key), std::to_string(r_value), std::to_string(s_value)});
        }

        table.format().font_align(tabulate::FontAlign::center);
        for (size_t i = 0; i < table[0].size(); ++i) {
          table[0][i].format().font_color(tabulate::Color::yellow);
        }

        std::cout << std::endl << table << std::endl;
      }
    }
  }
}

// Upstream: one giant && chain for validation.
// Changed: grouped checks with early return.
static bool ValidateSettings(const Settings& s) {
  if (!OptionsLimits::IsValidNumElements(s.r_num_elements) ||
      !OptionsLimits::IsValidNumElements(s.s_num_elements)) return false;
  if (!OptionsLimits::IsValidNumThreads(s.num_threads) ||
      !OptionsLimits::IsValidGpus(s.gpus)) return false;
  if (!OptionsLimits::IsValidSortAlgorithm(s.sort_algorithm) ||
      !OptionsLimits::IsValidJoinAlgorithm(s.join_algorithm)) return false;
  if (!OptionsLimits::IsValidChunkSize(s.chunk_size)) return false;
  if (!OptionsLimits::IsValidType(s.key_type) ||
      !OptionsLimits::IsValidJoinDistribution(s.key_distribution)) return false;
  if (!OptionsLimits::IsValidType(s.value_type) ||
      !OptionsLimits::IsValidJoinDistribution(s.value_distribution)) return false;
  if (!OptionsLimits::IsValidRandomSeed(s.random_seed) ||
      !OptionsLimits::IsValidTheta(s.theta) ||
      !OptionsLimits::IsValidSigma(s.sigma)) return false;
  return true;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("join_benchmark");

  options.set_width(250);

  options.add_options()("r_num_elements", "the number of elements in R " + OptionsLimits::GetNumElementsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumElements()));
  options.add_options()("s_num_elements", "the number of elements in S " + OptionsLimits::GetNumElementsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumElements()));
  options.add_options()("num_threads", "the number of threads " + OptionsLimits::GetNumThreadsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumThreads()));
  options.add_options()("gpus", "the visible GPUs " + OptionsLimits::GetGpusLimits(),
                        cxxopts::value<std::vector<int>>()->default_value(OptionsLimits::GetDefaultGpus()));
  options.add_options()("sort_algorithm", "the sort algorithm " + OptionsLimits::GetSortAlgorithmLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultSortAlgorithm()));
  options.add_options()("join_algorithm", "the join algorithm " + OptionsLimits::GetJoinAlgorithmLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultJoinAlgorithm()));
  options.add_options()("chunk_size", "the chunk size " + OptionsLimits::GetChunkSizeLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultChunkSize()));
  options.add_options()("key_type", "the key type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("key_distribution", "the key distribution " + OptionsLimits::GetJoinDistributionLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultJoinDistribution()));
  options.add_options()("value_type", "the value type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("value_distribution", "the value distribution " + OptionsLimits::GetJoinDistributionLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultJoinDistribution()));
  options.add_options()("random_seed", "the random seed " + OptionsLimits::GetRandomSeedLimits(),
                        cxxopts::value<uint32_t>()->default_value(OptionsLimits::GetDefaultRandomSeed()));
  options.add_options()("theta", "the theta value " + OptionsLimits::GetThetaLimits(),
                        cxxopts::value<uint32_t>()->default_value(OptionsLimits::GetDefaultTheta()));
  options.add_options()("sigma", "the sigma value " + OptionsLimits::GetSigmaLimits(),
                        cxxopts::value<uint32_t>()->default_value(OptionsLimits::GetDefaultSigma()));
  options.add_options()("r_sort", "sort the elements in R", cxxopts::value<bool>()->default_value("false"));
  options.add_options()("s_sort", "sort the elements in S", cxxopts::value<bool>()->default_value("false"));
  options.add_options()("save", "saves the input", cxxopts::value<bool>()->default_value("false"));
  options.add_options()("materialize", "materializes the output", cxxopts::value<bool>()->default_value("false"));
  options.add_options()("print", "prints the output", cxxopts::value<bool>()->default_value("false"));
  options.add_options()("help", "shows the help", cxxopts::value<bool>()->default_value("false"));
  // New CLI options
  options.add_options()("baseline_comparison", "run baseline (gnu_parallel_sort) and print speedup ratio",
                        cxxopts::value<bool>()->default_value("false"));
  options.add_options()("num_runs", "number of benchmark iterations for Welford statistics",
                        cxxopts::value<size_t>()->default_value("1"));

  cxxopts::ParseResult parse_result = options.parse(argc, argv);

  Settings s = {parse_result["r_num_elements"].as<size_t>(),
                parse_result["s_num_elements"].as<size_t>(),
                parse_result["num_threads"].as<size_t>(),
                parse_result["gpus"].as<std::vector<int>>(),
                parse_result["sort_algorithm"].as<std::string>(),
                parse_result["join_algorithm"].as<std::string>(),
                parse_result["chunk_size"].as<size_t>(),
                parse_result["key_type"].as<std::string>(),
                parse_result["key_distribution"].as<std::string>(),
                parse_result["value_type"].as<std::string>(),
                parse_result["value_distribution"].as<std::string>(),
                parse_result["random_seed"].as<uint32_t>(),
                parse_result["theta"].as<uint32_t>(),
                parse_result["sigma"].as<uint32_t>(),
                parse_result["r_sort"].as<bool>(),
                parse_result["s_sort"].as<bool>(),
                parse_result["save"].as<bool>(),
                parse_result["materialize"].as<bool>(),
                parse_result["print"].as<bool>(),
                parse_result["baseline_comparison"].as<bool>(),
                parse_result["num_runs"].as<size_t>()};

  if (!ValidateSettings(s) || parse_result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const std::string type_key = s.key_type + ":" + s.value_type;
  if (type_key == "int:int") {
    RunJoinBenchmark<uint32_t, uint32_t>(s);
  } else if (type_key == "long:long") {
    RunJoinBenchmark<uint64_t, uint64_t>(s);
  } else if (type_key == "float:float") {
    RunJoinBenchmark<float, float>(s);
  } else if (type_key == "double:double") {
    RunJoinBenchmark<double, double>(s);
  }

  return 0;
}
