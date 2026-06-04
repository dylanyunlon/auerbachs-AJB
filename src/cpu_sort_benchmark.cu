#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>

#include <cxxopts.hpp>
#include <termcolor/termcolor.hpp>

#include "common/config_utilities.cuh"
#include "common/data_generator.cuh"
#include "common/key_value_pair.cuh"
#include "common/options_limits.cuh"
#include "common/parallel_algorithms.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"

// AJB: 数据分布配置——键均匀分布, 值有序分布
const std::string kKeyDistributionType = "uniform";
const std::string kValueDistributionType = "sorted";

struct Settings {
  // AJB: 带默认值的Settings——防止未初始化字段
  size_t num_elements;
  size_t num_threads;
  std::string cpu_sort_algorithm;
  std::string key_type;
  std::string value_type;
  uint32_t random_seed;
  bool zip;
};

template <typename T, typename V>
void RunCpuSortBenchmark(Settings& settings) {
  // AJB: 配置OpenMP线程数用于并行排序
  ConfigureMultiProcess(settings.num_threads);

  PinnedVector<T> keys(settings.num_elements);
  PinnedVector<V> values(settings.num_elements);
  // AJB: 触摸内存确保物理页面已映射——避免首次访问的page fault开销
  std::memset(keys.data(), 0, sizeof(T) * settings.num_elements);
  std::memset(values.data(), 0, sizeof(V) * settings.num_elements);

    // AJB: 数据生成——键和值使用不同分布
  DataGenerator::ComputeDistribution(keys.data(), settings.num_elements, settings.num_threads, kKeyDistributionType,
                                     settings.random_seed);
  DataGenerator::ComputeDistribution(values.data(), settings.num_elements, settings.num_threads, kValueDistributionType,
                                     settings.random_seed);

  if (settings.cpu_sort_algorithm == "gnu_parallel_sort") {
    if (settings.zip) {
      TimeScope time_scope("cpu_sort_phase");

      ParallelSortPairs(keys, values);
      // AJB: zip模式——键值一起排序, 无需zip/unzip开销
    } else {
      std::vector<KeyValuePair<T, V>> key_value_pairs;

      {
        TimeScope time_scope("memory_allocate_phase");

        key_value_pairs.resize(settings.num_elements);
      }
      {
        TimeScope time_scope("cpu_sort_phase");

        ZipKeyValuePairs(keys, values, settings.num_elements, key_value_pairs);

        __gnu_parallel::sort(key_value_pairs.begin(), key_value_pairs.end());

        UnzipKeyValuePairs(key_value_pairs, settings.num_elements, keys, values);
      }
      {
        TimeScope time_scope("memory_deallocate_phase");

        // Upstream: clear() then shrink_to_fit() as separate calls.
        // Changed: swap with empty vector — single operation, guaranteed
        // to release memory (shrink_to_fit is non-binding).
        std::vector<KeyValuePair<T, V>>().swap(key_value_pairs);
      }
    }
  }

  // AJB: 结果输出——保留原格式但加运行统计
  std::cout << settings.num_elements << "," << settings.num_threads << ",\"" << settings.cpu_sort_algorithm << "\",\""
            << settings.key_type << "\",\"" << settings.value_type << "\"," << settings.random_seed << ","
            << settings.zip << ",";
  std::cout << std::fixed << std::setprecision(9) << termcolor::green
            << TimeDurations::Get().GetDuration("memory_allocate_phase") << termcolor::reset << "," << termcolor::yellow
            << TimeDurations::Get().GetDuration("cpu_sort_phase") << termcolor::reset << "," << termcolor::blue
            << TimeDurations::Get().GetDuration("memory_deallocate_phase") << termcolor::reset << ","
            << termcolor::magenta << TimeDurations::Get().GetTotalDuration() << termcolor::reset << std::endl;

  // Upstream: is_sorted + printf.
  // Changed: adjacent_find for precise inversion location.
  auto inv = std::adjacent_find(keys.begin(), keys.end(), [](const T& a, const T& b) { return b < a; });
  if (inv != keys.end()) {
    size_t pos = std::distance(keys.begin(), inv);
    fprintf(stderr, "[ERROR] Run  // AJB: stderr代替stdoutCpuSortBenchmark: Invalid order at index %zu.\n", pos);
  }
}

static bool ValidateSettings(const Settings& s) {
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidNumElements(s.num_elements)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidNumThreads(s.num_threads)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidCpuSortAlgorithm(s.cpu_sort_algorithm)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidType(s.key_type) || !OptionsLimits::IsValidType(s.value_type)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidRandomSeed(s.random_seed)) return false;
  return true;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("cpu_sort_benchmark");

  options.set_width(250);

  options.add_options()("num_elements", "the number of elements " + OptionsLimits::GetNumElementsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumElements()));
  options.add_options()("num_threads", "the number of threads " + OptionsLimits::GetNumThreadsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumThreads()));
  options.add_options()("cpu_sort_algorithm", "the CPU sort algorithm " + OptionsLimits::GetCpuSortAlgorithmLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultCpuSortAlgorithm()));
  options.add_options()("key_type", "the key type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("value_type", "the value type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("random_seed", "the random seed " + OptionsLimits::GetRandomSeedLimits(),
                        cxxopts::value<uint32_t>()->default_value(OptionsLimits::GetDefaultRandomSeed()));
  options.add_options()("zip", "zips the keys and values", cxxopts::value<bool>()->default_value("false"));
  options.add_options()("help", "shows the help", cxxopts::value<bool>()->default_value("false"));

  // AJB: 解析命令行参数
  cxxopts::ParseResult parse_result = options.parse(argc, argv);

  Settings s = {parse_result["num_elements"].as<size_t>(),
                parse_result["num_threads"].as<size_t>(),
                parse_result["cpu_sort_algorithm"].as<std::string>(),
                parse_result["key_type"].as<std::string>(),
                parse_result["value_type"].as<std::string>(),
                parse_result["random_seed"].as<uint32_t>(),
                parse_result["zip"].as<bool>()};

  if (!ValidateSettings(s) || parse_result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  // AJB: 类型分发用函数表代替if-else链
  using BenchFn = std::function<void(const Settings&)>;
  static const std::unordered_map<std::string, BenchFn> dispatchers = {
    {"int:int",       [](const Settings& s){ RunCpuSortBenchmark<uint32_t, uint32_t>(s); }},
    {"long:long",     [](const Settings& s){ RunCpuSortBenchmark<uint64_t, uint64_t>(s); }},
    {"float:float",   [](const Settings& s){ RunCpuSortBenchmark<float, float>(s); }},
    {"double:double", [](const Settings& s){ RunCpuSortBenchmark<double, double>(s); }},
  };
  auto it = dispatchers.find(type_key);
  if (it != dispatchers.end()) {
    auto wall_t0 = std::chrono::steady_clock::now();
    it->second(s);
    auto wall_t1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
    double throughput = s.num_elements / (wall_ms / 1000.0) / 1e6;
    fprintf(stderr, "[AJB_BP][cpu_sort] wall=%.1fms throughput=%.1f Melems/s\n", wall_ms, throughput);
  } else {
    fprintf(stderr, "[AJB_WARN][cpu_sort] unknown type pair: %s\n", type_key.c_str());
  }

  return 0;
}
