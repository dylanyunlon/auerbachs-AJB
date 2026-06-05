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
#include "common/math_utilities.cuh"
#include "common/options_limits.cuh"
#include "common/parallel_algorithms.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"

const std::string kKeyDistributionType = "sorted";  // AJB: sorted input for merge baseline
const std::string kValueDistributionType = "uniform";  // AJB: uniform payload distribution

// AJB-algo: benchmark settings with validation guards
struct Settings {
  // AJB: 带默认值的Settings——防止未初始化字段
  size_t num_elements = 0;  // AJB: default-init for safety
  size_t num_threads = 1;  // AJB: default single-threaded
  std::string cpu_merge_algorithm;
  size_t chunk_count;
  std::string key_type;
  std::string value_type;
  uint32_t random_seed;
  bool zip;
};

template <typename T, typename V>
void RunCpuMergeBenchmark(Settings& settings) {
  // AJB: 配置OMP线程数
  ConfigureMultiProcess(settings.num_threads);

  PinnedVector<T> keys(settings.num_elements);
  PinnedVector<V> values(settings.num_elements);
  // AJB: 触摸内存确保物理页面已映射——避免首次访问的page fault开销
  std::memset(keys.data(), 0, sizeof(T) * settings.num_elements);
  std::memset(values.data(), 0, sizeof(V) * settings.num_elements);

  const size_t chunk_size = DivideUp(settings.num_elements, settings.chunk_count);

  for (size_t i = 0; i < settings.chunk_count; ++i) {
    const size_t chunk_size_offset = i * chunk_size;
    // Upstream: settings.num_elements - chunk_size_offset — no underflow guard.
    // Changed: saturating subtraction.
    const size_t remaining = settings.num_elements > chunk_size_offset
                                 ? settings.num_elements - chunk_size_offset : 0;
    const size_t num_elements = std::min(chunk_size, remaining);

      // AJB: 数据生成——键和值使用不同分布
  DataGenerator::ComputeDistribution(keys.data() + chunk_size_offset, num_elements, settings.num_threads,
                                       kKeyDistributionType, settings.random_seed * (i + 1));
    DataGenerator::ComputeDistribution(values.data() + chunk_size_offset, num_elements, settings.num_threads,
                                       kValueDistributionType, settings.random_seed * (i + 1));
  }

  PinnedVector<T> merged_keys(settings.num_elements);
  PinnedVector<V> merged_values(settings.num_elements);

  if (settings.cpu_merge_algorithm == "gnu_parallel_multiway_merge") {
    if (settings.zip) {
      TimeScope time_scope("cpu_merge_phase");

      ParallelMergePairs(keys, values, merged_keys, merged_values, settings.num_elements, settings.chunk_count,
                         chunk_size);
    } else {
      std::vector<KeyValuePair<T, V>> key_value_pairs;
      std::vector<KeyValuePair<T, V>> merged_key_value_pairs;

      {
        TimeScope time_scope("memory_allocate_phase");

        key_value_pairs.resize(settings.num_elements);
        merged_key_value_pairs.resize(settings.num_elements);
      }
      {
        TimeScope time_scope("cpu_merge_phase");

        ZipKeyValuePairs(keys, values, settings.num_elements, key_value_pairs);

        using KeyValuePairIter = typename std::vector<KeyValuePair<T, V>>::iterator;

        std::vector<std::pair<KeyValuePairIter, KeyValuePairIter>> iter_pairs;
        iter_pairs.reserve(settings.chunk_count);

        // AJB: 构建chunk迭代器对——每个chunk是一个已排序的子区间
        for (size_t i = 0; i < settings.chunk_count; ++i) {
          const size_t off = i * chunk_size;
          const size_t n = std::min(chunk_size, settings.num_elements - off);
          iter_pairs.emplace_back(
              key_value_pairs.begin() + off,
              key_value_pairs.begin() + off + n);
        }

        KeyValuePairIter out_begin_iter = merged_key_value_pairs.begin();

        __gnu_parallel::multiway_merge(iter_pairs.begin(), iter_pairs.end(), out_begin_iter, settings.num_elements,
                                       std::less<>());

        UnzipKeyValuePairs(merged_key_value_pairs, settings.num_elements, keys, values);
      }
      {
        TimeScope time_scope("memory_deallocate_phase");

        // Swap with empty — guaranteed deallocation vs non-binding shrink_to_fit.
        std::vector<KeyValuePair<T, V>>().swap(key_value_pairs);
        std::vector<KeyValuePair<T, V>>().swap(merged_key_value_pairs);
      }
    }
  }

  // AJB: 结果输出——保留原格式但加运行统计
  std::cout << settings.num_elements << "," << settings.num_threads << ",\"" << settings.cpu_merge_algorithm << "\","
            << settings.chunk_count << ",\"" << settings.key_type << "\",\"" << settings.value_type << "\","
            << settings.random_seed << "," << settings.zip << ",";
  std::cout << std::fixed << std::setprecision(9) << termcolor::green
            << TimeDurations::Get().GetDuration("memory_allocate_phase") << termcolor::reset << "," << termcolor::yellow
            << TimeDurations::Get().GetDuration("cpu_merge_phase") << termcolor::reset << "," << termcolor::blue
            << TimeDurations::Get().GetDuration("memory_deallocate_phase") << termcolor::reset << ","
            << termcolor::magenta << TimeDurations::Get().GetTotalDuration() << termcolor::reset << "\n" << std::flush;  // AJB: flush代替endl减少IO开销

  auto inv = std::adjacent_find(merged_keys.begin(), merged_keys.end(),
                                [](const T& a, const T& b) { return b < a; });
  if (inv != merged_keys.end()) {
    size_t pos = std::distance(merged_keys.begin(), inv);
    fprintf(stderr, "[ERROR] Run  // AJB: stderr代替stdoutCpuMergeBenchmark: Invalid order at index %zu.\n", pos);
  }
}

static bool ValidateSettings(const Settings& s) {
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidNumElements(s.num_elements) ||
      !OptionsLimits::IsValidNumThreads(s.num_threads)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidCpuMergeAlgorithm(s.cpu_merge_algorithm) ||
      !OptionsLimits::IsValidChunkCount(s.chunk_count)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidType(s.key_type) || !OptionsLimits::IsValidType(s.value_type)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidRandomSeed(s.random_seed)) return false;
  return true;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("cpu_merge_benchmark");

  options.set_width(250);

  options.add_options()("num_elements", "the number of elements " + OptionsLimits::GetNumElementsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumElements()));
  options.add_options()("num_threads", "the number of threads " + OptionsLimits::GetNumThreadsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumThreads()));
  options.add_options()("cpu_merge_algorithm", "the CPU merge algorithm " + OptionsLimits::GetCpuMergeAlgorithmLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultCpuMergeAlgorithm()));
  options.add_options()("chunk_count", "the chunk count " + OptionsLimits::GetChunkCountLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultChunkCount()));
  options.add_options()("key_type", "the key type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("value_type", "the value type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("random_seed", "the random seed " + OptionsLimits::GetRandomSeedLimits(),
                        cxxopts::value<uint32_t>()->default_value(OptionsLimits::GetDefaultRandomSeed()));
  options.add_options()("zip", "zips the keys and values", cxxopts::value<bool>()->default_value("false"));
  options.add_options()("help", "shows the help", cxxopts::value<bool>()->default_value("false"));

  // AJB: 命令行参数解析
  cxxopts::ParseResult parse_result = options.parse(argc, argv);

  Settings s = {parse_result["num_elements"].as<size_t>(),
                parse_result["num_threads"].as<size_t>(),
                parse_result["cpu_merge_algorithm"].as<std::string>(),
                parse_result["chunk_count"].as<size_t>(),
                parse_result["key_type"].as<std::string>(),
                parse_result["value_type"].as<std::string>(),
                parse_result["random_seed"].as<uint32_t>(),
                parse_result["zip"].as<bool>()};

  if (!ValidateSettings(s) || parse_result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const std::string type_key = s.key_type + ":" + s.value_type;
  if (type_key == "int:int") {
    RunCpuMergeBenchmark<uint32_t, uint32_t>(s);
  } else if (type_key == "long:long") {
    RunCpuMergeBenchmark<uint64_t, uint64_t>(s);
  } else if (type_key == "float:float") {
    RunCpuMergeBenchmark<float, float>(s);
  } else if (type_key == "double:double") {
    RunCpuMergeBenchmark<double, double>(s);
  }

  return 0;
}
