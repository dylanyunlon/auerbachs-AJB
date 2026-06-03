#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
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

const std::string kKeyDistributionType = "uniform";
const std::string kValueDistributionType = "sorted";

struct Settings {
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
  ConfigureMultiProcess(settings.num_threads);

  PinnedVector<T> keys(settings.num_elements);
  PinnedVector<V> values(settings.num_elements);

  DataGenerator::ComputeDistribution(keys.data(), settings.num_elements, settings.num_threads, kKeyDistributionType,
                                     settings.random_seed);
  DataGenerator::ComputeDistribution(values.data(), settings.num_elements, settings.num_threads, kValueDistributionType,
                                     settings.random_seed);

  if (settings.cpu_sort_algorithm == "gnu_parallel_sort") {
    if (settings.zip) {
      TimeScope time_scope("cpu_sort_phase");

      ParallelSortPairs(keys, values);
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
    printf("[ERROR] RunCpuSortBenchmark: Invalid order at index %zu.\n", pos);
  }
}

static bool ValidateSettings(const Settings& s) {
  if (!OptionsLimits::IsValidNumElements(s.num_elements)) return false;
  if (!OptionsLimits::IsValidNumThreads(s.num_threads)) return false;
  if (!OptionsLimits::IsValidCpuSortAlgorithm(s.cpu_sort_algorithm)) return false;
  if (!OptionsLimits::IsValidType(s.key_type) || !OptionsLimits::IsValidType(s.value_type)) return false;
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

  const std::string type_key = s.key_type + ":" + s.value_type;
  if (type_key == "int:int") {
    RunCpuSortBenchmark<uint32_t, uint32_t>(s);
  } else if (type_key == "long:long") {
    RunCpuSortBenchmark<uint64_t, uint64_t>(s);
  } else if (type_key == "float:float") {
    RunCpuSortBenchmark<float, float>(s);
  } else if (type_key == "double:double") {
    RunCpuSortBenchmark<double, double>(s);
  }

  return 0;
}
