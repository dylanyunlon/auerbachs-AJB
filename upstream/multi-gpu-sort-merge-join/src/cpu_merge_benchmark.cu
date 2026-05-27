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
#include "common/math_utilities.cuh"
#include "common/options_limits.cuh"
#include "common/parallel_algorithms.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"

const std::string kKeyDistributionType = "sorted";
const std::string kValueDistributionType = "uniform";

struct Settings {
  size_t num_elements;
  size_t num_threads;
  std::string cpu_merge_algorithm;
  size_t chunk_count;
  std::string key_type;
  std::string value_type;
  uint32_t random_seed;
  bool zip;
};

template <typename T, typename V>
void RunCpuMergeBenchmark(Settings& settings) {
  ConfigureMultiProcess(settings.num_threads);

  PinnedVector<T> keys(settings.num_elements);
  PinnedVector<V> values(settings.num_elements);

  const size_t chunk_size = DivideUp(settings.num_elements, settings.chunk_count);

  for (size_t i = 0; i < settings.chunk_count; ++i) {
    const size_t chunk_size_offset = i * chunk_size;
    const size_t num_elements = std::min(chunk_size, settings.num_elements - chunk_size_offset);

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

        for (size_t i = 0; i < settings.chunk_count; ++i) {
          const size_t offset = i * chunk_size;
          const size_t num_elements = std::min(chunk_size, settings.num_elements - offset);

          KeyValuePairIter begin_iter = key_value_pairs.begin() + offset;
          KeyValuePairIter end_iter = key_value_pairs.begin() + offset + num_elements;

          iter_pairs.emplace_back(begin_iter, end_iter);
        }

        KeyValuePairIter out_begin_iter = merged_key_value_pairs.begin();

        __gnu_parallel::multiway_merge(iter_pairs.begin(), iter_pairs.end(), out_begin_iter, settings.num_elements,
                                       std::less<>());

        UnzipKeyValuePairs(merged_key_value_pairs, settings.num_elements, keys, values);
      }
      {
        TimeScope time_scope("memory_deallocate_phase");

        key_value_pairs.clear();
        key_value_pairs.shrink_to_fit();
        merged_key_value_pairs.clear();
        merged_key_value_pairs.shrink_to_fit();
      }
    }
  }

  std::cout << settings.num_elements << "," << settings.num_threads << ",\"" << settings.cpu_merge_algorithm << "\","
            << settings.chunk_count << ",\"" << settings.key_type << "\",\"" << settings.value_type << "\","
            << settings.random_seed << "," << settings.zip << ",";
  std::cout << std::fixed << std::setprecision(9) << termcolor::green
            << TimeDurations::Get().GetDuration("memory_allocate_phase") << termcolor::reset << "," << termcolor::yellow
            << TimeDurations::Get().GetDuration("cpu_merge_phase") << termcolor::reset << "," << termcolor::blue
            << TimeDurations::Get().GetDuration("memory_deallocate_phase") << termcolor::reset << ","
            << termcolor::magenta << TimeDurations::Get().GetTotalDuration() << termcolor::reset << std::endl;

  if (!std::is_sorted(merged_keys.begin(), merged_keys.end())) {
    printf("[ERROR] RunCpuMergeBenchmark: Invalid order.\n");
  }
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

  cxxopts::ParseResult parse_result = options.parse(argc, argv);

  Settings s = {parse_result["num_elements"].as<size_t>(),
                parse_result["num_threads"].as<size_t>(),
                parse_result["cpu_merge_algorithm"].as<std::string>(),
                parse_result["chunk_count"].as<size_t>(),
                parse_result["key_type"].as<std::string>(),
                parse_result["value_type"].as<std::string>(),
                parse_result["random_seed"].as<uint32_t>(),
                parse_result["zip"].as<bool>()};

  if (!OptionsLimits::IsValidNumElements(s.num_elements) || !OptionsLimits::IsValidNumThreads(s.num_threads) ||
      !OptionsLimits::IsValidCpuMergeAlgorithm(s.cpu_merge_algorithm) ||
      !OptionsLimits::IsValidChunkCount(s.chunk_count) || !OptionsLimits::IsValidType(s.key_type) ||
      !OptionsLimits::IsValidType(s.value_type) || !OptionsLimits::IsValidRandomSeed(s.random_seed) ||
      parse_result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (s.key_type == "int" && s.value_type == "int") {
    RunCpuMergeBenchmark<uint32_t, uint32_t>(s);
  } else if (s.key_type == "long" && s.value_type == "long") {
    RunCpuMergeBenchmark<uint64_t, uint64_t>(s);
  } else if (s.key_type == "float" && s.value_type == "float") {
    RunCpuMergeBenchmark<float, float>(s);
  } else if (s.key_type == "double" && s.value_type == "double") {
    RunCpuMergeBenchmark<double, double>(s);
  }

  return 0;
}
