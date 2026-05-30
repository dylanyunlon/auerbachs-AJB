#include <fstream>
#include <iomanip>
#include <iostream>
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
#include "common/options_limits.cuh"
#include "common/parallel_algorithms.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/stream_pool.cuh"
#include "hybrid_sort/hybrid_sort.cuh"

struct Settings {
  size_t num_elements;
  size_t num_threads;
  std::vector<int> gpus;
  std::string sort_algorithm;
  size_t chunk_size;
  std::string key_type;
  std::string key_distribution;
  std::string value_type;
  std::string value_distribution;
  uint32_t random_seed;
  bool save;
  bool print;
};

template <typename T, typename V>
void RunSortBenchmark(Settings& settings) {
  ConfigureMultiProcess(settings.num_threads);
  ConfigurePeerAccess(settings.gpus);

  PinnedVector<T> keys(settings.num_elements);
  PinnedVector<V> values(settings.num_elements);

  DataGenerator::ComputeDistribution(keys.data(), keys.size(), settings.num_threads, settings.key_distribution,
                                     settings.random_seed);
  DataGenerator::ComputeDistribution(values.data(), values.size(), settings.num_threads, settings.value_distribution,
                                     settings.random_seed);

  if (settings.save) {
    std::ofstream file("key_value_tuples.csv");

    auto writer = csv::make_csv_writer(file);
    for (size_t i = 0; i < settings.num_elements; ++i) {
      writer << std::make_tuple(keys[i], values[i]);
    }
  }

  PinnedVector<T> unsorted_keys(settings.print ? keys : PinnedVector<T>{});
  PinnedVector<V> unsorted_values(settings.print ? values : PinnedVector<V>{});

  PinnedVector<T> temporary_keys(settings.num_elements);
  PinnedVector<V> temporary_values(settings.num_elements);

  std::vector<HostAllocator> host_allocators(settings.gpus.size());
  std::vector<DeviceAllocator> device_allocators(settings.gpus.size());
  std::vector<StreamPool> stream_pools(settings.gpus.size());

  if (settings.sort_algorithm == "gnu_parallel_sort") {
    TimeScope time_scope("sort_phase");

    ParallelSortPairs<T, V>(keys, values);
  } else if (settings.sort_algorithm == "hybrid_merge_sort") {
    HybridSort<T, V, HybridSortKernel::kMerge>(keys, values, temporary_keys, temporary_values, settings.gpus,
                                               host_allocators, device_allocators, stream_pools, settings.num_elements,
                                               settings.chunk_size);
  } else if (settings.sort_algorithm == "hybrid_radix_sort") {
    HybridSort<T, V, HybridSortKernel::kRadix>(keys, values, temporary_keys, temporary_values, settings.gpus,
                                               host_allocators, device_allocators, stream_pools, settings.num_elements,
                                               settings.chunk_size);
  }

  std::cout << settings.num_elements << "," << settings.num_threads << ",\"";
  for (size_t i = 0; i < settings.gpus.size(); ++i) {
    std::cout << settings.gpus[i] << (i < settings.gpus.size() - 1 ? "," : "");
  }
  std::cout << "\",\"" << settings.sort_algorithm << "\"," << settings.chunk_size << ",\"" << settings.key_type
            << "\",\"" << settings.key_distribution << "\",\"" << settings.value_type << "\",\""
            << settings.value_distribution << "\"," << settings.random_seed << ",";
  std::cout << std::fixed << std::setprecision(9) << termcolor::green << TimeDurations::Get().GetDuration("sort_phase")
            << termcolor::reset << "," << termcolor::yellow << TimeDurations::Get().GetDuration("merge_phase")
            << termcolor::reset << "," << termcolor::magenta << TimeDurations::Get().GetTotalDuration()
            << termcolor::reset << std::endl;

  if (settings.print) {
    tabulate::Table table;

    table.add_row({"u_key", "u_value", "s_key", "s_value"});
    for (size_t i = 0; i < settings.num_elements; ++i) {
      table.add_row({std::to_string(unsorted_keys[i]), std::to_string(unsorted_values[i]), std::to_string(keys[i]),
                     std::to_string(values[i])});
    }

    table.format().font_align(tabulate::FontAlign::center);
    for (size_t i = 0; i < table[0].size(); ++i) {
      table[0][i].format().font_color(tabulate::Color::yellow);
    }

    std::cout << std::endl << table << std::endl;
  }

  if (!std::is_sorted(keys.begin(), keys.end())) {
    printf("[ERROR] RunSortBenchmark: Invalid sort order.\n");
  }
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("sort_benchmark");

  options.set_width(250);

  options.add_options()("num_elements", "the number of elements " + OptionsLimits::GetNumElementsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumElements()));
  options.add_options()("num_threads", "the number of threads " + OptionsLimits::GetNumThreadsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumThreads()));
  options.add_options()("gpus", "the visible GPUs " + OptionsLimits::GetGpusLimits(),
                        cxxopts::value<std::vector<int>>()->default_value(OptionsLimits::GetDefaultGpus()));
  options.add_options()("sort_algorithm", "the sort algorithm " + OptionsLimits::GetSortAlgorithmLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultSortAlgorithm()));
  options.add_options()("chunk_size", "the chunk size " + OptionsLimits::GetChunkSizeLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultChunkSize()));
  options.add_options()("key_type", "the key type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("key_distribution", "the key distribution " + OptionsLimits::GetSortDistributionLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultSortDistribution()));
  options.add_options()("value_type", "the value type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("value_distribution", "the value distribution " + OptionsLimits::GetSortDistributionLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultSortDistribution()));
  options.add_options()("random_seed", "the random seed " + OptionsLimits::GetRandomSeedLimits(),
                        cxxopts::value<uint32_t>()->default_value(OptionsLimits::GetDefaultRandomSeed()));
  options.add_options()("save", "saves the input", cxxopts::value<bool>()->default_value("false"));
  options.add_options()("print", "prints the output", cxxopts::value<bool>()->default_value("false"));
  options.add_options()("help", "shows the help", cxxopts::value<bool>()->default_value("false"));

  cxxopts::ParseResult parse_result = options.parse(argc, argv);

  Settings s = {parse_result["num_elements"].as<size_t>(),
                parse_result["num_threads"].as<size_t>(),
                parse_result["gpus"].as<std::vector<int>>(),
                parse_result["sort_algorithm"].as<std::string>(),
                parse_result["chunk_size"].as<size_t>(),
                parse_result["key_type"].as<std::string>(),
                parse_result["key_distribution"].as<std::string>(),
                parse_result["value_type"].as<std::string>(),
                parse_result["value_distribution"].as<std::string>(),
                parse_result["random_seed"].as<uint32_t>(),
                parse_result["save"].as<bool>(),
                parse_result["print"].as<bool>()};

  if (!OptionsLimits::IsValidNumElements(s.num_elements) || !OptionsLimits::IsValidNumThreads(s.num_threads) ||
      !OptionsLimits::IsValidGpus(s.gpus) || !OptionsLimits::IsValidSortAlgorithm(s.sort_algorithm) ||
      !OptionsLimits::IsValidChunkSize(s.chunk_size) || !OptionsLimits::IsValidType(s.key_type) ||
      !OptionsLimits::IsValidSortDistribution(s.key_distribution) || !OptionsLimits::IsValidType(s.value_type) ||
      !OptionsLimits::IsValidSortDistribution(s.value_distribution) ||
      !OptionsLimits::IsValidRandomSeed(s.random_seed) || parse_result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (s.key_type == "int" && s.value_type == "int") {
    RunSortBenchmark<uint32_t, uint32_t>(s);
  } else if (s.key_type == "long" && s.value_type == "long") {
    RunSortBenchmark<uint64_t, uint64_t>(s);
  } else if (s.key_type == "float" && s.value_type == "float") {
    RunSortBenchmark<float, float>(s);
  } else if (s.key_type == "double" && s.value_type == "double") {
    RunSortBenchmark<double, double>(s);
  }

  return 0;
}
