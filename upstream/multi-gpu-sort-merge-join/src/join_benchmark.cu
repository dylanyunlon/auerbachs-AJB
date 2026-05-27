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
#include "common/math_utilities.cuh"
#include "common/options_limits.cuh"
#include "common/parallel_algorithms.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/relation_generator.cuh"
#include "common/stream_pool.cuh"
#include "hybrid_sort/hybrid_sort.cuh"
#include "merge_join/merge_join.cuh"

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
};

template <typename T, typename V>
void RunJoinBenchmark(Settings& settings) {
  ConfigureMultiProcess(settings.num_threads);
  ConfigurePeerAccess(settings.gpus);

  Relation<T, V> r_relation(settings.r_num_elements);
  Relation<T, V> s_relation(settings.s_num_elements);

  const size_t num_matches = RelationGenerator::ComputeDistributions<T, V>(
      r_relation, s_relation, settings.num_threads, settings.key_distribution, settings.value_distribution,
      settings.random_seed, settings.theta, settings.sigma, settings.r_sort, settings.s_sort);

  if (settings.save) {
    std::ofstream r_file("r_relation.csv");

    auto r_writer = csv::make_csv_writer(r_file);
    for (size_t i = 0; i < r_relation.GetSize(); ++i) {
      r_writer << std::make_tuple(r_relation.GetKeys()[i], r_relation.GetValues()[i]);
    }

    std::ofstream s_file("s_relation.csv");

    auto s_writer = csv::make_csv_writer(s_file);
    for (size_t i = 0; i < s_relation.GetSize(); ++i) {
      s_writer << std::make_tuple(s_relation.GetKeys()[i], s_relation.GetValues()[i]);
    }
  }

  std::vector<HostAllocator> host_allocators(settings.gpus.size());
  std::vector<DeviceAllocator> device_allocators(settings.gpus.size());
  std::vector<StreamPool> stream_pools(settings.gpus.size());

  if (settings.join_algorithm == "hybrid_sort_merge_join") {
    const size_t temporary_num_elements = std::max(settings.r_num_elements, settings.s_num_elements);
    PinnedVector<T> temporary_keys(temporary_num_elements);
    PinnedVector<V> temporary_values(temporary_num_elements);

    if (settings.sort_algorithm == "gnu_parallel_sort") {
      TimeScope time_scope("sort_phase");

      if (!settings.r_sort) {
        ParallelSortPairs<T, V>(r_relation.GetKeys(), r_relation.GetValues());
      }
      if (!settings.s_sort) {
        ParallelSortPairs<T, V>(s_relation.GetKeys(), s_relation.GetValues());
      }
    } else if (settings.sort_algorithm == "hybrid_merge_sort") {
      if (!settings.r_sort) {
        HybridSort<T, V, HybridSortKernel::kMerge>(r_relation.GetKeys(), r_relation.GetValues(), temporary_keys,
                                                   temporary_values, settings.gpus, host_allocators, device_allocators,
                                                   stream_pools, settings.r_num_elements, settings.chunk_size);
      }
      if (!settings.s_sort) {
        HybridSort<T, V, HybridSortKernel::kMerge>(s_relation.GetKeys(), s_relation.GetValues(), temporary_keys,
                                                   temporary_values, settings.gpus, host_allocators, device_allocators,
                                                   stream_pools, settings.s_num_elements, settings.chunk_size);
      }
    } else if (settings.sort_algorithm == "hybrid_radix_sort") {
      if (!settings.r_sort) {
        HybridSort<T, V, HybridSortKernel::kRadix>(r_relation.GetKeys(), r_relation.GetValues(), temporary_keys,
                                                   temporary_values, settings.gpus, host_allocators, device_allocators,
                                                   stream_pools, settings.r_num_elements, settings.chunk_size);
      }
      if (!settings.s_sort) {
        HybridSort<T, V, HybridSortKernel::kRadix>(s_relation.GetKeys(), s_relation.GetValues(), temporary_keys,
                                                   temporary_values, settings.gpus, host_allocators, device_allocators,
                                                   stream_pools, settings.s_num_elements, settings.chunk_size);
      }
    }

    JoinResult<T> join_result =
        MergeJoin(r_relation.GetKeys(), r_relation.GetValues(), s_relation.GetKeys(), s_relation.GetValues(),
                  settings.gpus, device_allocators, stream_pools, settings.materialize);

    std::cout << settings.r_num_elements << "," << settings.s_num_elements << "," << settings.num_threads << ",\"";
    for (size_t i = 0; i < settings.gpus.size(); ++i) {
      std::cout << settings.gpus[i] << (i < settings.gpus.size() - 1 ? "," : "");
    }
    std::cout << "\",\"" << settings.sort_algorithm << "\",\"" << settings.join_algorithm << "\","
              << settings.chunk_size << ",\"" << settings.key_type << "\",\"" << settings.key_distribution << "\",\""
              << settings.value_type << "\",\"" << settings.value_distribution << "\"," << settings.random_seed << ","
              << settings.theta << "," << settings.sigma << "," << settings.r_sort << "," << settings.s_sort << ","
              << settings.materialize << ",";
    std::cout << std::fixed << std::setprecision(9) << termcolor::green
              << TimeDurations::Get().GetDuration("sort_phase") << termcolor::reset << "," << termcolor::yellow
              << TimeDurations::Get().GetDuration("merge_phase") << termcolor::reset << "," << termcolor::blue
              << TimeDurations::Get().GetDuration("join_phase") << termcolor::reset << "," << termcolor::magenta
              << TimeDurations::Get().GetTotalDuration() << termcolor::reset << std::endl;

    if (settings.materialize) {
      std::vector<std::tuple<T, V, V>> result_rows;
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
        for (const auto& [key, r_value, s_value] : result_rows) {
          table.add_row({std::to_string(key), std::to_string(r_value), std::to_string(s_value)});
        }

        table.format().font_align(tabulate::FontAlign::center);
        for (size_t i = 0; i < table[0].size(); ++i) {
          table[0][i].format().font_color(tabulate::Color::yellow);
        }

        std::cout << std::endl << table << std::endl;
      }
    }

    if (join_result.count_ != num_matches) {
      printf("[ERROR] RunJoinBenchmark: Invalid join count (%lu != %lu).\n", join_result.count_, num_matches);
    }
  }
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
                parse_result["print"].as<bool>()};

  if (!OptionsLimits::IsValidNumElements(s.r_num_elements) || !OptionsLimits::IsValidNumElements(s.s_num_elements) ||
      !OptionsLimits::IsValidNumThreads(s.num_threads) || !OptionsLimits::IsValidGpus(s.gpus) ||
      !OptionsLimits::IsValidSortAlgorithm(s.sort_algorithm) ||
      !OptionsLimits::IsValidJoinAlgorithm(s.join_algorithm) || !OptionsLimits::IsValidChunkSize(s.chunk_size) ||
      !OptionsLimits::IsValidType(s.key_type) || !OptionsLimits::IsValidJoinDistribution(s.key_distribution) ||
      !OptionsLimits::IsValidType(s.value_type) || !OptionsLimits::IsValidJoinDistribution(s.value_distribution) ||
      !OptionsLimits::IsValidRandomSeed(s.random_seed) || !OptionsLimits::IsValidTheta(s.theta) ||
      !OptionsLimits::IsValidSigma(s.sigma) || parse_result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (s.key_type == "int" && s.value_type == "int") {
    RunJoinBenchmark<uint32_t, uint32_t>(s);
  } else if (s.key_type == "long" && s.value_type == "long") {
    RunJoinBenchmark<uint64_t, uint64_t>(s);
  } else if (s.key_type == "float" && s.value_type == "float") {
    RunJoinBenchmark<float, float>(s);
  } else if (s.key_type == "double" && s.value_type == "double") {
    RunJoinBenchmark<double, double>(s);
  }

  return 0;
}
