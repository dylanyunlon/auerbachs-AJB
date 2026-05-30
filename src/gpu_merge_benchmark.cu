#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cub/cub.cuh>
#include <cxxopts.hpp>
#include <moderngpu/kernel_merge.hxx>
#include <termcolor/termcolor.hpp>
#include <thrust/merge.h>

#include "common/data_generator.cuh"
#include "common/debug_utilities.cuh"    // AJB: breakpoints + memory reporting
#include "common/device_allocator.cuh"
#include "common/error_utilities.cuh"
#include "common/math_utilities.cuh"
#include "common/options_limits.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/resource_context.cuh"
#include "common/stream_pool.cuh"

constexpr size_t kDeviceMemoryOverhead = 1024_MB;

const std::string kKeyDistributionType = "sorted";
const std::string kValueDistributionType = "uniform";

struct Settings {
  size_t num_elements;
  size_t num_threads;
  std::string gpu_merge_algorithm;
  std::string key_type;
  std::string value_type;
  uint32_t random_seed;
};

template <typename T, typename V>
void RunGpuMergeBenchmark(Settings& settings) {
  AJB_BREAKPOINT("[gpu_merge] start: n=%zu algo=%s", settings.num_elements, settings.gpu_merge_algorithm.c_str());
  AJBTimer timer_total("gpu_merge_total");
  AJBReportMemory("gpu_merge_benchmark_start");

  PinnedVector<T> keys(settings.num_elements);
  PinnedVector<V> values(settings.num_elements);

  const size_t chunk_size = DivideUp(settings.num_elements, 2);

  for (size_t i = 0; i < 2; ++i) {
    const size_t chunk_size_offset = i * chunk_size;
    const size_t num_elements = std::min(chunk_size, settings.num_elements - chunk_size_offset);

    DataGenerator::ComputeDistribution(keys.data() + chunk_size_offset, num_elements, settings.num_threads,
                                       kKeyDistributionType, settings.random_seed * (i + 1));
    DataGenerator::ComputeDistribution(values.data() + chunk_size_offset, num_elements, settings.num_threads,
                                       kValueDistributionType, settings.random_seed * (i + 1));
  }

  CheckCudaError(cudaSetDevice(0));

  DeviceAllocator device_allocator;
  device_allocator.Initialize(2 * settings.num_elements * (sizeof(T) + sizeof(V)) + kDeviceMemoryOverhead);

  StreamPool stream_pool;
  stream_pool.Initialize(1);

  cub::DoubleBuffer<T> keys_double_buffer(
      reinterpret_cast<T*>(device_allocator.allocate(settings.num_elements * sizeof(T))),
      reinterpret_cast<T*>(device_allocator.allocate(settings.num_elements * sizeof(T))));
  cub::DoubleBuffer<V> values_double_buffer(
      reinterpret_cast<V*>(device_allocator.allocate(settings.num_elements * sizeof(V))),
      reinterpret_cast<V*>(device_allocator.allocate(settings.num_elements * sizeof(V))));

  CheckCudaError(cudaMemcpyAsync(keys_double_buffer.Current(), keys.data(), settings.num_elements * sizeof(T),
                                 cudaMemcpyHostToDevice, stream_pool.GetStream(0)));
  CheckCudaError(cudaMemcpyAsync(values_double_buffer.Current(), values.data(), settings.num_elements * sizeof(V),
                                 cudaMemcpyHostToDevice, stream_pool.GetStream(0)));

  CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

  if (settings.gpu_merge_algorithm == "thrust_merge_by_key") {
    TimeScope time_scope("gpu_merge_phase");

    thrust::merge_by_key(
        thrust::cuda::par_nosync(device_allocator).on(stream_pool.GetStream(0)), keys_double_buffer.Current(),
        keys_double_buffer.Current() + chunk_size, keys_double_buffer.Current() + chunk_size,
        keys_double_buffer.Current() + settings.num_elements, values_double_buffer.Current(),
        values_double_buffer.Current() + chunk_size, keys_double_buffer.Alternate(), values_double_buffer.Alternate());

    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
  } else if (settings.gpu_merge_algorithm == "mgpu_merge") {
    TimeScope time_scope("gpu_merge_phase");

    ResourceContext resource_context(device_allocator, stream_pool.GetStream(0));

    mgpu::merge(keys_double_buffer.Current(), values_double_buffer.Current(), chunk_size,
                keys_double_buffer.Current() + chunk_size, values_double_buffer.Current() + chunk_size,
                settings.num_elements - chunk_size, keys_double_buffer.Alternate(), values_double_buffer.Alternate(),
                mgpu::less_t<T>(), resource_context);

    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
  }

  keys_double_buffer.selector ^= 1;
  values_double_buffer.selector ^= 1;

  CheckCudaError(cudaMemcpyAsync(keys.data(), keys_double_buffer.Current(), settings.num_elements * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream_pool.GetStream(0)));
  CheckCudaError(cudaMemcpyAsync(values.data(), values_double_buffer.Current(), settings.num_elements * sizeof(V),
                                 cudaMemcpyDeviceToHost, stream_pool.GetStream(0)));

  CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

  device_allocator.deallocate(reinterpret_cast<uint8_t*>(keys_double_buffer.Current()));
  device_allocator.deallocate(reinterpret_cast<uint8_t*>(keys_double_buffer.Alternate()));
  device_allocator.deallocate(reinterpret_cast<uint8_t*>(values_double_buffer.Current()));
  device_allocator.deallocate(reinterpret_cast<uint8_t*>(values_double_buffer.Alternate()));

  std::cout << settings.num_elements << "," << settings.num_threads << ",\"" << settings.gpu_merge_algorithm << "\",\""
            << settings.key_type << "\",\"" << settings.value_type << "\"," << settings.random_seed << ",";
  std::cout << std::fixed << std::setprecision(9) << termcolor::green
            << TimeDurations::Get().GetDuration("gpu_merge_phase") << termcolor::reset << std::endl;

  if (!std::is_sorted(keys.begin(), keys.end())) {
    printf("[ERROR] RunGpuMergeBenchmark: Invalid order.\n");
  }

  AJBReportGPUMemory(0, "gpu_merge_benchmark_end");
  AJB_BREAKPOINT("[gpu_merge] done: n=%zu elapsed=%.6fs", settings.num_elements, timer_total.ElapsedSec());
  TimeDurations::Get().PrintAllDurations();
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("gpu_merge_benchmark");

  options.set_width(250);

  options.add_options()("num_elements", "the number of elements " + OptionsLimits::GetNumElementsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumElements()));
  options.add_options()("num_threads", "the number of threads " + OptionsLimits::GetNumThreadsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumThreads()));
  options.add_options()("gpu_merge_algorithm", "the GPU merge algorithm " + OptionsLimits::GetGpuMergeAlgorithmLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultGpuMergeAlgorithm()));
  options.add_options()("key_type", "the key type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("value_type", "the value type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("random_seed", "the random seed " + OptionsLimits::GetRandomSeedLimits(),
                        cxxopts::value<uint32_t>()->default_value(OptionsLimits::GetDefaultRandomSeed()));
  options.add_options()("help", "shows the help", cxxopts::value<bool>()->default_value("false"));

  cxxopts::ParseResult parse_result = options.parse(argc, argv);

  Settings s = {parse_result["num_elements"].as<size_t>(),
                parse_result["num_threads"].as<size_t>(),
                parse_result["gpu_merge_algorithm"].as<std::string>(),
                parse_result["key_type"].as<std::string>(),
                parse_result["value_type"].as<std::string>(),
                parse_result["random_seed"].as<uint32_t>()};

  if (!OptionsLimits::IsValidNumElements(s.num_elements) || !OptionsLimits::IsValidNumThreads(s.num_threads) ||
      !OptionsLimits::IsValidGpuMergeAlgorithm(s.gpu_merge_algorithm) || !OptionsLimits::IsValidType(s.key_type) ||
      !OptionsLimits::IsValidType(s.value_type) || !OptionsLimits::IsValidRandomSeed(s.random_seed) ||
      parse_result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (s.key_type == "int" && s.value_type == "int") {
    RunGpuMergeBenchmark<int, int>(s);
  }

  return 0;
}
