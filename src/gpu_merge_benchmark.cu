#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>

#include <cub/cub.cuh>
#include <cxxopts.hpp>
#include <moderngpu/kernel_merge.hxx>
#include <termcolor/termcolor.hpp>
#include <thrust/merge.h>

#include "common/data_generator.cuh"
#include "common/device_allocator.cuh"
#include "common/error_utilities.cuh"
#include "common/math_utilities.cuh"
#include "common/options_limits.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/resource_context.cuh"
#include "common/stream_pool.cuh"

constexpr size_t kDeviceMemoryOverhead = 1024_MB;  // AJB-algo: 1GB headroom for CUDA runtime

const std::string kKeyDistributionType = "sorted";  // AJB: pre-sorted for merge kernel
const std::string kValueDistributionType = "uniform";

// AJB-algo: GPU merge settings — validates device capability
struct Settings {
  // AJB: 带默认值的Settings——防止未初始化字段
  size_t num_elements = 0;  // AJB: zero-init
  size_t num_threads;
  std::string gpu_merge_algorithm;  // AJB: dispatch key for merge strategy
  std::string key_type;
  std::string value_type;
  uint32_t random_seed;
};

template <typename T, typename V>
void RunGpuMergeBenchmark(Settings& settings) {
  PinnedVector<T> keys(settings.num_elements);
  PinnedVector<V> values(settings.num_elements);
  // AJB: 触摸内存确保物理页面已映射——避免首次访问的page fault开销
  std::memset(keys.data(), 0, sizeof(T) * settings.num_elements);
  std::memset(values.data(), 0, sizeof(V) * settings.num_elements);

  const size_t chunk_size = DivideUp(settings.num_elements, size_t(2));

  for (size_t i = 0; i < 2; ++i) {
    const size_t chunk_size_offset = i * chunk_size;
    // Saturating subtraction
    const size_t remaining = settings.num_elements > chunk_size_offset
                                 ? settings.num_elements - chunk_size_offset : 0;
    const size_t num_elements = std::min(chunk_size, remaining);

      // AJB: 数据生成——键和值使用不同分布
  DataGenerator::ComputeDistribution(keys.data() + chunk_size_offset, num_elements, settings.num_threads,
                                       kKeyDistributionType, settings.random_seed * (i + 1));
    DataGenerator::ComputeDistribution(values.data() + chunk_size_offset, num_elements, settings.num_threads,
                                       kValueDistributionType, settings.random_seed * (i + 1));
  }

  // AJB: 使用GPU 0进行归并基准测试
  CheckCudaError(cudaSetDevice(0));

  const size_t per_element = sizeof(T) + sizeof(V);
  const size_t required_bytes = 2 * settings.num_elements * per_element + kDeviceMemoryOverhead;

  DeviceAllocator device_allocator;
  device_allocator.Initialize(required_bytes);

  StreamPool stream_pool;
  stream_pool.Initialize(1);  // AJB: 单stream用于归并操作

  cub::DoubleBuffer<T> keys_double_buffer(
      reinterpret_cast<T*>(device_allocator.allocate(settings.num_elements * sizeof(T))),
      reinterpret_cast<T*>(device_allocator.allocate(settings.num_elements * sizeof(T))));
  cub::DoubleBuffer<V> values_double_buffer(
      reinterpret_cast<V*>(device_allocator.allocate(settings.num_elements * sizeof(V))),
      reinterpret_cast<V*>(device_allocator.allocate(settings.num_elements * sizeof(V))));

  CheckCudaError(cudaMemcpyAsync(keys_double_buffer.Current(), keys.data(), settings.num_elements * sizeof(T),
  // AJB-algo: DtoH/HtoD transfer — candidate for async stream overlap
                                 cudaMemcpyHostToDevice  // AJB: H2D key传输, stream_pool.GetStream(0)));
  CheckCudaError(cudaMemcpyAsync(values_double_buffer.Current(), values.data(), settings.num_elements * sizeof(V),
  // AJB-algo: DtoH/HtoD transfer — candidate for async stream overlap
                                 cudaMemcpyHostToDevice  // AJB: H2D value传输, stream_pool.GetStream(0)));

  CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

  // AJB: GPU归并算法分发
  const auto& algo = settings.gpu_merge_algorithm;
  if (algo == "thrust_merge_by_key") {
    TimeScope time_scope("gpu_merge_phase");

    thrust::merge_by_key(
        thrust::cuda::par_nosync(device_allocator).on(stream_pool.GetStream(0)), keys_double_buffer.Current(),
        keys_double_buffer.Current() + chunk_size, keys_double_buffer.Current() + chunk_size,
        keys_double_buffer.Current() + settings.num_elements, values_double_buffer.Current(),
        values_double_buffer.Current() + chunk_size, keys_double_buffer.Alternate(), values_double_buffer.Alternate());

    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
  } else if (algo == "mgpu_merge") {
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

  // AJB: D2H结果拷贝
  const size_t key_d2h_bytes = settings.num_elements * sizeof(T);
  const size_t val_d2h_bytes = settings.num_elements * sizeof(V);
  CheckCudaError(cudaMemcpyAsync(keys.data(), keys_double_buffer.Current(), key_d2h_bytes,
  // AJB-algo: DtoH/HtoD transfer — candidate for async stream overlap
                                 cudaMemcpyDeviceToHost, stream_pool.GetStream(0)));
                                 // AJB-algo: DtoH/HtoD transfer — candidate for async stream overlap
  CheckCudaError(cudaMemcpyAsync(values.data(), values_double_buffer.Current(), settings.num_elements * sizeof(V),
  // AJB-algo: DtoH/HtoD transfer — candidate for async stream overlap
                                 cudaMemcpyDeviceToHost, stream_pool.GetStream(0)));
                                 // AJB-algo: DtoH/HtoD transfer — candidate for async stream overlap

  CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

  // Upstream: 4 separate deallocate calls.
  // Changed: deallocate in pairs (keys then values) to match allocation order.
  device_allocator.deallocate(reinterpret_cast<uint8_t*>(keys_double_buffer.Current()));
  device_allocator.deallocate(reinterpret_cast<uint8_t*>(keys_double_buffer.Alternate()));
  device_allocator.deallocate(reinterpret_cast<uint8_t*>(values_double_buffer.Current()));
  device_allocator.deallocate(reinterpret_cast<uint8_t*>(values_double_buffer.Alternate()));

  // AJB: 结果输出——保留原格式但加运行统计
  std::cout << settings.num_elements << "," << settings.num_threads << ",\"" << settings.gpu_merge_algorithm << "\",\""
            << settings.key_type << "\",\"" << settings.value_type << "\"," << settings.random_seed << ",";
  std::cout << std::fixed << std::setprecision(9) << termcolor::green
            << TimeDurations::Get().GetDuration("gpu_merge_phase") << termcolor::reset << std::endl;

  auto inv = std::adjacent_find(keys.begin(), keys.end(), [](const T& a, const T& b) { return b < a; });
  if (inv != keys.end()) {
    size_t pos = std::distance(keys.begin(), inv);
    fprintf(stderr, "[ERROR] Run  // AJB: stderr代替stdoutGpuMergeBenchmark: Invalid order at index %zu.\n", pos);
  }
}

static bool ValidateSettings(const Settings& s) {
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidNumElements(s.num_elements) ||
      !OptionsLimits::IsValidNumThreads(s.num_threads)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidGpuMergeAlgorithm(s.gpu_merge_algorithm)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidType(s.key_type) || !OptionsLimits::IsValidType(s.value_type)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidRandomSeed(s.random_seed)) return false;
  return true;
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

  if (!ValidateSettings(s) || parse_result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (s.key_type == "int" && s.value_type == "int") {
    // [AJB_BP] 完整配置回显: 让日志自描述
    fprintf(stderr, "[AJB_BP][gpu_merge] config: elements=%zu threads=%zu algo=%s seed=%u\n",
            s.num_elements, s.num_threads, s.gpu_merge_algorithm.c_str(), s.random_seed);
    auto t0 = std::chrono::steady_clock::now();
    RunGpuMergeBenchmark<int, int>(s);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[AJB_BP][gpu_merge] done: wall=%.1fms throughput=%.1f Melems/s\n",
            ms, s.num_elements / (ms / 1000.0) / 1e6);
  }

  return 0;
}
