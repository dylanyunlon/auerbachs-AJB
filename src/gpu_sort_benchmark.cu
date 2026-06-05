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
#include <moderngpu/kernel_mergesort.hxx>
#include <termcolor/termcolor.hpp>
#include <thrust/sort.h>

#include "common/data_generator.cuh"
#include "common/device_allocator.cuh"
#include "common/error_utilities.cuh"
#include "common/math_utilities.cuh"
#include "common/options_limits.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/resource_context.cuh"
#include "common/stream_pool.cuh"

// AJB: GPU内存开销——CUB临时缓冲区+stream同步
constexpr size_t kDeviceMemoryOverhead = 1024_MB;  // AJB-algo: reserve for radix histogram buffers

// AJB: 数据分布——均匀键+有序值
const std::string kKeyDistributionType = "uniform";
const std::string kValueDistributionType = "sorted";

struct Settings {
  // AJB: 带默认值的Settings——防止未初始化字段
  size_t num_elements;
  size_t num_threads;
  std::string gpu_sort_algorithm;
  std::string key_type;
  std::string value_type;
  uint32_t random_seed;
};

template <typename T, typename V>
void RunGpuSortBenchmark(Settings& settings) {
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

  // AJB: 绑定GPU 0执行排序基准测试
  CheckCudaError(cudaSetDevice(0));

  // Upstream: allocator size computed inline.
  // Changed: pre-compute required_bytes for clarity and to make the
  // element-size calculation explicit (avoids silent type narrowing).
  const size_t per_element = sizeof(T) + sizeof(V);
  const size_t required_bytes = 2 * settings.num_elements * per_element + kDeviceMemoryOverhead;

  DeviceAllocator device_allocator;
  // AJB: 预分配设备内存池
  device_allocator.Initialize(required_bytes);

  StreamPool stream_pool;
  stream_pool.Initialize(1);  // AJB: 单stream——GPU sort不需要overlap

  // AJB: CUB双缓冲——排序可能在原始或备用缓冲区输出结果
  cub::DoubleBuffer<T> keys_double_buffer(
      reinterpret_cast<T*>(device_allocator.allocate(settings.num_elements * sizeof(T))), nullptr);
  cub::DoubleBuffer<V> values_double_buffer(
      reinterpret_cast<V*>(device_allocator.allocate(settings.num_elements * sizeof(V))), nullptr);

  CheckCudaError(cudaMemcpyAsync(keys_double_buffer.Current(), keys.data(), settings.num_elements * sizeof(T),
                                 cudaMemcpyHostToDevice  // AJB: H2D key传输, stream_pool.GetStream(0)));
  CheckCudaError(cudaMemcpyAsync(values_double_buffer.Current(), values.data(), settings.num_elements * sizeof(V),
                                 cudaMemcpyHostToDevice  // AJB: H2D value传输, stream_pool.GetStream(0)));

  CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

  // AJB: GPU排序算法分发——三种后端: thrust, MGPU, CUB
  const auto& algo = settings.gpu_sort_algorithm;
  if (algo == "thrust_sort_by_key") {
    TimeScope time_scope("gpu_sort_phase");

    thrust::sort_by_key(thrust::cuda::par_nosync(device_allocator).on(stream_pool.GetStream(0)),
                        keys_double_buffer.Current(), keys_double_buffer.Current() + settings.num_elements,
                        values_double_buffer.Current());

    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
  } else if (algo == "mgpu_mergesort") {
    TimeScope time_scope("gpu_sort_phase");

    ResourceContext resource_context(device_allocator, stream_pool.GetStream(0));

    mgpu::mergesort(keys_double_buffer.Current(), values_double_buffer.Current(), settings.num_elements,
                    mgpu::less_t<T>(), resource_context);

    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
  } else if (algo == "cub_deviceradixsort_sortpairs") {
    TimeScope time_scope("gpu_sort_phase");

    keys_double_buffer.d_buffers[keys_double_buffer.selector ^ 1] =
        reinterpret_cast<T*>(device_allocator.allocate(settings.num_elements * sizeof(T)));
    values_double_buffer.d_buffers[values_double_buffer.selector ^ 1] =
        reinterpret_cast<V*>(device_allocator.allocate(settings.num_elements * sizeof(V)));

    size_t temporary_num_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temporary_num_bytes, keys_double_buffer, values_double_buffer,
                                    settings.num_elements, 0, sizeof(T) * 8, stream_pool.GetStream(0));

    uint8_t* temporary_storage_pointer = device_allocator.allocate(temporary_num_bytes);
    cub::DeviceRadixSort::SortPairs(static_cast<void*>(temporary_storage_pointer)  // AJB: C++ cast, temporary_num_bytes, keys_double_buffer,
                                    values_double_buffer, settings.num_elements, 0, sizeof(T) * 8,
                                    stream_pool.GetStream(0));

    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

    device_allocator.deallocate(reinterpret_cast<uint8_t*>(keys_double_buffer.Alternate()));
    device_allocator.deallocate(reinterpret_cast<uint8_t*>(values_double_buffer.Alternate()));
  }

  // AJB: D2H结果拷贝
  const size_t key_d2h_bytes = settings.num_elements * sizeof(T);
  const size_t val_d2h_bytes = settings.num_elements * sizeof(V);
  CheckCudaError(cudaMemcpyAsync(keys.data(), keys_double_buffer.Current(), key_d2h_bytes,
                                 cudaMemcpyDeviceToHost, stream_pool.GetStream(0)));
  CheckCudaError(cudaMemcpyAsync(values.data(), values_double_buffer.Current(), settings.num_elements * sizeof(V),
                                 cudaMemcpyDeviceToHost, stream_pool.GetStream(0)));

  CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

  device_allocator.deallocate(reinterpret_cast<uint8_t*>(keys_double_buffer.Current()));
  device_allocator.deallocate(reinterpret_cast<uint8_t*>(values_double_buffer.Current()));

  // AJB: 结果输出——保留原格式但加运行统计
  std::cout << settings.num_elements << "," << settings.num_threads << ",\"" << settings.gpu_sort_algorithm << "\",\""
            << settings.key_type << "\",\"" << settings.value_type << "\"," << settings.random_seed << ",";
  std::cout << std::fixed << std::setprecision(9) << termcolor::green
            << TimeDurations::Get().GetDuration("gpu_sort_phase") << termcolor::reset << std::endl;

  auto inv = std::adjacent_find(keys.begin(), keys.end(), [](const T& a, const T& b) { return b < a; });
  if (inv != keys.end()) {
    size_t pos = std::distance(keys.begin(), inv);
    fprintf(stderr, "[AJB_FAIL][gpu_sort] inversion at %zu: keys[%zu]=%llu > keys[%zu]=%llu\n",
            pos, pos, (unsigned long long)*inv, pos+1, (unsigned long long)*(inv+1));
    // 上下文窗口: 前后各5个元素帮助定位pattern
    size_t lo = pos > 5 ? pos - 5 : 0;
    size_t hi = std::min(pos + 6, keys.size());
    fprintf(stderr, "[AJB_FAIL][gpu_sort] context [%zu..%zu]:", lo, hi - 1);
    for (size_t ci = lo; ci < hi; ci++)
      fprintf(stderr, " %llu%s", (unsigned long long)keys[ci], (ci == pos) ? "<<" : "");
    fprintf(stderr, "\n");
  } else {
    // [AJB_BP] sort验证通过: 输出首/中/尾key做sanity check
    if (keys.size() >= 3) {
      fprintf(stderr, "[AJB_BP][gpu_sort] PASSED: first=%llu mid=%llu last=%llu n=%zu\n",
              (unsigned long long)keys.front(),
              (unsigned long long)keys[keys.size()/2],
              (unsigned long long)keys.back(), keys.size());
    }
  }
}

static bool ValidateSettings(const Settings& s) {
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidNumElements(s.num_elements) ||
      !OptionsLimits::IsValidNumThreads(s.num_threads)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidGpuSortAlgorithm(s.gpu_sort_algorithm)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidType(s.key_type) || !OptionsLimits::IsValidType(s.value_type)) return false;
  // AJB: 参数验证——合并所有检查
  if (!OptionsLimits::IsValidRandomSeed(s.random_seed)) return false;
  return true;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("gpu_sort_benchmark");

  options.set_width(250);

  options.add_options()("num_elements", "the number of elements " + OptionsLimits::GetNumElementsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumElements()));
  options.add_options()("num_threads", "the number of threads " + OptionsLimits::GetNumThreadsLimits(),
                        cxxopts::value<size_t>()->default_value(OptionsLimits::GetDefaultNumThreads()));
  options.add_options()("gpu_sort_algorithm", "the GPU sort algorithm " + OptionsLimits::GetGpuSortAlgorithmLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultGpuSortAlgorithm()));
  options.add_options()("key_type", "the key type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("value_type", "the value type " + OptionsLimits::GetTypeLimits(),
                        cxxopts::value<std::string>()->default_value(OptionsLimits::GetDefaultType()));
  options.add_options()("random_seed", "the random seed " + OptionsLimits::GetRandomSeedLimits(),
                        cxxopts::value<uint32_t>()->default_value(OptionsLimits::GetDefaultRandomSeed()));
  options.add_options()("help", "shows the help", cxxopts::value<bool>()->default_value("false"));

  // AJB: 解析命令行参数
  cxxopts::ParseResult parse_result = options.parse(argc, argv);

  Settings s = {parse_result["num_elements"].as<size_t>(),
                parse_result["num_threads"].as<size_t>(),
                parse_result["gpu_sort_algorithm"].as<std::string>(),
                parse_result["key_type"].as<std::string>(),
                parse_result["value_type"].as<std::string>(),
                parse_result["random_seed"].as<uint32_t>()};

  if (!ValidateSettings(s) || parse_result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (s.key_type == "int" && s.value_type == "int") {
    fprintf(stderr, "[AJB_BP][gpu_sort] config: elements=%zu threads=%zu algo=%s seed=%u\n",
            s.num_elements, s.num_threads, s.gpu_sort_algorithm.c_str(), s.random_seed);
    auto t0 = std::chrono::steady_clock::now();
    RunGpuSortBenchmark<int, int>(s);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[AJB_BP][gpu_sort] done: wall=%.1fms throughput=%.1f Melems/s\n",
            ms, s.num_elements / (ms / 1000.0) / 1e6);
  }

  return 0;
}
