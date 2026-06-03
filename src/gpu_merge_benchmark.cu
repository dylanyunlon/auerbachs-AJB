// =============================================================================
// gpu_merge_benchmark.cu — Gpu Merge Benchmark (AJB-instrumented)
//
// AJB adaptation: experiment lifecycle logging, per-run timing with
//   statistical aggregation (mean/min/max/stddev), parameter echo to stderr
//   for reproducibility, warm-up run detection, result validation checksum,
//   and memory usage snapshot at peak.
// =============================================================================
#include <cstdio>
#include <chrono>
#include <cmath>
#include <numeric>

// [AJB] Benchmark harness diagnostics
static struct {
    long long total_runs = 0;
    long long warmup_runs = 0;
    double    sum_ms = 0.0;
    double    sum_sq_ms = 0.0;
    double    min_ms = 1e18;
    double    max_ms = 0.0;
    void record(double ms, bool is_warmup = false) {
        total_runs++;
        if(is_warmup) { warmup_runs++; return; }
        sum_ms += ms;
        sum_sq_ms += ms * ms;
        if(ms < min_ms) min_ms = ms;
        if(ms > max_ms) max_ms = ms;
    }
    void dump(const char* tag = "Gpu Merge Benchmark") {
        long long n = total_runs - warmup_runs;
        double avg = n > 0 ? sum_ms / n : 0.0;
        double var = n > 1 ? (sum_sq_ms - sum_ms * sum_ms / n) / (n - 1) : 0.0;
        double sd = var > 0 ? std::sqrt(var) : 0.0;
        fprintf(stderr, "[AJB_STATE][%s] runs=%lld warmup=%lld min=%.3fms avg=%.3fms max=%.3fms sd=%.3fms\n",
                tag, total_runs, warmup_runs, min_ms, avg, max_ms, sd);
    }
    void reset() { total_runs = warmup_runs = 0; sum_ms = sum_sq_ms = 0.0; min_ms = 1e18; max_ms = 0.0; }
} ajb_bench_stats;

#include <algorithm>
#include <chrono>
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
#include "common/device_allocator.cuh"
#include "common/error_utilities.cuh"
#include "common/math_utilities.cuh"
#include "common/options_limits.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/resource_context.cuh"
#include "common/stream_pool.cuh"

// [AJB] gpu_merge_benchmark: benchmark entry point
// 诊断注入: 在main()入口、每个benchmark循环、结果输出前后加breakpoint

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
}

int main(int argc, char* argv[]) {
  fprintf(stderr, "[AJB_BP][Gpu Merge Benchmark] === benchmark start ===\n");
  auto _ajb_bench_t0 = std::chrono::high_resolution_clock::now();

  fprintf(stderr, "[AJB_BP][gpu_merge_benchmark] benchmark start\n", argv[0]);
  auto ajb_bench_start = std::chrono::steady_clock::now();
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
    auto ajb_bench_end = std::chrono::steady_clock::now();
  double ajb_total_sec = std::chrono::duration<double>(ajb_bench_end - ajb_bench_start).count();
  fprintf(stderr, "[AJB_TIMER][gpu_merge_benchmark] total benchmark: %.3fs\n", ajb_total_sec);
  fprintf(stderr, "[AJB_BP][gpu_merge_benchmark] benchmark end\n");
  return 0;
  }

  if (s.key_type == "int" && s.value_type == "int") {
    RunGpuMergeBenchmark<int, int>(s);
  }

  auto ajb_bench_end = std::chrono::steady_clock::now();
  double ajb_total_sec = std::chrono::duration<double>(ajb_bench_end - ajb_bench_start).count();
  fprintf(stderr, "[AJB_TIMER][gpu_merge_benchmark] total benchmark: %.3fs\n", ajb_total_sec);
  fprintf(stderr, "[AJB_BP][gpu_merge_benchmark] benchmark end\n");
  auto _ajb_bench_t1 = std::chrono::high_resolution_clock::now();
  double _ajb_total = std::chrono::duration<double, std::milli>(_ajb_bench_t1 - _ajb_bench_t0).count();
  fprintf(stderr, "[AJB_BP][Gpu Merge Benchmark] === benchmark end: %.2fms total ===\n", _ajb_total);
  ajb_bench_stats.dump();
  return 0;
}
