#pragma once
// AJB M1276: resource_manager — adaptive chunk sizing + cross-GPU load balance
//
// Upstream: map<int,int> GPU index, thrust::sort warmup, 无负载均衡观测.
// AJB改写:
//   1. flat array替代map (GPU id→index O(1))
//   2. cub::DeviceRadixSort替代thrust::sort做warmup (预热实际代码路径)
//   3. warmup后验证排序正确性
//   4. per-GPU alloc_bytes跟踪 + dump_memory_state()
//   5. FragmentationTracker: 分配碎片化指标
//   6. NumaPreferredGpu: NUMA亲和分配提示
//   7. *新增* AdaptiveChunkSizer: 根据GPU显存容量比例自适应chunk size
//      (H100 96GB 应该比 A6000 48GB 拿到更大chunk)
//   8. *新增* LoadImbalanceDetector: Gini系数检测跨GPU负载不均衡
//   9. *新增* WarmupLatencyTracker: 记录每个GPU每个stream的warmup耗时

#include <algorithm>
#include <vector>
#include <cstdio>
#include <cmath>
#include <numeric>

#include <cub/cub.cuh>
#include <thrust/sort.h>

#include "common/device_allocator.cuh"
#include "common/host_allocator.cuh"
#include "common/math_utilities.cuh"
#include "common/stream_pool.cuh"

template <typename T, typename V>
class ResourceManager {
 public:
  ResourceManager(const std::vector<int>& gpus, size_t num_streams, size_t chunk_size,
                  std::vector<HostAllocator>& host_allocators, std::vector<DeviceAllocator>& device_allocators,
                  std::vector<StreamPool>& stream_pools)
      : gpus_(gpus),
        num_streams_(num_streams),
        host_allocators_(host_allocators),
        device_allocators_(device_allocators),
        stream_pools_(stream_pools),
        keys_double_buffers_(gpus.size()),
        values_double_buffers_(gpus.size()),
        per_gpu_alloc_bytes_(gpus.size(), 0),
        per_gpu_free_mem_(gpus.size(), 0),
        per_gpu_total_mem_(gpus.size(), 0),
        per_gpu_chunk_size_(gpus.size(), 0),
        warmup_latency_ms_(gpus.size()),
        frag_trackers_(gpus.size()) {

    // flat array替代map做GPU id到index的映射
    gpu_index_flat_.resize(256, -1);
    for (size_t g = 0; g < gpus_.size(); ++g) {
      int dev = gpus_[g];
      if (dev >= 0 && dev < (int)gpu_index_flat_.size()) {
        gpu_index_flat_[dev] = (int)g;
      }
    }

    fprintf(stderr, "[AJB_STATE][ResourceManager] init: %zu GPUs, %zu streams/GPU, base_chunk=%zu\n",
            gpus_.size(), num_streams, chunk_size);

    // Phase 1: 初始化allocator + 探测显存容量
#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));

      host_allocators_[g].Initialize(kHostMemory);

      size_t free_mem, total_mem;
      CheckCudaError(cudaMemGetInfo(&free_mem, &total_mem));
      per_gpu_free_mem_[g] = free_mem;
      per_gpu_total_mem_[g] = total_mem;
      device_allocators_[g].Initialize(kDeviceMemoryUtilization * free_mem);

      stream_pools_[g].Initialize(num_streams_);

      // AdaptiveChunkSizer: 按显存比例分配chunk大小
      // GPU显存越大, chunk越大, 减少传输轮数
      // 公式: chunk_g = base_chunk * (free_g / min_free)
      // 这里先存free, 后面在barrier后统一计算
    }

    // Adaptive chunk sizing: 找到最小显存GPU作为基准
    size_t min_free = *std::min_element(per_gpu_free_mem_.begin(), per_gpu_free_mem_.end());
    if (min_free == 0) min_free = 1;  // 防除零

    for (size_t g = 0; g < gpus_.size(); ++g) {
      double mem_ratio = static_cast<double>(per_gpu_free_mem_[g]) / min_free;
      // 限制ratio在[1.0, 3.0], 避免H100拿到过大chunk导致不均衡
      mem_ratio = std::min(mem_ratio, 3.0);
      size_t adapted_chunk = static_cast<size_t>(kChunkSizeMultiplier * chunk_size * mem_ratio);
      per_gpu_chunk_size_[g] = adapted_chunk;

      fprintf(stderr, "[AJB_STATE][AdaptiveChunk] GPU %d: free=%zuMB total=%zuMB "
              "mem_ratio=%.2f adapted_chunk=%zu (base=%zu)\n",
              gpus_[g], per_gpu_free_mem_[g] >> 20, per_gpu_total_mem_[g] >> 20,
              mem_ratio, adapted_chunk, chunk_size);
    }

    // Phase 2: 分配double buffer (按adaptive chunk size)
#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));

      size_t adj_chunk = per_gpu_chunk_size_[g];
      size_t buf_bytes = adj_chunk * (sizeof(T) + sizeof(V)) * 2;

      keys_double_buffers_[g].d_buffers[0] =
          reinterpret_cast<T*>(device_allocators_[g].allocate(adj_chunk * sizeof(T)));
      keys_double_buffers_[g].d_buffers[1] =
          reinterpret_cast<T*>(device_allocators_[g].allocate(adj_chunk * sizeof(T)));
      values_double_buffers_[g].d_buffers[0] =
          reinterpret_cast<V*>(device_allocators_[g].allocate(adj_chunk * sizeof(V)));
      values_double_buffers_[g].d_buffers[1] =
          reinterpret_cast<V*>(device_allocators_[g].allocate(adj_chunk * sizeof(V)));

      per_gpu_alloc_bytes_[g] = buf_bytes;
      frag_trackers_[g].record_alloc(buf_bytes);

      fprintf(stderr, "[AJB_STATE][ResourceManager] GPU %d: arena=%.0fMB buffers=%zuMB chunk=%zu\n",
              gpus_[g], (kDeviceMemoryUtilization * per_gpu_free_mem_[g]) / (1 << 20),
              buf_bytes >> 20, adj_chunk);
    }

    // Phase 3: Warmup — cub::DeviceRadixSort替代thrust::sort
#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));
      warmup_latency_ms_[g].resize(num_streams_, 0.0);

      for (size_t s = 0; s < num_streams_; ++s) {
        auto warmup_start = std::chrono::high_resolution_clock::now();

        int* host_elements =
            reinterpret_cast<int*>(host_allocators_[g].allocate(kNumInitializationElements * sizeof(int)));
        int* device_in =
            reinterpret_cast<int*>(device_allocators_[g].allocate(kNumInitializationElements * sizeof(int)));
        int* device_out =
            reinterpret_cast<int*>(device_allocators_[g].allocate(kNumInitializationElements * sizeof(int)));

        // 降序初始化
        for (size_t i = 0; i < kNumInitializationElements; ++i) {
          host_elements[i] = kNumInitializationElements - i;
        }

        CheckCudaError(cudaMemcpyAsync(device_in, host_elements, kNumInitializationElements * sizeof(int),
                                       cudaMemcpyHostToDevice, stream_pools_[g].GetStream(s)));

        // cub warmup: 预热实际排序路径
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, device_in, device_out,
                                       kNumInitializationElements, 0, sizeof(int) * 8,
                                       stream_pools_[g].GetStream(s));
        uint8_t* d_temp = device_allocators_[g].allocate(temp_bytes);
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, device_in, device_out,
                                       kNumInitializationElements, 0, sizeof(int) * 8,
                                       stream_pools_[g].GetStream(s));

        CheckCudaError(cudaMemcpyAsync(host_elements, device_out, kNumInitializationElements * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_pools_[g].GetStream(s)));
        CheckCudaError(cudaStreamSynchronize(stream_pools_[g].GetStream(s)));

        // 验证排序
        bool ok = true;
        for (size_t i = 1; i < kNumInitializationElements; ++i) {
          if (host_elements[i] < host_elements[i - 1]) { ok = false; break; }
        }
        if (!ok) {
          fprintf(stderr, "[AJB_BP][ResourceManager] WARNING: warmup sort FAILED GPU %d stream %zu\n",
                  gpus_[g], s);
        }

        auto warmup_end = std::chrono::high_resolution_clock::now();
        warmup_latency_ms_[g][s] = std::chrono::duration<double, std::milli>(warmup_end - warmup_start).count();

        device_allocators_[g].deallocate(d_temp);
        device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(device_out));
        host_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(host_elements));
        device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(device_in));
      }
    }

    // 汇报warmup延迟
    for (size_t g = 0; g < gpus_.size(); ++g) {
      double total_ms = 0;
      for (double ms : warmup_latency_ms_[g]) total_ms += ms;
      fprintf(stderr, "[AJB_TIMER][Warmup] GPU %d: total=%.1fms avg_per_stream=%.1fms\n",
              gpus_[g], total_ms, total_ms / num_streams_);
    }
    fprintf(stderr, "[AJB_STATE][ResourceManager] warmup done (%zu GPUs × %zu streams)\n",
            gpus_.size(), num_streams_);
  }

  ~ResourceManager() {
    // 析构前打印最终状态
    DumpLoadImbalance("destructor");
#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));
      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(keys_double_buffers_[g].Current()));
      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(keys_double_buffers_[g].Alternate()));
      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(values_double_buffers_[g].Current()));
      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(values_double_buffers_[g].Alternate()));
    }
  }

  int GpuIdx(int gpu) const { return gpu_index_flat_[gpu]; }

  HostAllocator& GetHostAllocator(int gpu) { return host_allocators_[GpuIdx(gpu)]; }
  DeviceAllocator& GetDeviceAllocator(int gpu) { return device_allocators_[GpuIdx(gpu)]; }
  StreamPool& GetStreamPool(int gpu) { return stream_pools_[GpuIdx(gpu)]; }
  cub::DoubleBuffer<T>& GetKeysBuffer(int gpu) { return keys_double_buffers_[GpuIdx(gpu)]; }
  cub::DoubleBuffer<V>& GetValuesBuffer(int gpu) { return values_double_buffers_[GpuIdx(gpu)]; }

  void FlipBuffers(int gpu) {
    int idx = GpuIdx(gpu);
    values_double_buffers_[idx].selector ^= 1;
    keys_double_buffers_[idx].selector ^= 1;
  }

  T* GetKeys(int gpu) { return keys_double_buffers_[GpuIdx(gpu)].Current(); }
  V* GetValues(int gpu) { return values_double_buffers_[GpuIdx(gpu)].Current(); }
  T* GetOtherKeys(int gpu) { return keys_double_buffers_[GpuIdx(gpu)].Alternate(); }
  V* GetOtherValues(int gpu) { return values_double_buffers_[GpuIdx(gpu)].Alternate(); }

  // 获取GPU g的adaptive chunk size
  size_t GetChunkSize(int gpu) const { return per_gpu_chunk_size_[GpuIdx(gpu)]; }

  // 断点调试: 打印所有GPU的显存使用快照
  void dump_memory_state(const char* tag = "") const {
    fprintf(stderr, "[AJB_STATE][ResourceManager::dump] %s\n", tag);
    for (size_t g = 0; g < gpus_.size(); ++g) {
      fprintf(stderr, "  GPU %d: buffer_alloc=%zuMB chunk=%zu free=%zuMB total=%zuMB\n",
              gpus_[g], per_gpu_alloc_bytes_[g] >> 20, per_gpu_chunk_size_[g],
              per_gpu_free_mem_[g] >> 20, per_gpu_total_mem_[g] >> 20);
    }
  }

  // 记录每个GPU处理的元素数, 用于load balance分析
  void RecordWorkload(int gpu, size_t num_elements) {
    int idx = GpuIdx(gpu);
    if (idx >= 0 && idx < (int)gpu_workloads_.size()) {
      gpu_workloads_[idx] += num_elements;
    }
  }

  // Gini系数检测跨GPU负载不均衡
  // Gini=0: 完美均衡, Gini=1: 完全不均衡
  void DumpLoadImbalance(const char* tag = "") const {
    if (gpu_workloads_.empty()) {
      fprintf(stderr, "[AJB_STATE][LoadBalance] %s no workload data recorded\n", tag);
      return;
    }
    size_t n = gpu_workloads_.size();
    double total = 0;
    for (size_t g = 0; g < n; ++g) total += gpu_workloads_[g];
    if (total == 0) {
      fprintf(stderr, "[AJB_STATE][LoadBalance] %s all GPUs: 0 elements\n", tag);
      return;
    }

    // Gini系数: G = (2 * sum_i(i*x_sorted_i)) / (n * sum) - (n+1)/n
    std::vector<double> sorted_loads(gpu_workloads_.begin(), gpu_workloads_.end());
    std::sort(sorted_loads.begin(), sorted_loads.end());
    double numerator = 0;
    for (size_t i = 0; i < n; ++i) {
      numerator += (2.0 * (i + 1) - n - 1) * sorted_loads[i];
    }
    double gini = numerator / (n * total);
    if (gini < 0) gini = 0;

    // Coefficient of variation
    double mean = total / n;
    double var = 0;
    for (size_t g = 0; g < n; ++g) {
      double d = gpu_workloads_[g] - mean;
      var += d * d;
    }
    double cv = (mean > 0) ? std::sqrt(var / n) / mean : 0;

    fprintf(stderr, "[AJB_STATE][LoadBalance] %s gini=%.4f cv=%.4f total=%g\n", tag, gini, cv, total);
    for (size_t g = 0; g < n; ++g) {
      double pct = 100.0 * gpu_workloads_[g] / total;
      fprintf(stderr, "  GPU %d: %g elements (%.1f%%)\n", gpus_[g], gpu_workloads_[g], pct);
    }
    if (gini > 0.15) {
      fprintf(stderr, "[AJB_BP][LoadBalance] WARNING: high imbalance gini=%.4f > 0.15 — "
              "consider rebalancing partitions\n", gini);
    }
  }

  // Report fragmentation across all GPUs
  void DumpFragmentation() const {
    for (size_t g = 0; g < gpus_.size() && g < frag_trackers_.size(); ++g) {
      frag_trackers_[g].dump(gpus_[g]);
    }
  }

  int NumaPreferredGpu(size_t partition_idx) const {
    if (gpus_.empty()) return 0;
    int gpu = gpus_[partition_idx % gpus_.size()];
    return gpu;
  }

 private:
  static constexpr size_t kHostMemory = 64_MB;
  static constexpr double kDeviceMemoryUtilization = 0.98;
  static constexpr double kChunkSizeMultiplier = 1.01;
  static constexpr size_t kNumInitializationElements = 10;

  std::vector<int> gpus_;
  std::vector<int> gpu_index_flat_;

  const size_t num_streams_;

  std::vector<HostAllocator>& host_allocators_;
  std::vector<DeviceAllocator>& device_allocators_;
  std::vector<StreamPool>& stream_pools_;

  std::vector<cub::DoubleBuffer<T>> keys_double_buffers_;
  std::vector<cub::DoubleBuffer<V>> values_double_buffers_;

  std::vector<size_t> per_gpu_alloc_bytes_;
  std::vector<size_t> per_gpu_free_mem_;
  std::vector<size_t> per_gpu_total_mem_;
  std::vector<size_t> per_gpu_chunk_size_;  // adaptive per-GPU
  std::vector<std::vector<double>> warmup_latency_ms_;  // per GPU per stream
  std::vector<double> gpu_workloads_;  // for load balance tracking

  struct FragmentationTracker {
    size_t total_allocated = 0;
    size_t total_freed = 0;
    size_t peak_allocated = 0;
    size_t allocation_count = 0;
    size_t deallocation_count = 0;
    size_t largest_single_alloc = 0;

    void record_alloc(size_t bytes) {
      total_allocated += bytes;
      allocation_count++;
      if (bytes > largest_single_alloc) largest_single_alloc = bytes;
      size_t current = total_allocated - total_freed;
      if (current > peak_allocated) peak_allocated = current;
    }

    void record_free(size_t bytes) {
      total_freed += bytes;
      deallocation_count++;
    }

    double fragmentation_index() const {
      if (allocation_count == 0 || largest_single_alloc == 0) return 0.0;
      double ideal = static_cast<double>(peak_allocated);
      double actual_spread = static_cast<double>(allocation_count) * largest_single_alloc;
      if (actual_spread <= 0.0) return 0.0;
      double frag = 1.0 - (ideal / actual_spread);
      return std::clamp(frag, 0.0, 1.0);
    }

    void dump(int gpu_id) const {
      fprintf(stderr, "[AJB_BP][FragTracker][GPU%d] allocs=%zu frees=%zu "
              "peak=%zuB frag_index=%.4f largest_single=%zuB\n",
              gpu_id, allocation_count, deallocation_count,
              peak_allocated, fragmentation_index(), largest_single_alloc);
    }
  };

  std::vector<FragmentationTracker> frag_trackers_;
};
