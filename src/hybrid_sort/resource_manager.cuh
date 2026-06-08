#pragma once

#include <algorithm>
#include <vector>
#include <cstdio>

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
        per_gpu_alloc_bytes_(gpus.size(), 0) {

    // Upstream: std::map<int,int> 做GPU id到index的映射.
    // AJB改写: 用flat数组+线性扫描替代map.
    // 理由: GPU数量<=64 (kMaxNumGpus), map的红黑树开销不值得,
    // 而且在OMP并行区内map的iterator invalidation是隐患.
    gpu_index_flat_.resize(256, -1);  // 覆盖常见GPU device id范围
    for (size_t g = 0; g < gpus_.size(); ++g) {
      int dev = gpus_[g];
      if (dev >= 0 && dev < (int)gpu_index_flat_.size()) {
        gpu_index_flat_[dev] = (int)g;
      }
    }

    fprintf(stderr, "[DEBUG][ResourceManager] init: %zu GPUs, %zu streams/GPU, chunk=%zu\n",
            gpus_.size(), num_streams, chunk_size);

#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));

      host_allocators_[g].Initialize(kHostMemory);

      size_t free_global_memory, total_global_memory;
      CheckCudaError(cudaMemGetInfo(&free_global_memory, &total_global_memory));
      device_allocators_[g].Initialize(kDeviceMemoryUtilization * free_global_memory);

      stream_pools_[g].Initialize(num_streams_);

      const size_t adjusted_chunk_size = kChunkSizeMultiplier * chunk_size;
      size_t buf_bytes = adjusted_chunk_size * (sizeof(T) + sizeof(V)) * 2;

      keys_double_buffers_[g].d_buffers[0] =
          reinterpret_cast<T*>(device_allocators_[g].allocate(adjusted_chunk_size * sizeof(T)));
      keys_double_buffers_[g].d_buffers[1] =
          reinterpret_cast<T*>(device_allocators_[g].allocate(adjusted_chunk_size * sizeof(T)));
      values_double_buffers_[g].d_buffers[0] =
          reinterpret_cast<V*>(device_allocators_[g].allocate(adjusted_chunk_size * sizeof(V)));
      values_double_buffers_[g].d_buffers[1] =
          reinterpret_cast<V*>(device_allocators_[g].allocate(adjusted_chunk_size * sizeof(V)));

      per_gpu_alloc_bytes_[g] = buf_bytes;

      fprintf(stderr, "[DEBUG][ResourceManager] GPU %d: free=%zuMB total=%zuMB "
              "arena=%.0fMB buffers=%zuMB\n",
              gpus_[g], free_global_memory >> 20, total_global_memory >> 20,
              (kDeviceMemoryUtilization * free_global_memory) / (1 << 20),
              buf_bytes >> 20);
    }

    // Warmup阶段: 对每个stream做一次小排序, 触发CUDA lazy init.
    // Upstream用倒序数组 + thrust::sort.
    // AJB改写: 用 cub::DeviceRadixSort 替代 thrust::sort 做warmup,
    // 因为实际benchmark用的是cub的radix sort, warmup应该预热同一条代码路径.
#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));

      for (size_t s = 0; s < num_streams_; ++s) {
        int* host_elements =
            reinterpret_cast<int*>(host_allocators_[g].allocate(kNumInitializationElements * sizeof(int)));
        int* device_in =
            reinterpret_cast<int*>(device_allocators_[g].allocate(kNumInitializationElements * sizeof(int)));
        int* device_out =
            reinterpret_cast<int*>(device_allocators_[g].allocate(kNumInitializationElements * sizeof(int)));

        for (size_t i = 0; i < kNumInitializationElements; ++i) {
          host_elements[i] = kNumInitializationElements - i;
        }

        CheckCudaError(cudaMemcpyAsync(device_in, host_elements, kNumInitializationElements * sizeof(int),
                                       cudaMemcpyHostToDevice, stream_pools_[g].GetStream(s)));

        // cub::DeviceRadixSort需要临时空间, 先query再分配
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, device_in, device_out,
                                       kNumInitializationElements, 0, sizeof(int) * 8,
                                       stream_pools_[g].GetStream(s));
        uint8_t* d_temp = device_allocators_[g].allocate(temp_bytes);
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, device_in, device_out,
                                       kNumInitializationElements, 0, sizeof(int) * 8,
                                       stream_pools_[g].GetStream(s));

        // 把结果拷回来验证warmup排序正确性
        CheckCudaError(cudaMemcpyAsync(host_elements, device_out, kNumInitializationElements * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_pools_[g].GetStream(s)));
        CheckCudaError(cudaStreamSynchronize(stream_pools_[g].GetStream(s)));

        // 验证: warmup数组应该有序
        bool warmup_ok = true;
        for (size_t i = 1; i < kNumInitializationElements; ++i) {
          if (host_elements[i] < host_elements[i - 1]) { warmup_ok = false; break; }
        }
        if (!warmup_ok) {
          fprintf(stderr, "[DEBUG][ResourceManager] WARNING: warmup sort FAILED on GPU %d stream %zu\n",
                  gpus_[g], s);
        }

        device_allocators_[g].deallocate(d_temp);
        device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(device_out));
        host_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(host_elements));
        device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(device_in));
      }
    }
    fprintf(stderr, "[DEBUG][ResourceManager] warmup done (%zu GPUs × %zu streams)\n",
            gpus_.size(), num_streams_);
  }

  ~ResourceManager() {
#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));

      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(keys_double_buffers_[g].Current()));
      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(keys_double_buffers_[g].Alternate()));
      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(values_double_buffers_[g].Current()));
      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(values_double_buffers_[g].Alternate()));
    }
  }

  // Upstream: map查找. AJB: flat array O(1)直查
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

  // 断点调试: 打印所有GPU的显存使用快照
  void dump_memory_state(const char* tag = "") const {
    fprintf(stderr, "[DEBUG][ResourceManager::dump] %s\n", tag);
    for (size_t g = 0; g < gpus_.size(); ++g) {
      fprintf(stderr, "  GPU %d: buffer_alloc=%zuMB\n",
              gpus_[g], per_gpu_alloc_bytes_[g] >> 20);
    }
  }

 private:
  static constexpr size_t kHostMemory = 64_MB;
  static constexpr double kDeviceMemoryUtilization = 0.98;
  static constexpr double kChunkSizeMultiplier = 1.01;
  static constexpr size_t kNumInitializationElements = 10;

  std::vector<int> gpus_;
  std::vector<int> gpu_index_flat_;  // 替代upstream的 std::map<int,int>

  const size_t num_streams_;

  std::vector<HostAllocator>& host_allocators_;
  std::vector<DeviceAllocator>& device_allocators_;
  std::vector<StreamPool>& stream_pools_;

  std::vector<cub::DoubleBuffer<T>> keys_double_buffers_;
  std::vector<cub::DoubleBuffer<V>> values_double_buffers_;

  std::vector<size_t> per_gpu_alloc_bytes_;

  // --- M1120: Memory fragmentation metrics ---
  // Tracks allocation/deallocation patterns to estimate fragmentation.
  // fragmentation_index = 1 - (largest_free_block / total_free)
  // 0 = no fragmentation (one contiguous block), 1 = fully fragmented
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

    // Fragmentation index: ratio of overhead from many small allocations
    // Approximation: 1.0 - (peak / (alloc_count * largest_single))
    double fragmentation_index() const {
      if (allocation_count == 0 || largest_single_alloc == 0) return 0.0;
      double ideal = static_cast<double>(peak_allocated);
      double actual_spread = static_cast<double>(allocation_count) * largest_single_alloc;
      if (actual_spread <= 0.0) return 0.0;
      double frag = 1.0 - (ideal / actual_spread);
      return frag < 0.0 ? 0.0 : (frag > 1.0 ? 1.0 : frag);
    }

    void dump(int gpu_id) const {
      fprintf(stderr, "[AJB_BP][FragTracker][GPU%d] allocs=%zu frees=%zu "
              "peak=%zuB frag_index=%.4f largest_single=%zuB\n",
              gpu_id, allocation_count, deallocation_count,
              peak_allocated, fragmentation_index(), largest_single_alloc);
    }
  };

  std::vector<FragmentationTracker> frag_trackers_;

 public:
  // NUMA-aware allocation hint: returns the preferred GPU for a given
  // data partition based on minimizing cross-NUMA transfer.
  // Heuristic: assign partition i to GPU (i % num_gpus), but prefer
  // GPUs on the same NUMA node if topology info is available.
  int NumaPreferredGpu(size_t partition_idx) const {
    if (gpus_.empty()) return 0;
    // Simple round-robin across available GPUs
    // A real implementation would query cudaDeviceGetAttribute for
    // cudaDevAttrNumaConfig and prefer co-located GPUs
    int gpu = gpus_[partition_idx % gpus_.size()];
    fprintf(stderr, "[AJB_BP][NUMA] partition=%zu -> preferred GPU=%d\n",
            partition_idx, gpu);
    return gpu;
  }

  // Report fragmentation across all GPUs
  void DumpFragmentation() const {
    for (size_t g = 0; g < gpus_.size() && g < frag_trackers_.size(); ++g) {
      frag_trackers_[g].dump(gpus_[g]);
    }
  }
};
