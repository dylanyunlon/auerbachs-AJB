// [AJB] hybrid_sort/hybrid_sort.cuh: 混合排序入口: 小数据用merge sort, 大数据用radix sort
#include <cstdio>
#pragma once

#include <algorithm>
#include <functional>
#include <vector>

#include "common/device_allocator.cuh"
#include "common/error_utilities.cuh"
#include "common/host_allocator.cuh"
#include "common/math_utilities.cuh"
#include "common/parallel_algorithms.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/stream_pool.cuh"
#include "merge_sort/merge_sort.cuh"
#include "radix_sort/radix_sort.cuh"
#include "resource_manager.cuh"

enum class HybridSortKernel { kMerge, kRadix };

size_t CalculateChunkSize(const std::vector<int>& gpus, size_t num_elements, size_t num_bytes_per_element) {
  constexpr double kChunkSizeDeviceMemoryUtilization = 0.8;

  size_t free_global_memory, total_global_memory;
  CheckCudaError(cudaMemGetInfo(&free_global_memory, &total_global_memory));

  const size_t max_chunk_size = free_global_memory / num_bytes_per_element * kChunkSizeDeviceMemoryUtilization / 2;
  const size_t preferred_chunk_size = DivideUp(num_elements, gpus.size());

  return std::min(max_chunk_size, preferred_chunk_size);
}

template <typename T, typename V, HybridSortKernel kernel>
void HybridSort(PinnedVector<T>& keys, PinnedVector<V>& values, PinnedVector<T>& temporary_keys,
                PinnedVector<V>& temporary_values, const std::vector<int>& gpus,
                std::vector<HostAllocator>& host_allocators, std::vector<DeviceAllocator>& device_allocators,
                std::vector<StreamPool>& stream_pools, size_t num_elements, size_t& chunk_size) {
  if (chunk_size == 0) {
    chunk_size = CalculateChunkSize(gpus, num_elements, sizeof(T) + sizeof(V));
  }
  const size_t num_elements_per_chunk_group = chunk_size * gpus.size();
  const size_t num_chunk_groups = DivideUp(num_elements, num_elements_per_chunk_group);

  size_t num_streams = 0;
  if (kernel == HybridSortKernel::kMerge) {
    num_streams = kNumMergeStreams;
  } else if (kernel == HybridSortKernel::kRadix) {
    num_streams = kNumRadixStreams;
  }

  ResourceManager<T, V> manager(gpus, num_streams, chunk_size, host_allocators, device_allocators, stream_pools);

  {
    TimeScope time_scope("sort_phase");

    std::function<void()> synchonize_transfers;

    size_t num_remaining_elements = num_elements;
    for (size_t i = 0; i < num_chunk_groups; ++i) {
      const size_t offset = i * num_elements_per_chunk_group;
      const size_t num_elements_to_process = std::min(num_remaining_elements, num_elements_per_chunk_group);

      T* in_keys = keys.data() + offset;
      V* in_values = values.data() + offset;

      T* out_keys = (num_chunk_groups > 1 ? temporary_keys.data() : keys.data()) + offset;
      T* out_values = (num_chunk_groups > 1 ? temporary_values.data() : values.data()) + offset;

      if (kernel == HybridSortKernel::kMerge) {
        synchonize_transfers =
            MergeSort<T, V>(in_keys, in_values, out_keys, out_values, num_elements_to_process, manager, gpus);
      } else if (kernel == HybridSortKernel::kRadix) {
        synchonize_transfers =
            RadixSort<T, V>(in_keys, in_values, out_keys, out_values, num_elements_to_process, manager, gpus);
      }

      num_remaining_elements -= num_elements_to_process;
    }

    synchonize_transfers();
  }

  {
    TimeScope time_scope("merge_phase");

    if (num_chunk_groups > 1) {
      ParallelMergePairs(temporary_keys, temporary_values, keys, values, num_elements, num_chunk_groups,
                         num_elements_per_chunk_group);
    }
  }
}

// [AJB] hybrid_sort_hybrid_sort 诊断报告
static inline void ajb_report_hybrid_sort_hybrid_sort(size_t n, double elapsed_ms, const char* phase) {
    fprintf(stderr, "[AJB_TIMER][hybrid_sort_hybrid_sort] %s: n=%zu elapsed=%.3fms throughput=%.2f M/s\n",
            phase, n, elapsed_ms, elapsed_ms > 0 ? n / elapsed_ms / 1000.0 : 0.0);
}
