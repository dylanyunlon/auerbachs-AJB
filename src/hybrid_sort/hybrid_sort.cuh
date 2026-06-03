#pragma once

#include <algorithm>
#include <functional>
#include <vector>
#include <cstdio>

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

// Upstream: 单次cudaMemGetInfo查空闲显存, 取min(free*0.8/2, n/gpus).
// AJB改写:
//   1. 遍历所有GPU取最小空闲显存(而非只查当前GPU), 避免多卡不均衡时OOM
//   2. 根据element大小自适应utilization比例 — 大element留更多余量给kernel临时显存
//   3. 打印chunk决策过程: 实验中chunk_size是调优第一参数, 必须能看到怎么算出来的
size_t CalculateChunkSize(const std::vector<int>& gpus, size_t num_elements, size_t num_bytes_per_element) {
  // 自适应利用率: 每element >64字节时降低利用率, 给kernel留临时空间
  double utilization = (num_bytes_per_element > 64) ? 0.65 : 0.80;

  size_t min_free = SIZE_MAX;
  for (size_t g = 0; g < gpus.size(); ++g) {
    cudaSetDevice(gpus[g]);
    size_t free_mem, total_mem;
    CheckCudaError(cudaMemGetInfo(&free_mem, &total_mem));
    if (free_mem < min_free) min_free = free_mem;
  }

  const size_t max_chunk_size = static_cast<size_t>(
      min_free / num_bytes_per_element * utilization) / 2;
  const size_t preferred_chunk_size = DivideUp(num_elements, gpus.size());
  size_t result = std::min(max_chunk_size, preferred_chunk_size);

  fprintf(stderr, "[DEBUG][ChunkSize] gpus=%zu elem_size=%zuB min_free=%zuMB "
          "util=%.0f%% max_chunk=%zuM preferred=%zuM → chosen=%zuM\n",
          gpus.size(), num_bytes_per_element, min_free >> 20,
          utilization * 100, max_chunk_size / 1000000,
          preferred_chunk_size / 1000000, result / 1000000);

  return result;
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

  fprintf(stderr, "[DEBUG][HybridSort] n=%zu chunk=%zu groups=%zu kernel=%s streams=%zu\n",
          num_elements, chunk_size, num_chunk_groups,
          (kernel == HybridSortKernel::kMerge ? "merge" : "radix"), num_streams);

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

      // Upstream: num_chunk_groups > 1 时写到temporary, 否则写回keys.
      // 逻辑不变, 但打印每个group的进度, 让长时间实验能看到还在跑.
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

      fprintf(stderr, "[DEBUG][HybridSort] group %zu/%zu done, processed=%zu remaining=%zu\n",
              i + 1, num_chunk_groups, num_elements_to_process, num_remaining_elements);
    }

    synchonize_transfers();
  }

  {
    TimeScope time_scope("merge_phase");

    if (num_chunk_groups > 1) {
      fprintf(stderr, "[DEBUG][HybridSort] multi-group merge: %zu groups × %zu elem\n",
              num_chunk_groups, num_elements_per_chunk_group);
      ParallelMergePairs(temporary_keys, temporary_values, keys, values, num_elements, num_chunk_groups,
                         num_elements_per_chunk_group);
    }
  }

  // 断点: 排序完成后采样检查有序性 — 不是全量验证(太慢), 而是等间距抽样
  if (num_elements > 100) {
    size_t stride = num_elements / 100;
    bool sorted = true;
    size_t first_inversion = 0;
    for (size_t i = stride; i < num_elements; i += stride) {
      if (keys[i] < keys[i - stride]) {
        sorted = false;
        first_inversion = i;
        break;
      }
    }
    fprintf(stderr, "[DEBUG][HybridSort] sample-check: %s",
            sorted ? "OK (100 samples in order)\n" :
                     "INVERSION detected!\n");
    if (!sorted) {
      fprintf(stderr, "  first_inversion at [%zu]: key[%zu-stride]=%llu > key[%zu]=%llu\n",
              first_inversion, first_inversion - stride,
              (unsigned long long)keys[first_inversion - stride],
              first_inversion, (unsigned long long)keys[first_inversion]);
    }
  }
}
