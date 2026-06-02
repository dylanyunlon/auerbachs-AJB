// [AJB] hybrid_sort/merge_sort/kernels.cuh: GPU merge sort kernel: 每个warp内的排序
#include <cstdio>
#pragma once

template <typename T>
inline __device__ void GetValueFromVirtualPartition(size_t partition_size, T** virtual_partition, size_t index,
                                                    T* value) {
  *value = virtual_partition[index / partition_size][index % partition_size];
}

template <typename T>
__global__ void SelectPivot(size_t partition_size, size_t num_partitions, T** local_virtual_partition,
                            T** remote_virtual_partition, size_t* pivot) {
  size_t low = 0;
  size_t high = partition_size * num_partitions;

  while (low < high) {
    const size_t mid = high - (high - low) / 2;

    T a;
    GetValueFromVirtualPartition<T>(partition_size, local_virtual_partition, partition_size * num_partitions - mid, &a);
    T b;
    GetValueFromVirtualPartition<T>(partition_size, remote_virtual_partition, mid - 1, &b);

    if (a <= b) {
      high = mid - 1;
    } else {
      low = mid;
    }
  }

  *pivot = low;
}

// [AJB] hybrid_sort_merge_sort_kernels 诊断报告
static inline void ajb_report_hybrid_sort_merge_sort_kernels(size_t n, double elapsed_ms, const char* phase) {
    fprintf(stderr, "[AJB_TIMER][hybrid_sort_merge_sort_kernels] %s: n=%zu elapsed=%.3fms throughput=%.2f M/s\n",
            phase, n, elapsed_ms, elapsed_ms > 0 ? n / elapsed_ms / 1000.0 : 0.0);
}
