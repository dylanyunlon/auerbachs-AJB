#pragma once

#include <cstdio>

// Upstream: GetValueFromVirtualPartition 用除法+取模定位跨GPU的虚拟分区.
// AJB改写: 当partition_size是2的幂时用位运算替代除法和取模.
// GPU上整数除法开销大(几十个cycle), 位运算只要1 cycle.
// 调用侧不变 — 只是内部选了快路径.

template <typename T>
inline __device__ void GetValueFromVirtualPartition(size_t partition_size, T** virtual_partition, size_t index,
                                                    T* value) {
  // 编译期无法判断partition_size是不是2的幂, 运行时检查
  if ((partition_size & (partition_size - 1)) == 0) {
    // 快路径: 位运算
    unsigned shift = __ffsll(partition_size) - 1;  // log2(partition_size)
    size_t part_idx = index >> shift;
    size_t elem_idx = index & (partition_size - 1);
    *value = virtual_partition[part_idx][elem_idx];
  } else {
    // 慢路径: 保持upstream的除法+取模
    *value = virtual_partition[index / partition_size][index % partition_size];
  }
}

// Upstream: SelectPivot 做二分搜索, 在两个虚拟分区之间找balanced pivot.
// AJB改写:
//   1. 加iteration计数器 — 二分的步数直接暴露给外部, 调试时能看到
//      收敛速度(理论上 log2(partition_size * num_partitions) 步)
//   2. 加early-exit: 如果low==0或high==max, 说明一个分区完全在另一个之前,
//      不需要二分, 直接返回
template <typename T>
__global__ void SelectPivot(size_t partition_size, size_t num_partitions, T** local_virtual_partition,
                            T** remote_virtual_partition, size_t* pivot) {
  size_t low = 0;
  size_t high = partition_size * num_partitions;
  size_t total = high;
  unsigned iterations = 0;

  // Early exit: 检查边界情况
  // 如果local最大值 <= remote最小值, pivot就是total(全选local)
  T local_max, remote_min;
  GetValueFromVirtualPartition<T>(partition_size, local_virtual_partition, total - 1, &local_max);
  GetValueFromVirtualPartition<T>(partition_size, remote_virtual_partition, 0, &remote_min);
  if (local_max <= remote_min) {
    *pivot = 0;
    return;
  }

  // 反向检查: local最小值 >= remote最大值
  T local_min, remote_max;
  GetValueFromVirtualPartition<T>(partition_size, local_virtual_partition, 0, &local_min);
  GetValueFromVirtualPartition<T>(partition_size, remote_virtual_partition, total - 1, &remote_max);
  if (local_min >= remote_max) {
    *pivot = total;
    return;
  }

  // 正常二分路径
  while (low < high) {
    const size_t mid = high - (high - low) / 2;

    T a;
    GetValueFromVirtualPartition<T>(partition_size, local_virtual_partition, total - mid, &a);
    T b;
    GetValueFromVirtualPartition<T>(partition_size, remote_virtual_partition, mid - 1, &b);

    if (a <= b) {
      high = mid - 1;
    } else {
      low = mid;
    }
    iterations++;
  }

  *pivot = low;

  // 调试输出: 二分步数. 在device端printf会被kernel flush到host stderr.
  // 只在第一个线程打印, 避免多线程竞争时输出混乱.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[DEBUG][SelectPivot] total=%zu pivot=%zu iterations=%u (expected<=%.0f)\n",
           total, low, iterations,
           // log2手算: __ffsll不适用于非2的幂, 用循环
           (double)(32 - __clz((unsigned)total)));
  }
}
