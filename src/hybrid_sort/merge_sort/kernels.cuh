#pragma once
// [AJB] merge sort GPU kernel辅助: GetValueFromVirtualPartition处理跨buffer读取

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
