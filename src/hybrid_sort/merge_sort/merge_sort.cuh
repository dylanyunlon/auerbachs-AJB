// [AJB] merge_sort.cuh: GPU merge sort — 小数据量时比radix sort更高效
// 多轮merge: 先warp内排序, 然后成倍merge直到全局有序
// 在hybrid_sort里, 数据量小于阈值时走这条路径
#include <cstdio>

// [AJB] merge sort诊断
static thread_local struct {
    long long sort_calls = 0;
    long long total_elements = 0;
    long long merge_rounds = 0;  // 总merge轮次
    int max_gpus = 0;
    void dump(const char* tag = "MergeSort") {
        fprintf(stderr, "[AJB_STATE][%s] calls=%lld elements=%lld rounds=%lld gpus=%d\n",
                tag, sort_calls, total_elements, merge_rounds, max_gpus);
    }
    void reset() { sort_calls = total_elements = merge_rounds = 0; max_gpus = 0; }
} ajb_msort_stats;

#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <vector>

#include <cub/cub.cuh>
#include <thrust/fill.h>
#include <thrust/merge.h>

#include "common/device_allocator.cuh"
#include "common/host_allocator.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/stream_pool.cuh"
#include "constants.cuh"
#include "hybrid_sort/resource_manager.cuh"
#include "kernels.cuh"

template <typename T, typename V>
size_t FindPivot(ResourceManager<T, V>& resource_manager, const std::vector<int>& gpus, size_t chunk_size) {
  const int gpu = gpus[0];
  HostAllocator& host_allocator = resource_manager.GetHostAllocator(gpu);
  DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
  StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

  CheckCudaError(cudaSetDevice(gpu));

  const size_t num_partitions = gpus.size() / 2;

  T** local_partitions = reinterpret_cast<T**>(host_allocator.allocate(num_partitions * sizeof(T*)));
  T** remote_partitions = reinterpret_cast<T**>(host_allocator.allocate(num_partitions * sizeof(T*)));
  size_t* host_pivot = reinterpret_cast<size_t*>(host_allocator.allocate(sizeof(size_t)));

  for (size_t i = 0; i < num_partitions; ++i) {
    local_partitions[i] = resource_manager.GetKeys(gpus[i]);
    remote_partitions[i] = resource_manager.GetKeys(gpus[i + num_partitions]);
  }

  T** local_virtual_partition = reinterpret_cast<T**>(device_allocator.allocate(num_partitions * sizeof(T*)));
  T** remote_virtual_partition = reinterpret_cast<T**>(device_allocator.allocate(num_partitions * sizeof(T*)));
  size_t* device_pivot = reinterpret_cast<size_t*>(device_allocator.allocate(sizeof(size_t)));

  CheckCudaError(cudaMemcpyAsync(local_virtual_partition, local_partitions, num_partitions * sizeof(T*),
                                 cudaMemcpyHostToDevice, stream_pool.GetStream(0)));
  CheckCudaError(cudaMemcpyAsync(remote_virtual_partition, remote_partitions, num_partitions * sizeof(T*),
                                 cudaMemcpyHostToDevice, stream_pool.GetStream(0)));

  SelectPivot<T><<<1, 1, 0, stream_pool.GetStream(0)>>>(chunk_size, num_partitions, local_virtual_partition,
      // [AJB_TRACE] merge sort kernel launch
                                                        remote_virtual_partition, device_pivot);

  CheckCudaError(
      cudaMemcpyAsync(host_pivot, device_pivot, sizeof(size_t), cudaMemcpyDeviceToHost, stream_pool.GetStream(0)));

  CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

  size_t pivot = *host_pivot;

  host_allocator.deallocate(reinterpret_cast<uint8_t*>(local_partitions));
  host_allocator.deallocate(reinterpret_cast<uint8_t*>(remote_partitions));
  host_allocator.deallocate(reinterpret_cast<uint8_t*>(host_pivot));

  device_allocator.deallocate(reinterpret_cast<uint8_t*>(local_virtual_partition));
  device_allocator.deallocate(reinterpret_cast<uint8_t*>(remote_virtual_partition));
  device_allocator.deallocate(reinterpret_cast<uint8_t*>(device_pivot));

  return pivot;
}

template <typename T, typename V>
std::array<int, 2> SwapPartitions(ResourceManager<T, V>& resource_manager, const std::vector<int>& gpus,
                                  size_t chunk_size, size_t pivot) {
  std::array<int, 2> gpus_to_merge;

  size_t partition_size = chunk_size;
  size_t gpus_to_swap = pivot / partition_size;

  if (pivot == partition_size * (gpus.size() / 2)) {
    --gpus_to_swap;
    pivot = partition_size;
  } else {
    pivot %= partition_size;
  }

  std::vector<int> gpus_to_sync;
  gpus_to_sync.reserve(gpus.size());

  for (size_t i = 0; i <= gpus_to_swap; ++i) {
    const int left_gpu = gpus[gpus.size() / 2 - i - 1];
    const int right_gpu = gpus[gpus.size() / 2 + i];

    const size_t num_elements = (i == gpus_to_swap) ? pivot : partition_size;
    const size_t offset = partition_size - num_elements;

    gpus_to_sync.push_back(left_gpu);
    gpus_to_sync.push_back(right_gpu);

    StreamPool& left_stream_pool = resource_manager.GetStreamPool(left_gpu);
    CheckCudaError(cudaMemcpyAsync(resource_manager.GetOtherKeys(left_gpu) + offset,
                                   resource_manager.GetKeys(right_gpu), sizeof(T) * num_elements,
                                   cudaMemcpyDeviceToDevice, left_stream_pool.GetStream(0)));
    CheckCudaError(cudaMemcpyAsync(resource_manager.GetOtherValues(left_gpu) + offset,
                                   resource_manager.GetValues(right_gpu), sizeof(V) * num_elements,
                                   cudaMemcpyDeviceToDevice, left_stream_pool.GetStream(0)));

    StreamPool& right_stream_pool = resource_manager.GetStreamPool(right_gpu);
    CheckCudaError(cudaMemcpyAsync(resource_manager.GetOtherKeys(right_gpu),
                                   resource_manager.GetKeys(left_gpu) + offset, sizeof(T) * num_elements,
                                   cudaMemcpyDeviceToDevice, right_stream_pool.GetStream(0)));
    CheckCudaError(cudaMemcpyAsync(resource_manager.GetOtherValues(right_gpu),
                                   resource_manager.GetValues(left_gpu) + offset, sizeof(V) * num_elements,
                                   cudaMemcpyDeviceToDevice, right_stream_pool.GetStream(0)));

    if (i == gpus_to_swap) {
      gpus_to_merge[0] = left_gpu;
      gpus_to_merge[1] = right_gpu;

      for (size_t j = 0; j < gpus_to_merge.size(); ++j) {
        const int gpu = gpus_to_merge[j];
        StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

        CheckCudaError(cudaMemcpyAsync(
            resource_manager.GetOtherKeys(gpu) + (j * pivot), resource_manager.GetKeys(gpu) + (j * pivot),
            sizeof(T) * (partition_size - pivot), cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
        CheckCudaError(cudaMemcpyAsync(
            resource_manager.GetOtherValues(gpu) + (j * pivot), resource_manager.GetValues(gpu) + (j * pivot),
            sizeof(V) * (partition_size - pivot), cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
      }
    }

    resource_manager.FlipBuffers(left_gpu);
    resource_manager.FlipBuffers(right_gpu);
  }

  for (const int gpu_to_sync : gpus_to_sync) {
    StreamPool& stream_pool = resource_manager.GetStreamPool(gpu_to_sync);

    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(1)));
    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
  }

  return gpus_to_merge;
  fprintf(stderr, "[AJB_BP][MergeSort] complete\n");
}

template <typename T, typename V>
void MergeLocalPartitions(ResourceManager<T, V>& resource_manager, const std::vector<int>& gpus,
                          const std::array<int, 2>& gpus_to_merge, size_t chunk_size, size_t pivot) {
  const size_t partition_size = chunk_size;
  pivot %= partition_size;

  for (size_t i = 0; i < gpus.size(); ++i) {
    const int gpu = gpus[i];
    DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
    StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

    if (gpu == gpus_to_merge[0] || gpu == gpus_to_merge[1]) {
      CheckCudaError(cudaSetDevice(gpu));

      const size_t offset = i >= gpus.size() / 2 ? pivot : partition_size - pivot;

      thrust::merge_by_key(thrust::cuda::par_nosync(device_allocator).on(stream_pool.GetStream(0)),
                           resource_manager.GetKeys(gpu), resource_manager.GetKeys(gpu) + offset,
                           resource_manager.GetKeys(gpu) + offset, resource_manager.GetKeys(gpu) + partition_size,
                           resource_manager.GetValues(gpu), resource_manager.GetValues(gpu) + offset,
                           resource_manager.GetOtherKeys(gpu), resource_manager.GetOtherValues(gpu));

      resource_manager.FlipBuffers(gpu);
    }
  }

  for (const int gpu : gpus) {
    CheckCudaError(cudaStreamSynchronize(resource_manager.GetStreamPool(gpu).GetStream(0)));
  }
}

template <typename T, typename V>
void MergePartitions(ResourceManager<T, V>& resource_manager, const std::vector<int>& gpus, size_t chunk_size) {
  if (gpus.size() > 2) {
#pragma omp parallel for num_threads(2)
    for (size_t i = 0; i < 2; ++i) {
      MergePartitions<T, V>(resource_manager,
                            {gpus.begin() + (i * (gpus.size() / 2)), gpus.begin() + ((i + 1) * (gpus.size() / 2))},
                            chunk_size);
    }
  }

  const size_t pivot = FindPivot<T>(resource_manager, gpus, chunk_size);
  if (pivot > 0) {
    const std::array<int, 2> gpus_to_merge = SwapPartitions<T, V>(resource_manager, gpus, chunk_size, pivot);
    MergeLocalPartitions<T, V>(resource_manager, gpus, gpus_to_merge, chunk_size, pivot);
  }

  if (gpus.size() > 2) {
#pragma omp parallel for num_threads(2)
    for (size_t i = 0; i < 2; ++i) {
      MergePartitions<T, V>(resource_manager,
                            {gpus.begin() + (i * (gpus.size() / 2)), gpus.begin() + ((i + 1) * (gpus.size() / 2))},
                            chunk_size);
    }
  }
}

template <typename T, typename V>
std::function<void()> MergeSort(T* in_keys, V* in_values, T* out_keys, V* out_values, const size_t num_elements,
                                ResourceManager<T, V>& resource_manager, std::vector<int> gpus) {
  ajb_msort_stats.sort_calls++;
  ajb_msort_stats.total_elements += num_elements;
  if((int)gpus.size() > ajb_msort_stats.max_gpus) ajb_msort_stats.max_gpus = gpus.size();
  fprintf(stderr, "[AJB_BP][MergeSort] start: n=%zu gpus=%zu\n", num_elements, gpus.size());
  size_t num_fillers = (num_elements % gpus.size() != 0) ? (gpus.size() - num_elements % gpus.size()) : 0;
  size_t chunk_size = (num_elements + num_fillers) / gpus.size();

  while (chunk_size < num_fillers) {
    gpus.resize(gpus.size() / 2);
    num_fillers = (num_elements % gpus.size() != 0) ? (gpus.size() - num_elements % gpus.size()) : 0;
    chunk_size = (num_elements + num_fillers) / gpus.size();
  }

  const size_t num_gpus = gpus.size();

#pragma omp parallel for num_threads(num_gpus)
  for (size_t i = 0; i < num_gpus; ++i) {
    const size_t offset = i * chunk_size;
    const size_t num_remaining_elements = num_elements > offset ? num_elements - offset : 0;
    const size_t num_elements_to_process = std::min(num_remaining_elements, chunk_size);

    const int gpu = gpus[i];
    DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
    StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

    CheckCudaError(cudaSetDevice(gpu));

    CheckCudaError(cudaMemcpyAsync(resource_manager.GetKeys(gpu), in_keys + offset, num_elements_to_process * sizeof(T),
                                   cudaMemcpyHostToDevice, stream_pool.GetStream(0)));
    CheckCudaError(cudaMemcpyAsync(resource_manager.GetValues(gpu), in_values + offset,
                                   num_elements_to_process * sizeof(V), cudaMemcpyHostToDevice,
                                   stream_pool.GetStream(0)));

    if (num_elements_to_process < chunk_size) {
      thrust::fill(thrust::cuda::par_nosync(device_allocator).on(stream_pool.GetStream(0)),
                   resource_manager.GetKeys(gpu) + num_elements_to_process, resource_manager.GetKeys(gpu) + chunk_size,
                   std::numeric_limits<T>::max());
    }

    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(2)));
    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

    size_t temporary_num_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temporary_num_bytes, resource_manager.GetKeysBuffer(gpu),
                                    resource_manager.GetValuesBuffer(gpu), chunk_size, 0, sizeof(T) * 8,
                                    stream_pool.GetStream(0));

    uint8_t* temporary_storage_pointer = device_allocator.allocate(temporary_num_bytes);
    cub::DeviceRadixSort::SortPairs((void*)temporary_storage_pointer, temporary_num_bytes,
                                    resource_manager.GetKeysBuffer(gpu), resource_manager.GetValuesBuffer(gpu),
                                    chunk_size, 0, sizeof(T) * 8, stream_pool.GetStream(0));

    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

    device_allocator.deallocate(reinterpret_cast<uint8_t*>(temporary_storage_pointer));
  }

  if (num_gpus > 1) {
    MergePartitions<T, V>(resource_manager, gpus, chunk_size);
  }

#pragma omp parallel for num_threads(num_gpus)
  for (size_t i = 0; i < num_gpus; ++i) {
    const size_t offset = i * chunk_size;
    const size_t num_remaining_elements = num_elements - offset;
    const size_t num_elements_to_process = std::min(num_remaining_elements, chunk_size);

    const int gpu = gpus[i];
    StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

    CheckCudaError(cudaSetDevice(gpu));

    CheckCudaError(cudaMemcpyAsync(out_keys + offset, resource_manager.GetKeys(gpu),
                                   num_elements_to_process * sizeof(T), cudaMemcpyDeviceToHost,
                                   stream_pool.GetStream(2)));
    CheckCudaError(cudaMemcpyAsync(out_values + offset, resource_manager.GetValues(gpu),
                                   num_elements_to_process * sizeof(V), cudaMemcpyDeviceToHost,
                                   stream_pool.GetStream(2)));

    resource_manager.FlipBuffers(gpu);
  }

  return [&resource_manager, gpus]() {
    for (const int gpu : gpus) {
      CheckCudaError(cudaStreamSynchronize(resource_manager.GetStreamPool(gpu).GetStream(2)));
    }
  };
}
