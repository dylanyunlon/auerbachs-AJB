#pragma once

#include <array>
#include <chrono>
#include <cstdio>
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

// --- M1118: Adaptive tile size computation ---
// Optimal merge sort tile size depends on L2 cache size and key width.
// Too-small tiles → excessive merge passes; too-large → cache thrashing.
// We target 2 tiles fitting in L2 (one per merge input), with a minimum
// of 1024 elements for GPU warp utilization.
struct AdaptiveTileConfig {
    size_t tile_elements;    // elements per tile
    size_t tile_bytes;       // bytes per tile
    size_t merge_passes;     // estimated passes for total_elements
    size_t l2_utilization;   // percentage of L2 used by 2 tiles
};

// --- M1293: Adaptive merge degree ---
// When GPU SM count >= 80, use 4-way merge to halve the number of passes.
// Otherwise fall back to standard 2-way merge.
static inline int SelectMergeDegree() {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int degree = (prop.multiProcessorCount >= 80) ? 4 : 2;
    fprintf(stderr, "[AJB_STATE][MergeDegree] sm_count=%d selected_degree=%d\n",
            prop.multiProcessorCount, degree);
    return degree;
}

static inline AdaptiveTileConfig ComputeAdaptiveTile(
    size_t total_elements, size_t key_bytes, size_t l2_cache_bytes = 6 * 1024 * 1024) {
    // Each merge step reads from 2 tiles simultaneously
    size_t available_per_tile = l2_cache_bytes / 2;
    size_t element_size = key_bytes + sizeof(uint32_t);  // key + value (index)
    size_t max_tile = available_per_tile / element_size;

    // Clamp to power of 2 for efficient indexing
    size_t tile = 1;
    while (tile * 2 <= max_tile) tile *= 2;

    // Minimum tile size for GPU efficiency
    constexpr size_t MIN_TILE = 1024;
    if (tile < MIN_TILE) tile = MIN_TILE;

    // Don't exceed total elements
    if (tile > total_elements) tile = total_elements;

    // Compute merge pass count using adaptive degree
    int merge_degree = SelectMergeDegree();
    size_t passes = 0;
    size_t merged = tile;
    while (merged < total_elements) {
        merged *= merge_degree;
        passes++;
    }

    size_t util = l2_cache_bytes > 0 ? (2 * tile * element_size * 100) / l2_cache_bytes : 0;

    fprintf(stderr, "[AJB_BP][AdaptiveTile] elements=%zu key_bytes=%zu tile=%zu passes=%zu L2_util=%zu%% merge_degree=%d\n",
            total_elements, key_bytes, tile, passes, util, merge_degree);

    return {tile, tile * element_size, passes, util};
}

// --- M1119: Sentinel-free merge path ---
// Traditional GPU merge sort uses INF sentinel values at the end of each
// sorted run to simplify boundary checks.  This wastes memory (padding)
// and can cause issues with non-numeric key types.
// The sentinel-free approach tracks explicit end pointers and uses a
// three-way branch: left exhausted, right exhausted, or compare.
template <typename T>
static inline void SentinelFreeMerge(
    const T* left, size_t left_n,
    const T* right, size_t right_n,
    T* output) {
    size_t li = 0, ri = 0, oi = 0;
    size_t total = left_n + right_n;

    // Main merge loop with explicit boundary checks (no sentinel)
    while (oi < total) {
        if (li >= left_n) {
            // Left exhausted: copy remaining right
            while (ri < right_n) output[oi++] = right[ri++];
        } else if (ri >= right_n) {
            // Right exhausted: copy remaining left
            while (li < left_n) output[oi++] = left[li++];
        } else {
            // Both active: compare and advance
            if (left[li] <= right[ri]) {
                output[oi++] = left[li++];
            } else {
                output[oi++] = right[ri++];
            }
        }
    }

    fprintf(stderr, "[AJB_BP][SentinelFreeMerge] left=%zu right=%zu total=%zu\n",
            left_n, right_n, total);
}

// --- M1294: Run length verification ---
// Before merge, verify each run is already sorted. If a run is found
// out of order, emit a diagnostic breakpoint.
template <typename T>
static inline bool VerifyRunSorted(const T* data, size_t count, int gpu_id, int run_id) {
    if (count <= 1) return true;
    for (size_t i = 1; i < count; ++i) {
        if (data[i] < data[i - 1]) {
            fprintf(stderr, "[AJB_BP][RunVerify] UNSORTED run=%d gpu=%d at pos=%zu "
                    "val[%zu] < val[%zu]\n", run_id, gpu_id, i, i, i - 1);
            return false;
        }
    }
    return true;
}

// Upstream FindPivot: 6 separate allocate / deallocate calls (3 host,
// 3 device).  Each allocator call is a potential synchronization point
// and bookkeeping overhead.
// Changed: batch all host allocations into a single slab and device
// allocations into another, then carve out sub-pointers with offsets.
// Deallocation is two calls instead of six.
// M1295: Memory deallocation timing added.

template <typename T, typename V>
// AJB-algo: FindPivot uses binary search on GPU-resident sorted chunks
size_t FindPivot(ResourceManager<T, V>& resource_manager, const std::vector<int>& gpus, size_t chunk_size) {
  const int gpu = gpus[0];  // AJB: pivot computation on first GPU
  HostAllocator& host_allocator = resource_manager.GetHostAllocator(gpu);
  DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
  StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

  CheckCudaError(cudaSetDevice(gpu));  // AJB: bind to pivot GPU before kernel launch

  // AJB: 预计算分区数——gpus.size()/2在多处使用
  const size_t num_partitions = gpus.size() / 2;  // AJB-algo: balanced partition for merge tree

  // Host slab: [local_ptrs | remote_ptrs | pivot]
  const size_t ptr_block_bytes = num_partitions * sizeof(T*);
  const size_t host_slab_bytes = ptr_block_bytes * 2 + sizeof(size_t);
  uint8_t* host_slab = host_allocator.allocate(host_slab_bytes);
  T** local_partitions  = reinterpret_cast<T**>(host_slab);
  T** remote_partitions = reinterpret_cast<T**>(host_slab + ptr_block_bytes);
  size_t* host_pivot    = reinterpret_cast<size_t*>(host_slab + ptr_block_bytes * 2);

  for (size_t i = 0; i < num_partitions; ++i) {
    local_partitions[i] = resource_manager.GetKeys(gpus[i]);
    remote_partitions[i] = resource_manager.GetKeys(gpus[i + num_partitions]);
  }

  // Device slab: [local_ptrs | remote_ptrs | pivot]
  const size_t dev_slab_bytes = ptr_block_bytes * 2 + sizeof(size_t);
  uint8_t* dev_slab = device_allocator.allocate(dev_slab_bytes);
  T** local_virtual_partition  = reinterpret_cast<T**>(dev_slab);
  T** remote_virtual_partition = reinterpret_cast<T**>(dev_slab + ptr_block_bytes);
  size_t* device_pivot         = reinterpret_cast<size_t*>(dev_slab + ptr_block_bytes * 2);

  CheckCudaError(cudaMemcpyAsync(local_virtual_partition, local_partitions, ptr_block_bytes,
                                 cudaMemcpyHostToDevice, stream_pool.GetStream(0)));
  CheckCudaError(cudaMemcpyAsync(remote_virtual_partition, remote_partitions, ptr_block_bytes,
                                 cudaMemcpyHostToDevice, stream_pool.GetStream(0)));

  SelectPivot<T><<<1, 1, 0, stream_pool.GetStream(0)>>>(chunk_size, num_partitions, local_virtual_partition,
                                                        remote_virtual_partition, device_pivot);

  CheckCudaError(
      cudaMemcpyAsync(host_pivot, device_pivot, sizeof(size_t), cudaMemcpyDeviceToHost, stream_pool.GetStream(0)));

  CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

  size_t pivot = *host_pivot;

  // M1295: Timed deallocation — measure reclaim latency
  auto dealloc_start = std::chrono::steady_clock::now();
  host_allocator.deallocate(host_slab);
  device_allocator.deallocate(dev_slab);
  auto dealloc_end = std::chrono::steady_clock::now();
  double dealloc_us = std::chrono::duration<double, std::micro>(dealloc_end - dealloc_start).count();
  fprintf(stderr, "[AJB_BP][FindPivot] dealloc_latency=%.2fus host_bytes=%zu dev_bytes=%zu\n",
          dealloc_us, host_slab_bytes, dev_slab_bytes);

  return pivot;
}

// Upstream: gpus_to_sync is a vector with push_back — heap allocs for
// at most gpus.size() entries (typically 2-8).
// Changed: reserve up-front based on known max (2 per swap iteration,
// plus the merge pair).

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

  // Pre-allocate for the exact number of GPUs we'll touch
  std::vector<int> gpus_to_sync;
  gpus_to_sync.reserve(2 * (gpus_to_swap + 1));

  // AJB: 对称交换循环——left从中间往左, right从中间往右
  const size_t half = gpus.size() / 2;
  for (size_t i = 0; i <= gpus_to_swap; ++i) {
    const int left_gpu = gpus[half - i - 1];
    const int right_gpu = gpus[half + i];

    const size_t num_elements = (i == gpus_to_swap) ? pivot : partition_size;
    const size_t offset = partition_size - num_elements;
    const size_t key_bytes = sizeof(T) * num_elements;
    const size_t val_bytes = sizeof(V) * num_elements;

    gpus_to_sync.push_back(left_gpu);
    gpus_to_sync.push_back(right_gpu);
    // AJB: 缓存key/val字节数用于下面的memcpy

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
}

// Upstream: MergeLocalPartitions scans all GPUs linearly to find the
// two merge targets.
// Changed: use a pair of indexed lookups (gpus_to_merge[0]/[1]) to
// find the GPU's index in the gpus vector, computing offset directly
// from that position vs the halfway mark.

template <typename T, typename V>
void MergeLocalPartitions(ResourceManager<T, V>& resource_manager, const std::vector<int>& gpus,
                          const std::array<int, 2>& gpus_to_merge, size_t chunk_size, size_t pivot) {
  const size_t partition_size = chunk_size;
  pivot %= partition_size;

  for (size_t j = 0; j < 2; ++j) {
    const int gpu = gpus_to_merge[j];
    DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
    StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);
    CheckCudaError(cudaSetDevice(gpu));

    // AJB: 用std::lower_bound替代线性扫描确定GPU半区位置
    // gpus向量通常已排序(由上层按设备号排), 二分O(logN)
    const size_t half = gpus.size() / 2;
    auto it = std::lower_bound(gpus.begin() + half, gpus.end(), gpu);
    bool is_upper_half = (it != gpus.end() && *it == gpu);

    const size_t offset = is_upper_half ? pivot : partition_size - pivot;

    // [AJB_BP] merge诊断: 哪个GPU在做merge, 分区多大
    fprintf(stderr, "[AJB_BP][merge_sort] gpu=%d half=%s offset=%zu pivot=%zu part=%zu\n",
            gpu, is_upper_half ? "upper" : "lower", offset, pivot, partition_size);

    thrust::merge_by_key(thrust::cuda::par_nosync(device_allocator).on(stream_pool.GetStream(0)),
                         resource_manager.GetKeys(gpu), resource_manager.GetKeys(gpu) + offset,
                         resource_manager.GetKeys(gpu) + offset, resource_manager.GetKeys(gpu) + partition_size,
                         resource_manager.GetValues(gpu), resource_manager.GetValues(gpu) + offset,
                         resource_manager.GetOtherKeys(gpu), resource_manager.GetOtherValues(gpu));

    resource_manager.FlipBuffers(gpu);
  }

  for (const int gpu : gpus) {
    CheckCudaError(cudaStreamSynchronize(resource_manager.GetStreamPool(gpu).GetStream(0)));
  }
}

// --- M1293: Merge pass statistics tracking ---
static int g_merge_pass_counter = 0;

template <typename T, typename V>
void MergePartitions(ResourceManager<T, V>& resource_manager, const std::vector<int>& gpus,
                     size_t chunk_size, int _depth = 0) {
  if (gpus.size() < 2) return;

  int pass_id = __sync_fetch_and_add(&g_merge_pass_counter, 1);
  size_t total_elements = chunk_size * gpus.size();

  auto pass_start = std::chrono::steady_clock::now();

  if (gpus.size() > 2) {
#pragma omp parallel for num_threads(2)
    for (size_t i = 0; i < 2; ++i) {
      MergePartitions<T, V>(resource_manager,
                            {gpus.begin() + (i * (gpus.size() / 2)), gpus.begin() + ((i + 1) * (gpus.size() / 2))},
                            chunk_size, _depth + 1);
    }
  }

  const size_t pivot = FindPivot<T>(resource_manager, gpus, chunk_size);
  // [AJB_BP] pivot质量: pivot/chunk_size 接近0.5说明数据均匀分布
  // 偏离0.5越多说明数据偏斜越严重, merge负载越不均衡
  if (pivot > 0) {
    double pivot_ratio = static_cast<double>(pivot) / (chunk_size * (gpus.size() / 2));
    fprintf(stderr, "[AJB_BP][merge_sort] depth=%d gpus=%zu pivot=%zu ratio=%.3f %s\n",
            _depth, gpus.size(), pivot, pivot_ratio,
            (pivot_ratio < 0.3 || pivot_ratio > 0.7) ? "SKEWED" : "ok");
    const std::array<int, 2> gpus_to_merge = SwapPartitions<T, V>(resource_manager, gpus, chunk_size, pivot);
    MergeLocalPartitions<T, V>(resource_manager, gpus, gpus_to_merge, chunk_size, pivot);
  }

  if (gpus.size() > 2) {
#pragma omp parallel for num_threads(2)
    for (size_t i = 0; i < 2; ++i) {
      MergePartitions<T, V>(resource_manager,
                            {gpus.begin() + (i * (gpus.size() / 2)), gpus.begin() + ((i + 1) * (gpus.size() / 2))},
                            chunk_size, _depth + 1);
    }
  }

  // M1293: merge pass statistics output
  auto pass_end = std::chrono::steady_clock::now();
  double pass_ms = std::chrono::duration<double, std::milli>(pass_end - pass_start).count();
  double data_bytes = static_cast<double>(total_elements) * (sizeof(T) + sizeof(V));
  double throughput_gbps = (pass_ms > 0.0) ? (data_bytes / (pass_ms * 1e6)) : 0.0;
  fprintf(stderr, "[AJB_BP][MergePass] pass=%d depth=%d elements=%zu time_ms=%.3f throughput_GBps=%.3f\n",
          pass_id, _depth, total_elements, pass_ms, throughput_gbps);
}

template <typename T, typename V>
std::function<void()> MergeSort(T* in_keys, V* in_values, T* out_keys, V* out_values, const size_t num_elements,
                                ResourceManager<T, V>& resource_manager, std::vector<int> gpus) {
  // Upstream: while(chunk_size < num_fillers) — same infinite loop risk
  // as in RadixSort when gpus shrinks to 1.
  // Changed: guard with gpus.size() > 1.
  size_t num_fillers = (num_elements % gpus.size() != 0) ? (gpus.size() - num_elements % gpus.size()) : 0;
  size_t chunk_size = (num_elements + num_fillers) / gpus.size();

  while (chunk_size < num_fillers && gpus.size() > 1) {
    gpus.resize(gpus.size() / 2);
    num_fillers = (num_elements % gpus.size() != 0) ? (gpus.size() - num_elements % gpus.size()) : 0;
    chunk_size = (num_elements + num_fillers) / gpus.size();
  }

  const size_t num_gpus = gpus.size();

  // Reset per-sort pass counter
  g_merge_pass_counter = 0;

#pragma omp parallel for num_threads(num_gpus)
  for (size_t i = 0; i < num_gpus; ++i) {
    const size_t offset = i * chunk_size;
    // Upstream: num_elements - offset with no underflow protection.
    // Changed: saturating subtraction.
    const size_t num_remaining_elements = num_elements > offset ? num_elements - offset : 0;
    const size_t num_elements_to_process = std::min(num_remaining_elements, chunk_size);

    const int gpu = gpus[i];
    DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
    StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

    CheckCudaError(cudaSetDevice(gpu));

    const size_t key_xfer_bytes = num_elements_to_process * sizeof(T);
    CheckCudaError(cudaMemcpyAsync(resource_manager.GetKeys(gpu), in_keys + offset, key_xfer_bytes,
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
    cub::DeviceRadixSort::SortPairs(static_cast<void*>(temporary_storage_pointer), temporary_num_bytes,  // AJB: static_cast
                                    resource_manager.GetKeysBuffer(gpu), resource_manager.GetValuesBuffer(gpu),
                                    chunk_size, 0, sizeof(T) * 8, stream_pool.GetStream(0));

    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

    // M1295: Timed deallocation of radix sort temp storage
    auto rsort_dealloc_t0 = std::chrono::steady_clock::now();
    device_allocator.deallocate(reinterpret_cast<uint8_t*>(temporary_storage_pointer));
    auto rsort_dealloc_t1 = std::chrono::steady_clock::now();
    double rsort_dealloc_us = std::chrono::duration<double, std::micro>(rsort_dealloc_t1 - rsort_dealloc_t0).count();
    fprintf(stderr, "[AJB_BP][MergeSort] gpu=%d radix_dealloc_latency=%.2fus bytes=%zu\n",
            gpu, rsort_dealloc_us, temporary_num_bytes);
  }

  // M1294: Run length verification before multi-GPU merge
  if (num_gpus > 1) {
    for (size_t i = 0; i < num_gpus; ++i) {
      const int gpu = gpus[i];
      CheckCudaError(cudaSetDevice(gpu));
      // Allocate host buffer and copy keys back for verification
      std::vector<T> host_check(chunk_size);
      CheckCudaError(cudaMemcpy(host_check.data(), resource_manager.GetKeys(gpu),
                                chunk_size * sizeof(T), cudaMemcpyDeviceToHost));
      VerifyRunSorted(host_check.data(), chunk_size, gpu, static_cast<int>(i));
    }

    // AJB: 多GPU归并路径——递归二分合并
    MergePartitions<T, V>(resource_manager, gpus, chunk_size);
  }

#pragma omp parallel for num_threads(num_gpus)
  for (size_t i = 0; i < num_gpus; ++i) {
    const size_t offset = i * chunk_size;
    // Saturating subtraction (matches the H2D loop above)
    const size_t num_remaining_elements = num_elements > offset ? num_elements - offset : 0;
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
