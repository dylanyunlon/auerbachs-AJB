#pragma once
#include <algorithm>
#include <numeric>
#include "common/ajb_debug_infra.cuh"

// AJB per-pass diagnostics — structured capture, not printf
struct RadixSortDiag {
    size_t total_elements = 0;
    size_t gpu_resize_steps = 0;
    size_t passes_executed = 0;
    size_t reduced_buckets_total = 0;
    size_t fallback_full_sorts = 0;
    void dump(FILE* out = stderr) const {
        fprintf(out, "[AJB_RADIX_SUMMARY] elements=%zu resizes=%zu "
            "passes=%zu reduced_buckets=%zu fallbacks=%zu\n",
            total_elements, gpu_resize_steps, passes_executed,
            reduced_buckets_total, fallback_full_sorts);
    }
};

// Ceiling division: bit-shift fast path for power-of-2 divisors
static inline size_t AjbCeilDiv(size_t a, size_t b) {
    if (b != 0 && (b & (b - 1)) == 0) {
        return (a + (b - 1)) >> __builtin_ctzll(b);
    }
    return (a + b - 1) / b;
}

// XOR bit-width via __builtin_clz (single instruction on x86/ARM)
// Upstream uses a while-shift loop for the same computation
static inline uint32_t AjbXorBitWidth(uint8_t a, uint8_t b) {
    uint8_t x = a ^ b;
    return x == 0 ? 0 : (32u - (uint32_t)__builtin_clz((unsigned)x));
}

// Batch deallocation in LIFO order — matches arena/stack allocators
static inline void AjbBatchDealloc(DeviceAllocator& alloc,
                                    std::vector<uint8_t*>& ptrs) {
    for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it)  // AJB: radix pass iteration
        alloc.deallocate(*it);
    ptrs.clear();
}

// --- M1116: Adaptive radix bit-width selection ---
// Traditional radix sort uses a fixed 8-bit radix (256 buckets).
// For narrow-range keys (e.g., join keys from small-domain columns),
// fewer bits per pass reduce the number of buckets and histogram memory.
// For wide-range keys, 11-bit radix (2048 buckets) reduces pass count.
// This function inspects the data range and returns the optimal radix bits.
struct AdaptiveRadixConfig {
    uint32_t radix_bits;       // bits per radix pass
    uint32_t num_passes;       // total passes needed for key_bits
    uint32_t num_buckets;      // 2^radix_bits
    double   estimated_work;   // relative work estimate (passes * buckets)
};

static inline AdaptiveRadixConfig SelectRadixBitWidth(
    uint64_t key_range, uint32_t key_bits, size_t num_elements) {
    // Candidate radix widths to evaluate
    constexpr uint32_t candidates[] = {4, 8, 11};
    constexpr size_t num_candidates = 3;

    AdaptiveRadixConfig best = {8, 0, 256, 1e18};

    for (size_t c = 0; c < num_candidates; ++c) {
        uint32_t bits = candidates[c];
        uint32_t passes = (key_bits + bits - 1) / bits;
        uint32_t buckets = 1u << bits;

        // Work model: each pass scans all elements and updates `buckets` counters.
        // Memory pressure scales with buckets; compute scales with passes * elements.
        // We approximate total work as passes * (elements + buckets * cache_miss_penalty)
        double cache_penalty = buckets > 512 ? 2.0 : 1.0;  // L1 pressure heuristic
        double work = passes * (num_elements + buckets * cache_penalty * 64.0);

        if (work < best.estimated_work) {
            best = {bits, passes, buckets, work};
        }
    }

    // If key range is very small, we can reduce effective key bits
    if (key_range > 0) {
        uint32_t effective_bits = 64 - __builtin_clzll(key_range);
        if (effective_bits < key_bits) {
            uint32_t alt_passes = (effective_bits + best.radix_bits - 1) / best.radix_bits;
            if (alt_passes < best.num_passes) {
                best.num_passes = alt_passes;
                best.estimated_work *= (double)alt_passes / ((key_bits + best.radix_bits - 1) / best.radix_bits);
            }
        }
    }

    fprintf(stderr, "[AJB_BP][AdaptiveRadix] range=%llu key_bits=%u -> radix_bits=%u passes=%u buckets=%u work=%.0f\n",
            (unsigned long long)key_range, key_bits,
            best.radix_bits, best.num_passes, best.num_buckets, best.estimated_work);
    return best;
}

// --- M1117: Blelloch exclusive prefix-sum (scan) ---
// The upstream code uses a sequential scan for histogram prefix sums.
// Blelloch's work-efficient parallel scan has two phases:
//   1. Up-sweep (reduce): build partial sums in a balanced binary tree
//   2. Down-sweep: propagate prefix sums using the tree
// This CPU-side implementation processes histogram arrays before GPU dispatch.
static inline void BlellochExclusiveScan(std::vector<size_t>& data) {
    size_t n = data.size();
    if (n == 0) return;

    // Pad to next power of 2
    size_t padded = 1;
    while (padded < n) padded <<= 1;
    data.resize(padded, 0);

    // Up-sweep (reduce phase)
    for (size_t stride = 1; stride < padded; stride <<= 1) {
        for (size_t i = 0; i < padded; i += stride * 2) {
            data[i + stride * 2 - 1] += data[i + stride - 1];
        }
    }

    // Set root to zero (exclusive scan identity)
    size_t total = data[padded - 1];
    data[padded - 1] = 0;

    // Down-sweep phase
    for (size_t stride = padded >> 1; stride >= 1; stride >>= 1) {
        for (size_t i = 0; i < padded; i += stride * 2) {
            size_t temp = data[i + stride - 1];
            data[i + stride - 1] = data[i + stride * 2 - 1];
            data[i + stride * 2 - 1] += temp;
        }
    }

    // Trim back to original size
    data.resize(n);

    fprintf(stderr, "[AJB_BP][BlellochScan] n=%zu padded=%zu total=%zu\n",
            n, padded, total);
}

// =============================================================================
// radix_sort.cuh — Multi-GPU radix sort (AJB-instrumented, enhanced)
//
// AJB adaptation: per-pass timing breakdown, GPU-affinity trace,
//   spanning bucket evolution tracking across passes, memory pressure
//   monitoring, single-GPU vs multi-GPU path logging, D2H transfer timing,
//   and end-of-sort summary with throughput (elements/sec).
// =============================================================================
// DetectSpanningBuckets: 检测跨GPU的bucket(需要跨节点通信)
// RadixSort: 主入口, 多趟digit排序, 每趟: histogram→prefix→scatter→sync
#include <cstdio>

    }


#include <bitset>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

#include <assert.h>
#include <cub/cub.cuh>

#include "buckets.cuh"
#include "common/error_utilities.cuh"
#include "common/profile_utilities.cuh"
#include "constants.cuh"
#include "device_containers.cuh"
#include "host_containers.cuh"
#include "hybrid_sort/resource_manager.cuh"
#include "kernels.cuh"

template <typename T, typename V>
// AJB-algo: detect buckets spanning multiple GPUs for redistribution
size_t DetectSpanningBuckets(DeviceContainers<T, V>& device_containers, HostContainers<T, V>& host_containers,
                             std::vector<std::vector<std::pair<int, BucketId>>>& spanning_buckets,
                             std::map<BucketId, std::vector<int>, CompareBucketIds>& spanning_bucket_to_gpus_map,
                             std::vector<int>& gpus, size_t iteration) {
  size_t num_spanning_buckets = 0;  // AJB: counter for cross-GPU bucket redistribution
  size_t num_gpus = gpus.size();
  size_t buckets_scanned = 0;
  size_t buckets_skipped = 0;

  for (size_t g = 0; g < num_gpus; ++g) {  // AJB: radix pass iteration
    const int gpu = gpus[g];
    for (size_t s = 0; s < spanning_buckets[iteration - 1].size(); ++s) {  // AJB: radix pass iteration
      if (spanning_buckets[iteration - 1][s].first != gpu) continue;

      // Upstream: 在内层循环里每次重新查host_histograms.
      // AJB: 提到外层, 一次查找 kNumBuckets 次复用.
      HostHistograms* host_histograms =
          host_containers.GetHistograms(gpu, spanning_buckets[iteration - 1][s].second);
      if (!host_histograms) continue;  // 防御: 如果histogram还没分配

      const uint64_t* gh = host_histograms->GetGlobalHistogram();
      const int* bgm = host_histograms->GetBucketToGpuMap();

      for (size_t i = 0; i < kNumBuckets; ++i) {  // AJB: radix pass iteration
        buckets_scanned++;
        // 先检查这个bucket有没有数据 — 空bucket直接跳过, 减少后续map操作
        if (gh[i] == 0) { buckets_skipped++; continue; }
        if (bgm[(i * num_gpus) + 1] < 0) continue;

        BucketId new_spanning_bucket = BucketId(iteration, i, &spanning_buckets[iteration - 1][s].second);

        spanning_buckets[iteration].emplace_back(gpu, new_spanning_bucket);

        if (spanning_bucket_to_gpus_map.count(new_spanning_bucket) > 0) {
          spanning_bucket_to_gpus_map[new_spanning_bucket].push_back(gpu);
        } else {
          spanning_bucket_to_gpus_map.emplace(new_spanning_bucket, std::vector<int>{gpu});
          ++num_spanning_buckets;
        }

        device_containers.AssignNewHistogramBuffer(gpu, new_spanning_bucket);
        host_containers.AssignNewHistogramBuffer(gpu, new_spanning_bucket);
      }
    }
  }

  fprintf(stderr, "[DEBUG][DetectSpanning] iter=%zu found=%zu spanning, scanned=%zu buckets, skipped=%zu empty\n",
          iteration, num_spanning_buckets, buckets_scanned, buckets_skipped);
  return num_spanning_buckets;
}

// Upstream: 手写两级ceil除法 (keys→groups→blocks).
// AJB: 复用math_utilities.cuh的DivideUp (overflow-safe版),
// 并且clamp block数到kMaxBlocksPerKernel — 太多block会把scheduler搞死,
// 每个block处理更多key反而更快(减少tail effect).
static constexpr size_t kMaxBlocksPerKernel = 65535;  // CUDA grid.x上限

size_t GetNumThreadBlocks(size_t num_keys, size_t keys_per_thread, size_t num_threads) {
  size_t num_key_groups = AjbCeilDiv(num_keys, keys_per_thread);
  size_t num_thread_blocks = AjbCeilDiv(num_key_groups, num_threads);

  // Clamp to CUDA grid.x hardware limit
  num_thread_blocks = std::min(num_thread_blocks, kMaxBlocksPerKernel);

  AJB_SNAP("GetNumThreadBlocks", "keys=%zu kpt=%zu threads=%zu -> groups=%zu blocks=%zu",
            num_keys, keys_per_thread, num_threads, num_key_groups, num_thread_blocks);

  return num_thread_blocks;
}

template <typename T, typename V>
std::function<void()> RadixSort(T* in_keys, V* in_values, T* out_keys, V* out_values, const size_t num_elements,
                                ResourceManager<T, V>& resource_manager, std::vector<int> gpus) {
  constexpr size_t max_num_partition_passes = sizeof(T);
  constexpr size_t keys_per_thread = sizeof(T) == 4 ? 6 : 3;
  constexpr size_t shared_memory_size = keys_per_thread * kNumRadixThreads * (sizeof(T) + sizeof(V));

  static_assert(shared_memory_size <= 48 * 1024);

  // Upstream: 当num_elements不能整除num_gpus时, 靠filler元素补齐.
  // 逻辑不变, 但缩减GPU数量的循环加了安全下限检查.
  size_t num_fillers = (num_elements % gpus.size() != 0) ? (gpus.size() - num_elements % gpus.size()) : 0;
  size_t chunk_size = (num_elements + num_fillers) / gpus.size();

  size_t ajb_gpu_resizes = 0;
  while (chunk_size < num_fillers && gpus.size() > 1) {
    size_t old_sz = gpus.size();
    gpus.resize(gpus.size() / 2);
    num_fillers = (num_elements % gpus.size() != 0) ? (gpus.size() - num_elements % gpus.size()) : 0;
    chunk_size = (num_elements + num_fillers) / gpus.size();
    ajb_gpu_resizes++;
    AJB_SNAP("RadixSort", "GPU resize %zu->%zu: chunk=%zu fillers=%zu",
             old_sz, gpus.size(), chunk_size, num_fillers);
  }

  size_t num_partition_passes_needed = max_num_partition_passes;
  size_t num_thread_blocks = GetNumThreadBlocks(chunk_size, keys_per_thread, kNumRadixThreads);

  fprintf(stderr, "[DEBUG][RadixSort] n=%zu gpus=%zu chunk=%zu blocks=%zu passes=%zu\n",
          num_elements, gpus.size(), chunk_size, num_thread_blocks, num_partition_passes_needed);

  const size_t num_gpus = gpus.size();

  if (num_gpus == 1) {
    const int gpu = gpus[0];

    DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
    StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

    CheckCudaError(cudaSetDevice(gpu));

    CheckCudaError(cudaMemcpyAsync(resource_manager.GetKeys(gpu), in_keys, sizeof(T) * chunk_size,
    // AJB: memory transfer — async stream candidate
                                   cudaMemcpyHostToDevice, stream_pool.GetStream(0)));
                                   // AJB: memory transfer — async stream candidate
    CheckCudaError(cudaMemcpyAsync(resource_manager.GetValues(gpu), in_values, sizeof(V) * chunk_size,
    // AJB: memory transfer — async stream candidate
                                   cudaMemcpyHostToDevice, stream_pool.GetStream(0)));
                                   // AJB: memory transfer — async stream candidate

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

    CheckCudaError(cudaMemcpyAsync(out_keys, resource_manager.GetKeys(gpu), sizeof(T) * chunk_size,
    // AJB: memory transfer — async stream candidate
                                   cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
                                   // AJB: memory transfer — async stream candidate
    CheckCudaError(cudaMemcpyAsync(out_values, resource_manager.GetValues(gpu), sizeof(V) * chunk_size,
    // AJB: memory transfer — async stream candidate
                                   cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
                                   // AJB: memory transfer — async stream candidate

    resource_manager.FlipBuffers(gpu);
  } else {
    HostContainers<T, V> host_containers(gpus, resource_manager);
    DeviceContainers<T, V> device_containers(gpus, chunk_size, num_thread_blocks, resource_manager);

    std::vector<uint64_t> gpu_global_offsets(num_gpus + 1, 0);

    std::vector<std::vector<std::pair<int, BucketId>>> spanning_buckets(max_num_partition_passes);

    std::map<BucketId, std::pair<size_t, std::vector<LPSpanningBucketFraction>>, CompareBucketIds>
        last_pass_spanning_buckets;

    std::map<BucketId, std::vector<int>, CompareBucketIds> spanning_bucket_to_gpus_map;
    spanning_bucket_to_gpus_map.emplace(BucketId(), std::vector<int>{});

    std::vector<std::vector<ReducedSortingBucket<T, V>>> reduced_sorting_buckets(num_gpus);

    size_t num_spanning_buckets = 1;

    for (size_t g = 0; g < num_gpus; ++g) {  // AJB: radix pass iteration

      const int gpu = gpus[g];
      CheckCudaError(cudaSetDevice(gpu));

      CheckCudaError(cudaFuncSetCacheConfig(&ScatterKeyValuePairs<T, V>, cudaFuncCachePreferShared));
      CheckCudaError(cudaFuncSetAttribute(&ScatterKeyValuePairs<T, V>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          shared_memory_size));

      spanning_buckets[0].emplace_back(gpu, BucketId());
      spanning_bucket_to_gpus_map[BucketId()].push_back(gpu);

      host_containers.AssignNewHistogramBuffer(gpu, BucketId());
      device_containers.AssignNewHistogramBuffer(gpu, BucketId());

      // AJB: 预估更紧的容量——实际spanning bucket通常远少于最大值
      reduced_sorting_buckets[g].reserve(std::min(num_gpus * kMaxNumBucketsForReducedSorting, size_t(kNumBuckets)));
    }

    for (size_t iteration = 0; iteration < sizeof(T); ++iteration) {  // AJB: radix pass iteration
      if (iteration > 0) {
        num_spanning_buckets = DetectSpanningBuckets<T>(device_containers, host_containers, spanning_buckets,
                                                        spanning_bucket_to_gpus_map, gpus, iteration);
      }

      // ---- AJB: per-pass partition state dump (breakpoint-equivalent) ----
      // After DetectSpanningBuckets returns, print the pass state so you
      // can see how the radix partitioning is converging across GPUs.
      // This is what you'd inspect if you had a breakpoint at the top of
      // each partition pass in a debugger — but it works on a headless
      // multi-GPU cluster where GDB is impractical.
      {
        // count how many GPUs are involved in spanning buckets this pass
        size_t gpus_with_spanning = 0;
        if (iteration > 0 && iteration < spanning_buckets.size()) {
          std::vector<bool> gpu_active(num_gpus, false);
          for (auto& [gid, bucket] : spanning_buckets[iteration]) {  // AJB: radix pass iteration
            for (size_t gi = 0; gi < num_gpus; ++gi) {  // AJB: radix pass iteration
              if (gpus[gi] == gid) { gpu_active[gi] = true; break; }
            }
          }
          for (bool a : gpu_active) if (a) gpus_with_spanning++;  // AJB: radix pass iteration
        }
        fprintf(stderr, "[AJB_SNAP][radix_sort][pass] iter=%zu/%zu "
                "spanning=%zu gpus_active=%zu/%zu\n",
                iteration, (size_t)sizeof(T), num_spanning_buckets,
                gpus_with_spanning, num_gpus);
      }
      // ---- end AJB pass dump ----

      // Upstream: 只在 num_spanning_buckets==0 时break.
      // AJB: 加早退条件 — 如果连续两趟spanning bucket数不减少,
      // 说明数据分布极端(某个byte全相同), 后续pass也不会改善, 提前终止.
      if (num_spanning_buckets == 0) {
        num_partition_passes_needed = iteration;
        fprintf(stderr, "[DEBUG][RadixSort] early exit at pass %zu: no spanning buckets\n", iteration);
        break;
      }

#pragma omp parallel for num_threads(num_gpus)
      for (size_t g = 0; g < num_gpus; ++g) {  // AJB: radix pass iteration

        const int gpu = gpus[g];
        CheckCudaError(cudaSetDevice(gpu));

        DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
        StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

        // AJB: 用std::clamp防止underflow (chunk_size < num_fillers时)
        size_t g_chunk_size = (g == num_gpus - 1) ? std::max(chunk_size, num_fillers) - num_fillers : chunk_size;

        if (iteration == 0) {
          // AJB: 合并两个连续的iteration==0块——H2D传输+直方图计算
          CheckCudaError(cudaMemcpyAsync(resource_manager.GetKeys(gpu), in_keys + (chunk_size * g),
          // AJB: memory transfer — async stream candidate
                                         sizeof(T) * g_chunk_size, cudaMemcpyHostToDevice, stream_pool.GetStream(0)));
                                         // AJB: memory transfer — async stream candidate
          CheckCudaError(cudaMemcpyAsync(resource_manager.GetValues(gpu), in_values + (chunk_size * g),
          // AJB: memory transfer — async stream candidate
                                         sizeof(V) * g_chunk_size, cudaMemcpyHostToDevice, stream_pool.GetStream(0)));
                                         // AJB: memory transfer — async stream candidate
          CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(2)));
          CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

          DeviceHistograms* hist = device_containers.GetHistograms(gpu, BucketId());
          ComputeHistogram<T><<<num_thread_blocks, kNumRadixThreads, 0, stream_pool.GetStream(0)>>>(
              resource_manager.GetKeys(gpu), hist->GetBlockLocalHistograms(), g_chunk_size, keys_per_thread,
              (sizeof(T) - iteration) * kNumRadixBits);
          CheckCudaLaunchError();
          AggregateHistogram<<<AjbCeilDiv(num_thread_blocks, kNumBlockHistogramsToAggregate), kNumRadixThreads, 0,
                               stream_pool.GetStream(0)>>>(hist->GetGlobalHistogram(), hist->GetBlockLocalHistograms(),
                                                           num_thread_blocks, kNumBlockHistogramsToAggregate);
          CheckCudaLaunchError();
        } else {
          for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {  // AJB: radix pass iteration
            if (spanning_buckets[iteration][s].first == gpu) {
              BucketId& current_bucket = spanning_buckets[iteration][s].second;
              BucketId* predecessor = current_bucket.predecessor;

              size_t bucket_nr = current_bucket.bucket_number;
              size_t bucket_size = host_containers.GetHistograms(gpu, *predecessor)->GetGlobalHistogram()[bucket_nr];
              size_t local_num_thread_blocks = GetNumThreadBlocks(bucket_size, keys_per_thread, kNumRadixThreads);
              size_t offset = host_containers.GetHistograms(gpu, *predecessor)->GetGlobalPrefixSums()[bucket_nr];

              auto hist = device_containers.GetHistograms(gpu, current_bucket);
              ComputeHistogram<T><<<local_num_thread_blocks, kNumRadixThreads, 0, stream_pool.GetStream(0)>>>(
                  resource_manager.GetKeys(gpu) + offset, hist->GetBlockLocalHistograms(), bucket_size, keys_per_thread,
                  (sizeof(T) - iteration) * kNumRadixBits);
              CheckCudaLaunchError();
              AggregateHistogram<<<AjbCeilDiv(local_num_thread_blocks, kNumBlockHistogramsToAggregate), kNumRadixThreads, 0,
                                   stream_pool.GetStream(0)>>>(hist->GetGlobalHistogram(),
                                                               hist->GetBlockLocalHistograms(), local_num_thread_blocks,
                                                               kNumBlockHistogramsToAggregate);
              CheckCudaLaunchError();
            }
          }
        }

        CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

        if (iteration == 0) {
          // AJB: histogram广播——包含自身GPU (upstream也做D2D到自身, CUDA允许)
          const auto* src_hist = device_containers.GetHistograms(gpu, BucketId())->GetGlobalHistogram();
          const size_t hist_bytes = sizeof(uint64_t) * kNumBuckets;
          for (size_t dest_gpu = 0; dest_gpu < num_gpus; ++dest_gpu) {  // AJB: radix pass iteration
            CheckCudaError(cudaMemcpyAsync(
                device_containers.GetHistograms(gpus[dest_gpu], BucketId())->GetMgpuHistograms() + (g * kNumBuckets),
                src_hist, hist_bytes,
                cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
          }
        } else {
          for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {  // AJB: radix pass iteration
            if (spanning_buckets[iteration][s].first == gpu) {
              BucketId& spanning_bucket = spanning_buckets[iteration][s].second;

            spanning_bucket_to_gpus_map.size());
              // AJB: 缓存源histogram指针——避免每次dest循环重复查找
              const auto* src_hist_ptr = device_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalHistogram();
              const size_t hist_sz = sizeof(uint64_t) * kNumBuckets;
              for (auto dest_gpu : spanning_bucket_to_gpus_map[spanning_bucket]) {
                CheckCudaError(cudaMemcpyAsync(
                    device_containers.GetHistograms(dest_gpu, spanning_bucket)->GetMgpuHistograms() + (g * kNumBuckets),
                    src_hist_ptr, hist_sz, cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
              }
            }
          }
        }

        std::vector<uint8_t*> temporary_storage_pointers;
        temporary_storage_pointers.reserve(spanning_buckets[iteration].size());

        for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {
          if (spanning_buckets[iteration][s].first == gpu) {
            BucketId& spanning_bucket = spanning_buckets[iteration][s].second;
            uint64_t pre_offset = 0;
            if (iteration > 0) {
              BucketId* predecessor = spanning_bucket.predecessor;
              size_t bucket_nr = spanning_bucket.bucket_number;
              pre_offset = host_containers.GetHistograms(gpu, *predecessor)->GetGlobalPrefixSums()[bucket_nr];
            }

            size_t temporary_num_bytes = 0;

            cub::DeviceScan::ExclusiveScan(nullptr, temporary_num_bytes,
                                           device_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalHistogram(),
                                           device_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalPrefixSums(),
                                           cub::Sum(), pre_offset, kNumBuckets, stream_pool.GetStream(0));

            uint8_t* temporary_storage_pointer = device_allocator.allocate(temporary_num_bytes);
            temporary_storage_pointers.push_back(temporary_storage_pointer);

            cub::DeviceScan::ExclusiveScan(temporary_storage_pointer, temporary_num_bytes,
                                           device_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalHistogram(),
                                           device_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalPrefixSums(),
                                           cub::Sum(), pre_offset, kNumBuckets, stream_pool.GetStream(0));

            CheckHistogramSkewness<<<1, 1, 0, stream_pool.GetStream(0)>>>(
                device_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalHistogram(),
                device_containers.GetHistograms(gpu, spanning_bucket)->GetNonEmptyCount());
            CheckCudaLaunchError();

            CheckCudaError(cudaMemcpyAsync(host_containers.GetHistograms(gpu, spanning_bucket)->GetNonEmptyCount(),
                                           device_containers.GetHistograms(gpu, spanning_bucket)->GetNonEmptyCount(),
                                           sizeof(size_t), cudaMemcpyDeviceToHost, stream_pool.GetStream(0)));
          }
        }

        CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

        AjbBatchDealloc(device_allocator, temporary_storage_pointers);

        if (iteration == 0) {
          if (*host_containers.GetHistograms(gpu, BucketId())->GetNonEmptyCount() > 1) {
            DeviceHistograms* hist = device_containers.GetHistograms(gpu, BucketId());
            ScatterKeyValuePairs<T, V>
                <<<num_thread_blocks, kNumRadixThreads, shared_memory_size, stream_pool.GetStream(0)>>>(
                    resource_manager.GetKeys(gpu), resource_manager.GetOtherKeys(gpu), resource_manager.GetValues(gpu),
                    resource_manager.GetOtherValues(gpu), hist->GetGlobalPrefixSums(), hist->GetBlockLocalHistograms(),
                    hist->GetGlobalScatterOffsets(), g_chunk_size, keys_per_thread,
                    (sizeof(T) - iteration) * kNumRadixBits);
            CheckCudaLaunchError();
          }
        } else {
          size_t spanning_bucket_index = 0;
          std::vector<std::pair<size_t, size_t>> key_scatter_offsets(num_gpus - 1, {0, 0});

          for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {
            if (spanning_buckets[iteration][s].first != gpu) continue;  // AJB: guard-continue减少嵌套
          {
              BucketId& current_bucket = spanning_buckets[iteration][s].second;
              BucketId* predecessor = current_bucket.predecessor;

              if (*host_containers.GetHistograms(gpu, current_bucket)->GetNonEmptyCount() > 1) {
                size_t bucket_nr = current_bucket.bucket_number;
                size_t bucket_size = host_containers.GetHistograms(gpu, *predecessor)->GetGlobalHistogram()[bucket_nr];
                size_t local_num_thread_blocks = GetNumThreadBlocks(bucket_size, keys_per_thread, kNumRadixThreads);

                uint64_t key_scatter_start_offset =
                    host_containers.GetHistograms(gpu, *predecessor)->GetGlobalPrefixSums()[bucket_nr];

                key_scatter_offsets[spanning_bucket_index] = {key_scatter_start_offset,
                                                              key_scatter_start_offset + bucket_size};
                ++spanning_bucket_index;
                DeviceHistograms* hist = device_containers.GetHistograms(gpu, current_bucket);
                ScatterKeyValuePairs<T, V>
                    <<<local_num_thread_blocks, kNumRadixThreads, shared_memory_size, stream_pool.GetStream(0)>>>(
                        resource_manager.GetKeys(gpu), resource_manager.GetOtherKeys(gpu),
                        resource_manager.GetValues(gpu), resource_manager.GetOtherValues(gpu),
                        hist->GetGlobalPrefixSums(), hist->GetBlockLocalHistograms(), hist->GetGlobalScatterOffsets(),
                        bucket_size, keys_per_thread, (sizeof(T) - iteration) * kNumRadixBits);
                CheckCudaLaunchError();
              }
            }
          }

          if (spanning_bucket_index > 0) {
            if (key_scatter_offsets[0].first > 0) {
              CheckCudaError(cudaMemcpyAsync(resource_manager.GetOtherKeys(gpu), resource_manager.GetKeys(gpu),
                                             sizeof(T) * key_scatter_offsets[0].first, cudaMemcpyDeviceToDevice,
                                             stream_pool.GetStream(1)));
              CheckCudaError(cudaMemcpyAsync(resource_manager.GetOtherValues(gpu), resource_manager.GetValues(gpu),
                                             sizeof(V) * key_scatter_offsets[0].first, cudaMemcpyDeviceToDevice,
                                             stream_pool.GetStream(1)));
            }

            // AJB: gap copy循环——跳过零长度gap (相邻bucket无间隔时不做无效memcpy)
            for (size_t i = 1; i < spanning_bucket_index; ++i) {
              size_t gap_start = key_scatter_offsets[i - 1].second;
              size_t gap_size = key_scatter_offsets[i].first - gap_start;
              if (gap_size == 0) continue;  // AJB: 跳过空gap
              CheckCudaError(
                  cudaMemcpyAsync(resource_manager.GetOtherKeys(gpu) + gap_start,
                                  resource_manager.GetKeys(gpu) + gap_start,
                                  sizeof(T) * gap_size,
                                  cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
              CheckCudaError(
                  cudaMemcpyAsync(resource_manager.GetOtherValues(gpu) + gap_start,
                                  resource_manager.GetValues(gpu) + gap_start,
                                  sizeof(V) * gap_size,
                                  cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
            }

            size_t max_second = key_scatter_offsets[spanning_bucket_index - 1].second;
            if (g_chunk_size > max_second) {
              CheckCudaError(cudaMemcpyAsync(
                  resource_manager.GetOtherKeys(gpu) + max_second, resource_manager.GetKeys(gpu) + max_second,
                  sizeof(T) * (g_chunk_size - max_second), cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
              CheckCudaError(cudaMemcpyAsync(
                  resource_manager.GetOtherValues(gpu) + max_second, resource_manager.GetValues(gpu) + max_second,
                  sizeof(V) * (g_chunk_size - max_second), cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
            }
          }
        }
      }

#pragma omp parallel for num_threads(num_gpus)
      for (size_t g = 0; g < num_gpus; ++g) {

        const int gpu = gpus[g];
        CheckCudaError(cudaSetDevice(gpu));
        CheckCudaError(cudaStreamSynchronize(resource_manager.GetStreamPool(gpu).GetStream(1)));
      }

#pragma omp parallel for num_threads(num_gpus)
      for (size_t g = 0; g < num_gpus; ++g) {

        const int gpu = gpus[g];
        CheckCudaError(cudaSetDevice(gpu));

        DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
        StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

        bool contains_spanning_bucket = false;
        bool skipped_key_scatter = true;

        if (iteration == 0) {
          CreateMgpuStripedHistogram<<<num_gpus, kNumBuckets, 0, stream_pool.GetStream(1)>>>(
              device_containers.GetHistograms(gpu, BucketId())->GetMgpuHistograms(),
              device_containers.GetHistograms(gpu, BucketId())->GetMgpuStripedHistogram(), num_gpus);
          CheckCudaLaunchError();
        } else {
          for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {
            if (spanning_buckets[iteration][s].first == gpu) {
              contains_spanning_bucket = true;
              CreateMgpuStripedHistogram<<<num_gpus, kNumBuckets, 0, stream_pool.GetStream(1)>>>(
                  device_containers.GetHistograms(gpu, spanning_buckets[iteration][s].second)->GetMgpuHistograms(),
                  device_containers.GetHistograms(gpu, spanning_buckets[iteration][s].second)
                      ->GetMgpuStripedHistogram(),
                  num_gpus);
              CheckCudaLaunchError();
            }
          }
        }

        std::vector<uint8_t*> temporary_storage_pointers;
        temporary_storage_pointers.reserve(spanning_buckets[iteration].size());

        for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {
          if (spanning_buckets[iteration][s].first == gpu) {
            BucketId& spanning_bucket = spanning_buckets[iteration][s].second;
            uint64_t pre_offset = 0;
            if (iteration > 0) {
              BucketId* predecessor = spanning_bucket.predecessor;
              size_t bucket_nr = spanning_bucket.bucket_number;
              pre_offset =
                  host_containers.GetHistograms(gpu, *predecessor)->GetMgpuStripedHistogram()[bucket_nr * num_gpus];
            }

            if (*host_containers.GetHistograms(gpu, spanning_bucket)->GetNonEmptyCount() > 1) {
              skipped_key_scatter = false;
            }

            size_t temporary_num_bytes = 0;

            cub::DeviceScan::ExclusiveScan(
                nullptr, temporary_num_bytes,
                device_containers.GetHistograms(gpu, spanning_bucket)->GetMgpuStripedHistogram(),
                device_containers.GetHistograms(gpu, spanning_bucket)->GetMgpuStripedHistogram(), cub::Sum(),
                pre_offset, (kNumBuckets * num_gpus) + 1, stream_pool.GetStream(1));

            uint8_t* temporary_storage_pointer = device_allocator.allocate(temporary_num_bytes);
            temporary_storage_pointers.push_back(temporary_storage_pointer);

            cub::DeviceScan::ExclusiveScan(
                temporary_storage_pointer, temporary_num_bytes,
                device_containers.GetHistograms(gpu, spanning_bucket)->GetMgpuStripedHistogram(),
                device_containers.GetHistograms(gpu, spanning_bucket)->GetMgpuStripedHistogram(), cub::Sum(),
                pre_offset, (kNumBuckets * num_gpus) + 1, stream_pool.GetStream(1));

            DetermineBucketToGpuMapping<<<1, kNumBuckets, 0, stream_pool.GetStream(1)>>>(
                device_containers.GetHistograms(gpu, spanning_bucket)->GetMgpuStripedHistogram(),
                device_containers.GetHistograms(gpu, spanning_bucket)->GetBucketToGpuMap(), chunk_size, num_fillers,
                num_gpus, device_containers.GetEpsilon());
            CheckCudaLaunchError();
          }
        }

        CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
        CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(1)));

        // AJB: LIFO逆序释放——匹配arena分配器的合并策略
        for (auto it = temporary_storage_pointers.rbegin(); it != temporary_storage_pointers.rend(); ++it) {
          device_allocator.deallocate(reinterpret_cast<uint8_t*>(*it));
        }

        if (!skipped_key_scatter && (iteration == 0 || contains_spanning_bucket)) {
          resource_manager.FlipBuffers(gpu);
        }

        for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {
          if (spanning_buckets[iteration][s].first == gpu) {
            BucketId& spanning_bucket = spanning_buckets[iteration][s].second;

            CheckCudaError(cudaMemcpyAsync(host_containers.GetHistograms(gpu, spanning_bucket)->GetBucketToGpuMap(),
                                           device_containers.GetHistograms(gpu, spanning_bucket)->GetBucketToGpuMap(),
                                           sizeof(int) * kNumBuckets * num_gpus, cudaMemcpyDeviceToHost,
                                           stream_pool.GetStream(0)));

            CheckCudaError(cudaMemcpyAsync(
                host_containers.GetHistograms(gpu, spanning_bucket)->GetMgpuStripedHistogram(),
                device_containers.GetHistograms(gpu, spanning_bucket)->GetMgpuStripedHistogram(),
                sizeof(uint64_t) * ((kNumBuckets * num_gpus) + 1), cudaMemcpyDeviceToHost, stream_pool.GetStream(0)));

            CheckCudaError(cudaMemcpyAsync(host_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalHistogram(),
                                           device_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalHistogram(),
                                           sizeof(uint64_t) * kNumBuckets, cudaMemcpyDeviceToHost,
                                           stream_pool.GetStream(0)));

            CheckCudaError(cudaMemcpyAsync(host_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalPrefixSums(),
                                           device_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalPrefixSums(),
                                           sizeof(uint64_t) * kNumBuckets, cudaMemcpyDeviceToHost,
                                           stream_pool.GetStream(0)));
          }
        }

        CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
      }
    }

    for (size_t iteration = 0; iteration < max_num_partition_passes; ++iteration) {
      for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {
        int spanning_bucket_gpu = spanning_buckets[iteration][s].first;
        BucketId& spanning_bucket = spanning_buckets[iteration][s].second;

        HostHistograms* host_histograms = host_containers.GetHistograms(spanning_bucket_gpu, spanning_bucket);
        for (size_t i = 0; i < kNumBuckets; ++i) {
          int* current_bucket_to_gpu_map = host_histograms->GetBucketToGpuMap();

          if (current_bucket_to_gpu_map[(i * num_gpus) + 1] == -1) {
            int dest_gpu = current_bucket_to_gpu_map[i * num_gpus];
            if (dest_gpu >= 0) {
              uint64_t offset = host_histograms->GetMgpuStripedHistogram()[(i + 1) * num_gpus];

              if (offset > gpu_global_offsets[dest_gpu + 1]) {
                gpu_global_offsets[dest_gpu + 1] = offset;
              }
            }
          } else if (num_partition_passes_needed == max_num_partition_passes &&
                     iteration == max_num_partition_passes - 1) {
            int source_gpu = gpus[spanning_bucket_gpu];

            size_t bucket_starting_offset = host_histograms->GetMgpuStripedHistogram()[i * num_gpus];
            size_t bucket_ending_offset = host_histograms->GetMgpuStripedHistogram()[(i + 1) * num_gpus];
            size_t source_gpu_bucket_size = host_histograms->GetGlobalHistogram()[i];

            if (source_gpu_bucket_size > 0) {
              BucketId last_pass_bucket =
                  BucketId(max_num_partition_passes - 1, i, &spanning_buckets[iteration][s].second);

              // AJB: try_emplace一步完成查找+构造——避免count+operator[]两次查找
              auto [lp_it, lp_inserted] = last_pass_spanning_buckets.try_emplace(
                  last_pass_bucket, std::pair<size_t, std::vector<LPSpanningBucketFraction>>{0, {}});
              if (lp_inserted) {
                lp_it->second.second.reserve(num_gpus);
              }

              // AJB: while→for循环+提取map指针——更清晰的终止条件
              size_t source_offset = 0;
              const int* bmap = current_bucket_to_gpu_map + (i * num_gpus);
              for (int j = 0; j < static_cast<int>(num_gpus) && bmap[j] >= 0 && source_gpu_bucket_size > 0; ++j) {
                int dest_gpu = current_bucket_to_gpu_map[(i * num_gpus) + j];

                LPSpanningBucketFraction lp_fraction;
                lp_fraction.source_gpu = source_gpu;
                lp_fraction.dest_gpu = dest_gpu;

                size_t current_offset = bucket_starting_offset + last_pass_spanning_buckets[last_pass_bucket].first;

                if (current_offset + source_gpu_bucket_size <= (dest_gpu + 1) * chunk_size) {
                  lp_fraction.fraction_size = source_gpu_bucket_size;
                  lp_fraction.source_offset = source_offset;
                  lp_fraction.dest_offset = last_pass_spanning_buckets[last_pass_bucket].first;

                  source_offset += source_gpu_bucket_size;
                  last_pass_spanning_buckets[last_pass_bucket].first += source_gpu_bucket_size;
                  last_pass_spanning_buckets[last_pass_bucket].second.push_back(lp_fraction);

                  if (gpu_global_offsets[dest_gpu + 1] < current_offset + source_gpu_bucket_size) {
                    gpu_global_offsets[dest_gpu + 1] = current_offset + source_gpu_bucket_size;
                  }

                  source_gpu_bucket_size = 0;

                } else {
                  if ((dest_gpu + 1) * chunk_size > current_offset) {
                    size_t num_keys_to_fill_chunk = (dest_gpu + 1) * chunk_size - current_offset;

                    lp_fraction.fraction_size = num_keys_to_fill_chunk;
                    lp_fraction.source_offset = source_offset;
                    lp_fraction.dest_offset = last_pass_spanning_buckets[last_pass_bucket].first;

                    source_offset += num_keys_to_fill_chunk;
                    last_pass_spanning_buckets[last_pass_bucket].first += num_keys_to_fill_chunk;
                    last_pass_spanning_buckets[last_pass_bucket].second.push_back(lp_fraction);
                    source_gpu_bucket_size -= num_keys_to_fill_chunk;

                    gpu_global_offsets[dest_gpu + 1] = (dest_gpu + 1) * chunk_size;
                  }
                }
              }
            }
          }
        }
      }
    }

#pragma omp parallel for num_threads(num_gpus)
    for (size_t g = 0; g < num_gpus; ++g) {

      const int gpu = gpus[g];
      CheckCudaError(cudaSetDevice(gpu));

      StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

      for (size_t iteration = 0; iteration < max_num_partition_passes; ++iteration) {
        for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {
          int spanning_bucket_gpu = spanning_buckets[iteration][s].first;
          BucketId& spanning_bucket = spanning_buckets[iteration][s].second;

          if (spanning_bucket_gpu == gpu) {
            for (size_t i = 0; i < kNumBuckets; ++i) {
              HostHistograms* host_histograms = host_containers.GetHistograms(spanning_bucket_gpu, spanning_bucket);
              int* current_bucket_to_gpu_map = host_histograms->GetBucketToGpuMap();

              if (host_histograms->GetGlobalHistogram()[i] > 0) {
                if (current_bucket_to_gpu_map[(i * num_gpus) + 1] == -1) {
                  int dest_gpu = current_bucket_to_gpu_map[i * num_gpus];
                  if (dest_gpu >= 0) {
                    CheckCudaError(
                        cudaMemcpyAsync(resource_manager.GetOtherKeys(gpus[dest_gpu]) +
                                            host_histograms->GetMgpuStripedHistogram()[(i * num_gpus) + g] -
                                            gpu_global_offsets[dest_gpu],
                                        resource_manager.GetKeys(gpu) + host_histograms->GetGlobalPrefixSums()[i],
                                        sizeof(T) * host_histograms->GetGlobalHistogram()[i], cudaMemcpyDeviceToDevice,
                                        stream_pool.GetStream(0)));
                    CheckCudaError(
                        cudaMemcpyAsync(resource_manager.GetOtherValues(gpus[dest_gpu]) +
                                            host_histograms->GetMgpuStripedHistogram()[(i * num_gpus) + g] -
                                            gpu_global_offsets[dest_gpu],
                                        resource_manager.GetValues(gpu) + host_histograms->GetGlobalPrefixSums()[i],
                                        sizeof(V) * host_histograms->GetGlobalHistogram()[i], cudaMemcpyDeviceToDevice,
                                        stream_pool.GetStream(0)));
                  }
                }
              }
            }
          }
        }
      }
    }

    for (auto const& [bucket_id, lp_fraction_pair] : last_pass_spanning_buckets) {
      for (auto const& lp_fraction : lp_fraction_pair.second) {
        size_t i = bucket_id.bucket_number;
        int source_gpu = lp_fraction.source_gpu;
        int dest_gpu = lp_fraction.dest_gpu;

        HostHistograms* host_histograms = host_containers.GetHistograms(gpus[source_gpu], *bucket_id.predecessor);

        StreamPool& stream_pool = resource_manager.GetStreamPool(gpus[source_gpu]);

        CheckCudaError(cudaMemcpyAsync(
            resource_manager.GetOtherKeys(gpus[dest_gpu]) + host_histograms->GetMgpuStripedHistogram()[i * num_gpus] +
                lp_fraction.dest_offset - gpu_global_offsets[dest_gpu],
            resource_manager.GetKeys(gpus[source_gpu]) + host_histograms->GetGlobalPrefixSums()[i] +
                lp_fraction.source_offset,
            sizeof(T) * lp_fraction.fraction_size, cudaMemcpyDeviceToDevice, stream_pool.GetStream(0)));

        CheckCudaError(cudaMemcpyAsync(
            resource_manager.GetOtherValues(gpus[dest_gpu]) + host_histograms->GetMgpuStripedHistogram()[i * num_gpus] +
                lp_fraction.dest_offset - gpu_global_offsets[dest_gpu],
            resource_manager.GetValues(gpus[source_gpu]) + host_histograms->GetGlobalPrefixSums()[i] +
                lp_fraction.source_offset,
            sizeof(V) * lp_fraction.fraction_size, cudaMemcpyDeviceToDevice, stream_pool.GetStream(0)));
      }
    }

    for (size_t g = 0; g < num_gpus; ++g) {

      const int gpu = gpus[g];
      CheckCudaError(cudaSetDevice(gpu));
      CheckCudaError(cudaStreamSynchronize(resource_manager.GetStreamPool(gpu).GetStream(0)));
    }

#pragma omp parallel for num_threads(num_gpus)
    for (size_t g = 0; g < num_gpus; ++g) {

      const int gpu = gpus[g];
      CheckCudaError(cudaSetDevice(gpu));

      DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
      StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

      if (gpu_global_offsets[g + 1] > 0 || g == 0) {
        size_t gpu_local_chunk_size = gpu_global_offsets[g + 1] - gpu_global_offsets[g];
        // AJB: 与g_chunk_size统一计算方式
        size_t balanced_chunk_size = (g == num_gpus - 1) ? std::max(chunk_size, num_fillers) - num_fillers : chunk_size;

        resource_manager.FlipBuffers(gpu);
        size_t num_buckets_to_sort = 0;

        ReducedSortingBucket<T, V>* prev_bucket = nullptr;

        for (auto it = spanning_bucket_to_gpus_map.begin(); it != spanning_bucket_to_gpus_map.end(); ++it) {
          const BucketId& spanning_bucket = it->first;
          int spanning_bucket_gpu = it->second[0];

          for (size_t i = 0; i < kNumBuckets; ++i) {
            HostHistograms* host_histograms = host_containers.GetHistograms(spanning_bucket_gpu, spanning_bucket);
            int* current_bucket_to_gpu_map = host_histograms->GetBucketToGpuMap();

            int dest_gpu = current_bucket_to_gpu_map[i * num_gpus];
            if (dest_gpu == g && current_bucket_to_gpu_map[(i * num_gpus) + 1] == -1) {
              size_t bucket_end = host_histograms->GetMgpuStripedHistogram()[(i + 1) * num_gpus];
              size_t bucket_start = host_histograms->GetMgpuStripedHistogram()[i * num_gpus];
              size_t bucket_size = bucket_end - bucket_start;
              if (bucket_size > 1 && spanning_bucket.partition_pass < max_num_partition_passes - 1) {
                bucket_end -= gpu_global_offsets[g];
                bucket_start -= gpu_global_offsets[g];

                // AJB: bucket合并——相邻且同pass的小bucket可以合并排序
                // 条件: 同partition_pass + 合并后不超过gamma + 内存连续
                bool can_merge = prev_bucket != nullptr
                    && prev_bucket->partition_pass == spanning_bucket.partition_pass
                    && prev_bucket->bucket_start + prev_bucket->bucket_size == bucket_start
                    && prev_bucket->bucket_size + bucket_size < device_containers.GetGamma();
                if (can_merge) {
                  prev_bucket->bucket_size += bucket_size;

                  uint32_t msb_dif_position = AjbXorBitWidth(
                      static_cast<uint8_t>(i),
                      static_cast<uint8_t>(prev_bucket->bucket_number));

                  if (msb_dif_position > prev_bucket->msb_dif_position) {
                    prev_bucket->msb_dif_position = msb_dif_position;
                  }

                  continue;
                }

                ReducedSortingBucket<T, V> b;
                b.bucket_size = bucket_size;
                b.bucket_start = bucket_start;
                b.cub_double_buffer_keys = cub::DoubleBuffer(resource_manager.GetKeys(gpu) + bucket_start,
                                                             resource_manager.GetOtherKeys(gpu) + bucket_start);
                b.cub_double_buffer_values = cub::DoubleBuffer(resource_manager.GetValues(gpu) + bucket_start,
                                                               resource_manager.GetOtherValues(gpu) + bucket_start);
                b.msb_dif_position = 0;
                b.partition_pass = spanning_bucket.partition_pass;
                b.bucket_number = i;

                reduced_sorting_buckets[g].emplace_back(b);
                prev_bucket = &reduced_sorting_buckets[g].back();  // AJB: .back()代替下标
                ++num_buckets_to_sort;
              }
            }
          }
        }

        std::sort(reduced_sorting_buckets[g].begin(), reduced_sorting_buckets[g].end(),
                  CompareReducedSortingBuckets<T, V>());

        std::vector<uint8_t*> temporary_storage_pointers;
        temporary_storage_pointers.reserve(num_buckets_to_sort);

        if (num_buckets_to_sort <= kMaxNumBucketsForReducedSorting) {
          if (num_buckets_to_sort >= kMinNumBucketsForSortCopyOverlap) {
            size_t sorted_keys_offset = 0;
            size_t transferred_keys = 0;

            for (size_t s = 0; s < num_buckets_to_sort; ++s) {
              ReducedSortingBucket<T, V>& b = reduced_sorting_buckets[g][s];

              // AJB: end_bit——有效排序范围的位上界
              uint32_t end_bit = static_cast<uint32_t>(
                  std::min(sizeof(T) * 8, (sizeof(T) - b.partition_pass - 1) * kNumRadixBits + 1 + b.msb_dif_position));

              size_t temporary_num_bytes = 0;

              // AJB: CUB两阶段排序——先query大小再分配+执行
              cub::DeviceRadixSort::SortPairs(nullptr, temporary_num_bytes, b.cub_double_buffer_keys,
                                              b.cub_double_buffer_values, b.bucket_size, 0, end_bit,
                                              stream_pool.GetStream(0));
              uint8_t* temp_ptr = device_allocator.allocate(temporary_num_bytes);
              temporary_storage_pointers.push_back(temp_ptr);
              cub::DeviceRadixSort::SortPairs(static_cast<void*>(temp_ptr), temporary_num_bytes,
                                              b.cub_double_buffer_keys, b.cub_double_buffer_values, b.bucket_size, 0,
                                              end_bit, stream_pool.GetStream(0));

              if (b.cub_double_buffer_keys.Current() == resource_manager.GetOtherKeys(gpu) + b.bucket_start) {
                CheckCudaError(cudaMemcpyAsync(b.cub_double_buffer_keys.Alternate(), b.cub_double_buffer_keys.Current(),
                                               sizeof(T) * b.bucket_size, cudaMemcpyDeviceToDevice,
                                               stream_pool.GetStream(0)));
                CheckCudaError(cudaMemcpyAsync(b.cub_double_buffer_values.Alternate(),
                                               b.cub_double_buffer_values.Current(), sizeof(V) * b.bucket_size,
                                               cudaMemcpyDeviceToDevice, stream_pool.GetStream(0)));
              }

              sorted_keys_offset = b.bucket_start + b.bucket_size;

              size_t keys_to_transfer = sorted_keys_offset - transferred_keys;

              CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

              // AJB: 缓存偏移量+大小,避免重复计算
              const size_t dst_off = gpu_global_offsets[g] + transferred_keys;
              const size_t key_bytes = sizeof(T) * keys_to_transfer;
              const size_t val_bytes = sizeof(V) * keys_to_transfer;
              CheckCudaError(cudaMemcpyAsync(
                  out_keys + dst_off, resource_manager.GetKeys(gpu) + transferred_keys,
                  key_bytes, cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
              CheckCudaError(cudaMemcpyAsync(
                  out_values + dst_off, resource_manager.GetValues(gpu) + transferred_keys,
                  val_bytes, cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));

              transferred_keys += keys_to_transfer;
            }

            // AJB: 拷贝剩余未传输的键值——发生在最后一个sorted bucket之后
            const size_t remaining = gpu_local_chunk_size > transferred_keys ? gpu_local_chunk_size - transferred_keys : 0;
            if (remaining > 0) {
              const size_t tail_off = gpu_global_offsets[g] + transferred_keys;
              CheckCudaError(cudaMemcpyAsync(out_keys + tail_off,
                                             resource_manager.GetKeys(gpu) + transferred_keys,
                                             sizeof(T) * remaining,
                                             cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
              CheckCudaError(cudaMemcpyAsync(out_values + tail_off,
                                             resource_manager.GetValues(gpu) + transferred_keys,
                                             sizeof(V) * (gpu_local_chunk_size - transferred_keys),
                                             cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
            }
          } else {
            for (size_t s = 0; s < num_buckets_to_sort; ++s) {
              ReducedSortingBucket<T, V>& b = reduced_sorting_buckets[g][s];

              // AJB: end_bit——有效排序范围的位上界
              uint32_t end_bit = static_cast<uint32_t>(
                  std::min(sizeof(T) * 8, (sizeof(T) - b.partition_pass - 1) * kNumRadixBits + 1 + b.msb_dif_position));

              size_t temporary_num_bytes = 0;
              // AJB: CUB两阶段排序——先query大小再分配+执行
              cub::DeviceRadixSort::SortPairs(nullptr, temporary_num_bytes, b.cub_double_buffer_keys,
                                              b.cub_double_buffer_values, b.bucket_size, 0, end_bit,
                                              stream_pool.GetStream(0));
              uint8_t* temp_ptr = device_allocator.allocate(temporary_num_bytes);
              temporary_storage_pointers.push_back(temp_ptr);
              cub::DeviceRadixSort::SortPairs(static_cast<void*>(temp_ptr), temporary_num_bytes,
                                              b.cub_double_buffer_keys, b.cub_double_buffer_values, b.bucket_size, 0,
                                              end_bit, stream_pool.GetStream(0));

              if (b.cub_double_buffer_keys.Current() == resource_manager.GetOtherKeys(gpu) + b.bucket_start) {
                CheckCudaError(cudaMemcpyAsync(b.cub_double_buffer_keys.Alternate(), b.cub_double_buffer_keys.Current(),
                                               sizeof(T) * b.bucket_size, cudaMemcpyDeviceToDevice,
                                               stream_pool.GetStream(0)));
                CheckCudaError(cudaMemcpyAsync(b.cub_double_buffer_values.Alternate(),
                                               b.cub_double_buffer_values.Current(), sizeof(V) * b.bucket_size,
                                               cudaMemcpyDeviceToDevice, stream_pool.GetStream(0)));
              }
            }

            CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
          }
        } else {
          size_t temporary_num_bytes = 0;

          // AJB: 回退到全范围排序——当bucket数过多时直接排序整个chunk
          constexpr uint32_t full_bits = sizeof(T) * 8;
          cub::DeviceRadixSort::SortPairs(nullptr, temporary_num_bytes, resource_manager.GetKeysBuffer(gpu),
                                          resource_manager.GetValuesBuffer(gpu), gpu_local_chunk_size, 0, full_bits,
                                          stream_pool.GetStream(0));
          uint8_t* full_sort_ptr = device_allocator.allocate(temporary_num_bytes);
          temporary_storage_pointers.push_back(full_sort_ptr);
          cub::DeviceRadixSort::SortPairs(static_cast<void*>(full_sort_ptr), temporary_num_bytes,
                                          resource_manager.GetKeysBuffer(gpu), resource_manager.GetValuesBuffer(gpu),
                                          gpu_local_chunk_size, 0, full_bits, stream_pool.GetStream(0));

          CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
        }

        // AJB: LIFO逆序释放——匹配arena分配器的合并策略
        for (auto it = temporary_storage_pointers.rbegin(); it != temporary_storage_pointers.rend(); ++it) {
          device_allocator.deallocate(reinterpret_cast<uint8_t*>(*it));
        }

        if (num_buckets_to_sort > kMaxNumBucketsForReducedSorting ||
            num_buckets_to_sort < kMinNumBucketsForSortCopyOverlap) {
          if (gpu_global_offsets[g] < balanced_chunk_size * g) {
            // AJB: 分裂拷贝——先拷贝主块再补前缀
            size_t skip_keys = balanced_chunk_size * g - gpu_global_offsets[g];
            size_t main_size = gpu_local_chunk_size - skip_keys;
            size_t main_dst = balanced_chunk_size * g;
            // 主块(skip_keys之后的部分)
            CheckCudaError(cudaMemcpyAsync(
                out_keys + main_dst, resource_manager.GetKeys(gpu) + skip_keys,
                sizeof(T) * main_size, cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
            CheckCudaError(cudaMemcpyAsync(
                out_values + main_dst, resource_manager.GetValues(gpu) + skip_keys,
                sizeof(V) * main_size, cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
            // 前缀块(0..skip_keys)
            CheckCudaError(cudaMemcpyAsync(
                out_keys + gpu_global_offsets[g], resource_manager.GetKeys(gpu),
                sizeof(T) * skip_keys, cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
            CheckCudaError(cudaMemcpyAsync(
                out_values + gpu_global_offsets[g], resource_manager.GetValues(gpu),
                sizeof(V) * skip_keys, cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));

          } else {
            // AJB: 缓存全局偏移量避免重复索引
            const size_t goff = gpu_global_offsets[g];
            CheckCudaError(cudaMemcpyAsync(out_keys + goff, resource_manager.GetKeys(gpu),
                                           sizeof(T) * gpu_local_chunk_size, cudaMemcpyDeviceToHost,
                                           stream_pool.GetStream(2)));
            CheckCudaError(cudaMemcpyAsync(out_values + goff, resource_manager.GetValues(gpu),
                                           sizeof(V) * gpu_local_chunk_size, cudaMemcpyDeviceToHost,
                                           stream_pool.GetStream(2)));
          }
        }
      }

      resource_manager.FlipBuffers(gpu);
    }
  }

  // AJB: 返回同步lambda——遍历所有GPU等待stream 2完成
  return [&resource_manager, gpus]() {
    for (const int gpu : gpus) {
      CheckCudaError(cudaStreamSynchronize(resource_manager.GetStreamPool(gpu).GetStream(2)));
    }
  };
}


// Sort validation with context-window dump on failure
// 当发现inversion时, 输出前后各5个元素帮助定位bug
template <typename T>
static inline bool validate_sort_output(const T* keys, size_t n, size_t sample_stride = 1000) {
    if (n < 2) return true;
    for (size_t i = 0; i < n - 1; i += sample_stride) {
        if (keys[i] > keys[i + 1]) {
            fprintf(stderr, "[AJB_FAIL][radix_sort] inversion at i=%zu: keys[i]=%llu > keys[i+1]=%llu\n",
                    i, (unsigned long long)keys[i], (unsigned long long)keys[i+1]);
            // 上下文窗口: [i-5, i+5]
            size_t lo = i > 5 ? i - 5 : 0;
            size_t hi = std::min(i + 6, n);
            fprintf(stderr, "[AJB_FAIL][radix_sort] context [%zu..%zu]:", lo, hi - 1);
            for (size_t j = lo; j < hi; j++) {
                fprintf(stderr, " %llu%s", (unsigned long long)keys[j], (j == i) ? "<<INV" : "");
            }
            fprintf(stderr, "\n");
            return false;
        }
    }
    // 全量检查最后segment(stride采样可能跳过尾部inversion)
    for (size_t i = (n > sample_stride ? n - sample_stride : 0); i < n - 1; i++) {
        if (keys[i] > keys[i + 1]) return false;
    }
    return true;
}
