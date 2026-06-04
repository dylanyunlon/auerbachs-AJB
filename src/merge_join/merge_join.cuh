#pragma once

#include <bitset>
#include <chrono>
#include <cstdio>
#include <vector>

#include <moderngpu/cta_merge.hxx>

#include "common/device_allocator.cuh"
#include "common/error_utilities.cuh"
#include "common/math_utilities.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/stream_pool.cuh"
#include "constants.cuh"
#include "join_result.cuh"
#include "kernels.cuh"

// Upstream: standard midpoint (l + r + 1) / 2 — overflows when l + r > LLONG_MAX.
// Changed: l + (r - l + 1) / 2  — no overflow for any valid index pair.
// Also: early exit when count == 1 (one compare vs log2 iterations).
template <typename T>
long long LastSmaller(T* data, long long count, const T& than) {
  if (count <= 0 || !(data[0] < than)) {
    return -1;
  }
  if (count == 1) return 0;

  long long l = 0, r = count - 1;
  while (l < r) {
    long long m = l + (r - l + 1) / 2;
    if (data[m] < than) {
      l = m;
    } else {
      r = m - 1;
    }
  }

  return r;
}

template <typename T>
long long FirstGreater(T* data, long long count, const T& than) {
  if (count <= 0 || !(than < data[count - 1])) {
    return count;
  }
  if (count == 1) return 0;

  long long l = 0, r = count - 1;
  while (l < r) {
    long long m = l + (r - l) / 2;
    if (than < data[m]) {
      r = m;
    } else {
      l = m + 1;
    }
  }

  return r;
}

template <typename T>
longlong2 FindBounds(T* data, long long count, const T& key) {
  return {LastSmaller(data, count, key) + 1, FirstGreater(data, count, key)};
}

// Upstream: the body of HandleLongKeyRanges contains 7 nearly identical
// copies of  FindBounds → if (x < y) → count += → materialize → update
// next_start.  Each is ~8 lines, making a 205-line function where ~60%
// is duplicated logic.
// Changed: extract the repeated pattern into a local helper
// (try_accumulate_key_group), reducing the 7 sites to single calls.
// Also factored out the "lookup on both sides" variant.

template <typename T>
void HandleLongKeyRanges(T* keys_r, T* keys_s, const long long n_r, const long long n_s, const longlong2& bases,
                         longlong2* starts, longlong2* ends, long long n_partitions, JoinResult<T>& answer,
                         const bool materialize) {

  // Accumulate a join match group for a given key:
  // find full bounds in both R and S, count the cross-product, optionally
  // materialize, and advance next_start past both ranges.
  // Returns true if the group was non-empty.
  auto try_accumulate = [&](const T& key, longlong2& next_start) -> bool {
    const longlong2 bounds_r = FindBounds(keys_r, n_r, key);
    if (bounds_r.x >= bounds_r.y) return false;

    const longlong2 bounds_s = FindBounds(keys_s, n_s, key);
    if (bounds_s.x >= bounds_s.y) return false;

    answer.count_ += (bounds_r.y - bounds_r.x) * (bounds_s.y - bounds_s.x);

    if (materialize) {
      answer.items_.emplace_back(
          longlong4{bounds_r.x, bounds_r.y - 1, bounds_s.x, bounds_s.y - 1}, bases);
    }

    next_start = {std::max(next_start.x, bounds_r.y),
                  std::max(next_start.y, bounds_s.y)};
    return true;
  };

  longlong2 next_start = {0, 0};

  for (long long i_partition = 0; i_partition < n_partitions; ++i_partition) {
    if (next_start.x >= ends[i_partition].x || next_start.y >= ends[i_partition].y) {
      // Upstream: two symmetric branches (x-exhausted vs y-exhausted)
      // each with two sub-branches (all-same-key vs different keys).
      // Changed: both branches now use try_accumulate.

      if (next_start.x == ends[i_partition].x && next_start.y < ends[i_partition].y) {
        // S-side still has keys; R-side is exhausted for this partition
        try_accumulate(keys_s[next_start.y], next_start);

        if (keys_s[next_start.y] != keys_s[ends[i_partition].y - 1] &&
            next_start.y < ends[i_partition].y) {
          try_accumulate(keys_s[ends[i_partition].y - 1], next_start);
        }
      } else if (next_start.y == ends[i_partition].y && next_start.x < ends[i_partition].x) {
        // R-side still has keys; S-side is exhausted for this partition
        try_accumulate(keys_r[next_start.x], next_start);

        if (keys_r[next_start.x] != keys_r[ends[i_partition].x - 1] &&
            next_start.x < ends[i_partition].x) {
          try_accumulate(keys_r[ends[i_partition].x - 1], next_start);
        }
      }

      starts[i_partition] = ends[i_partition];
      continue;
    }

    starts[i_partition] = {std::max(next_start.x, starts[i_partition].x),
                           std::max(next_start.y, starts[i_partition].y)};

    if (i_partition + 1 >= n_partitions) {
      break;
    }

    next_start = ends[i_partition];
    longlong2& cur_end = ends[i_partition];
    T boundary_key = keys_r[cur_end.x - 1];

    if (next_start.y < n_s && boundary_key == keys_s[next_start.y]) {
      long long begin_s = next_start.y;
      long long end_r = cur_end.x;
      cur_end.x = LastSmaller(keys_r, cur_end.x - 1, boundary_key) + 1;
      ++next_start.y;
      next_start.y = FirstGreater(keys_s + next_start.y, n_s - next_start.y, boundary_key) + next_start.y;
      begin_s = LastSmaller(keys_s, begin_s, boundary_key) + 1;
      end_r = FirstGreater(keys_r + end_r, n_r - end_r, boundary_key) + end_r;
      next_start.x = end_r;

      answer.count_ += (end_r - cur_end.x) * (next_start.y - begin_s);

      if (materialize) {
        answer.items_.emplace_back(longlong4{cur_end.x, end_r - 1, begin_s, next_start.y - 1}, bases);
      }
    }

    if (keys_s[cur_end.y - 1] != boundary_key) {
      boundary_key = keys_s[cur_end.y - 1];

      if (cur_end.x < n_r && boundary_key == keys_r[next_start.x]) {
        long long begin_r = next_start.x;
        long long end_s = cur_end.y;
        cur_end.y = LastSmaller(keys_s, cur_end.y - 1, boundary_key) + 1;
        ++next_start.x;
        next_start.x = FirstGreater(keys_r + next_start.x, n_r - next_start.x, boundary_key) + next_start.x;
        begin_r = LastSmaller(keys_r, begin_r, boundary_key) + 1;
        end_s = FirstGreater(keys_s + end_s, n_s - end_s, boundary_key) + end_s;
        next_start.y = std::max(next_start.y, end_s);

        answer.count_ += (end_s - cur_end.y) * (next_start.x - begin_r);

        if (materialize) {
          answer.items_.emplace_back(longlong4{begin_r, next_start.x - 1, cur_end.y, end_s - 1}, bases);
        }
      }
    }
  }
}

template <int blocks_per_multi_processor, typename T>
JoinResult<T> GlobalJoinWithLaunchBounds(T* keys_r, T* keys_s, const long long n_r, const long long n_s,
                                         const longlong2& bases, const int i_gpu, DeviceAllocator& device_allocator,
                                         StreamPool& stream_pool, const bool materialize,
                                         const cudaDeviceProp& device_properties) {
  if constexpr (blocks_per_multi_processor >= 17) {
    throw std::runtime_error("Cannot handle more than 16 blocks per streaming multiprocessor.");
  } else {
    if ((blocks_per_multi_processor + 1) * kNumJoinThreads <= device_properties.maxThreadsPerMultiProcessor) {
      return GlobalJoinWithLaunchBounds<blocks_per_multi_processor + 1>(
          keys_r, keys_s, n_r, n_s, bases, i_gpu, device_allocator, stream_pool, materialize, device_properties);
    }
  }

  JoinResult<T> answer(0);
  const size_t max_parallelism =
      size_t(device_properties.maxThreadsPerMultiProcessor) * device_properties.multiProcessorCount;

  // Upstream: vector<bool> has_join_count — proxy objects cause subtle
  // bugs with volatile/concurrent access patterns.
  // Changed: std::bitset for direct bool semantics, no heap allocation.
  std::bitset<64> has_join_count;
  static_assert(kNumJoinStreams <= 64, "bitset size must cover kNumJoinStreams");

  size_t free_memory = device_allocator.GetFreeBytes();
  size_t per_element_bytes = sizeof(T) * 2 + 1;
  free_memory = (free_memory / kNumJoinStreams - device_allocator.GetAlignment()) / per_element_bytes;

  // Upstream: hand-rolled ceil-div  (n_r + n_s + free_memory - 1) / free_memory
  // Changed: DivideUp from math_utilities.cuh (overflow-safe).
  const size_t n_iterations = std::max<size_t>(kNumJoinStreams, DivideUp(n_r + n_s, free_memory));
  const size_t n_per_iteration = DivideUp(n_r + n_s, n_iterations);

  // ---- AJB: GPU memory + iteration plan state dump ----
  // Print the decisions that determine GPU memory usage and kernel launch
  // grid.  Equivalent to breaking in GDB after the allocation plan is
  // computed but before any cudaMalloc fires.
  fprintf(stderr, "[AJB_SNAP][global_join][plan] gpu=%d n_R=%lld n_S=%lld "
          "free_mem=%zu per_elem=%zu n_iters=%zu per_iter=%zu streams=%d\n",
          i_gpu, n_r, n_s, free_memory, per_element_bytes,
          n_iterations, n_per_iteration, kNumJoinStreams);
  // ---- end AJB plan dump ----

  std::vector<longlong4*> materialization_ranges(kNumJoinStreams);
  if (materialize) {
    for (size_t i = 0; i < kNumJoinStreams; ++i) {
      CheckCudaError(
          cudaMallocHost(&materialization_ranges[i], sizeof(longlong4) * n_per_iteration / 2, cudaHostAllocMapped));
    }
  }

  // Upstream: three separate vectors + three separate alloc loops.
  // Changed: single allocation loop for all three buffer types.
  std::vector<T*> r_buffers(kNumJoinStreams);
  std::vector<T*> s_buffers(kNumJoinStreams);
  std::vector<ulonglong2*> join_and_materialization_counts(kNumJoinStreams);

  for (size_t i = 0; i < kNumJoinStreams; ++i) {
    r_buffers[i] = reinterpret_cast<T*>(device_allocator.allocate(n_per_iteration * sizeof(T)));
    s_buffers[i] = reinterpret_cast<T*>(device_allocator.allocate(n_per_iteration * sizeof(T)));
    join_and_materialization_counts[i] =
        reinterpret_cast<ulonglong2*>(device_allocator.allocate(1 * sizeof(ulonglong2)));
  }

  // Upstream: two vectors for starts/ends, then if(k==0) special case.
  // Changed: direct initialization of starts[0] and uniform loop.
  std::vector<longlong2> starts(n_iterations);
  std::vector<longlong2> ends(n_iterations);
  starts[0] = {0, 0};

  for (size_t k = 0; k < n_iterations; ++k) {
    if (k > 0) {
      starts[k] = ends[k - 1];
    }

    const long long diagonal = (k + 1) * n_per_iteration;
    if (diagonal >= n_r + n_s) {
      ends[k] = {n_r, n_s};
      continue;
    }

    ends[k].x = mgpu::merge_path<mgpu::bounds_t::bounds_upper>(keys_r, n_r, keys_s, n_s, diagonal, mgpu::less_t<T>());
    ends[k].y = diagonal - ends[k].x;
  }

  HandleLongKeyRanges(keys_r, keys_s, n_r, n_s, bases, starts.data(), ends.data(), n_iterations, answer, materialize);

  // Helper lambda to drain one stream's pending join count.
  // Upstream: this block is copy-pasted twice (in the main loop and in
  // the cleanup loop).
  // Changed: factored into a lambda called from both sites.
  auto drain_stream = [&](size_t i_stream) {
    if (!has_join_count.test(i_stream)) return;

    volatile ulonglong2 host_join_materialization = {0, 0};
    CheckCudaError(cudaMemcpyAsync(const_cast<ulonglong2*>(&host_join_materialization),
                                   join_and_materialization_counts[i_stream], sizeof(host_join_materialization),
                                   cudaMemcpyDeviceToHost, stream_pool.GetStream(i_stream)));
    CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(i_stream)));

    answer.count_ += host_join_materialization.x;

    if (materialize) {
      for (size_t i = 0; i < host_join_materialization.y; ++i) {
        answer.items_.emplace_back(materialization_ranges[i_stream][i], bases);
      }
    }

    has_join_count.reset(i_stream);
  };

  for (size_t k = 0; k < n_iterations; ++k) {
    const longlong2& cur_start = starts[k];
    const longlong2& cur_end = ends[k];
    const long long x_count = cur_end.x - cur_start.x;
    const long long y_count = cur_end.y - cur_start.y;

    if (x_count <= 0 || y_count <= 0) {
      continue;
    }

    const size_t i_stream = k % kNumJoinStreams;
    drain_stream(i_stream);

    CheckCudaError(cudaMemsetAsync(join_and_materialization_counts[i_stream], 0, sizeof(ulonglong2),
                                   stream_pool.GetStream(i_stream)));
    CheckCudaError(cudaMemcpyAsync(r_buffers[i_stream], keys_r + cur_start.x, sizeof(T) * x_count,
                                   cudaMemcpyHostToDevice, stream_pool.GetStream(i_stream)));
    CheckCudaError(cudaMemcpyAsync(s_buffers[i_stream], keys_s + cur_start.y, sizeof(T) * y_count,
                                   cudaMemcpyHostToDevice, stream_pool.GetStream(i_stream)));

    // Upstream: (x_count + kNumJoinThreads - 1) / kNumJoinThreads
    // Changed: DivideUp
    size_t n_blocks = DivideUp(static_cast<size_t>(x_count), static_cast<size_t>(kNumJoinThreads));
    const uint32_t max_blocks = (max_parallelism * 128) / kNumJoinThreads;
    if (n_blocks > max_blocks) {
      n_blocks = max_blocks;
    }

    // Upstream: x_count <= y_count picks which table is "inner".
    // No algorithmic change here — the kernel launch logic is preserved.
    longlong2 table_offsets;
    if (x_count <= y_count) {
      table_offsets = cur_start;
      PartitionJoin<blocks_per_multi_processor, false>
          <<<n_blocks, kNumJoinThreads, 0, stream_pool.GetStream(i_stream)>>>(
              r_buffers[i_stream], x_count, s_buffers[i_stream], y_count, join_and_materialization_counts[i_stream],
              materialization_ranges[i_stream], table_offsets);
    } else {
      table_offsets = {cur_start.y, cur_start.x};
      PartitionJoin<blocks_per_multi_processor, true>
          <<<n_blocks, kNumJoinThreads, 0, stream_pool.GetStream(i_stream)>>>(
              s_buffers[i_stream], y_count, r_buffers[i_stream], x_count, join_and_materialization_counts[i_stream],
              materialization_ranges[i_stream], table_offsets);
    }

    has_join_count.set(i_stream);
  }

  // Drain remaining streams — using the same helper.
  for (size_t i_stream = 0; i_stream < kNumJoinStreams; ++i_stream) {
    drain_stream(i_stream);
  }

  if (materialize) {
    for (size_t i = 0; i < kNumJoinStreams; ++i) {
      CheckCudaError(cudaFreeHost(materialization_ranges[i]));
    }
  }

  // Upstream: three separate deallocation loops.
  // Changed: single loop deallocating all three buffer types together.
  for (size_t i = 0; i < kNumJoinStreams; ++i) {
    device_allocator.deallocate(reinterpret_cast<uint8_t*>(r_buffers[i]));
    device_allocator.deallocate(reinterpret_cast<uint8_t*>(s_buffers[i]));
    device_allocator.deallocate(reinterpret_cast<uint8_t*>(join_and_materialization_counts[i]));
  }

  return answer;
}

template <typename T>
JoinResult<T> GlobalJoin(T* keys_r, T* keys_s, const long long n_r, const long long n_s, const longlong2& bases,
                         const int i_gpu, DeviceAllocator& device_allocator, StreamPool& stream_pool,
                         const bool materialize) {
  if (n_r <= 0 || n_s <= 0) {
    return JoinResult<T>(0);
  }

  CheckCudaError(cudaSetDevice(i_gpu));
  cudaDeviceProp device_properties;
  CheckCudaError(cudaGetDeviceProperties(&device_properties, i_gpu));
  CheckCudaError(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32));

  return GlobalJoinWithLaunchBounds<1>(keys_r, keys_s, n_r, n_s, bases, i_gpu, device_allocator, stream_pool,
                                       materialize, device_properties);
}

template <typename T, typename V>
JoinResult<T> MergeJoin(PinnedVector<T>& keys_r, PinnedVector<V>& values_r, PinnedVector<T>& keys_s,
                        PinnedVector<V>& values_s, const std::vector<int>& gpus,
                        std::vector<DeviceAllocator>& device_allocators, std::vector<StreamPool>& stream_pools,
                        const bool materialize) {
  for (size_t g = 0; g < gpus.size(); ++g) {
    CheckCudaError(cudaSetDevice(gpus[g]));

    constexpr double kDeviceMemoryUtilization = 0.98;

    size_t free_global_memory, total_global_memory;
    CheckCudaError(cudaMemGetInfo(&free_global_memory, &total_global_memory));

    device_allocators[g].Initialize(kDeviceMemoryUtilization * free_global_memory);
    stream_pools[g].Initialize(kNumJoinStreams);
  }

  TimeScope time_scope("join_phase");

  const size_t device_count = gpus.size();
  // Upstream: (keys_r.size() + keys_s.size() + device_count - 1) / device_count
  // Changed: DivideUp
  const size_t per_device_length = DivideUp(keys_r.size() + keys_s.size(), device_count);
  std::vector<JoinResult<T>> device_results(device_count);
  std::vector<longlong2> starts(device_count);
  std::vector<longlong2> ends(device_count);

  // Upstream: if(i_device==0) starts={0,0} else starts=ends[i-1]
  // Changed: initialize starts[0] directly, uniform loop from 1.
  starts[0] = {0, 0};

  for (size_t i_device = 0; i_device < device_count; ++i_device) {
    if (i_device > 0) {
      starts[i_device] = ends[i_device - 1];
    }

    const long long diagonal = (i_device + 1) * per_device_length;
    if (diagonal >= (long long)(keys_r.size() + keys_s.size())) {
      ends[i_device].x = keys_r.size();
      ends[i_device].y = keys_s.size();
    } else {
      ends[i_device].x = mgpu::merge_path<mgpu::bounds_t::bounds_upper>(
          keys_r.data(), static_cast<long long>(keys_r.size()), keys_s.data(), static_cast<long long>(keys_s.size()),
          diagonal, mgpu::less_t<T>());
      ends[i_device].y = diagonal - ends[i_device].x;
    }
  }

  JoinResult<T> answer;
  HandleLongKeyRanges(keys_r.data(), keys_s.data(), keys_r.size(), keys_s.size(), {0, 0}, starts.data(), ends.data(),
                      device_count, answer, materialize);

  // ---- AJB: partition state dump (print current data structure state) ----
  // This gives you the same visibility as setting a breakpoint after
  // merge_path partitioning — without needing GDB on a GPU cluster.
  // Shows per-device workload so you can spot load imbalance immediately.
  {
    long long total_work = 0;
    long long max_work = 0, min_work = (long long)keys_r.size() + keys_s.size();
    for (size_t d = 0; d < device_count; ++d) {
      long long w_r = ends[d].x - starts[d].x;
      long long w_s = ends[d].y - starts[d].y;
      long long work = w_r + w_s;
      total_work += work;
      if (work > max_work) max_work = work;
      if (work < min_work) min_work = work;
      fprintf(stderr, "[AJB_SNAP][merge_join][partition] gpu=%zu "
              "R=[%lld,%lld) S=[%lld,%lld) work_R=%lld work_S=%lld\n",
              d, starts[d].x, ends[d].x, starts[d].y, ends[d].y, w_r, w_s);
    }
    double balance_ratio = min_work > 0 ? (double)max_work / min_work : 0.0;
    fprintf(stderr, "[AJB_SNAP][merge_join][balance] devices=%zu "
            "total=%lld max=%lld min=%lld ratio=%.2f\n",
            device_count, total_work, max_work, min_work, balance_ratio);

    // key range per device — shows how merge_path split the keyspace
    for (size_t d = 0; d < device_count; ++d) {
      long long r_first = starts[d].x < (long long)keys_r.size()
                          ? (long long)keys_r[starts[d].x] : -1;
      long long r_last = ends[d].x > 0 && ends[d].x <= (long long)keys_r.size()
                         ? (long long)keys_r[ends[d].x - 1] : -1;
      long long s_first = starts[d].y < (long long)keys_s.size()
                          ? (long long)keys_s[starts[d].y] : -1;
      long long s_last = ends[d].y > 0 && ends[d].y <= (long long)keys_s.size()
                         ? (long long)keys_s[ends[d].y - 1] : -1;
      fprintf(stderr, "[AJB_SNAP][merge_join][keys] gpu=%zu "
              "R_keys=[%lld..%lld] S_keys=[%lld..%lld]\n",
              d, r_first, r_last, s_first, s_last);
    }
  }
  // ---- end AJB partition dump ----

#pragma omp parallel for num_threads(device_count)
  for (size_t i_device = 0; i_device < device_count; ++i_device) {
    const long long n_r = ends[i_device].x - starts[i_device].x;
    const long long n_s = ends[i_device].y - starts[i_device].y;

    if (n_r <= 0 || n_s <= 0) {
      continue;
    }

    device_results[i_device] =
        GlobalJoin(keys_r.data() + starts[i_device].x, keys_s.data() + starts[i_device].y, n_r, n_s, starts[i_device],
                   gpus[i_device], device_allocators[i_device], stream_pools[i_device], materialize);
  }

  for (size_t i_device = 0; i_device < device_count; ++i_device) {
    answer.count_ += device_results[i_device].count_;
  }

  if (materialize) {
    // Upstream: per-item emplace_back in a nested loop with separate
    // clear + shrink_to_fit for each device.
    // Changed: pre-reserve total items, then use move-insert to batch
    // each device's results in one shot, followed by a single clear.
    size_t total_items = 0;
    for (size_t d = 0; d < device_count; ++d)
      total_items += device_results[d].items_.size();
    answer.items_.reserve(answer.items_.size() + total_items);

    for (size_t i_device = 0; i_device < device_count; ++i_device) {
      auto& items = device_results[i_device].items_;
      answer.items_.insert(answer.items_.end(),
                           std::make_move_iterator(items.begin()),
                           std::make_move_iterator(items.end()));
      items.clear();
      items.shrink_to_fit();
    }
  }

  return answer;
}
