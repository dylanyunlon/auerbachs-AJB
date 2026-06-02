#pragma once

#include <vector>

#include <moderngpu/cta_merge.hxx>

#include "common/device_allocator.cuh"
#include "common/error_utilities.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/stream_pool.cuh"
#include "constants.cuh"
#include "join_result.cuh"
#include "kernels.cuh"

template <typename T>
long long LastSmaller(T* data, long long count, const T& than) {
  if (count <= 0 || !(data[0] < than)) {
    return -1;
  }

  long long l = 0, r = count - 1;
  while (l < r) {
    long long m = (l + r + 1) / 2;
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

  long long l = 0, r = count - 1;
  while (l < r) {
    long long m = (l + r) / 2;
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

template <typename T>
void HandleLongKeyRanges(T* keys_r, T* keys_s, const long long n_r, const long long n_s, const longlong2& bases,
                         longlong2* starts, longlong2* ends, long long n_partitions, JoinResult<T>& answer,
                         const bool materialize) {
  longlong2 next_start = {0, 0};

  for (long long i_partition = 0; i_partition < n_partitions; ++i_partition) {
    if (next_start.x >= ends[i_partition].x || next_start.y >= ends[i_partition].y) {
      if (next_start.x == ends[i_partition].x && next_start.y < ends[i_partition].y) {
        if (keys_s[next_start.y] == keys_s[ends[i_partition].y - 1]) {
          const longlong2 bounds_r = FindBounds(keys_r, n_r, keys_s[next_start.y]);

          if (bounds_r.x < bounds_r.y) {
            const longlong2 bounds_s = FindBounds(keys_s, n_s, keys_s[next_start.y]);
            answer.count_ += (bounds_r.y - bounds_r.x) * (bounds_s.y - bounds_s.x);

            if (materialize) {
              answer.items_.emplace_back(longlong4{bounds_r.x, bounds_r.y - 1, bounds_s.x, bounds_s.y - 1}, bases);
            }

            next_start = {std::max(next_start.x, bounds_r.y), std::max(next_start.y, bounds_s.y)};
          }
        } else {
          longlong2 bounds_r = FindBounds(keys_r, n_r, keys_s[next_start.y]);

          if (bounds_r.x < bounds_r.y) {
            const longlong2 bounds_s = FindBounds(keys_s, n_s, keys_s[next_start.y]);
            answer.count_ += (bounds_r.y - bounds_r.x) * (bounds_s.y - bounds_s.x);

            if (materialize) {
              answer.items_.emplace_back(longlong4{bounds_r.x, bounds_r.y - 1, bounds_s.x, bounds_s.y - 1}, bases);
            }

            next_start = {std::max(next_start.x, bounds_r.y), std::max(next_start.y, bounds_s.y)};
          }
          if (next_start.y < ends[i_partition].y) {
            bounds_r = FindBounds(keys_r, n_r, keys_s[ends[i_partition].y - 1]);

            if (bounds_r.x < bounds_r.y) {
              const longlong2 bounds_s = FindBounds(keys_s, n_s, keys_s[ends[i_partition].y - 1]);
              answer.count_ += (bounds_r.y - bounds_r.x) * (bounds_s.y - bounds_s.x);

              if (materialize) {
                answer.items_.emplace_back(longlong4{bounds_r.x, bounds_r.y - 1, bounds_s.x, bounds_s.y - 1}, bases);
              }

              next_start = {std::max(next_start.x, bounds_r.y), std::max(next_start.y, bounds_s.y)};
            }
          }
        }
      } else if (next_start.y == ends[i_partition].y && next_start.x < ends[i_partition].x) {
        if (keys_r[next_start.x] == keys_r[ends[i_partition].x - 1]) {
          const longlong2 bounds_s = FindBounds(keys_s, n_s, keys_r[next_start.x]);

          if (bounds_s.x < bounds_s.y) {
            const longlong2 bounds_r = FindBounds(keys_r, n_r, keys_r[next_start.x]);
            answer.count_ += (bounds_r.y - bounds_r.x) * (bounds_s.y - bounds_s.x);

            if (materialize) {
              answer.items_.emplace_back(longlong4{bounds_r.x, bounds_r.y - 1, bounds_s.x, bounds_s.y - 1}, bases);
            }

            next_start = {std::max(next_start.x, bounds_r.y), std::max(next_start.y, bounds_s.y)};
          }
        } else {
          longlong2 bounds_s = FindBounds(keys_s, n_s, keys_r[next_start.x]);

          if (bounds_s.x < bounds_s.y) {
            const longlong2 bounds_r = FindBounds(keys_r, n_r, keys_r[next_start.x]);
            answer.count_ += (bounds_r.y - bounds_r.x) * (bounds_s.y - bounds_s.x);

            if (materialize) {
              answer.items_.emplace_back(longlong4{bounds_r.x, bounds_r.y - 1, bounds_s.x, bounds_s.y - 1}, bases);
            }

            next_start = {std::max(next_start.x, bounds_r.y), std::max(next_start.y, bounds_s.y)};
          }
          if (next_start.x < ends[i_partition].x) {
            bounds_s = FindBounds(keys_s, n_s, keys_r[ends[i_partition].x - 1]);

            if (bounds_s.x < bounds_s.y) {
              const longlong2 bounds_r = FindBounds(keys_r, n_r, keys_r[ends[i_partition].x - 1]);
              answer.count_ += (bounds_r.y - bounds_r.x) * (bounds_s.y - bounds_s.x);

              if (materialize) {
                answer.items_.emplace_back(longlong4{bounds_r.x, bounds_r.y - 1, bounds_s.x, bounds_s.y - 1}, bases);
              }

              next_start = {std::max(next_start.x, bounds_r.y), std::max(next_start.y, bounds_s.y)};
            }
          }
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
  std::vector<bool> has_join_count(kNumJoinStreams, false);

  size_t free_memory = device_allocator.GetFreeBytes();
  size_t per_element_bytes = sizeof(T) * 2 + 1;
  free_memory = (free_memory / kNumJoinStreams - device_allocator.GetAlignment()) / per_element_bytes;

  const size_t n_iterations = std::max<size_t>(kNumJoinStreams, (n_r + n_s + free_memory - 1) / free_memory);
  const size_t n_per_iteration = (n_r + n_s + n_iterations - 1) / n_iterations;

  std::vector<longlong4*> materialization_ranges(kNumJoinStreams);
  if (materialize) {
    for (size_t i = 0; i < kNumJoinStreams; ++i) {
      CheckCudaError(
          cudaMallocHost(&materialization_ranges[i], sizeof(longlong4) * n_per_iteration / 2, cudaHostAllocMapped));
    }
  }

  std::vector<T*> r_buffers(kNumJoinStreams);
  std::vector<T*> s_buffers(kNumJoinStreams);
  std::vector<ulonglong2*> join_and_materialization_counts(kNumJoinStreams);

  for (size_t i = 0; i < kNumJoinStreams; ++i) {
    r_buffers[i] = reinterpret_cast<T*>(device_allocator.allocate(n_per_iteration * sizeof(T)));
    s_buffers[i] = reinterpret_cast<T*>(device_allocator.allocate(n_per_iteration * sizeof(T)));
    join_and_materialization_counts[i] =
        reinterpret_cast<ulonglong2*>(device_allocator.allocate(1 * sizeof(ulonglong2)));
  }

  std::vector<longlong2> starts(n_iterations);
  std::vector<longlong2> ends(n_iterations);

  for (size_t k = 0; k < n_iterations; ++k) {
    if (k == 0) {
      starts[k] = {0, 0};
    } else {
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

  for (size_t k = 0; k < n_iterations; ++k) {
    const longlong2& cur_start = starts[k];
    const longlong2& cur_end = ends[k];
    const long long x_count = cur_end.x - cur_start.x;
    const long long y_count = cur_end.y - cur_start.y;

    if (x_count <= 0 || y_count <= 0) {
      continue;
    }

    const int i_stream = k % kNumJoinStreams;
    volatile ulonglong2 host_join_materialization = {0, 0};

    if (has_join_count[i_stream]) {
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

      has_join_count[i_stream] = false;
    }

    CheckCudaError(cudaMemsetAsync(join_and_materialization_counts[i_stream], 0, sizeof(ulonglong2),
                                   stream_pool.GetStream(i_stream)));
    CheckCudaError(cudaMemcpyAsync(r_buffers[i_stream], keys_r + cur_start.x, sizeof(T) * x_count,
                                   cudaMemcpyHostToDevice, stream_pool.GetStream(i_stream)));
    CheckCudaError(cudaMemcpyAsync(s_buffers[i_stream], keys_s + cur_start.y, sizeof(T) * y_count,
                                   cudaMemcpyHostToDevice, stream_pool.GetStream(i_stream)));

    size_t n_blocks = (x_count + kNumJoinThreads - 1) / kNumJoinThreads;
    const uint32_t max_blocks = (max_parallelism * 128) / kNumJoinThreads;
    if (n_blocks > max_blocks) {
      n_blocks = max_blocks;
    }

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

    has_join_count[i_stream] = true;
  }

  for (size_t i_stream = 0; i_stream < kNumJoinStreams; ++i_stream) {
    if (!has_join_count[i_stream]) {
      continue;
    }

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
  }

  if (materialize) {
    for (size_t i = 0; i < kNumJoinStreams; ++i) {
      CheckCudaError(cudaFreeHost(materialization_ranges[i]));
    }
  }

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
  // [AJB] merge-join入口: R和S必须已排序,merge-path按对角线切分给各GPU
  fprintf(stderr, "[AJB_BP][MergeJoin] |R|=%zu |S|=%zu gpus=%zu materialize=%d\n",
          keys_r.size(), keys_s.size(), gpus.size(), (int)materialize);

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
  const size_t per_device_length = (keys_r.size() + keys_s.size() + device_count - 1) / device_count;
  std::vector<JoinResult<T>> device_results(device_count);
  std::vector<longlong2> starts(device_count);
  std::vector<longlong2> ends(device_count);

  for (size_t i_device = 0; i_device < device_count; ++i_device) {
    if (i_device == 0) {
      starts[i_device] = {0, 0};
    } else {
      starts[i_device] = ends[i_device - 1];
    }

    const long long diagonal = (i_device + 1) * per_device_length;
    if (diagonal >= keys_r.size() + keys_s.size()) {
      ends[i_device].x = keys_r.size();
      ends[i_device].y = keys_s.size();
    } else {
      ends[i_device].x = mgpu::merge_path<mgpu::bounds_t::bounds_upper>(
          keys_r.data(), static_cast<long long>(keys_r.size()), keys_s.data(), static_cast<long long>(keys_s.size()),
          diagonal, mgpu::less_t<T>());
      ends[i_device].y = diagonal - ends[i_device].x;
    }
    // [AJB] merge-path分区: 每GPU拿到R和S各一段,大小由对角线切分决定
    fprintf(stderr, "[AJB_STATE][MergeJoin] gpu[%zu]: R[%lld..%lld]=%lld  S[%lld..%lld]=%lld\n",
            i_device, starts[i_device].x, ends[i_device].x,
            ends[i_device].x - starts[i_device].x,
            starts[i_device].y, ends[i_device].y,
            ends[i_device].y - starts[i_device].y);
  }

  JoinResult<T> answer;
  HandleLongKeyRanges(keys_r.data(), keys_s.data(), keys_r.size(), keys_s.size(), {0, 0}, starts.data(), ends.data(),
                      device_count, answer, materialize);

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
  fprintf(stderr, "[AJB_TRACE][MergeJoin] total_matches=%zu\n", answer.count_);

  if (materialize) {
    answer.items_.reserve(answer.count_);
    for (size_t i_device = 0; i_device < device_count; ++i_device) {
      for (size_t item = 0; item < device_results[i_device].items_.size(); ++item) {
        answer.items_.emplace_back(std::move(device_results[i_device].items_[item]));
      }
      
      device_results[i_device].items_.clear();
      device_results[i_device].items_.shrink_to_fit();
    }
  }

  return answer;
}
