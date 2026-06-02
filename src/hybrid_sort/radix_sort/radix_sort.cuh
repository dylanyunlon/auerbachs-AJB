#pragma once

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
size_t DetectSpanningBuckets(DeviceContainers<T, V>& device_containers, HostContainers<T, V>& host_containers,
                             std::vector<std::vector<std::pair<int, BucketId>>>& spanning_buckets,
                             std::map<BucketId, std::vector<int>, CompareBucketIds>& spanning_bucket_to_gpus_map,
                             std::vector<int>& gpus, size_t iteration) {
  size_t num_spanning_buckets = 0;
  size_t num_gpus = gpus.size();

  for (size_t g = 0; g < num_gpus; ++g) {
    const int gpu = gpus[g];
    for (size_t s = 0; s < spanning_buckets[iteration - 1].size(); ++s) {
      if (spanning_buckets[iteration - 1][s].first == gpu) {
        for (size_t i = 0; i < kNumBuckets; ++i) {
          HostHistograms* host_histograms =
              host_containers.GetHistograms(gpu, spanning_buckets[iteration - 1][s].second);
          if (host_histograms->GetBucketToGpuMap()[(i * num_gpus) + 1] >= 0 &&
              host_histograms->GetGlobalHistogram()[i] > 0) {
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
    }
  }

  return num_spanning_buckets;
}

size_t GetNumThreadBlocks(size_t num_keys, size_t keys_per_thread, size_t num_threads) {
  size_t num_key_groups = num_keys / keys_per_thread;
  if (num_keys % keys_per_thread != 0) {
    ++num_key_groups;
  }

  size_t num_thread_blocks = 1;
  num_thread_blocks = num_key_groups / num_threads;
  if (num_key_groups % num_threads != 0) {
    ++num_thread_blocks;
  }

  return num_thread_blocks;
}

template <typename T, typename V>
std::function<void()> RadixSort(T* in_keys, V* in_values, T* out_keys, V* out_values, const size_t num_elements,
                                ResourceManager<T, V>& resource_manager, std::vector<int> gpus) {
  constexpr size_t max_num_partition_passes = sizeof(T);
  constexpr size_t keys_per_thread = sizeof(T) == 4 ? 6 : 3;
  constexpr size_t shared_memory_size = keys_per_thread * kNumRadixThreads * (sizeof(T) + sizeof(V));

  static_assert(shared_memory_size <= 48 * 1024);

  size_t num_fillers = (num_elements % gpus.size() != 0) ? (gpus.size() - num_elements % gpus.size()) : 0;
  size_t chunk_size = (num_elements + num_fillers) / gpus.size();

  while (chunk_size < num_fillers) {
    gpus.resize(gpus.size() / 2);
    num_fillers = (num_elements % gpus.size() != 0) ? (gpus.size() - num_elements % gpus.size()) : 0;
    chunk_size = (num_elements + num_fillers) / gpus.size();
  }

  size_t num_partition_passes_needed = max_num_partition_passes;
  size_t num_thread_blocks = GetNumThreadBlocks(chunk_size, keys_per_thread, kNumRadixThreads);

  const size_t num_gpus = gpus.size();

  // [AJB] radix sort config: max_passes = sizeof(T) = how many bytes to partition through
  fprintf(stderr, "[AJB_BP][RadixSort] n=%zu gpus=%zu chunk=%zu max_passes=%zu key_bytes=%zu smem=%zu\n",
          num_elements, num_gpus, chunk_size, max_num_partition_passes, sizeof(T), shared_memory_size);

  if (num_gpus == 1) {
    const int gpu = gpus[0];
    fprintf(stderr, "[AJB_TRACE][RadixSort] single-GPU path: gpu=%d, using cub::DeviceRadixSort directly\n", gpu);

    DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
    StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

    CheckCudaError(cudaSetDevice(gpu));

    CheckCudaError(cudaMemcpyAsync(resource_manager.GetKeys(gpu), in_keys, sizeof(T) * chunk_size,
                                   cudaMemcpyHostToDevice, stream_pool.GetStream(0)));
    CheckCudaError(cudaMemcpyAsync(resource_manager.GetValues(gpu), in_values, sizeof(V) * chunk_size,
                                   cudaMemcpyHostToDevice, stream_pool.GetStream(0)));

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
                                   cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
    CheckCudaError(cudaMemcpyAsync(out_values, resource_manager.GetValues(gpu), sizeof(V) * chunk_size,
                                   cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));

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

    for (size_t g = 0; g < num_gpus; ++g) {
      const int gpu = gpus[g];
      CheckCudaError(cudaSetDevice(gpu));

      CheckCudaError(cudaFuncSetCacheConfig(&ScatterKeyValuePairs<T, V>, cudaFuncCachePreferShared));
      CheckCudaError(cudaFuncSetAttribute(&ScatterKeyValuePairs<T, V>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          shared_memory_size));

      spanning_buckets[0].emplace_back(gpu, BucketId());
      spanning_bucket_to_gpus_map[BucketId()].push_back(gpu);

      host_containers.AssignNewHistogramBuffer(gpu, BucketId());
      device_containers.AssignNewHistogramBuffer(gpu, BucketId());

      reduced_sorting_buckets[g].reserve(num_gpus * kMaxNumBucketsForReducedSorting);
    }

    for (size_t iteration = 0; iteration < sizeof(T); ++iteration) {
      if (iteration > 0) {
        num_spanning_buckets = DetectSpanningBuckets<T>(device_containers, host_containers, spanning_buckets,
                                                        spanning_bucket_to_gpus_map, gpus, iteration);
      }

      // [AJB] partition pass状态: spanning_buckets>0 说明还有桶横跨多个GPU,需要继续分
      fprintf(stderr, "[AJB_TRACE][RadixSort] pass %zu: spanning_buckets=%zu\n", iteration, num_spanning_buckets);

      if (num_spanning_buckets == 0) {
        num_partition_passes_needed = iteration;
        fprintf(stderr, "[AJB_TRACE][RadixSort] partition converged at pass %zu (all buckets single-GPU)\n", iteration);
        break;
      }

#pragma omp parallel for num_threads(num_gpus)
      for (size_t g = 0; g < num_gpus; ++g) {
        const int gpu = gpus[g];
        CheckCudaError(cudaSetDevice(gpu));

        DeviceAllocator& device_allocator = resource_manager.GetDeviceAllocator(gpu);
        StreamPool& stream_pool = resource_manager.GetStreamPool(gpu);

        size_t g_chunk_size = chunk_size - (g == num_gpus - 1 ? num_fillers : 0);

        if (iteration == 0) {
          CheckCudaError(cudaMemcpyAsync(resource_manager.GetKeys(gpu), in_keys + (chunk_size * g),
                                         sizeof(T) * g_chunk_size, cudaMemcpyHostToDevice, stream_pool.GetStream(0)));

          CheckCudaError(cudaMemcpyAsync(resource_manager.GetValues(gpu), in_values + (chunk_size * g),
                                         sizeof(V) * g_chunk_size, cudaMemcpyHostToDevice, stream_pool.GetStream(0)));

          CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(2)));
          CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
        }

        if (iteration == 0) {
          DeviceHistograms* hist = device_containers.GetHistograms(gpu, BucketId());
          ComputeHistogram<T><<<num_thread_blocks, kNumRadixThreads, 0, stream_pool.GetStream(0)>>>(
              resource_manager.GetKeys(gpu), hist->GetBlockLocalHistograms(), g_chunk_size, keys_per_thread,
              (sizeof(T) - iteration) * kNumRadixBits);
          CheckCudaLaunchError();
          AggregateHistogram<<<(num_thread_blocks / kNumBlockHistogramsToAggregate) + 1, kNumRadixThreads, 0,
                               stream_pool.GetStream(0)>>>(hist->GetGlobalHistogram(), hist->GetBlockLocalHistograms(),
                                                           num_thread_blocks, kNumBlockHistogramsToAggregate);
          CheckCudaLaunchError();
        } else {
          for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {
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
              AggregateHistogram<<<(local_num_thread_blocks / kNumBlockHistogramsToAggregate) + 1, kNumRadixThreads, 0,
                                   stream_pool.GetStream(0)>>>(hist->GetGlobalHistogram(),
                                                               hist->GetBlockLocalHistograms(), local_num_thread_blocks,
                                                               kNumBlockHistogramsToAggregate);
              CheckCudaLaunchError();
            }
          }
        }

        CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));

        if (iteration == 0) {
          for (size_t dest_gpu = 0; dest_gpu < num_gpus; ++dest_gpu) {
            CheckCudaError(cudaMemcpyAsync(
                device_containers.GetHistograms(gpus[dest_gpu], BucketId())->GetMgpuHistograms() + (g * kNumBuckets),
                device_containers.GetHistograms(gpu, BucketId())->GetGlobalHistogram(), sizeof(uint64_t) * kNumBuckets,
                cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
          }
        } else {
          for (size_t s = 0; s < spanning_buckets[iteration].size(); ++s) {
            if (spanning_buckets[iteration][s].first == gpu) {
              BucketId& spanning_bucket = spanning_buckets[iteration][s].second;

              for (auto dest_gpu : spanning_bucket_to_gpus_map[spanning_bucket]) {
                CheckCudaError(cudaMemcpyAsync(
                    device_containers.GetHistograms(dest_gpu, spanning_bucket)->GetMgpuHistograms() + (g * kNumBuckets),
                    device_containers.GetHistograms(gpu, spanning_bucket)->GetGlobalHistogram(),
                    sizeof(uint64_t) * kNumBuckets, cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
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

        for (uint8_t* temporary_storage_pointer : temporary_storage_pointers) {
          device_allocator.deallocate(reinterpret_cast<uint8_t*>(temporary_storage_pointer));
        }

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
            if (spanning_buckets[iteration][s].first == gpu) {
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

            for (size_t i = 1; i < spanning_bucket_index; ++i) {
              CheckCudaError(
                  cudaMemcpyAsync(resource_manager.GetOtherKeys(gpu) + key_scatter_offsets[i - 1].second,
                                  resource_manager.GetKeys(gpu) + key_scatter_offsets[i - 1].second,
                                  sizeof(T) * (key_scatter_offsets[i].first - key_scatter_offsets[i - 1].second),
                                  cudaMemcpyDeviceToDevice, stream_pool.GetStream(1)));
              CheckCudaError(
                  cudaMemcpyAsync(resource_manager.GetOtherValues(gpu) + key_scatter_offsets[i - 1].second,
                                  resource_manager.GetValues(gpu) + key_scatter_offsets[i - 1].second,
                                  sizeof(V) * (key_scatter_offsets[i].first - key_scatter_offsets[i - 1].second),
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

        for (uint8_t* temporary_storage_pointer : temporary_storage_pointers) {
          device_allocator.deallocate(reinterpret_cast<uint8_t*>(temporary_storage_pointer));
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

              if (last_pass_spanning_buckets.count(last_pass_bucket) == 0) {
                last_pass_spanning_buckets[last_pass_bucket] = {0, {}};
                last_pass_spanning_buckets[last_pass_bucket].second.reserve(num_gpus);
              }

              int j = 0;
              size_t source_offset = 0;
              while (current_bucket_to_gpu_map[(i * num_gpus) + j] >= 0 && j < num_gpus && source_gpu_bucket_size > 0) {
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
                ++j;
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
        size_t balanced_chunk_size = chunk_size - (g == num_gpus - 1 ? num_fillers : 0);

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

                if (prev_bucket != nullptr && prev_bucket->partition_pass == spanning_bucket.partition_pass &&
                    prev_bucket->bucket_size + bucket_size < device_containers.GetGamma() &&
                    prev_bucket->bucket_start + prev_bucket->bucket_size == bucket_start) {
                  prev_bucket->bucket_size += bucket_size;

                  uint8_t xor_bucket = static_cast<uint8_t>(i) ^ static_cast<uint8_t>(prev_bucket->bucket_number);
                  uint32_t msb_dif_position = 0;
                  while (xor_bucket) {
                    xor_bucket >>= 1;
                    ++msb_dif_position;
                  }

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
                prev_bucket = &reduced_sorting_buckets[g][num_buckets_to_sort];
                ++num_buckets_to_sort;
              }
            }
          }
        }

        std::sort(reduced_sorting_buckets[g].begin(), reduced_sorting_buckets[g].end(),
                  CompareReducedSortingBuckets<T, V>());

        // [AJB] reduced sort: 每个桶太小不值得multi-GPU partition,改用cub局部排序
        // num_buckets_to_sort > kMaxNumBucketsForReducedSorting 时退化为全量cub排序
        fprintf(stderr, "[AJB_STATE][RadixSort] gpu=%d reduced_buckets=%zu local_chunk=%zu (threshold=%zu)\n",
                gpu, num_buckets_to_sort, gpu_local_chunk_size, (size_t)kMaxNumBucketsForReducedSorting);

        std::vector<uint8_t*> temporary_storage_pointers;
        temporary_storage_pointers.reserve(num_buckets_to_sort);

        if (num_buckets_to_sort <= kMaxNumBucketsForReducedSorting) {
          if (num_buckets_to_sort >= kMinNumBucketsForSortCopyOverlap) {
            size_t sorted_keys_offset = 0;
            size_t transferred_keys = 0;

            for (size_t s = 0; s < num_buckets_to_sort; ++s) {
              ReducedSortingBucket<T, V>& b = reduced_sorting_buckets[g][s];

              uint32_t end_bit =
                  std::min(sizeof(T) * 8, (sizeof(T) - b.partition_pass - 1) * kNumRadixBits + 1 + b.msb_dif_position);

              size_t temporary_num_bytes = 0;

              cub::DeviceRadixSort::SortPairs(nullptr, temporary_num_bytes, b.cub_double_buffer_keys,
                                              b.cub_double_buffer_values, b.bucket_size, 0, end_bit,
                                              stream_pool.GetStream(0));

              uint8_t* temporary_storage_pointer = device_allocator.allocate(temporary_num_bytes);
              temporary_storage_pointers.push_back(temporary_storage_pointer);

              cub::DeviceRadixSort::SortPairs((void*)temporary_storage_pointer, temporary_num_bytes,
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

              CheckCudaError(cudaMemcpyAsync(
                  out_keys + gpu_global_offsets[g] + transferred_keys, resource_manager.GetKeys(gpu) + transferred_keys,
                  sizeof(T) * keys_to_transfer, cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
              CheckCudaError(cudaMemcpyAsync(out_values + gpu_global_offsets[g] + transferred_keys,
                                             resource_manager.GetValues(gpu) + transferred_keys,
                                             sizeof(V) * keys_to_transfer, cudaMemcpyDeviceToHost,
                                             stream_pool.GetStream(2)));

              transferred_keys += keys_to_transfer;
            }

            if (gpu_local_chunk_size > transferred_keys) {
              CheckCudaError(cudaMemcpyAsync(out_keys + gpu_global_offsets[g] + transferred_keys,
                                             resource_manager.GetKeys(gpu) + transferred_keys,
                                             sizeof(T) * (gpu_local_chunk_size - transferred_keys),
                                             cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
              CheckCudaError(cudaMemcpyAsync(out_values + gpu_global_offsets[g] + transferred_keys,
                                             resource_manager.GetValues(gpu) + transferred_keys,
                                             sizeof(V) * (gpu_local_chunk_size - transferred_keys),
                                             cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
            }
          } else {
            for (size_t s = 0; s < num_buckets_to_sort; ++s) {
              ReducedSortingBucket<T, V>& b = reduced_sorting_buckets[g][s];

              uint32_t end_bit =
                  std::min(sizeof(T) * 8, (sizeof(T) - b.partition_pass - 1) * kNumRadixBits + 1 + b.msb_dif_position);

              size_t temporary_num_bytes = 0;
              cub::DeviceRadixSort::SortPairs(nullptr, temporary_num_bytes, b.cub_double_buffer_keys,
                                              b.cub_double_buffer_values, b.bucket_size, 0, end_bit,
                                              stream_pool.GetStream(0));

              uint8_t* temporary_storage_pointer = device_allocator.allocate(temporary_num_bytes);
              temporary_storage_pointers.push_back(temporary_storage_pointer);

              cub::DeviceRadixSort::SortPairs((void*)temporary_storage_pointer, temporary_num_bytes,
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

          cub::DeviceRadixSort::SortPairs(nullptr, temporary_num_bytes, resource_manager.GetKeysBuffer(gpu),
                                          resource_manager.GetValuesBuffer(gpu), gpu_local_chunk_size, 0, sizeof(T) * 8,
                                          stream_pool.GetStream(0));

          uint8_t* temporary_storage_pointer = device_allocator.allocate(temporary_num_bytes);
          temporary_storage_pointers.push_back(temporary_storage_pointer);

          cub::DeviceRadixSort::SortPairs((void*)temporary_storage_pointer, temporary_num_bytes,
                                          resource_manager.GetKeysBuffer(gpu), resource_manager.GetValuesBuffer(gpu),
                                          gpu_local_chunk_size, 0, sizeof(T) * 8, stream_pool.GetStream(0));

          CheckCudaError(cudaStreamSynchronize(stream_pool.GetStream(0)));
        }

        for (uint8_t* temporary_storage_pointer : temporary_storage_pointers) {
          device_allocator.deallocate(reinterpret_cast<uint8_t*>(temporary_storage_pointer));
        }

        if (num_buckets_to_sort > kMaxNumBucketsForReducedSorting ||
            num_buckets_to_sort < kMinNumBucketsForSortCopyOverlap) {
          if (gpu_global_offsets[g] < balanced_chunk_size * g) {
            size_t skip_keys = balanced_chunk_size * g - gpu_global_offsets[g];

            CheckCudaError(cudaMemcpyAsync(
                out_keys + balanced_chunk_size * g, resource_manager.GetKeys(gpu) + skip_keys,
                sizeof(T) * (gpu_local_chunk_size - skip_keys), cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
            CheckCudaError(cudaMemcpyAsync(
                out_values + balanced_chunk_size * g, resource_manager.GetValues(gpu) + skip_keys,
                sizeof(V) * (gpu_local_chunk_size - skip_keys), cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));

            CheckCudaError(cudaMemcpyAsync(out_keys + gpu_global_offsets[g], resource_manager.GetKeys(gpu),
                                           sizeof(T) * skip_keys, cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));
            CheckCudaError(cudaMemcpyAsync(out_values + gpu_global_offsets[g], resource_manager.GetValues(gpu),
                                           sizeof(V) * skip_keys, cudaMemcpyDeviceToHost, stream_pool.GetStream(2)));

          } else {
            CheckCudaError(cudaMemcpyAsync(out_keys + gpu_global_offsets[g], resource_manager.GetKeys(gpu),
                                           sizeof(T) * gpu_local_chunk_size, cudaMemcpyDeviceToHost,
                                           stream_pool.GetStream(2)));
            CheckCudaError(cudaMemcpyAsync(out_values + gpu_global_offsets[g], resource_manager.GetValues(gpu),
                                           sizeof(V) * gpu_local_chunk_size, cudaMemcpyDeviceToHost,
                                           stream_pool.GetStream(2)));
          }
        }
      }

      resource_manager.FlipBuffers(gpu);
    }
  }

  return [&resource_manager, gpus]() {
    for (const int gpu : gpus) {
      CheckCudaError(cudaStreamSynchronize(resource_manager.GetStreamPool(gpu).GetStream(2)));
    }
  };
}
