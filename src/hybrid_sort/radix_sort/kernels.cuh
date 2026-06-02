#pragma once
// [AJB] 基数排序GPU kernel: ComputeHistogram做局部计数, ScatterKeyValuePairs做桶内重排

#include <stdio.h>

#include "common/error_utilities.cuh"

__device__ __forceinline__ int GetAbsolute(int v) { return v < 0 ? -v : v; }

template <std::uint8_t num_bytes>
using uint_type = typename std::conditional<num_bytes == 4, uint32_t, uint64_t>::type;

template <typename T>
__device__ __forceinline__ void GetRadixBucket(const T& key_value, uint_type<sizeof(T)> radix_mask, size_t radix_msb,
                                               uint_type<sizeof(T)>* bucket) {
  *bucket = radix_mask & key_value;
  *bucket = *bucket >> (radix_msb - kNumRadixBits);
}

template <>
__device__ __forceinline__ void GetRadixBucket<float>(const float& key_value, uint32_t radix_mask, size_t radix_msb,
                                                      uint32_t* bucket) {
  uint32_t key = *(reinterpret_cast<const uint32_t*>(&key_value));
  *bucket = radix_mask & key;
  *bucket = *bucket >> (radix_msb - kNumRadixBits);
}

template <>
__device__ __forceinline__ void GetRadixBucket<double>(const double& key_value, uint64_t radix_mask, size_t radix_msb,
                                                       uint64_t* bucket) {
  uint64_t key = *(reinterpret_cast<const uint64_t*>(&key_value));
  *bucket = radix_mask & key;
  *bucket = *bucket >> (radix_msb - kNumRadixBits);
}

template <typename T>
__global__ void ComputeHistogram(const T* input, uint32_t* block_local_histograms, size_t n, size_t k,
                                 size_t radix_msb) {
  const size_t index = (k * (size_t)blockDim.x * (size_t)blockIdx.x) + threadIdx.x;

  __shared__ __align__(sizeof(uint32_t)) uint32_t smem_histogram[kNumBuckets];

  uint_type<sizeof(T)> radix_mask = (1 << kNumRadixBits) - 1;
  radix_mask = radix_mask << (radix_msb - kNumRadixBits);
  uint_type<sizeof(T)> start_bucket = 0;
  uint_type<sizeof(T)> bucket = 0;

  if (threadIdx.x < kNumBuckets) {
    smem_histogram[threadIdx.x] = 0;
  }

  __syncthreads();

  if (index < n) {
    int count = 1;
    GetRadixBucket<T>(input[index], radix_mask, radix_msb, &start_bucket);

    for (size_t i = 1; i < k && index + (i * blockDim.x) < n; ++i) {
      GetRadixBucket<T>(input[index + (i * blockDim.x)], radix_mask, radix_msb, &bucket);

      if (bucket == start_bucket) {
        ++count;
      } else {
        atomicAdd(&smem_histogram[bucket], 1);
      }
    }
    atomicAdd(&smem_histogram[start_bucket], count);
  }

  __syncthreads();

  if (threadIdx.x < kWarpSize) {
    for (size_t bucket = threadIdx.x; bucket < kNumBuckets; bucket += kWarpSize) {
      block_local_histograms[kNumBuckets * blockIdx.x + bucket] = smem_histogram[bucket];
    }
  }
}

__global__ void AggregateHistogram(uint64_t* global_histogram, const uint32_t* block_local_histograms,
                                   size_t total_num_blocks, size_t blocks_per_thread) {
  typedef unsigned long long int uint64_cu;
  uint64_cu* ptr = (uint64_cu*)global_histogram;
  if (threadIdx.x < kNumBuckets) {
    size_t offset = blockIdx.x * kNumBuckets * blocks_per_thread;
    uint64_cu count = 0;
    for (size_t i = 0; i < blocks_per_thread && blockIdx.x * blocks_per_thread + i < total_num_blocks; ++i) {
      count += block_local_histograms[offset + (kNumBuckets * i) + threadIdx.x];
    }
    atomicAdd(&ptr[threadIdx.x], count);
  }
}

__global__ void CheckHistogramSkewness(uint64_t* histogram, size_t* non_empty_count) {
  size_t skew_count = 0;
  for (size_t i = 0; i < kNumBuckets; ++i) {
    skew_count += histogram[i] > 0;
  }
  *non_empty_count = skew_count;
}

template <typename T, typename V>
__global__ __launch_bounds__(kNumRadixThreads, kNumRadixBlocksPerMultiProcessor) void ScatterKeyValuePairs(
    const T* input_keys, T* output_keys, const V* input_vals, V* output_vals, const uint64_t* global_prefix_sums,
    const uint32_t* block_local_histograms, uint64_t* global_offsets, size_t n, size_t k, uint8_t radix_msb) {
  typedef unsigned long long int uint64_cu;
  const size_t thread_index = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t index = thread_index * k;

  extern __shared__ __align__(sizeof(uint64_t)) uint8_t smem_buffer[];
  const size_t smem_offset = k * kNumRadixThreads * sizeof(T);
  T* key_buffer = reinterpret_cast<T*>(smem_buffer);
  V* val_buffer = reinterpret_cast<V*>(smem_buffer + smem_offset);

  __shared__ int thread_bucket_map[kNumRadixThreads];

  __shared__ uint32_t local_histogram[kNumBuckets];
  __shared__ uint32_t local_offsets[kNumBuckets];
  __shared__ uint64_t global_offset_per_bucket[kNumBuckets];

  const uint64_t global_input_start = global_prefix_sums[0];

  if (threadIdx.x < kNumBuckets) {
    local_histogram[threadIdx.x] = block_local_histograms[kNumBuckets * blockIdx.x + threadIdx.x];
    global_offset_per_bucket[threadIdx.x] =
        global_prefix_sums[threadIdx.x] +
        atomicAdd((uint64_cu*)&global_offsets[threadIdx.x], (uint64_cu)local_histogram[threadIdx.x]);
  }

  const uint32_t buckets_to_handle_per_warp = kNumBuckets / (kNumRadixThreads / kWarpSize);

  __syncthreads();

  if (threadIdx.x == 0) {
    uint64_t prefix_sum = 0;
    for (size_t i = 0; i < kNumBuckets; ++i) {
      local_offsets[i] = prefix_sum;
      prefix_sum += local_histogram[i];
    }

    int bucket = 0;
    for (size_t i = 0; i < kNumRadixThreads; ++i) {
      if (i % kWarpSize == 0) {
        thread_bucket_map[i] = bucket;
        bucket += buckets_to_handle_per_warp;
      }
    }
  }

  __syncthreads();

  uint_type<sizeof(T)> bucket = 0;
  {
    uint_type<sizeof(T)> radix_mask = (1 << kNumRadixBits) - 1;
    radix_mask = radix_mask << (radix_msb - kNumRadixBits);

    for (size_t i = 0; i < k && index + i < n; ++i) {
      const size_t local_index = global_input_start + index + i;
      const T& key = input_keys[local_index];
      const V& value = input_vals[local_index];
      GetRadixBucket<T>(key, radix_mask, radix_msb, &bucket);

      uint64_t offset = atomicAdd(&local_offsets[bucket], 1);
      key_buffer[offset] = key;
      val_buffer[offset] = value;
    }
  }

  __syncthreads();

  for (size_t b = 0; b < buckets_to_handle_per_warp; ++b) {
    uint32_t bucket = thread_bucket_map[threadIdx.x - (threadIdx.x % kWarpSize)] + b;
    const uint32_t local_offset = local_offsets[bucket] - local_histogram[bucket];

    for (size_t i = threadIdx.x % kWarpSize; i < local_histogram[bucket]; i += kWarpSize) {
      output_keys[global_offset_per_bucket[bucket] + i] = key_buffer[local_offset + i];
    }
  }

  __syncthreads();

  for (size_t b = 0; b < buckets_to_handle_per_warp; ++b) {
    uint32_t bucket = thread_bucket_map[threadIdx.x - (threadIdx.x % kWarpSize)] + b;
    const uint32_t local_offset = local_offsets[bucket] - local_histogram[bucket];

    for (size_t i = threadIdx.x % kWarpSize; i < local_histogram[bucket]; i += kWarpSize) {
      output_vals[global_offset_per_bucket[bucket] + i] = val_buffer[local_offset + i];
    }
  }
}

__global__ void CreateMgpuStripedHistogram(uint64_t* mgpu_histograms, uint64_t* mgpu_striped_histogram,
                                           size_t num_gpus) {
  if (blockDim.x != kNumBuckets) {
    printf("ERROR: CreateStripedGpuHistogram requires exactly kNumBuckets threads per block!\n");
  }

  if (gridDim.x != num_gpus) {
    printf("ERROR: CreateStripedGpuHistogram requires exactly num_gpus thread blocks!\n");
  }

  __shared__ uint64_t smem_mgpu_histogram[kNumBuckets];

  if (threadIdx.x < kNumBuckets) {
    smem_mgpu_histogram[threadIdx.x] = (uint64_t)mgpu_histograms[blockIdx.x * kNumBuckets + threadIdx.x];
  }

  __syncthreads();

  if (threadIdx.x < kNumBuckets) {
    mgpu_striped_histogram[threadIdx.x * num_gpus + blockIdx.x] = smem_mgpu_histogram[threadIdx.x];
  }
}

__device__ void GetStartGpuDistance(uint64_t* splitters, uint64_t value, int* distance, size_t num_gpus) {
  *distance = 0;
  for (size_t i = 0; i < num_gpus; ++i) {
    *distance = i;
    if (value < splitters[i]) break;
  }
}

__device__ void GetEndGpuDistance(uint64_t* splitters, uint64_t value, int* distance, size_t num_gpus) {
  *distance = 0;
  for (size_t i = 0; i < num_gpus; ++i) {
    *distance = i;
    if (value <= splitters[i]) break;
  }
}

__global__ void DetermineBucketToGpuMapping(uint64_t* mgpu_striped_prefix_sums, int* bucket_to_gpu_map,
                                            size_t chunk_size, size_t num_fillers, size_t num_gpus, size_t epsilon) {
  if (blockDim.x != kNumBuckets) {
    printf("ERROR: DetermineBucketToGpuMapping requires exactly kNumBuckets threads per block!\n");
  }

  if (gridDim.x != 1) {
    printf("ERROR: DetermineBucketToGpuMapping requires exactly 1 thread block!\n");
  }

  __shared__ uint64_t smem_mgpu_striped_prefix_sums[kNumBuckets + 1];
  __shared__ uint64_t splitters[kMaxNumGpus];

  if (threadIdx.x == 0) {
    smem_mgpu_striped_prefix_sums[kNumBuckets] = mgpu_striped_prefix_sums[kNumBuckets * num_gpus];

    for (size_t g = 0; g < num_gpus; ++g) {
      splitters[g] = (g + 1) * chunk_size;
    }
  }

  if (threadIdx.x < kNumBuckets) {
    smem_mgpu_striped_prefix_sums[threadIdx.x] = mgpu_striped_prefix_sums[threadIdx.x * num_gpus];
  }

  __syncthreads();

  if (threadIdx.x < kNumBuckets) {
    size_t bucket_size = smem_mgpu_striped_prefix_sums[threadIdx.x + 1] - smem_mgpu_striped_prefix_sums[threadIdx.x];

    if (bucket_size > 0) {
      int start_gpu = 0;
      GetStartGpuDistance(&splitters[0], smem_mgpu_striped_prefix_sums[threadIdx.x], &start_gpu, num_gpus);

      int end_gpu = 0;
      GetEndGpuDistance(&splitters[0], smem_mgpu_striped_prefix_sums[threadIdx.x + 1], &end_gpu, num_gpus);

      if (end_gpu - start_gpu == 1) {
        int start_overflow = splitters[start_gpu] - smem_mgpu_striped_prefix_sums[threadIdx.x];
        int end_overflow = GetAbsolute(splitters[start_gpu] - smem_mgpu_striped_prefix_sums[threadIdx.x + 1]);

        if (start_overflow > end_overflow) {
          if (end_overflow <= epsilon) {
            end_gpu--;
          }
        } else {
          if (start_overflow <= epsilon) {
            ++start_gpu;
          }
        }
      } else if (end_gpu - start_gpu >= 2) {
        int start_overflow = splitters[start_gpu] - smem_mgpu_striped_prefix_sums[threadIdx.x];
        int end_overflow = GetAbsolute(splitters[end_gpu - 1] - smem_mgpu_striped_prefix_sums[threadIdx.x + 1]);

        if (start_overflow <= epsilon) {
          ++start_gpu;
        }

        if (end_overflow <= epsilon) {
          end_gpu--;
        }
      }

      for (size_t g = start_gpu; g <= end_gpu; ++g) {
        bucket_to_gpu_map[(threadIdx.x * num_gpus) + g - start_gpu] = g;
      }
    }
  }
}
