#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include <cub/cub.cuh>
#include <thrust/sort.h>

#include "common/device_allocator.cuh"
#include "common/host_allocator.cuh"
#include "common/math_utilities.cuh"
#include "common/stream_pool.cuh"

template <typename T, typename V>
class ResourceManager {
 public:
  ResourceManager(const std::vector<int>& gpus, size_t num_streams, size_t chunk_size,
                  std::vector<HostAllocator>& host_allocators, std::vector<DeviceAllocator>& device_allocators,
                  std::vector<StreamPool>& stream_pools)
      : gpus_(gpus),
        num_streams_(num_streams),
        host_allocators_(host_allocators),
        device_allocators_(device_allocators),
        stream_pools_(stream_pools),
        keys_double_buffers_(gpus.size()),
        values_double_buffers_(gpus.size()) {
    for (size_t g = 0; g < gpus_.size(); ++g) {
      gpu_index_[gpus_[g]] = g;
    }

#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));

      host_allocators_[g].Initialize(kHostMemory);

      size_t free_global_memory, total_global_memory;
      CheckCudaError(cudaMemGetInfo(&free_global_memory, &total_global_memory));
      device_allocators_[g].Initialize(kDeviceMemoryUtilization * free_global_memory);

      stream_pools_[g].Initialize(num_streams_);

      const size_t adjusted_chunk_size = kChunkSizeMultiplier * chunk_size;

      keys_double_buffers_[g].d_buffers[0] =
          reinterpret_cast<T*>(device_allocators_[g].allocate(adjusted_chunk_size * sizeof(T)));
      keys_double_buffers_[g].d_buffers[1] =
          reinterpret_cast<T*>(device_allocators_[g].allocate(adjusted_chunk_size * sizeof(T)));
      values_double_buffers_[g].d_buffers[0] =
          reinterpret_cast<V*>(device_allocators_[g].allocate(adjusted_chunk_size * sizeof(V)));
      values_double_buffers_[g].d_buffers[1] =
          reinterpret_cast<V*>(device_allocators_[g].allocate(adjusted_chunk_size * sizeof(V)));
    }

#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));

      for (size_t s = 0; s < num_streams_; ++s) {
        int* host_elements =
            reinterpret_cast<int*>(host_allocators_[g].allocate(kNumInitializationElements * sizeof(int)));
        int* device_elements =
            reinterpret_cast<int*>(device_allocators_[g].allocate(kNumInitializationElements * sizeof(int)));

        for (size_t i = 0; i < kNumInitializationElements; ++i) {
          host_elements[i] = kNumInitializationElements - i;
        }

        CheckCudaError(cudaMemcpyAsync(device_elements, host_elements, kNumInitializationElements * sizeof(int),
                                       cudaMemcpyHostToDevice, stream_pools_[g].GetStream(s)));

        thrust::sort(thrust::cuda::par_nosync(device_allocators_[g]).on(stream_pools_[g].GetStream(s)), device_elements,
                     device_elements + kNumInitializationElements);

        CheckCudaError(cudaMemcpyAsync(host_elements, device_elements, kNumInitializationElements * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_pools_[g].GetStream(s)));

        CheckCudaError(cudaStreamSynchronize(stream_pools_[g].GetStream(s)));

        host_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(host_elements));
        device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(device_elements));
      }
    }
  }

  ~ResourceManager() {
#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));

      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(keys_double_buffers_[g].Current()));
      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(keys_double_buffers_[g].Alternate()));
      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(values_double_buffers_[g].Current()));
      device_allocators_[g].deallocate(reinterpret_cast<uint8_t*>(values_double_buffers_[g].Alternate()));
    }
  }

  HostAllocator& GetHostAllocator(int gpu) { return host_allocators_[gpu_index_[gpu]]; }

  DeviceAllocator& GetDeviceAllocator(int gpu) { return device_allocators_[gpu_index_[gpu]]; }

  StreamPool& GetStreamPool(int gpu) { return stream_pools_[gpu_index_[gpu]]; }

  cub::DoubleBuffer<T>& GetKeysBuffer(int gpu) { return keys_double_buffers_[gpu_index_[gpu]]; }

  cub::DoubleBuffer<V>& GetValuesBuffer(int gpu) { return values_double_buffers_[gpu_index_[gpu]]; }

  void FlipBuffers(int gpu) {
    values_double_buffers_[gpu_index_[gpu]].selector ^= 1;
    keys_double_buffers_[gpu_index_[gpu]].selector ^= 1;
  }

  T* GetKeys(int gpu) { return keys_double_buffers_[gpu_index_[gpu]].Current(); }

  V* GetValues(int gpu) { return values_double_buffers_[gpu_index_[gpu]].Current(); }

  T* GetOtherKeys(int gpu) { return keys_double_buffers_[gpu_index_[gpu]].Alternate(); }

  V* GetOtherValues(int gpu) { return values_double_buffers_[gpu_index_[gpu]].Alternate(); }

 private:
  static constexpr size_t kHostMemory = 64_MB;
  static constexpr double kDeviceMemoryUtilization = 0.98;
  static constexpr double kChunkSizeMultiplier = 1.01;

  static constexpr size_t kNumInitializationElements = 10;

  std::vector<int> gpus_;
  std::map<int, int> gpu_index_;

  const size_t num_streams_;

  std::vector<HostAllocator>& host_allocators_;
  std::vector<DeviceAllocator>& device_allocators_;
  std::vector<StreamPool>& stream_pools_;

  std::vector<cub::DoubleBuffer<T>> keys_double_buffers_;
  std::vector<cub::DoubleBuffer<V>> values_double_buffers_;
};
