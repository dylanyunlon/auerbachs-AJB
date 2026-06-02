// [AJB] hybrid_sort/radix_sort/device_containers.cuh: GPU侧radix sort数据容器
#include <cstdio>
#pragma once

#include <map>
#include <vector>

#include <cub/cub.cuh>

#include "buckets.cuh"
#include "constants.cuh"
#include "device_histograms.cuh"
#include "hybrid_sort/resource_manager.cuh"

template <typename T, typename V>
class DeviceContainers {
 public:
  DeviceContainers(const std::vector<int>& gpus, size_t chunk_size, size_t num_blocks,
                   ResourceManager<T, V>& resource_manager)
      : gpus_(gpus),
        histogram_buffers_(gpus.size()),
        histogram_maps_(gpus.size()),
        next_histogram_index_(gpus.size(), 0),
        epsilon_(kEpsilon * chunk_size),
        gamma_(kGamma * chunk_size) {
    for (size_t g = 0; g < gpus_.size(); ++g) {
      gpu_index_[gpus_[g]] = g;
    }

    const size_t max_num_partition_passes = sizeof(T);
    max_histograms_per_gpu_ = (gpus_.size() - 1) * (max_num_partition_passes - 1) + 1;

#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));

      histogram_buffers_[g].reserve(max_histograms_per_gpu_);
      for (size_t i = 0; i < max_histograms_per_gpu_; ++i) {
        histogram_buffers_[g].emplace_back(gpus_[g], gpus_.size(), num_blocks,
                                           resource_manager.GetDeviceAllocator(gpus_[g]),
                                           resource_manager.GetStreamPool(gpus_[g]));
      }
    }
  }

  ~DeviceContainers() {
#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      CheckCudaError(cudaSetDevice(gpus_[g]));

      histogram_buffers_[g].clear();
      histogram_buffers_[g].shrink_to_fit();
    }
  }

  DeviceHistograms* GetHistograms(int gpu, const BucketId& bucket_id) {
    const int i = gpu_index_[gpu];
    auto bucket_id_iter = histogram_maps_[i].find(bucket_id);
    if (bucket_id_iter != histogram_maps_[i].end()) {
      return &histogram_buffers_[i][bucket_id_iter->second];
    }
    return nullptr;
  }

  void AssignNewHistogramBuffer(int gpu, const BucketId& bucket_id) {
    const int i = gpu_index_[gpu];
    size_t& next_index = next_histogram_index_[i];
    if (next_index < max_histograms_per_gpu_) {
      histogram_maps_[i].emplace(bucket_id, next_index);
      ++next_index;
    }
  }

  inline size_t GetEpsilon() const { return epsilon_; }

  inline size_t GetGamma() const { return gamma_; }

 private:
  static constexpr double kEpsilon = 0.005;
  static constexpr double kGamma = 0.01;

  std::vector<int> gpus_;
  std::map<int, int> gpu_index_;

  std::vector<std::vector<DeviceHistograms>> histogram_buffers_;
  std::vector<std::map<BucketId, size_t, CompareBucketIds>> histogram_maps_;
  std::vector<size_t> next_histogram_index_;
  size_t max_histograms_per_gpu_;

  const size_t epsilon_;
  const size_t gamma_;
};

// [AJB] hybrid_sort_radix_sort_device_containers 诊断报告
static inline void ajb_report_hybrid_sort_radix_sort_device_containers(size_t n, double elapsed_ms, const char* phase) {
    fprintf(stderr, "[AJB_TIMER][hybrid_sort_radix_sort_device_containers] %s: n=%zu elapsed=%.3fms throughput=%.2f M/s\n",
            phase, n, elapsed_ms, elapsed_ms > 0 ? n / elapsed_ms / 1000.0 : 0.0);
}
