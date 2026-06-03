#pragma once

#include <vector>
#include <cstdio>

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

    // Upstream: std::map<int,int> 做GPU id→index映射.
    // AJB: flat array替代map, 与ResourceManager保持一致的优化策略.
    gpu_index_flat_.resize(256, -1);
    for (size_t g = 0; g < gpus_.size(); ++g) {
      int dev = gpus_[g];
      if (dev >= 0 && dev < (int)gpu_index_flat_.size())
        gpu_index_flat_[dev] = (int)g;
    }

    const size_t max_num_partition_passes = sizeof(T);
    max_histograms_per_gpu_ = (gpus_.size() - 1) * (max_num_partition_passes - 1) + 1;

    fprintf(stderr, "[DEBUG][DeviceContainers] gpus=%zu max_hist/gpu=%zu "
            "epsilon=%zu gamma=%zu chunk=%zu\n",
            gpus_.size(), max_histograms_per_gpu_, epsilon_, gamma_, chunk_size);

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
    const int i = GpuIdx(gpu);
    auto bucket_id_iter = histogram_maps_[i].find(bucket_id);
    if (bucket_id_iter != histogram_maps_[i].end()) {
      return &histogram_buffers_[i][bucket_id_iter->second];
    }
    return nullptr;
  }

  void AssignNewHistogramBuffer(int gpu, const BucketId& bucket_id) {
    const int i = GpuIdx(gpu);
    size_t& next_index = next_histogram_index_[i];
    if (next_index < max_histograms_per_gpu_) {
      histogram_maps_[i].emplace(bucket_id, next_index);
      ++next_index;
    }
  }

  // Upstream: 无法知道histogram buffer的使用率.
  // AJB: 暴露使用率 — 接近100%说明partition pass太多, 数据skew严重.
  double histogram_utilization(int gpu) const {
    const int i = GpuIdx(gpu);
    if (max_histograms_per_gpu_ == 0) return 0.0;
    return static_cast<double>(next_histogram_index_[i]) / max_histograms_per_gpu_;
  }

  inline size_t GetEpsilon() const { return epsilon_; }
  inline size_t GetGamma() const { return gamma_; }

 private:
  int GpuIdx(int gpu) const { return gpu_index_flat_[gpu]; }

  static constexpr double kEpsilon = 0.005;
  static constexpr double kGamma = 0.01;

  std::vector<int> gpus_;
  std::vector<int> gpu_index_flat_;  // 替代upstream的std::map<int,int>

  std::vector<std::vector<DeviceHistograms>> histogram_buffers_;
  std::vector<std::map<BucketId, size_t, CompareBucketIds>> histogram_maps_;
  std::vector<size_t> next_histogram_index_;
  size_t max_histograms_per_gpu_;

  const size_t epsilon_;
  const size_t gamma_;
};
