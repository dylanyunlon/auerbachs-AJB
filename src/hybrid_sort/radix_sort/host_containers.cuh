#pragma once

#include <vector>
#include <cstdio>

#include "buckets.cuh"
#include "constants.cuh"
#include "host_histograms.cuh"
#include "hybrid_sort/resource_manager.cuh"

template <typename T, typename V>
class HostContainers {
 public:
  HostContainers(const std::vector<int>& gpus, ResourceManager<T, V>& resource_manager)
      : gpus_(gpus),
        histogram_buffers_(gpus.size()),
        histogram_maps_(gpus.size()),
        next_histogram_index_(gpus.size(), 0) {

    // Upstream: std::map<int,int> → AJB: flat array
    gpu_index_flat_.resize(256, -1);
    for (size_t g = 0; g < gpus_.size(); ++g) {
      int dev = gpus_[g];
      if (dev >= 0 && dev < (int)gpu_index_flat_.size())
        gpu_index_flat_[dev] = (int)g;
    }

    const size_t num_partition_passes = sizeof(T);
    max_histograms_per_gpu_ = (gpus_.size() - 1) * (num_partition_passes - 1) + 1;

#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      histogram_buffers_[g].reserve(max_histograms_per_gpu_);

      for (size_t i = 0; i < max_histograms_per_gpu_; ++i) {
        histogram_buffers_[g].emplace_back(gpus_.size(), resource_manager.GetHostAllocator(gpus_[g]));
      }
    }
  }

  ~HostContainers() {
#pragma omp parallel for num_threads(gpus_.size())
    for (size_t g = 0; g < gpus_.size(); ++g) {
      histogram_buffers_[g].clear();
      histogram_buffers_[g].shrink_to_fit();
    }
  }

  HostHistograms* GetHistograms(int gpu, const BucketId& bucket_id) {
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

  // host端histogram使用率
  double histogram_utilization(int gpu) const {
    const int i = GpuIdx(gpu);
    if (max_histograms_per_gpu_ == 0) return 0.0;
    return static_cast<double>(next_histogram_index_[i]) / max_histograms_per_gpu_;
  }

  // 所有GPU的总histogram使用数
  size_t total_assigned() const {
    size_t sum = 0;
    for (size_t g = 0; g < gpus_.size(); ++g)
      sum += next_histogram_index_[g];
    return sum;
  }

 private:
  int GpuIdx(int gpu) const { return gpu_index_flat_[gpu]; }

  std::vector<int> gpus_;
  std::vector<int> gpu_index_flat_;  // 替代upstream的std::map<int,int>

  std::vector<std::vector<HostHistograms>> histogram_buffers_;
  std::vector<std::map<BucketId, size_t, CompareBucketIds>> histogram_maps_;
  std::vector<size_t> next_histogram_index_;
  size_t max_histograms_per_gpu_;
};
