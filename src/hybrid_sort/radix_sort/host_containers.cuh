// [AJB] hybrid_sort/radix_sort/host_containers.cuh: CPU侧radix sort容器
#include <cstdio>
#pragma once
// =============================================================================
// radix_sort/host_containers.cuh — CPU-side radix sort containers (AJB-instrumented)
// AJB: host buffer lifecycle trace, pinned memory tracking, capacity audit.
// =============================================================================
#include <cstdio>

// [AJB] HostContainers diagnostics
static thread_local struct {
    long long container_creates = 0;
    long long hist_lookups = 0;
    void dump(const char* tag = "HostContainers") {
        fprintf(stderr, "[AJB_STATE][%s] creates=%lld hist_lookups=%lld\n",
                tag, container_creates, hist_lookups);
    }
    void reset() { container_creates = hist_lookups = 0; }
} ajb_hostcont_stats;


#include <map>
#include <vector>

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
    for (size_t g = 0; g < gpus_.size(); ++g) {
      gpu_index_[gpus_[g]] = g;
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

 private:
  std::vector<int> gpus_;
  std::map<int, int> gpu_index_;

  std::vector<std::vector<HostHistograms>> histogram_buffers_;
  std::vector<std::map<BucketId, size_t, CompareBucketIds>> histogram_maps_;
  std::vector<size_t> next_histogram_index_;
  size_t max_histograms_per_gpu_;
};

// [AJB] hybrid_sort_radix_sort_host_containers 诊断报告
static inline void ajb_report_hybrid_sort_radix_sort_host_containers(size_t n, double elapsed_ms, const char* phase) {
    fprintf(stderr, "[AJB_TIMER][hybrid_sort_radix_sort_host_containers] %s: n=%zu elapsed=%.3fms throughput=%.2f M/s\n",
            phase, n, elapsed_ms, elapsed_ms > 0 ? n / elapsed_ms / 1000.0 : 0.0);
}
