// [AJB] hybrid_sort/radix_sort/device_histograms.cuh: GPU侧histogram计算
#include <cstdio>
#pragma once

#include <thrust/device_vector.h>

#include "common/device_allocator.cuh"
#include "common/stream_pool.cuh"
#include "constants.cuh"

class DeviceHistograms {
 public:
  DeviceHistograms(int gpu, size_t num_gpus, size_t num_blocks, DeviceAllocator& device_allocator,
                   StreamPool& stream_pool)
      : gpu_(gpu), device_allocator_(device_allocator), stream_pool_(stream_pool) {
    CheckCudaError(cudaSetDevice(gpu));

    const size_t non_empty_count_size = sizeof(size_t);
    non_empty_count_ = reinterpret_cast<size_t*>(device_allocator_.allocate(non_empty_count_size));
    CheckCudaError(cudaMemsetAsync(non_empty_count_, 0, non_empty_count_size, stream_pool_.GetStream(0)));

    const size_t global_histogram_size = kNumBuckets * sizeof(uint64_t);
    global_histogram_ = reinterpret_cast<uint64_t*>(device_allocator_.allocate(global_histogram_size));
    CheckCudaError(cudaMemsetAsync(global_histogram_, 0, global_histogram_size, stream_pool_.GetStream(0)));

    const size_t global_prefix_sums_size = kNumBuckets * sizeof(uint64_t);
    global_prefix_sums_ = reinterpret_cast<uint64_t*>(device_allocator_.allocate(global_prefix_sums_size));
    CheckCudaError(cudaMemsetAsync(global_prefix_sums_, 0, global_prefix_sums_size, stream_pool_.GetStream(0)));

    const size_t global_scatter_offsets_size = kNumBuckets * sizeof(uint64_t);
    global_scatter_offsets_ = reinterpret_cast<uint64_t*>(device_allocator_.allocate(global_scatter_offsets_size));
    CheckCudaError(cudaMemsetAsync(global_scatter_offsets_, 0, global_scatter_offsets_size, stream_pool_.GetStream(0)));

    const size_t block_local_histograms_size = kNumBuckets * num_blocks * sizeof(uint32_t);
    block_local_histograms_ = reinterpret_cast<uint32_t*>(device_allocator_.allocate(block_local_histograms_size));
    CheckCudaError(cudaMemsetAsync(block_local_histograms_, 0, block_local_histograms_size, stream_pool_.GetStream(0)));

    const size_t mgpu_histograms_size = kNumBuckets * num_gpus * sizeof(uint64_t);
    mgpu_histograms_ = reinterpret_cast<uint64_t*>(device_allocator_.allocate(mgpu_histograms_size));
    CheckCudaError(cudaMemsetAsync(mgpu_histograms_, 0, mgpu_histograms_size, stream_pool_.GetStream(0)));

    const size_t mgpu_striped_histogram_size = ((kNumBuckets * num_gpus) + 1) * sizeof(uint64_t);
    mgpu_striped_histogram_ = reinterpret_cast<uint64_t*>(device_allocator_.allocate(mgpu_striped_histogram_size));
    CheckCudaError(cudaMemsetAsync(mgpu_striped_histogram_, 0, mgpu_striped_histogram_size, stream_pool_.GetStream(0)));

    const size_t bucket_to_gpu_map_size = kNumBuckets * num_gpus * sizeof(int);
    bucket_to_gpu_map_ = reinterpret_cast<int*>(device_allocator_.allocate(bucket_to_gpu_map_size));
    CheckCudaError(cudaMemsetAsync(bucket_to_gpu_map_, -1, bucket_to_gpu_map_size, stream_pool_.GetStream(0)));
  }

  ~DeviceHistograms() {
    CheckCudaError(cudaSetDevice(gpu_));

    device_allocator_.deallocate(reinterpret_cast<uint8_t*>(non_empty_count_));
    device_allocator_.deallocate(reinterpret_cast<uint8_t*>(global_histogram_));
    device_allocator_.deallocate(reinterpret_cast<uint8_t*>(global_prefix_sums_));
    device_allocator_.deallocate(reinterpret_cast<uint8_t*>(global_scatter_offsets_));
    device_allocator_.deallocate(reinterpret_cast<uint8_t*>(block_local_histograms_));
    device_allocator_.deallocate(reinterpret_cast<uint8_t*>(mgpu_histograms_));
    device_allocator_.deallocate(reinterpret_cast<uint8_t*>(mgpu_striped_histogram_));
    device_allocator_.deallocate(reinterpret_cast<uint8_t*>(bucket_to_gpu_map_));
  }

  size_t* GetNonEmptyCount() const { return non_empty_count_; }

  uint64_t* GetGlobalHistogram() { return global_histogram_; }

  uint64_t* GetGlobalPrefixSums() { return global_prefix_sums_; }

  uint64_t* GetGlobalScatterOffsets() { return global_scatter_offsets_; }

  uint32_t* GetBlockLocalHistograms() { return block_local_histograms_; }

  uint64_t* GetMgpuHistograms() { return mgpu_histograms_; }

  uint64_t* GetMgpuStripedHistogram() { return mgpu_striped_histogram_; }

  int* GetBucketToGpuMap() { return bucket_to_gpu_map_; }

 private:
  const int gpu_;

  DeviceAllocator& device_allocator_;
  StreamPool& stream_pool_;

  size_t* non_empty_count_;
  uint64_t* global_histogram_;
  uint64_t* global_prefix_sums_;
  uint64_t* global_scatter_offsets_;
  uint32_t* block_local_histograms_;
  uint64_t* mgpu_histograms_;
  uint64_t* mgpu_striped_histogram_;
  int* bucket_to_gpu_map_;
};

// [AJB] hybrid_sort_radix_sort_device_histograms 诊断报告
static inline void ajb_report_hybrid_sort_radix_sort_device_histograms(size_t n, double elapsed_ms, const char* phase) {
    fprintf(stderr, "[AJB_TIMER][hybrid_sort_radix_sort_device_histograms] %s: n=%zu elapsed=%.3fms throughput=%.2f M/s\n",
            phase, n, elapsed_ms, elapsed_ms > 0 ? n / elapsed_ms / 1000.0 : 0.0);
}
