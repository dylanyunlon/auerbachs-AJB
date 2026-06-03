#pragma once

#include <cstring>
#include <cstdio>

#include "common/host_allocator.cuh"
#include "common/pinned_vector.cuh"
#include "constants.cuh"

class HostHistograms {
 public:
  HostHistograms(size_t num_gpus, HostAllocator& host_allocator)
      : num_gpus_(num_gpus), host_allocator_(host_allocator) {

    // Upstream: 5个独立的alloc+memset.
    // AJB: 跟踪总分配量, 与DeviceHistograms保持同样的诊断能力.
    total_alloc_bytes_ = 0;

    const size_t non_empty_count_size = sizeof(size_t);
    non_empty_count_ = reinterpret_cast<size_t*>(host_allocator_.allocate(non_empty_count_size));
    std::memset(non_empty_count_, 0, non_empty_count_size);
    total_alloc_bytes_ += non_empty_count_size;

    const size_t global_histogram_size = kNumBuckets * sizeof(uint64_t);
    global_histogram_ = reinterpret_cast<uint64_t*>(host_allocator_.allocate(global_histogram_size));
    std::memset(global_histogram_, 0, global_histogram_size);
    total_alloc_bytes_ += global_histogram_size;

    const size_t global_prefix_sums_size = kNumBuckets * sizeof(uint64_t);
    global_prefix_sums_ = reinterpret_cast<uint64_t*>(host_allocator_.allocate(global_prefix_sums_size));
    std::memset(global_prefix_sums_, 0, global_prefix_sums_size);
    total_alloc_bytes_ += global_prefix_sums_size;

    const size_t mgpu_striped_histogram_size = ((kNumBuckets * num_gpus) + 1) * sizeof(uint64_t);
    mgpu_striped_histogram_ = reinterpret_cast<uint64_t*>(host_allocator_.allocate(mgpu_striped_histogram_size));
    std::memset(mgpu_striped_histogram_, 0, mgpu_striped_histogram_size);
    total_alloc_bytes_ += mgpu_striped_histogram_size;

    const size_t bucket_to_gpu_map_size = kNumBuckets * num_gpus * sizeof(int);
    bucket_to_gpu_map_ = reinterpret_cast<int*>(host_allocator_.allocate(bucket_to_gpu_map_size));
    std::memset(bucket_to_gpu_map_, -1, bucket_to_gpu_map_size);
    total_alloc_bytes_ += bucket_to_gpu_map_size;
  }

  ~HostHistograms() {
    host_allocator_.deallocate(reinterpret_cast<uint8_t*>(non_empty_count_));
    host_allocator_.deallocate(reinterpret_cast<uint8_t*>(global_histogram_));
    host_allocator_.deallocate(reinterpret_cast<uint8_t*>(global_prefix_sums_));
    host_allocator_.deallocate(reinterpret_cast<uint8_t*>(mgpu_striped_histogram_));
    host_allocator_.deallocate(reinterpret_cast<uint8_t*>(bucket_to_gpu_map_));
  }

  size_t* GetNonEmptyCount() { return non_empty_count_; }
  uint64_t* GetGlobalHistogram() { return global_histogram_; }
  uint64_t* GetGlobalPrefixSums() { return global_prefix_sums_; }
  uint64_t* GetMgpuStripedHistogram() { return mgpu_striped_histogram_; }
  int* GetBucketToGpuMap() { return bucket_to_gpu_map_; }

  size_t total_bytes() const { return total_alloc_bytes_; }

  // 检查global_histogram是否为全零 — 表示这个pass没有产生有效bucket
  bool is_empty() const {
    for (size_t i = 0; i < kNumBuckets; ++i) {
      if (global_histogram_[i] > 0) return false;
    }
    return true;
  }

  // 非空bucket计数 — 衡量数据在这个radix pass上的分散程度
  // 1 = 所有数据落入同一bucket(skew极端), 256 = 均匀分布
  size_t num_nonempty_buckets() const {
    size_t count = 0;
    for (size_t i = 0; i < kNumBuckets; ++i) {
      if (global_histogram_[i] > 0) ++count;
    }
    return count;
  }

 private:
  const size_t num_gpus_;
  HostAllocator& host_allocator_;
  size_t total_alloc_bytes_;

  size_t* non_empty_count_;
  uint64_t* global_histogram_;
  uint64_t* global_prefix_sums_;
  uint64_t* mgpu_striped_histogram_;
  int* bucket_to_gpu_map_;
};
