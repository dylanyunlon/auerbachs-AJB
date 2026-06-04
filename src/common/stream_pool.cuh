#pragma once
#include "common/ajb_debug_infra.cuh"

#include <vector>
#include <cassert>
#include <cstdio>
#include <algorithm>

#include "error_utilities.cuh"

class StreamPool {
 public:
  StreamPool() = default;

  ~StreamPool() { Free(); }

  void Initialize(size_t num_streams) {
#ifdef DEBUG_BUILD
    printf("[StreamPool] Initialize with %lu streams.\n", num_streams);
#endif
    if (streams_.size() < num_streams) {
      Free();
      streams_.resize(num_streams);
      use_count_.assign(num_streams, 0);
      for (cudaStream_t& stream : streams_) {
        CheckCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      }
      capacity_ = num_streams;
    }
  }

  void Free() {
    for (cudaStream_t& stream : streams_) {
      CheckCudaError(cudaStreamDestroy(stream));
    }
    streams_.clear();
    streams_.shrink_to_fit();
    use_count_.clear();
    capacity_ = 0;
  }

  // 带bounds-check和负载追踪的stream获取
  cudaStream_t GetStream(size_t index) {
    assert(index < streams_.size() && "StreamPool: index out of bounds");
    use_count_[index]++;
    return streams_[index];
  }

  // round-robin分配: 自动选择使用次数最少的stream
  cudaStream_t GetLeastUsed() {
    assert(!streams_.empty() && "StreamPool: no streams allocated");
    size_t min_idx = 0;
    size_t min_val = use_count_[0];
    for (size_t i = 1; i < use_count_.size(); i++) {
      if (use_count_[i] < min_val) {
        min_val = use_count_[i];
        min_idx = i;
      }
    }
    use_count_[min_idx]++;
    return streams_[min_idx];
  }

  size_t Size() const { return streams_.size(); }

  // [AJB_BP] 输出每个stream的使用次数, 判断负载是否均衡
  void DumpUsage(int device_id = 0) const {
    if (streams_.empty()) return;
    size_t total = 0, peak = 0;
    for (size_t c : use_count_) {
      total += c;
      if (c > peak) peak = c;
    }
    double avg = streams_.size() > 0 ? (double)total / streams_.size() : 0.0;
    double imbalance = avg > 0.0 ? (double)peak / avg : 0.0;
    fprintf(stderr, "[AJB_BP][StreamPool] dev=%d streams=%zu total_ops=%zu avg=%.1f peak=%zu imbalance=%.2f\n",
            device_id, streams_.size(), total, avg, peak, imbalance);
  }

 private:
  std::vector<cudaStream_t> streams_;
  std::vector<size_t> use_count_;
  size_t capacity_ = 0;
};
