#pragma once
#include "common/ajb_debug_infra.cuh"

#include <vector>

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
      for (cudaStream_t& stream : streams_) {
        CheckCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      }
    }
  }

  void Free() {
    for (cudaStream_t& stream : streams_) {
      CheckCudaError(cudaStreamDestroy(stream));
    }
    streams_.clear();
    streams_.shrink_to_fit();
  }

  cudaStream_t GetStream(size_t index) { if (ajb_stream_use_count_.size() > index)
        ajb_stream_use_count_[index]++;
    return streams_[index]; }

 private:
  mutable std::vector<size_t> ajb_stream_use_count_;
  std::vector<cudaStream_t> streams_;
};

// [AJB] stream pool诊断: 检查stream数量是否与GPU SM数量匹配
#include <cstdio>
static inline void ajb_report_stream_pool(size_t num_streams, int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    fprintf(stderr, "[AJB_STATE][StreamPool] device=%d streams=%zu SMs=%d (%.1f streams/SM)\n",
            device_id, num_streams, prop.multiProcessorCount,
            num_streams > 0 ? (double)num_streams / prop.multiProcessorCount : 0.0);
}
