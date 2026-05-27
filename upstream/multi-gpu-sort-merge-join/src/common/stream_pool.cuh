#pragma once

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

  cudaStream_t GetStream(size_t index) { return streams_[index]; }

 private:
  std::vector<cudaStream_t> streams_;
};
