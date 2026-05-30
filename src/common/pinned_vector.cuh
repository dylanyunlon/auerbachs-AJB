#pragma once

#include <vector>

#include "error_utilities.cuh"

template <typename T>
struct PinnedAllocator {
 public:
  using value_type = T;

  value_type* allocate(size_t num_elements) {
    value_type* begin_pointer;
    CheckCudaError(cudaMallocHost(&begin_pointer, num_elements * sizeof(value_type)));
    return begin_pointer;
  }

  void deallocate(value_type* begin_pointer, size_t num_elements) { CheckCudaError(cudaFreeHost(begin_pointer)); }

  bool operator==(const PinnedAllocator& other) const { return true; }
};

template <typename T>
using PinnedVector = std::vector<T, PinnedAllocator<T>>;
