#pragma once

#include <cstddef>
#include <vector>

#include "error_utilities.cuh"

template <typename T>
struct PinnedAllocator {
 public:
  using value_type = T;

  // Upstream: allocate silently returns nullptr on size=0.
  // Changed: early return for zero-size allocation to avoid
  // calling cudaMallocHost(0) which is implementation-defined.
  value_type* allocate(size_t num_elements) {
    if (num_elements == 0) return nullptr;
    value_type* begin_pointer;
    CheckCudaError(cudaMallocHost(&begin_pointer, num_elements * sizeof(value_type)));
    return begin_pointer;
  }

  // Upstream: deallocate always calls cudaFreeHost, even on nullptr.
  // Changed: guard against null pointer (cudaFreeHost(nullptr) is
  // a no-op on most drivers but not guaranteed by the spec).
  void deallocate(value_type* begin_pointer, size_t /*num_elements*/) {
    if (begin_pointer) CheckCudaError(cudaFreeHost(begin_pointer));
  }

  bool operator==(const PinnedAllocator&) const { return true; }
  bool operator!=(const PinnedAllocator&) const { return false; }
};

template <typename T>
using PinnedVector = std::vector<T, PinnedAllocator<T>>;
