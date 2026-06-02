// [AJB] PinnedVector: cudaMallocHost分配的std::vector, H2D传输零拷贝
// 对比普通vector, pinned memory的PCIe传输速度可以快2-3x
// 但总量受限于系统RAM的pinned比例(通常<50%)
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

#include <cstdio>
// [AJB] pinned memory使用量跟踪
static inline void ajb_report_pinned_usage(size_t count, size_t elem_size, const char* tag) {
    size_t bytes = count * elem_size;
    fprintf(stderr, "[AJB_MEM][Pinned] %s: %zu elements x %zu = %.2f MB\n",
            tag, count, elem_size, bytes / 1048576.0);
}
