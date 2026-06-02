// [AJB] HostAllocator: pinned host内存arena, 16字节对齐, 用于H2D/D2H传输buffer
// cudaMallocHost分配的内存可以实现zero-copy DMA传输
#pragma once

#include <string>

#include "error_utilities.cuh"
#include "memory_allocator.cuh"

struct HostAllocator : MemoryAllocator {
  ~HostAllocator() { Free(); }

  size_t GetAlignment() const override { return kByteAlignment; }

  const char* GetType() const override { return kType.c_str(); }

 private:
  void InitializeMemory(value_type** pointer, size_t max_bytes) override {
    CheckCudaError(cudaMallocHost(pointer, max_bytes));
  }

  void FreeMemory(value_type* pointer) override { CheckCudaError(cudaFreeHost(pointer)); }

  static constexpr size_t kByteAlignment = 16;
  const std::string kType = "HostAllocator";
};

#ifdef AJB_TRACE_ALLOC
#include <cstdio>
static inline void ajb_report_host_alloc(size_t bytes, const char* tag) {
    fprintf(stderr, "[AJB_MEM][HostAlloc] %s: %zu bytes (%.2f MB)\n", tag, bytes, bytes / 1048576.0);
}
#else
static inline void ajb_report_host_alloc(size_t, const char*) {}
#endif
