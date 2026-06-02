// [AJB] DeviceAllocator: GPU显存arena, 128字节对齐, 用于sort/join所有临时buffer
// 所有GPU侧的临时内存都通过这个allocator分配, OOM时这里是第一调试点
#pragma once

#include <string>

#include "error_utilities.cuh"
#include "memory_allocator.cuh"

struct DeviceAllocator : MemoryAllocator {
  ~DeviceAllocator() { Free(); }

  size_t GetAlignment() const override { return kByteAlignment; }

  const char* GetType() const override { return kType.c_str(); }

 private:
  void InitializeMemory(value_type** pointer, size_t max_bytes) override {
    CheckCudaError(cudaMalloc(pointer, max_bytes));
  }

  void FreeMemory(value_type* pointer) override { CheckCudaError(cudaFree(pointer)); }

  static constexpr size_t kByteAlignment = 128;
  const std::string kType = "DeviceAllocator";
};

// [AJB] 显存分配跟踪: 在debug模式下可以打开来追踪每次alloc/free
#ifdef AJB_TRACE_ALLOC
#include <cstdio>
static inline void ajb_report_device_alloc(size_t bytes, const char* tag) {
    fprintf(stderr, "[AJB_MEM][DeviceAlloc] %s: %zu bytes (%.2f MB)\n", tag, bytes, bytes / 1048576.0);
}
#else
static inline void ajb_report_device_alloc(size_t, const char*) {}
#endif
