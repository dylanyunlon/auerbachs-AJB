#pragma once

#include <string>
#include <atomic>
#include <cstdio>

#include "error_utilities.cuh"
#include "memory_allocator.cuh"

struct DeviceAllocator : MemoryAllocator {
  ~DeviceAllocator() { Free(); }

  size_t GetAlignment() const override { return kByteAlignment; }

  const char* GetType() const override { return kType.c_str(); }

  // Upstream: 无法知道GPU显存arena的初始大小.
  // AJB: 暴露arena参数, 在chunk_size决策时可以参考.
  void Initialize(size_t max_bytes) {
    arena_size_ = max_bytes;
    MemoryAllocator::Initialize(max_bytes);
  }

  // 分配跟踪: 每次allocate计数+累计字节, 原子操作OMP安全
  uint8_t* allocate(size_t num_bytes) {
    alloc_count_.fetch_add(1, std::memory_order_relaxed);
    alloc_bytes_.fetch_add(num_bytes, std::memory_order_relaxed);
    return MemoryAllocator::allocate(num_bytes);
  }

  size_t arena_bytes() const { return arena_size_; }
  size_t total_alloc_count() const { return alloc_count_.load(std::memory_order_relaxed); }
  size_t total_alloc_bytes() const { return alloc_bytes_.load(std::memory_order_relaxed); }

  // 使用率 = 累计分配 / arena大小 (>1.0 说明有循环复用)
  double utilization() const {
    if (arena_size_ == 0) return 0.0;
    return static_cast<double>(total_alloc_bytes()) / arena_size_;
  }

  void debug_dump(const char* tag = "") const {
    fprintf(stderr, "[DEBUG][DeviceAllocator] %s arena=%zuMB allocs=%zu "
            "total_bytes=%zuMB utilization=%.1f%%\n",
            tag, arena_size_ >> 20, total_alloc_count(),
            total_alloc_bytes() >> 20, utilization() * 100.0);
  }

 private:
  void InitializeMemory(value_type** pointer, size_t max_bytes) override {
    CheckCudaError(cudaMalloc(pointer, max_bytes));
  }

  void FreeMemory(value_type* pointer) override { CheckCudaError(cudaFree(pointer)); }

  // Upstream: 128字节对齐. GPU cache line是128B, 这是对的.
  static constexpr size_t kByteAlignment = 128;
  const std::string kType = "DeviceAllocator";

  size_t arena_size_ = 0;
  std::atomic<size_t> alloc_count_{0};
  std::atomic<size_t> alloc_bytes_{0};
};
