#pragma once

#include <string>
#include <atomic>
#include <cstdio>

#include "error_utilities.cuh"
#include "memory_allocator.cuh"

struct HostAllocator : MemoryAllocator {
  ~HostAllocator() { Free(); }

  size_t GetAlignment() const override { return kByteAlignment; }

  const char* GetType() const override { return kType.c_str(); }

  // Upstream: 无法知道pinned memory的分配量.
  // AJB: 跟踪pinnedhost的初始化大小, 在OOM诊断时可以快速确认
  // host端arena是否太小(导致频繁的cudaMallocHost fallback).
  void Initialize(size_t max_bytes) {
    arena_size_ = max_bytes;
    MemoryAllocator::Initialize(max_bytes);
  }

  // 累计分配量跟踪 — 原子操作, OMP安全
  uint8_t* allocate(size_t num_bytes) {
    alloc_count_.fetch_add(1, std::memory_order_relaxed);
    alloc_bytes_.fetch_add(num_bytes, std::memory_order_relaxed);
    return MemoryAllocator::allocate(num_bytes);
  }

  size_t arena_bytes() const { return arena_size_; }
  size_t total_alloc_count() const { return alloc_count_.load(std::memory_order_relaxed); }
  size_t total_alloc_bytes() const { return alloc_bytes_.load(std::memory_order_relaxed); }

  // 断点调试: host allocator状态快照
  void debug_dump(const char* tag = "") const {
    fprintf(stderr, "[DEBUG][HostAllocator] %s arena=%zuMB allocs=%zu total_bytes=%zuMB\n",
            tag, arena_size_ >> 20, total_alloc_count(), total_alloc_bytes() >> 20);
  }

 private:
  void InitializeMemory(value_type** pointer, size_t max_bytes) override {
    CheckCudaError(cudaMallocHost(pointer, max_bytes));
  }

  void FreeMemory(value_type* pointer) override { CheckCudaError(cudaFreeHost(pointer)); }

  static constexpr size_t kByteAlignment = 16;
  const std::string kType = "HostAllocator";

  size_t arena_size_ = 0;
  std::atomic<size_t> alloc_count_{0};
  std::atomic<size_t> alloc_bytes_{0};
};
