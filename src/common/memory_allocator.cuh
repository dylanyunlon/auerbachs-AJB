#pragma once

#include <algorithm>
#include <list>
#include <map>
#include <numeric>
#include <string>

#include "math_utilities.cuh"

struct MemoryAllocator {
  using value_type = uint8_t;

  MemoryAllocator() = default;
  ~MemoryAllocator() = default;

  MemoryAllocator(const MemoryAllocator&) = delete;
  MemoryAllocator& operator=(const MemoryAllocator&) = delete;

  MemoryAllocator(MemoryAllocator&& src)
      : capacity_(src.capacity_),
        offset_(src.offset_),
        pointer_(src.pointer_),
        allocations_(std::move(src.allocations_)) {
    src.capacity_ = 0;
    src.offset_ = 0;
    src.pointer_ = nullptr;
    src.allocations_.clear();
  }
  MemoryAllocator& operator=(MemoryAllocator&& src) {
    if (this != &src) {
      std::swap(src.capacity_, capacity_);
      std::swap(src.offset_, offset_);
      std::swap(src.pointer_, pointer_);
      std::swap(src.allocations_, allocations_);
    }
    return *this;
  }

  void Initialize(size_t max_bytes) {
    if (capacity_ < max_bytes) {
#ifdef DEBUG_BUILD
      printf("[%s] initialize max_bytes = %lu bytes.\n", GetType(), max_bytes);
#endif
      Free();
      InitializeMemory(&pointer_, max_bytes);
      capacity_ = max_bytes;
    }
  }

  void Free() {
    capacity_ = 0;
    offset_ = 0;
    FreeMemory(pointer_);
    pointer_ = nullptr;
    allocations_.clear();
  }

  value_type* allocate(size_t num_bytes) {
    if (num_bytes == 0) {
      return nullptr;
    }
    value_type* begin_pointer = pointer_ + offset_;
    const size_t aligned_num_bytes = RoundUp(num_bytes, GetAlignment());
#ifdef DEBUG_BUILD
    printf("[%s] allocate num_bytes = %lu bytes.\n", GetType(), aligned_num_bytes);
#endif
    if (capacity_ - offset_ < aligned_num_bytes) {
#ifdef DEBUG_BUILD
      printf("[ERROR][%s] num_bytes = %lu is too large.\n", GetType(), aligned_num_bytes);
#endif
      return nullptr;
    }
    offset_ += aligned_num_bytes;
    allocations_.push_back({begin_pointer, aligned_num_bytes});
    return begin_pointer;
  }

  void deallocate(value_type* begin_pointer, size_t num_bytes = 0) {
    auto reverse_it = std::find(allocations_.rbegin(), allocations_.rend(), Allocation{begin_pointer});
    if (reverse_it == allocations_.rend()) {
#ifdef DEBUG_BUILD
      printf("[ERROR][%s] begin_pointer = %p is invalid.\n", GetType(), reinterpret_cast<void*>(begin_pointer));
#endif
      return;
    }
#ifdef DEBUG_BUILD
    printf("[%s] deallocate num_bytes = %lu bytes.\n", GetType(), reverse_it->aligned_num_bytes);
#endif
    const bool shift_offset = reverse_it == allocations_.rbegin();
    allocations_.erase(std::next(reverse_it).base());
    if (shift_offset) {
      offset_ = 0;
      for (const auto& allocation : allocations_) {
        offset_ += allocation.aligned_num_bytes;
      }
#ifdef DEBUG_BUILD
      printf("[%s] set offset_ = %lu bytes.\n", GetType(), offset_);
#endif
    }
  }

  size_t GetFreeBytes() const { return capacity_ - offset_; }

  virtual size_t GetAlignment() const = 0;

  virtual const char* GetType() const = 0;

 private:
  virtual void InitializeMemory(value_type** pointer, size_t max_bytes) = 0;

  virtual void FreeMemory(value_type* pointer) = 0;

  struct Allocation {
    value_type* begin_pointer = nullptr;
    size_t aligned_num_bytes = 0;

    bool operator==(const Allocation& other) const { return (begin_pointer == other.begin_pointer); }
  };

  size_t capacity_ = 0;
  size_t offset_ = 0;
  value_type* pointer_ = nullptr;
  std::list<Allocation> allocations_;
};

// [AJB] 内存分配器状态dump
#include <cstdio>
static inline void ajb_report_memory_allocator(size_t allocated, size_t peak, const char* tag) {
    fprintf(stderr, "[AJB_MEM][Allocator] %s: current=%.2fMB peak=%.2fMB\n",
            tag, allocated / 1048576.0, peak / 1048576.0);
}
