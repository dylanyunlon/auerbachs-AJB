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
