// [AJB] ResourceContext: moderngpu的context适配器
// 把mgpu的alloc/free转接到DeviceAllocator
// mgpu的sort/merge kernel通过这个context获取临时显存
#pragma once

#include <moderngpu/context.hxx>

#include "device_allocator.cuh"

struct ResourceContext : public mgpu::standard_context_t {
 public:
  explicit ResourceContext(DeviceAllocator& device_allocator, cudaStream_t stream)
      : standard_context_t(false, stream), device_allocator_(device_allocator) {}

  void* alloc(size_t num_bytes, mgpu::memory_space_t memory_space) override {
    if (memory_space == mgpu::memory_space_device) {
      return reinterpret_cast<void*>(device_allocator_.allocate(num_bytes));
    } else {
      return mgpu::standard_context_t::alloc(num_bytes, memory_space);
    }
  }

  void free(void* begin_pointer, mgpu::memory_space_t memory_space) override {
    if (memory_space == mgpu::memory_space_device) {
      device_allocator_.deallocate(reinterpret_cast<uint8_t*>(begin_pointer));
    } else {
      mgpu::standard_context_t::free(begin_pointer, memory_space);
    }
  }

 private:
  DeviceAllocator& device_allocator_;
};

#include <cstdio>
// [AJB] context创建跟踪
static inline void ajb_report_context_create(int device_id, const char* tag) {
    fprintf(stderr, "[AJB_TRACE][ResourceContext] created for device %d (%s)\n", device_id, tag);
}
