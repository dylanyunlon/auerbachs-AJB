#pragma once

#include <moderngpu/context.hxx>
#include <cstdio>
#include <atomic>

#include "device_allocator.cuh"

// Upstream: 直接转发alloc/free到DeviceAllocator, 无任何观测手段.
// AJB改写:
//   1. 引入alloc/free计数 + 累计字节跟踪(原子操作, OMP安全)
//   2. alloc时检查watermark, 接近显存上限时打印warning
//   3. 提供 dump_state() — 在实验运行中途可以看到mgpu内部
//      到底分配了多少临时显存, 帮助判断chunk_size是否合理
struct ResourceContext : public mgpu::standard_context_t {
 public:
  explicit ResourceContext(DeviceAllocator& device_allocator, cudaStream_t stream,
                           size_t watermark_bytes = 0)
      : standard_context_t(false, stream),
        device_allocator_(device_allocator),
        watermark_(watermark_bytes) {}

  void* alloc(size_t num_bytes, mgpu::memory_space_t memory_space) override {
    if (memory_space == mgpu::memory_space_device) {
      void* ptr = reinterpret_cast<void*>(device_allocator_.allocate(num_bytes));
      alloc_count_.fetch_add(1, std::memory_order_relaxed);
      size_t prev = live_bytes_.fetch_add(num_bytes, std::memory_order_relaxed);
      size_t now = prev + num_bytes;

      // 峰值跟踪: CAS循环更新peak
      size_t cur_peak = peak_bytes_.load(std::memory_order_relaxed);
      while (now > cur_peak &&
             !peak_bytes_.compare_exchange_weak(cur_peak, now, std::memory_order_relaxed)) {}

      // watermark告警: 超过阈值时打印, 帮助在GPU OOM前定位问题
      if (watermark_ > 0 && now > watermark_) {
        fprintf(stderr, "[DEBUG][ResourceContext::alloc] watermark breach: "
                "live=%zuMB > limit=%zuMB (this alloc=%zuKB)\n",
                now >> 20, watermark_ >> 20, num_bytes >> 10);
      }
      return ptr;
    } else {
      return mgpu::standard_context_t::alloc(num_bytes, memory_space);
    }
  }

  void free(void* begin_pointer, mgpu::memory_space_t memory_space) override {
    if (memory_space == mgpu::memory_space_device) {
      device_allocator_.deallocate(reinterpret_cast<uint8_t*>(begin_pointer));
      free_count_.fetch_add(1, std::memory_order_relaxed);
      // 注意: 这里无法知道被free的精确字节数(DeviceAllocator不返回size)
      // 所以live_bytes_只增不减, peak_bytes_才是有意义的指标
    } else {
      mgpu::standard_context_t::free(begin_pointer, memory_space);
    }
  }

  // 断点调试: 打印当前context的alloc/free全貌
  void dump_state(const char* tag = "") const {
    fprintf(stderr, "[DEBUG][ResourceContext] %s allocs=%zu frees=%zu "
            "live_bytes=%zuMB peak=%zuMB watermark=%zuMB\n",
            tag,
            alloc_count_.load(std::memory_order_relaxed),
            free_count_.load(std::memory_order_relaxed),
            live_bytes_.load(std::memory_order_relaxed) >> 20,
            peak_bytes_.load(std::memory_order_relaxed) >> 20,
            watermark_ >> 20);
  }

 private:
  DeviceAllocator& device_allocator_;
  size_t watermark_;  // 0 = 不检查

  std::atomic<size_t> alloc_count_{0};
  std::atomic<size_t> free_count_{0};
  std::atomic<size_t> live_bytes_{0};
  std::atomic<size_t> peak_bytes_{0};
};
