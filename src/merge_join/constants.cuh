#pragma once

#include "common/math_utilities.cuh"

// Upstream: 裸常量, 无约束无推导.
// AJB: 加约束检查 + 提供运行时可查询的内存开销估算,
// 与radix_sort/constants.cuh保持同样的工程规范.
constexpr size_t kNumJoinStreams = 3;
constexpr size_t kNumJoinThreads = 256;

// 每个join stream需要的最小显存: 两个relation的chunk + 输出buffer
// 调参时用这个估算是否会OOM
constexpr size_t JoinStreamMemoryPerElement(size_t key_bytes, size_t val_bytes) {
  // 输入: R和S各一份 + 输出: 最坏情况=|R|*|S|(但实际分chunk后远小于此)
  // 这里只估算输入buffer
  return 2 * (key_bytes + val_bytes);
}

static_assert(kNumJoinThreads > 0 && (kNumJoinThreads & (kNumJoinThreads - 1)) == 0,
              "kNumJoinThreads must be power of 2 for warp alignment");
static_assert(kNumJoinStreams >= 2,
              "need ≥2 streams for compute/transfer overlap");
