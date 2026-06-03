#pragma once

#include <cstddef>

// --- Radix sort 核心参数 ---
// Upstream: 12个裸constexpr, 无关联、无约束检查.
// AJB改写: 将有约束关系的常量用constexpr函数验证一致性,
// 防止调参时破坏隐含不变量(比如改kNumRadixBits忘了改kNumBuckets).

constexpr size_t kNumRadixBits = 8;
constexpr size_t kNumBuckets = 1u << kNumRadixBits;  // 从kNumRadixBits推导, 不再硬编码256

constexpr size_t kNumRadixStreams = 3;
constexpr size_t kNumRadixThreads = 1024;
constexpr size_t kNumRadixBlocksPerMultiProcessor = 2;
constexpr size_t kMaxNumGpus = 64;
constexpr size_t kWarpSize = 32;
constexpr size_t kNumBlockHistogramsToAggregate = 256;
constexpr size_t kMaxNumBucketsForReducedSorting = 128;
constexpr size_t kMinNumBucketsForSortCopyOverlap = 4;

// 编译期约束检查 — upstream里这些不变量全靠口头约定
static_assert(kNumBuckets == (1u << kNumRadixBits),
              "kNumBuckets must equal 2^kNumRadixBits");
static_assert(kNumRadixThreads % kWarpSize == 0,
              "kNumRadixThreads must be multiple of kWarpSize");
static_assert(kMaxNumBucketsForReducedSorting <= kNumBuckets,
              "reduced sorting threshold can't exceed total buckets");
static_assert(kMinNumBucketsForSortCopyOverlap >= 2,
              "overlap needs at least 2 buckets");
static_assert(kNumRadixBlocksPerMultiProcessor >= 1,
              "need at least 1 block per SM");

// 运行时参数打印 — 在benchmark启动时调一次, 确认用的是哪组参数
inline void DumpRadixConstants() {
  fprintf(stderr, "[DEBUG][RadixConstants] bits=%zu buckets=%zu threads=%zu "
          "blocks_per_sm=%zu max_gpu=%zu warp=%zu "
          "hist_agg=%zu reduced_sort=%zu overlap=%zu\n",
          kNumRadixBits, kNumBuckets, kNumRadixThreads,
          kNumRadixBlocksPerMultiProcessor, kMaxNumGpus, kWarpSize,
          kNumBlockHistogramsToAggregate, kMaxNumBucketsForReducedSorting,
          kMinNumBucketsForSortCopyOverlap);
}

// shared memory需求估算(字节) — 调参时先check会不会超过SM上限
constexpr size_t RadixSharedMemPerBlock() {
  // 每个线程块需要: kNumBuckets个int做local histogram
  return kNumBuckets * sizeof(unsigned int);
}
