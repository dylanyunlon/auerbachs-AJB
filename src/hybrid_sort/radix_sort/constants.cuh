#pragma once
// [AJB] 256桶=8bit基数, kNumRadixThreads=1024是kernel线程数, kMaxNumGpus=64是硬上限

constexpr size_t kNumBuckets = 256;
constexpr size_t kNumRadixBits = 8;
constexpr size_t kNumRadixStreams = 3;
constexpr size_t kNumRadixThreads = 1024;
constexpr size_t kNumRadixBlocksPerMultiProcessor = 2;
constexpr size_t kMaxNumGpus = 64;
constexpr size_t kWarpSize = 32;
constexpr size_t kNumBlockHistogramsToAggregate = 256;
constexpr size_t kMaxNumBucketsForReducedSorting = 128;
constexpr size_t kMinNumBucketsForSortCopyOverlap = 4;
