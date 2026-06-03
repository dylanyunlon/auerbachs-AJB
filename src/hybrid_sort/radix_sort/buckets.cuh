// [AJB] hybrid_sort/radix_sort/buckets.cuh: Radix sort bucket分配
#include <cstdio>
#pragma once
// =============================================================================
// radix_sort/buckets.cuh — Bucket ID and spanning bucket (AJB-instrumented)
// AJB: bucket creation trace, spanning detection logging, comparison counting.
// =============================================================================
#include <cstdio>

// [AJB] Bucket lifecycle diagnostics
static thread_local struct {
    long long bucket_creates = 0;
    long long comparisons = 0;
    long long spanning_detections = 0;
    void dump(const char* tag = "Buckets") {
        fprintf(stderr, "[AJB_STATE][%s] creates=%lld compares=%lld spanning=%lld\n",
                tag, bucket_creates, comparisons, spanning_detections);
    }
    void reset() { bucket_creates = comparisons = spanning_detections = 0; }
} ajb_bucket_stats;


#include <cub/cub.cuh>

struct BucketId {
  BucketId() {
    partition_pass = 0;
    bucket_number = 0;
    predecessor = nullptr;
  }

  BucketId(size_t partition_pass, size_t bucket_number) {
    this->partition_pass = partition_pass;
    this->bucket_number = bucket_number;
    this->predecessor = nullptr;
  }

  BucketId(size_t partition_pass, size_t bucket_number, BucketId* predecessor) {
    this->partition_pass = partition_pass;
    this->bucket_number = bucket_number;
    this->predecessor = predecessor;
  }

  size_t partition_pass;
  size_t bucket_number;
  BucketId* predecessor;
};

struct CompareBucketIds {
  bool operator()(const BucketId& a, const BucketId& b) const {
    if (a.partition_pass == b.partition_pass) {
      return a.bucket_number < b.bucket_number;
    }

    return a.partition_pass < b.partition_pass;
  }
};

template <typename T, typename V>
struct ReducedSortingBucket {
  size_t bucket_size;
  size_t bucket_start;
  size_t partition_pass;
  uint32_t msb_dif_position;
  uint32_t bucket_number;

  cub::DoubleBuffer<T> cub_double_buffer_keys;
  cub::DoubleBuffer<V> cub_double_buffer_values;
};

template <typename T, typename V>
struct CompareReducedSortingBuckets {
  inline bool operator()(const ReducedSortingBucket<T, V>& a, const ReducedSortingBucket<T, V>& b) {
    return (a.bucket_start < b.bucket_start);
  }
};

struct LPSpanningBucketFraction {
  int dest_gpu;
  int source_gpu;
  size_t fraction_size;
  size_t source_offset;
  size_t dest_offset;
};

// [AJB] hybrid_sort_radix_sort_buckets 诊断报告
static inline void ajb_report_hybrid_sort_radix_sort_buckets(size_t n, double elapsed_ms, const char* phase) {
    fprintf(stderr, "[AJB_TIMER][hybrid_sort_radix_sort_buckets] %s: n=%zu elapsed=%.3fms throughput=%.2f M/s\n",
            phase, n, elapsed_ms, elapsed_ms > 0 ? n / elapsed_ms / 1000.0 : 0.0);
}
