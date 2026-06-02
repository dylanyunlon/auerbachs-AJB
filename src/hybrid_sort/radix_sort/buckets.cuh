#pragma once
// [AJB] BucketId: 多pass基数排序的桶标识, predecessor指向上一pass的父桶

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
