#pragma once

#include <cub/cub.cuh>

struct BucketId {
  BucketId() : partition_pass(0), bucket_number(0), predecessor(nullptr), depth_(0) {}

  BucketId(size_t partition_pass, size_t bucket_number)
      : partition_pass(partition_pass), bucket_number(bucket_number),
        predecessor(nullptr), depth_(1) {}

  BucketId(size_t partition_pass, size_t bucket_number, BucketId* predecessor)
      : partition_pass(partition_pass), bucket_number(bucket_number),
        predecessor(predecessor),
        depth_(predecessor ? predecessor->depth_ + 1 : 1) {}

  // Upstream: 无法知道一个bucket经过了多少次partition.
  // AJB: 递归深度跟踪 — 在skew场景下某些bucket会被反复partition,
  // depth过深说明数据分布极端, 可能需要切换到merge sort路径.
  size_t depth() const { return depth_; }

  // 完整路径: 从当前bucket回溯到root, 返回每一级的bucket_number
  // 在调试时可以看到某个bucket是怎么一步步split下来的
  void trace_path(size_t* out_path, size_t max_len) const {
    const BucketId* cur = this;
    size_t i = 0;
    while (cur && i < max_len) {
      out_path[i++] = cur->bucket_number;
      cur = cur->predecessor;
    }
    // 反转: out_path现在是root→leaf顺序
    for (size_t l = 0, r = i - 1; l < r; ++l, --r) {
      size_t tmp = out_path[l];
      out_path[l] = out_path[r];
      out_path[r] = tmp;
    }
  }

  size_t partition_pass;
  size_t bucket_number;
  BucketId* predecessor;

 private:
  size_t depth_;
};

struct CompareBucketIds {
  bool operator()(const BucketId& a, const BucketId& b) const {
    if (a.partition_pass != b.partition_pass) {
      return a.partition_pass < b.partition_pass;
    }
    return a.bucket_number < b.bucket_number;
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

  // Upstream: 无法估算这个bucket的cub sort需要多少临时显存.
  // AJB: 根据bucket_size估算 — cub radix sort临时空间约=2*n*sizeof(key)
  size_t estimated_temp_bytes() const {
    return 2 * bucket_size * sizeof(T);
  }
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

  // 传输量(字节) — 在bandwidth分析时需要
  size_t transfer_bytes(size_t element_size) const {
    return fraction_size * element_size;
  }
};
