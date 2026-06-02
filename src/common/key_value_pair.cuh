// [AJB] KVP zip/unzip: sort/join用key排序时需要value跟着走
// zip把两个数组合成pair数组, unzip拆回去
// 这是sort pipeline的关键步骤, zip/unzip的带宽开销不容忽视
#pragma once

#include <vector>

#include "pinned_vector.cuh"

template <typename T, typename V>
struct KeyValuePair {
  bool operator<(const KeyValuePair& other) const { return (key < other.key); }

  bool operator<=(const KeyValuePair& other) const { return (key <= other.key); }

  T key;
  V value;
};

template <typename T, typename V>
void ZipKeyValuePairs(const PinnedVector<T>& keys, const PinnedVector<V>& values, const size_t num_elements,
                      std::vector<KeyValuePair<T, V>>& key_value_pairs) {
#pragma omp parallel for
  for (size_t i = 0; i < num_elements; ++i) {
    key_value_pairs[i].key = keys[i];
    key_value_pairs[i].value = values[i];
  }
}

template <typename T, typename V>
void UnzipKeyValuePairs(const std::vector<KeyValuePair<T, V>>& key_value_pairs, const size_t num_elements,
                        PinnedVector<T>& keys, PinnedVector<V>& values) {
#pragma omp parallel for
  for (size_t i = 0; i < num_elements; ++i) {
    keys[i] = key_value_pairs[i].key;
    values[i] = key_value_pairs[i].value;
  }
}

// [AJB] zip/unzip bandwidth估算
#include <cstdio>
static inline void ajb_report_kvp_bandwidth(size_t n, double elapsed_ms, const char* op) {
    double gb = (double)n * sizeof(int64_t) * 2 / 1e9; // 读+写
    fprintf(stderr, "[AJB_TIMER][KVP] %s: n=%zu elapsed=%.3fms bandwidth=%.2f GB/s\n",
            op, n, elapsed_ms, elapsed_ms > 0 ? gb / (elapsed_ms / 1000.0) : 0.0);
}
