#pragma once

#include <vector>

#include "pinned_vector.cuh"

template <typename T, typename V>
struct KeyValuePair {
  bool operator<(const KeyValuePair& other) const { return (key < other.key); }

  bool operator<=(const KeyValuePair& other) const { return (key <= other.key); }

  // Upstream: no equality operator.
  // Changed: add operator== for use with std::adjacent_find and other
  // algorithms that need equality comparison.
  bool operator==(const KeyValuePair& other) const { return key == other.key && value == other.value; }

  T key;
  V value;
};

// Upstream: Zip and Unzip access key_value_pairs[i].key and .value as
// separate stores.  On x86 with wide SIMD, this prevents vectorization
// because the struct fields are interleaved in memory.
// Changed: access through a reference to coalesce the two stores into
// a single struct write per iteration.

template <typename T, typename V>
void ZipKeyValuePairs(const PinnedVector<T>& keys, const PinnedVector<V>& values, const size_t num_elements,
                      std::vector<KeyValuePair<T, V>>& key_value_pairs) {
#pragma omp parallel for
  for (size_t i = 0; i < num_elements; ++i) {
    auto& kvp = key_value_pairs[i];
    kvp = {keys[i], values[i]};
  }
}

template <typename T, typename V>
void UnzipKeyValuePairs(const std::vector<KeyValuePair<T, V>>& key_value_pairs, const size_t num_elements,
                        PinnedVector<T>& keys, PinnedVector<V>& values) {
#pragma omp parallel for
  for (size_t i = 0; i < num_elements; ++i) {
    const auto& kvp = key_value_pairs[i];
    keys[i] = kvp.key;
    values[i] = kvp.value;
  }
}
