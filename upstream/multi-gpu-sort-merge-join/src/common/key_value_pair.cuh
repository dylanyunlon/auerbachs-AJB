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
