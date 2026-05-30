#pragma once

#include <vector>

#include <parallel/multiway_merge.h>
#include <parallel/random_shuffle.h>
#include <parallel/sort.h>
#include <zipiterator/ZipIterator.hpp>

#include "pinned_vector.cuh"

template <typename T, typename V>
using KeyValueZipIter = ZipIter<typename PinnedVector<T>::iterator, typename PinnedVector<V>::iterator>;

template <typename T, typename V>
void ParallelShufflePairs(PinnedVector<T>& keys, PinnedVector<V>& values, const uint32_t random_seed) {
  KeyValueZipIter<T, V> begin_zip_iter(keys.begin(), values.begin());
  KeyValueZipIter<T, V> end_zip_iter(keys.end(), values.end());

  __gnu_parallel::random_shuffle(begin_zip_iter, end_zip_iter, __gnu_parallel::_RandomNumber(random_seed));
}

template <typename T, typename V>
void ParallelSortPairs(PinnedVector<T>& keys, PinnedVector<V>& values) {
  KeyValueZipIter<T, V> begin_zip_iter(keys.begin(), values.begin());
  KeyValueZipIter<T, V> end_zip_iter(keys.end(), values.end());

  __gnu_parallel::sort(begin_zip_iter, end_zip_iter);
}

template <typename T, typename V>
void ParallelMergePairs(PinnedVector<T>& in_keys, PinnedVector<V>& in_values, PinnedVector<T>& out_keys,
                        PinnedVector<V>& out_values, const size_t num_elements, const size_t num_chunk_groups,
                        const size_t num_elements_per_chunk_group) {
  std::vector<std::pair<KeyValueZipIter<T, V>, KeyValueZipIter<T, V>>> zip_iter_pairs;
  zip_iter_pairs.reserve(num_chunk_groups);

  for (size_t i = 0; i < num_chunk_groups; ++i) {
    const size_t begin_offset = i * num_elements_per_chunk_group;
    const size_t end_offset = std::min((i + 1) * num_elements_per_chunk_group, num_elements);

    KeyValueZipIter<T, V> begin_zip_iter(in_keys.begin() + begin_offset, in_values.begin() + begin_offset);
    KeyValueZipIter<T, V> end_zip_iter(in_keys.begin() + end_offset, in_values.begin() + end_offset);

    zip_iter_pairs.emplace_back(begin_zip_iter, end_zip_iter);
  }

  KeyValueZipIter<T, V> out_begin_zip_iter(out_keys.begin(), out_values.begin());

  __gnu_parallel::multiway_merge(zip_iter_pairs.begin(), zip_iter_pairs.end(), out_begin_zip_iter, num_elements,
                                 std::less<>());
}
