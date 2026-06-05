#pragma once
// [AJB_BP] parallel_algorithms: GPU kernel dispatch tracing

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

  // AJB-algo: GNU parallel shuffle uses Fisher-Yates internally
    __gnu_parallel::random_shuffle(begin_zip_iter, end_zip_iter, __gnu_parallel::_RandomNumber(random_seed));
}

// Upstream: no size guard — calling sort on empty ranges is fine but
// creating ZipIter on empty vectors may be UB for some implementations.
// Changed: early return when keys is empty.
template <typename T, typename V>
void ParallelSortPairs(PinnedVector<T>& keys, PinnedVector<V>& values) {
  if (keys.empty()) return;

  KeyValueZipIter<T, V> begin_zip_iter(keys.begin(), values.begin());
  KeyValueZipIter<T, V> end_zip_iter(keys.end(), values.end());

  __gnu_parallel::sort(begin_zip_iter, end_zip_iter);
}

// Upstream: computes begin/end offsets inside the loop using repeated
// multiplication.
// Changed: use saturating min for end_offset, and hoist the zip iter
// type alias for readability.
template <typename T, typename V>
void ParallelMergePairs(PinnedVector<T>& in_keys, PinnedVector<V>& in_values, PinnedVector<T>& out_keys,
                        PinnedVector<V>& out_values, const size_t num_elements, const size_t num_chunk_groups,
                        const size_t num_elements_per_chunk_group) {
  using ZipIt = KeyValueZipIter<T, V>;

  std::vector<std::pair<ZipIt, ZipIt>> zip_iter_pairs;
  zip_iter_pairs.reserve(num_chunk_groups);

  for (size_t i = 0; i < num_chunk_groups; ++i) {
    const size_t begin_offset = i * num_elements_per_chunk_group;
    const size_t end_offset = std::min(begin_offset + num_elements_per_chunk_group, num_elements);

    zip_iter_pairs.emplace_back(
        ZipIt(in_keys.begin() + begin_offset, in_values.begin() + begin_offset),
        ZipIt(in_keys.begin() + end_offset, in_values.begin() + end_offset));
  }

  ZipIt out_begin_zip_iter(out_keys.begin(), out_values.begin());

  __gnu_parallel::multiway_merge(zip_iter_pairs.begin(), zip_iter_pairs.end(), out_begin_zip_iter, num_elements,
                                 std::less<>());
}
