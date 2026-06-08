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

// --- M1114: Work-stealing scheduling heuristic ---
// When threads in a parallel merge finish early (because chunk sizes are
// unbalanced), idle threads should steal from the heaviest queue.
// This helper computes a steal-aware partition plan: given N items and
// P workers, it assigns ceil(N/P) to the first (N mod P) workers and
// floor(N/P) to the rest, then returns per-worker [begin,end) offsets.
// The caller (GPU kernel launcher) uses these to dispatch work tiles.
struct WorkStealPartition {
  size_t begin;
  size_t end;
  size_t weight;  // estimated cost (e.g., element count * key_width)
};

inline std::vector<WorkStealPartition> ComputeStealAwarePartitions(
    size_t total_items, size_t num_workers, size_t per_item_weight = 1) {
  std::vector<WorkStealPartition> parts;
  if (num_workers == 0 || total_items == 0) return parts;
  parts.reserve(num_workers);

  size_t base = total_items / num_workers;
  size_t remainder = total_items % num_workers;
  size_t offset = 0;

  for (size_t w = 0; w < num_workers; ++w) {
    size_t chunk = base + (w < remainder ? 1 : 0);
    parts.push_back({offset, offset + chunk, chunk * per_item_weight});
    offset += chunk;
  }

  // [AJB_BP] dump partition balance for diagnostics
  if (num_workers > 1) {
    size_t max_w = 0, min_w = ~size_t(0);
    for (auto& p : parts) {
      if (p.weight > max_w) max_w = p.weight;
      if (p.weight < min_w) min_w = p.weight;
    }
    double imbalance = max_w > 0 ? (double)(max_w - min_w) / max_w : 0.0;
    fprintf(stderr, "[AJB_BP][WorkSteal] workers=%zu items=%zu imbalance=%.4f max_weight=%zu\n",
            num_workers, total_items, imbalance, max_w);
  }
  return parts;
}

// --- M1115: Adaptive grain size selection ---
// Determines optimal parallel granularity based on data volume and key width.
// Too-small grain → scheduling overhead dominates; too-large → poor load balance.
// Heuristic: target ~64KB of data per work unit (fits in L1 cache), but
// never fewer than 256 elements (to amortize thread launch) or more than
// total/4 (ensures at least 4 work units for parallelism).
inline size_t AdaptiveGrainSize(size_t total_elements, size_t key_bytes) {
  constexpr size_t TARGET_BYTES_PER_GRAIN = 64 * 1024;  // 64 KB
  constexpr size_t MIN_ELEMENTS = 256;
  constexpr size_t MIN_WORK_UNITS = 4;

  size_t bytes_per_element = key_bytes > 0 ? key_bytes : 8;
  size_t grain = TARGET_BYTES_PER_GRAIN / bytes_per_element;

  // Clamp: at least MIN_ELEMENTS
  if (grain < MIN_ELEMENTS) grain = MIN_ELEMENTS;

  // Clamp: ensure at least MIN_WORK_UNITS work units
  size_t max_grain = total_elements / MIN_WORK_UNITS;
  if (max_grain < MIN_ELEMENTS) max_grain = MIN_ELEMENTS;
  if (grain > max_grain) grain = max_grain;

  fprintf(stderr, "[AJB_BP][AdaptiveGrain] elements=%zu key_bytes=%zu grain=%zu work_units=%zu\n",
          total_elements, key_bytes, grain,
          grain > 0 ? (total_elements + grain - 1) / grain : 0);
  return grain;
}
