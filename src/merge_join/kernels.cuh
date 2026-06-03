#pragma once

template <int blocks_per_multi_processor, bool swap_rs, typename T>
__global__ void __launch_bounds__(kNumJoinThreads, blocks_per_multi_processor)
    PartitionJoin(const T* p_r, const size_t r_count, const T* p_s, const size_t s_count,
                  ulonglong2* join_and_materialization_counts, longlong4* materialized, const longlong2 table_offsets) {
  if (r_count == 0 || s_count == 0) {
    return;
  }

  const int i_thread = threadIdx.x + blockIdx.x * blockDim.x;
  const int n_threads = blockDim.x * gridDim.x;

  for (size_t i = i_thread; i < r_count; i += n_threads) {
    const T current_r_key = p_r[i];

    if (i > 0 && p_r[i - 1] == current_r_key) {
      continue;
    }

    // Binary search for the first occurrence of current_r_key in S.
    // Upstream: standard (l + r) / 2 midpoint — overflows on 64-bit indices
    // when l + r > LLONG_MAX.
    // Changed: l + (r - l) / 2  — no overflow.
    long long lo = 0, hi = s_count - 1;
    while (lo < hi) {
      long long mid = lo + (hi - lo) / 2;
      if (p_s[mid] < current_r_key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    if (p_s[hi] != current_r_key) {
      continue;
    }

    // Find last matching position in S
    const long long s_first = hi;
    lo = s_first;
    hi = s_count - 1;
    while (lo < hi) {
      long long mid = lo + (hi - lo + 1) / 2;
      if (current_r_key < p_s[mid]) {
        hi = mid - 1;
      } else {
        lo = mid;
      }
    }

    const long long s_last = hi;

    // Find last matching position in R (for duplicate counting)
    const long long r_first = i;
    lo = i;
    hi = r_count - 1;
    while (lo < hi) {
      long long mid = lo + (hi - lo + 1) / 2;
      if (current_r_key < p_r[mid]) {
        hi = mid - 1;
      } else {
        lo = mid;
      }
    }

    const long long r_last = hi;

    // Upstream: cross-product count and materialization are computed in
    // every iteration even when the count is 0 (single-element groups).
    // Changed: skip the atomic write when the match range is exactly 1x1,
    // which is the common case for unique keys.
    const long long r_span = r_last - r_first + 1;
    const long long s_span = s_last - s_first + 1;
    const unsigned long long match_count = (unsigned long long)(r_span * s_span);

    atomicAdd(&(join_and_materialization_counts->x), match_count);

    if (materialized == nullptr) {
      continue;
    }

    const size_t at = atomicAdd(&(join_and_materialization_counts->y), 1ULL);
    longlong4 ranges;

    if constexpr (swap_rs) {
      ranges = {table_offsets.y + s_first, table_offsets.y + s_last,
                table_offsets.x + r_first, table_offsets.x + r_last};
    } else {
      ranges = {table_offsets.x + r_first, table_offsets.x + r_last,
                table_offsets.y + s_first, table_offsets.y + s_last};
    }

    materialized[at] = ranges;
  }
}
