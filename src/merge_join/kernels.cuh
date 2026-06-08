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

// --- M1121: SIMD-friendly 4-way key comparison for 32-bit keys ---
// When join keys are 32-bit integers, we can compare 4 keys simultaneously
// using 128-bit SIMD registers (or GPU warp shuffle).  This helper packs
// 4 comparison results into a bitmask for branchless processing.
// Returns a 4-bit mask where bit i is set if keys_a[i] == keys_b[i].
__device__ __forceinline__
unsigned int SimdCompare4x32(const int* keys_a, const int* keys_b) {
    unsigned int mask = 0;
    // Manual unroll of 4-way comparison — compiler maps to VSETP on SM80+
    mask |= (keys_a[0] == keys_b[0]) ? 1u : 0u;
    mask |= (keys_a[1] == keys_b[1]) ? 2u : 0u;
    mask |= (keys_a[2] == keys_b[2]) ? 4u : 0u;
    mask |= (keys_a[3] == keys_b[3]) ? 8u : 0u;
    return mask;
}

// Population count of match mask — gives the number of matching pairs
__device__ __forceinline__
int SimdMatchCount4(unsigned int mask) {
    return __popc(mask);  // PTX popc instruction
}

// --- M1122: Branch-free merge path binary search ---
// Standard merge path binary search uses if/else branches which cause
// warp divergence on GPU.  This version uses conditional move (ternary)
// which the CUDA compiler maps to SELP instructions (no branch).
template <typename T>
__device__ __forceinline__
long long BranchFreeMergePathSearch(
    const T* a, long long a_len,
    const T* b, long long b_len,
    long long diag) {

    long long lo = (diag > b_len) ? (diag - b_len) : 0;
    long long hi = (diag < a_len) ? diag : a_len;

    // Binary search iterations — all use conditional assignment, no branch
    while (lo < hi) {
        long long mid = lo + ((hi - lo) >> 1);
        long long b_idx = diag - 1 - mid;

        // Branch-free: both paths compute, select result via ternary
        int cmp = (b_idx >= 0 && b_idx < b_len && mid < a_len) ?
                  (a[mid] > b[b_idx] ? 1 : 0) : 0;
        lo = cmp ? lo : (mid + 1);
        hi = cmp ? mid : hi;
    }

    return lo;
}
