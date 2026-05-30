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

    long long l = 0, r = s_count - 1;
    while (l < r) {
      size_t m = (l + r) / 2;
      if (p_s[m] < current_r_key) {
        l = m + 1;
      } else {
        r = m;
      }
    }

    if (p_s[r] != current_r_key) {
      continue;
    }

    const long long s_first = r;
    l = s_first;
    r = s_count - 1;
    while (l < r) {
      size_t m = (l + r + 1) / 2;
      if (current_r_key < p_s[m]) {
        r = m - 1;
      } else {
        l = m;
      }
    }

    const long long s_last = r;
    const long long r_first = i;
    l = i;
    r = r_count - 1;
    while (l < r) {
      size_t m = (l + r + 1) / 2;
      if (current_r_key < p_r[m]) {
        r = m - 1;
      } else {
        l = m;
      }
    }

    const long long r_last = r;
    atomicAdd(&(join_and_materialization_counts->x), (r_last - r_first + 1) * (s_last - s_first + 1));

    if (materialized == nullptr) {
      continue;
    }

    const size_t at = atomicAdd(&(join_and_materialization_counts->y), 1);
    longlong4 ranges;

    if constexpr (swap_rs) {
      ranges = {table_offsets.y + s_first, table_offsets.y + s_last, table_offsets.x + r_first,
                table_offsets.x + r_last};
    } else {
      ranges = {table_offsets.x + r_first, table_offsets.x + r_last, table_offsets.y + s_first,
                table_offsets.y + s_last};
    }

    materialized[at] = ranges;
  }
}
