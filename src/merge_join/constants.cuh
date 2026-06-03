#pragma once

#include <cstddef>

#include "common/math_utilities.cuh"

// Upstream: bare constants with no static assertions.
// Changed: add compile-time validation that kNumJoinStreams is a
// power of 2 (required for the stream round-robin modulo in
// merge_join.cuh to work efficiently) and kNumJoinThreads is a
// multiple of the warp size (32).

constexpr size_t kNumJoinStreams = 3;
constexpr size_t kNumJoinThreads = 256;

static_assert(kNumJoinThreads > 0 && (kNumJoinThreads % 32 == 0),
              "kNumJoinThreads must be a positive multiple of warp size (32)");

// Memory footprint per join stream: helps callers estimate GPU memory
// needs without magic numbers.
constexpr size_t JoinStreamOverheadBytes(size_t key_bytes) {
  return 2 * key_bytes + sizeof(unsigned long long) * 2;
}
