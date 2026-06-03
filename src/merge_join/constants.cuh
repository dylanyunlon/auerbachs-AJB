#pragma once
// =============================================================================
// merge_join/constants.cuh — Join kernel tuning constants (AJB-instrumented)
// AJB: document performance impact of each constant, provide tuning guidance.
// =============================================================================

// [AJB] kNumJoinThreads: threads per block for join kernel — affects occupancy
//   Higher = better latency hiding but more register pressure
//   Optimal: 256 for most GPUs (A100/H100), 128 for older (V100)
// [AJB] kNumJoinStreams: concurrent join operations — overlap compute+transfer
//   Higher = better pipeline utilization but more memory pressure
// [AJB] join也用3个stream, 和sort共享stream编号约定

#include "common/math_utilities.cuh"

constexpr size_t kNumJoinStreams = 3;
constexpr size_t kNumJoinThreads = 256;
