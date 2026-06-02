#pragma once
// [AJB] join也用3个stream, 和sort共享stream编号约定

#include "common/math_utilities.cuh"

constexpr size_t kNumJoinStreams = 3;
constexpr size_t kNumJoinThreads = 256;
