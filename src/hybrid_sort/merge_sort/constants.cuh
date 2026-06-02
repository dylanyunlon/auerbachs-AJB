#pragma once
// [AJB] merge sort用3个CUDA stream: 0=计算, 1=histogram传输, 2=D2H结果回传

constexpr size_t kNumMergeStreams = 3;
