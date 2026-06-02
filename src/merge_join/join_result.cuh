// [AJB] merge_join/join_result.cuh: Join结果容器: 存储匹配的(rid, sid)对
#include <cstdio>
#pragma once

#include <vector>

template <typename T>
struct JoinMatch {
  explicit JoinMatch(const longlong4& ranges, const longlong2& bases)
      : r_first_(ranges.x + bases.x),
        r_last_(ranges.y + bases.x),
        s_first_(ranges.z + bases.y),
        s_last_(ranges.w + bases.y) {}

  JoinMatch(JoinMatch&&) = default;

  JoinMatch& operator=(JoinMatch&&) = default;

  size_t r_first_;
  size_t r_last_;
  size_t s_first_;
  size_t s_last_;
};

template <typename T>
struct JoinResult {
  explicit JoinResult(const size_t count = 0) : count_(count) {}

  JoinResult(JoinResult&&) = default;

  JoinResult& operator=(JoinResult&&) = default;

  size_t count_;
  std::vector<JoinMatch<T>> items_;
};

// [AJB] merge_join_join_result 诊断报告
static inline void ajb_report_merge_join_join_result(size_t n, double elapsed_ms, const char* phase) {
    fprintf(stderr, "[AJB_TIMER][merge_join_join_result] %s: n=%zu elapsed=%.3fms throughput=%.2f M/s\n",
            phase, n, elapsed_ms, elapsed_ms > 0 ? n / elapsed_ms / 1000.0 : 0.0);
}
