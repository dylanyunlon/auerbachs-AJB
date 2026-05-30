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
