#pragma once

#include <vector>
#include <algorithm>
#include <numeric>

template <typename T>
struct JoinMatch {
  // Upstream用4个独立size_t存储range端点,
  // AJB改为: 只存相对偏移(delta编码), 在访问时还原.
  // 原因: merge join产生大量JoinMatch, delta编码在skew场景下
  // 压缩率高(相邻match的base接近), 减少内存带宽压力.
  explicit JoinMatch(const longlong4& ranges, const longlong2& bases)
      : r_base_(bases.x),
        r_off_first_(static_cast<uint32_t>(ranges.x)),
        r_off_last_(static_cast<uint32_t>(ranges.y)),
        s_base_(bases.y),
        s_off_first_(static_cast<uint32_t>(ranges.z)),
        s_off_last_(static_cast<uint32_t>(ranges.w)) {}

  JoinMatch(JoinMatch&&) = default;
  JoinMatch& operator=(JoinMatch&&) = default;

  // 还原为绝对坐标 — 调用侧语义不变
  size_t r_first() const { return r_base_ + r_off_first_; }
  size_t r_last()  const { return r_base_ + r_off_last_; }
  size_t s_first() const { return s_base_ + s_off_first_; }
  size_t s_last()  const { return s_base_ + s_off_last_; }

  // 这个match覆盖的笛卡尔积大小
  size_t cardinality() const {
    return (r_last() - r_first()) * (s_last() - s_first());
  }

  // 向后兼容: 保留原字段名作为 inline accessor
  size_t r_first_ __attribute__((deprecated("use r_first()"))) = 0;
  size_t r_last_  __attribute__((deprecated("use r_last()")))  = 0;
  size_t s_first_ __attribute__((deprecated("use s_first()"))) = 0;
  size_t s_last_  __attribute__((deprecated("use s_last()")))  = 0;

private:
  size_t   r_base_;
  uint32_t r_off_first_;
  uint32_t r_off_last_;
  size_t   s_base_;
  uint32_t s_off_first_;
  uint32_t s_off_last_;
};

template <typename T>
struct JoinResult {
  explicit JoinResult(const size_t count = 0) : count_(count) {}

  JoinResult(JoinResult&&) = default;
  JoinResult& operator=(JoinResult&&) = default;

  size_t count_;
  std::vector<JoinMatch<T>> items_;

  // --- AJB算法补充: 结果集统计, 用于skew检测决策 ---

  // 返回所有match中最大的单个笛卡尔积
  size_t max_cardinality() const {
    size_t mx = 0;
    for (size_t i = 0; i < items_.size(); ++i) {
      size_t c = items_[i].cardinality();
      if (c > mx) mx = c;
    }
    return mx;
  }

  // 累计笛卡尔积 — O(n)遍历, 不缓存, 因为items_可能还在增长
  size_t total_cardinality() const {
    size_t sum = 0;
    for (size_t i = 0; i < items_.size(); ++i) {
      sum += items_[i].cardinality();
    }
    return sum;
  }

  // Gini系数: 衡量match间cardinality的不均匀程度
  // 0.0 = 完全均匀, 1.0 = 完全集中在一个match
  // skew_detector用这个决定是否切换到adaptive merge路径
  double cardinality_gini() const {
    if (items_.size() <= 1) return 0.0;
    const size_t n = items_.size();
    std::vector<size_t> cards(n);
    for (size_t i = 0; i < n; ++i) cards[i] = items_[i].cardinality();
    std::sort(cards.begin(), cards.end());

    double sum_of_abs_diff = 0.0;
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
      total += cards[i];
      // Gini公式的离散形式: Σ(2i - n - 1) * x_i
      sum_of_abs_diff += static_cast<double>(2 * i - n + 1) * cards[i];
    }
    if (total < 1.0) return 0.0;
    return sum_of_abs_diff / (n * total);
  }

  // 断点调试: 打印结果集的完整统计快照
  void debug_dump(const char* label = "JoinResult") const {
    fprintf(stderr, "[DEBUG][%s] count=%zu items=%zu total_card=%zu max_card=%zu gini=%.4f\n",
            label, count_, items_.size(), total_cardinality(), max_cardinality(),
            cardinality_gini());
    // 打印前3个和后3个match的详细信息
    size_t show = std::min<size_t>(items_.size(), 3);
    for (size_t i = 0; i < show; ++i) {
      fprintf(stderr, "  [%zu] R[%zu..%zu) S[%zu..%zu) card=%zu\n",
              i, items_[i].r_first(), items_[i].r_last(),
              items_[i].s_first(), items_[i].s_last(),
              items_[i].cardinality());
    }
    if (items_.size() > 6) fprintf(stderr, "  ... (%zu omitted)\n", items_.size() - 6);
    for (size_t i = (items_.size() > 3 ? items_.size() - 3 : show); i < items_.size(); ++i) {
      fprintf(stderr, "  [%zu] R[%zu..%zu) S[%zu..%zu) card=%zu\n",
              i, items_[i].r_first(), items_[i].r_last(),
              items_[i].s_first(), items_[i].s_last(),
              items_[i].cardinality());
    }
  }
};

// --- M1123: Flajolet-Martin probabilistic distinct count sketch ---
// Estimates the number of distinct join keys without storing all keys.
// Uses multiple hash functions (simulated via seed variation) and takes
// the harmonic mean of the max-trailing-zeros estimates (LogLog variant).
// Memory: O(num_buckets) instead of O(distinct_keys).
class FlajoletMartinSketch {
 public:
  explicit FlajoletMartinSketch(size_t num_buckets = 64)
      : num_buckets_(num_buckets), max_trailing_(num_buckets, 0) {}

  void Insert(size_t key_hash) {
    // Use top bits for bucket selection, bottom bits for trailing zeros
    size_t bucket = key_hash % num_buckets_;
    // Count trailing zeros as the rank estimator
    int tz = 0;
    size_t shifted = key_hash / num_buckets_;  // remove bucket bits
    if (shifted == 0) {
      tz = 0;
    } else {
      while ((shifted & 1) == 0) { tz++; shifted >>= 1; }
    }
    if (tz > max_trailing_[bucket]) {
      max_trailing_[bucket] = tz;
    }
    total_inserted_++;
  }

  // Estimate distinct count using harmonic mean (SuperLogLog correction)
  double EstimateDistinct() const {
    // Harmonic mean of 2^max_trailing across buckets
    double sum_inv = 0.0;
    int active_buckets = 0;
    for (size_t b = 0; b < num_buckets_; ++b) {
      if (max_trailing_[b] > 0 || total_inserted_ > 0) {
        sum_inv += 1.0 / (1ULL << max_trailing_[b]);
        active_buckets++;
      }
    }
    if (active_buckets == 0 || sum_inv <= 0.0) return 0.0;

    // Alpha correction factor (depends on bucket count)
    double alpha = 0.7213 / (1.0 + 1.079 / num_buckets_);
    double estimate = alpha * active_buckets * active_buckets / sum_inv;

    fprintf(stderr, "[AJB_BP][FMSketch] buckets=%zu inserted=%zu estimate=%.0f active=%d\n",
            num_buckets_, total_inserted_, estimate, active_buckets);
    return estimate;
  }

  // Merge two sketches (for parallel processing)
  void Merge(const FlajoletMartinSketch& other) {
    for (size_t b = 0; b < num_buckets_ && b < other.num_buckets_; ++b) {
      if (other.max_trailing_[b] > max_trailing_[b]) {
        max_trailing_[b] = other.max_trailing_[b];
      }
    }
    total_inserted_ += other.total_inserted_;
  }

 private:
  size_t num_buckets_;
  std::vector<int> max_trailing_;
  size_t total_inserted_ = 0;
};
