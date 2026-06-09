#pragma once
// AJB M1275: profile_utilities — hierarchical timer tree + phase overlap detection
//
// Upstream: flat map of tag→duration, no hierarchy, no overlap detection.
// AJB改写:
//   1. Meyer's singleton thread-safe (C++11保证, upstream也是, 保留)
//   2. Tic/Toc累积 + 新增per-invocation tracking (count+min+max)
//   3. PrintAllDurations: 表格输出含调用次数/均值/最大值
//   4. ExportCSV: CSV导出所有timer数据
//   5. HierarchicalScope: 用显式栈跟踪嵌套scope, 输出缩进树
//   6. OverlapDetector: 检测两个phase是否在时间上重叠执行
//      (并行pipeline中compute和transfer重叠是期望行为, 但sort和join重叠是bug)
//   7. RingBufferTimer: 滑动窗口统计 (mean/var/p95)
//   8. EmaAnomalyDetector: 指数移动均值异常检测

#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <stack>

#include <nvtx3/nvToolsExt.h>

class TimeDurations {
 public:
  static TimeDurations& Get() {
    static TimeDurations instance;  // Meyer's singleton
    return instance;
  }

  TimeDurations(const TimeDurations&) = delete;
  void operator=(const TimeDurations&) = delete;

  inline void Tic(const std::string& tag) {
    begin_times_[tag] = std::chrono::high_resolution_clock::now();
    // 记录区间起点用于overlap检测
    active_intervals_[tag] = begin_times_[tag];
  }

  inline std::chrono::duration<double> Toc(const std::string& tag) {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = now - begin_times_[tag];
    double dur_sec = duration.count();

    auto iter = durations_.find(tag);
    if (iter == durations_.end()) {
      durations_[tag] = duration;
    } else {
      iter->second += duration;
    }

    // Per-invocation统计
    auto& stats = invocation_stats_[tag];
    stats.count++;
    stats.total_sec += dur_sec;
    if (dur_sec < stats.min_sec || stats.count == 1) stats.min_sec = dur_sec;
    if (dur_sec > stats.max_sec) stats.max_sec = dur_sec;

    // 清除active区间
    active_intervals_.erase(tag);

    return duration;
  }

  double GetDuration(const std::string& tag) { std::chrono::duration<double> d = durations_[tag]; return d.count(); }

  double GetTotalDuration() {
    double total = 0.0;
    for (auto it = durations_.begin(); it != durations_.end(); ++it) {
      std::chrono::duration<double> d = it->second;
      total += d.count();
    }
    return total;
  }

  // 带invocation统计的表格输出
  void PrintAllDurations(const char* header = "TimeDurations") {
    fprintf(stderr, "\n[%s] Accumulated timing breakdown:\n", header);
    fprintf(stderr, "  %-30s %8s %12s %12s %12s %12s\n",
            "Phase", "Calls", "Total(s)", "Mean(s)", "Min(s)", "Max(s)");
    fprintf(stderr, "  %-30s %8s %12s %12s %12s %12s\n",
            "-----", "-----", "--------", "-------", "------", "------");
    double total = 0.0;
    for (auto it = durations_.begin(); it != durations_.end(); ++it) {
      std::string tag = it->first;
      std::chrono::duration<double> dur = it->second;
      auto sit = invocation_stats_.find(tag);
      if (sit != invocation_stats_.end()) {
        const InvocationStats& s = sit->second;
        fprintf(stderr, "  %-30s %8zu %12.6f %12.6f %12.6f %12.6f\n",
                tag.c_str(), s.count, s.total_sec,
                s.count > 0 ? s.total_sec / s.count : 0.0,
                s.min_sec, s.max_sec);
      } else {
        fprintf(stderr, "  %-30s %8s %12.6f\n", tag.c_str(), "1", dur.count());
      }
      total += dur.count();
    }
    fprintf(stderr, "  %-30s %8s %12.6f\n", "TOTAL", "", total);
    fprintf(stderr, "\n");
  }

  // CSV导出
  void ExportCSV(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) { fprintf(stderr, "[TimeDurations] Cannot open %s\n", path.c_str()); return; }
    fprintf(fp, "phase,calls,total_sec,mean_sec,min_sec,max_sec\n");
    for (auto it = durations_.begin(); it != durations_.end(); ++it) {
      std::string tag = it->first;
      std::chrono::duration<double> dur = it->second;
      auto sit = invocation_stats_.find(tag);
      if (sit != invocation_stats_.end()) {
        const InvocationStats& s = sit->second;
        fprintf(fp, "%s,%zu,%.9f,%.9f,%.9f,%.9f\n",
                tag.c_str(), s.count, s.total_sec,
                s.count > 0 ? s.total_sec / s.count : 0.0,
                s.min_sec, s.max_sec);
      } else {
        double dur_sec = dur.count();
        fprintf(fp, "%s,1,%.9f,%.9f,%.9f,%.9f\n",
                tag.c_str(), dur_sec, dur_sec, dur_sec, dur_sec);
      }
    }
    fclose(fp);
    fprintf(stderr, "[AJB_STATE][TimeDurations] Exported to %s\n", path.c_str());
  }

  bool HasDuration(const std::string& tag) const {
    return durations_.find(tag) != durations_.end();
  }

  void Reset() {
    begin_times_.clear();
    durations_.clear();
    invocation_stats_.clear();
    active_intervals_.clear();
  }

  // Overlap检测: 检查两个tag是否同时在Tic状态(=并行执行)
  // 返回值: 当前有多少个phase同时活跃
  size_t ActivePhaseCount() const { return active_intervals_.size(); }

  void DumpActivePhases() const {
    if (active_intervals_.empty()) return;
    fprintf(stderr, "[AJB_BP][Overlap] %zu phases active simultaneously:",
            active_intervals_.size());
    for (auto it = active_intervals_.begin(); it != active_intervals_.end(); ++it) {
      fprintf(stderr, " [%s]", it->first.c_str());
    }
    fprintf(stderr, "\n");
  }

 private:
  TimeDurations() = default;
  ~TimeDurations() = default;

  struct InvocationStats {
    size_t count = 0;
    double total_sec = 0.0;
    double min_sec = 1e30;
    double max_sec = 0.0;
  };

  std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> begin_times_;
  std::map<std::string, std::chrono::duration<double>> durations_;
  std::map<std::string, InvocationStats> invocation_stats_;
  // 当前正在计时的phase (Tic后Toc前)
  std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> active_intervals_;
};

// RAII scope timer with NVTX annotation
struct TimeScope {
  explicit TimeScope(const std::string& tag) : tag_(tag) {
    TimeDurations::Get().Tic(tag_);
    nvtxRangePushA(tag_.c_str());
  }

  ~TimeScope() {
    nvtxRangePop();
    TimeDurations::Get().Toc(tag_);
  }

 private:
  std::string tag_;
};

// Hierarchical scope: 自动跟踪嵌套深度,输出缩进树
// 用法: { HierarchicalScope hs("outer"); { HierarchicalScope hs2("inner"); } }
// 输出: [AJB_TIMER] outer: 1.234s
//       [AJB_TIMER]   inner: 0.567s
class HierarchicalScope {
 public:
  explicit HierarchicalScope(const std::string& tag)
      : tag_(tag), depth_(current_depth_++) {
    start_ = std::chrono::high_resolution_clock::now();
    nvtxRangePushA(tag_.c_str());
  }

  ~HierarchicalScope() {
    nvtxRangePop();
    auto elapsed = std::chrono::high_resolution_clock::now() - start_;
    double sec = std::chrono::duration<double>(elapsed).count();
    // 缩进: 每层2空格
    char indent[64];
    int indent_len = depth_ * 2;
    if (indent_len > 60) indent_len = 60;
    memset(indent, ' ', indent_len);
    indent[indent_len] = '\0';
    fprintf(stderr, "[AJB_TIMER] %s%s: %.6fs\n", indent, tag_.c_str(), sec);
    current_depth_--;
  }

 private:
  std::string tag_;
  int depth_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  static inline int current_depth_ = 0;  // C++17 inline static
};

// Ring buffer timer for sliding window statistics
class RingBufferTimer {
 public:
  explicit RingBufferTimer(size_t capacity = 128) : cap_(capacity) {
    samples_.reserve(capacity);
  }

  void Record(double duration_seconds) {
    if (samples_.size() < cap_) {
      samples_.push_back(duration_seconds);
    } else {
      samples_[write_pos_ % cap_] = duration_seconds;
    }
    write_pos_++;
    total_recorded_++;
  }

  size_t Count() const { return std::min(total_recorded_, cap_); }

  // Welford在线均值 (对ring buffer使用two-pass, buffer已在内存中)
  double Mean() const {
    size_t n = Count();
    if (n == 0) return 0.0;
    // Kahan补偿求和避免浮点精度丢失
    double sum = 0.0, comp = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double y = samples_[i] - comp;
      double t = sum + y;
      comp = (t - sum) - y;
      sum = t;
    }
    return sum / n;
  }

  double Variance() const {
    size_t n = Count();
    if (n <= 1) return 0.0;
    double m = Mean();
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double d = samples_[i] - m;
      sum_sq += d * d;
    }
    return sum_sq / (n - 1);
  }

  double Stddev() const { return std::sqrt(std::max(0.0, Variance())); }

  // P95 via partial_sort (O(n) amortized instead of full sort)
  double Percentile95() const {
    size_t n = Count();
    if (n == 0) return 0.0;
    std::vector<double> sorted(samples_.begin(), samples_.begin() + n);
    size_t idx = static_cast<size_t>(0.95 * (n - 1));
    std::nth_element(sorted.begin(), sorted.begin() + idx, sorted.end());
    return sorted[idx];
  }

  void DumpStats(const char* label) const {
    fprintf(stderr, "[AJB_BP][RingTimer][%s] n=%zu mean=%.6fs stddev=%.6fs p95=%.6fs\n",
            label, Count(), Mean(), Stddev(), Percentile95());
  }

 private:
  size_t cap_;
  size_t write_pos_ = 0;
  size_t total_recorded_ = 0;
  std::vector<double> samples_;
};

// EMA-based anomaly detector
class EmaAnomalyDetector {
 public:
  explicit EmaAnomalyDetector(double alpha = 0.1, double k_sigma = 3.0)
      : alpha_(alpha), k_(k_sigma) {}

  bool Observe(double duration) {
    total_observations_++;
    if (total_observations_ == 1) {
      ema_mean_ = duration;
      ema_var_ = 0.0;
      return false;
    }
    double prev_mean = ema_mean_;
    ema_mean_ = alpha_ * duration + (1.0 - alpha_) * ema_mean_;
    double diff = duration - prev_mean;
    ema_var_ = (1.0 - alpha_) * (ema_var_ + alpha_ * diff * diff);

    double ewma_std = std::sqrt(ema_var_);
    double threshold = ema_mean_ + k_ * ewma_std;

    if (duration > threshold && ewma_std > 1e-12) {
      anomaly_count_++;
      fprintf(stderr, "[AJB_BP][EmaAnomaly] ALERT: duration=%.6fs > threshold=%.6fs "
              "(ema=%.6f ewma_std=%.6f k=%.1f count=%zu)\n",
              duration, threshold, ema_mean_, ewma_std, k_, anomaly_count_);
      return true;
    }
    return false;
  }

  double ema_mean() const { return ema_mean_; }
  double ema_std() const { return std::sqrt(ema_var_); }
  size_t anomaly_count() const { return anomaly_count_; }

 private:
  double alpha_, k_;
  double ema_mean_ = 0.0, ema_var_ = 0.0;
  size_t total_observations_ = 0, anomaly_count_ = 0;
};

// Phase overlap detector: 记录phase的[start,end]区间, 检测哪些对存在时间重叠
class PhaseOverlapDetector {
 public:
  void RecordPhase(const std::string& tag, double start_sec, double end_sec) {
    intervals_.push_back({tag, start_sec, end_sec});
  }

  // 扫描所有phase对, 报告重叠
  void DetectOverlaps() const {
    size_t n = intervals_.size();
    int overlaps_found = 0;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        // 区间重叠条件: a.start < b.end && b.start < a.end
        if (intervals_[i].start < intervals_[j].end &&
            intervals_[j].start < intervals_[i].end) {
          double overlap_sec = std::min(intervals_[i].end, intervals_[j].end)
                             - std::max(intervals_[i].start, intervals_[j].start);
          fprintf(stderr, "[AJB_BP][Overlap] %s [%.3f-%.3f] ∩ %s [%.3f-%.3f] = %.6fs\n",
                  intervals_[i].tag.c_str(), intervals_[i].start, intervals_[i].end,
                  intervals_[j].tag.c_str(), intervals_[j].start, intervals_[j].end,
                  overlap_sec);
          overlaps_found++;
        }
      }
    }
    fprintf(stderr, "[AJB_STATE][OverlapDetector] total_phases=%zu overlapping_pairs=%d\n",
            n, overlaps_found);
  }

 private:
  struct Interval { std::string tag; double start; double end; };
  std::vector<Interval> intervals_;
};
