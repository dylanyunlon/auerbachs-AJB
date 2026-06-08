#pragma once
// [AJB_BP] profile_utilities v2: extended precision timing

#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <nvtx3/nvToolsExt.h>

class TimeDurations {
 public:
  static TimeDurations& Get() {
    static TimeDurations instance;  // AJB-algo: Meyer's singleton — thread-safe in C++11
    return instance;
  }

  TimeDurations(const TimeDurations& time_durations) = delete;
  void operator=(const TimeDurations& time_durations) = delete;

  inline void Tic(const std::string& tag) { begin_times_[tag] = std::chrono::high_resolution_clock::now(); }

  inline std::chrono::duration<double> Toc(const std::string& tag) {
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - begin_times_[tag];

    auto iter = durations_.find(tag);
    if (iter == durations_.end()) {
      durations_[tag] = duration;
    } else {
      iter->second += duration;
    }

    return duration;
  }

  double GetDuration(const std::string& tag) { return durations_[tag].count(); }

  double GetTotalDuration() {
    auto total_duration = std::chrono::duration<double>::zero();
    for (const auto& [tag, duration] : durations_) {
      total_duration += duration;
    }
    return total_duration.count();
  }

  // ---- AJB extensions ----

  // Print ALL accumulated timing tags (for end-of-run diagnostics)
  void PrintAllDurations(const char* header = "TimeDurations") {
    printf("\n[%s] Accumulated timing breakdown:\n", header);
    printf("  %-30s  %12s\n", "Phase", "Seconds");
    printf("  %-30s  %12s\n", "-----", "-------");
    double total = 0.0;
    for (const auto& [tag, dur] : durations_) {
      printf("  %-30s  %12.6f\n", tag.c_str(), dur.count());
      total += dur.count();
    }
    printf("  %-30s  %12.6f\n", "TOTAL", total);
    printf("\n");
  }

  // Export all timing data as CSV
  void ExportCSV(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) {
      printf("[TimeDurations] Cannot open %s\n", path.c_str());
      return;
    }
    fprintf(fp, "phase,duration_seconds\n");
    for (const auto& [tag, dur] : durations_) {
      fprintf(fp, "%s,%.9f\n", tag.c_str(), dur.count());
    }
    fclose(fp);
    printf("[TimeDurations] Exported to %s\n", path.c_str());
  }

  // Check if a specific phase has been timed
  bool HasDuration(const std::string& tag) const {
    return durations_.find(tag) != durations_.end();
  }

  // Reset all timers (useful for repeated benchmark runs)
  void Reset() {
    begin_times_.clear();
    durations_.clear();
  }

 private:
  TimeDurations() = default;
  ~TimeDurations() = default;

  std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> begin_times_;
  std::map<std::string, std::chrono::duration<double>> durations_;
};

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

// --- M1115: Ring buffer timer for sliding window statistics ---
// Tracks the most recent N samples of a named phase, providing
// mean/variance/p95 over a moving window instead of cumulative stats.
// Useful for detecting performance regression during long runs.
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

  double Mean() const {
    size_t n = Count();
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += samples_[i];
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

  double Stddev() const {
    double v = Variance();
    return v > 0.0 ? std::sqrt(v) : 0.0;
  }

  // Approximate P95 via sorted sample (exact for window sizes < 1000)
  double Percentile95() const {
    size_t n = Count();
    if (n == 0) return 0.0;
    std::vector<double> sorted(samples_.begin(), samples_.begin() + n);
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(0.95 * (n - 1));
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

// --- M1115: EMA-based anomaly detector ---
// Maintains exponential moving average and EWMA standard deviation.
// Flags a sample as anomalous when duration > ema_mean + k * ewma_std.
// Default k=3 (three-sigma rule).
class EmaAnomalyDetector {
 public:
  explicit EmaAnomalyDetector(double alpha = 0.1, double k_sigma = 3.0)
      : alpha_(alpha), k_(k_sigma) {}

  // Returns true if the sample is anomalous
  bool Observe(double duration) {
    total_observations_++;
    if (total_observations_ == 1) {
      ema_mean_ = duration;
      ema_var_ = 0.0;
      return false;  // first sample is never anomalous
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
  double alpha_;
  double k_;
  double ema_mean_ = 0.0;
  double ema_var_ = 0.0;
  size_t total_observations_ = 0;
  size_t anomaly_count_ = 0;
};
