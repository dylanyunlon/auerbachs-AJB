#pragma once
// [AJB_BP] profile_utilities v2: extended precision timing

#include <chrono>
#include <map>
#include <string>

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
