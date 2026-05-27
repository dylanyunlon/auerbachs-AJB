#pragma once

#include <chrono>
#include <map>
#include <string>

#include <nvtx3/nvToolsExt.h>

class TimeDurations {
 public:
  static TimeDurations& Get() {
    static TimeDurations instance;
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
