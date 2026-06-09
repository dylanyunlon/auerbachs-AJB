#pragma once
// AJB M1320: profile_utilities — minimal nvcc 11.5 compatible version
// 完全避免: auto推导map迭代器, std::chrono在map中, 链式成员调用
// 只用: double数组, char*数组, 索引遍历

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include <nvtx3/nvToolsExt.h>

class TimeDurations {
 public:
  static TimeDurations& Get() {
    static TimeDurations instance;
    return instance;
  }

  TimeDurations(const TimeDurations&) = delete;
  void operator=(const TimeDurations&) = delete;

  void Tic(const std::string& tag) {
    int idx = FindOrCreate(tag);
    starts_[idx] = std::chrono::high_resolution_clock::now();
    active_[idx] = true;
  }

  double Toc(const std::string& tag) {
    std::chrono::time_point<std::chrono::high_resolution_clock> now =
        std::chrono::high_resolution_clock::now();
    int idx = FindOrCreate(tag);
    std::chrono::duration<double> elapsed = now - starts_[idx];
    double sec = elapsed.count();

    totals_[idx] += sec;
    counts_[idx]++;
    if (sec < mins_[idx] || counts_[idx] == 1) mins_[idx] = sec;
    if (sec > maxs_[idx]) maxs_[idx] = sec;
    active_[idx] = false;

    return sec;
  }

  double GetDuration(const std::string& tag) {
    int idx = Find(tag);
    return (idx >= 0) ? totals_[idx] : 0.0;
  }

  double GetTotalDuration() {
    double t = 0.0;
    for (int i = 0; i < n_; i++) t += totals_[i];
    return t;
  }

  void PrintAllDurations(const char* header = "TimeDurations") {
    fprintf(stderr, "\n[%s] Accumulated timing breakdown:\n", header);
    fprintf(stderr, "  %-30s %8s %12s %12s %12s %12s\n",
            "Phase", "Calls", "Total(s)", "Mean(s)", "Min(s)", "Max(s)");
    fprintf(stderr, "  %-30s %8s %12s %12s %12s %12s\n",
            "-----", "-----", "--------", "-------", "------", "------");
    double total = 0.0;
    for (int i = 0; i < n_; i++) {
      double mean = counts_[i] > 0 ? totals_[i] / counts_[i] : 0.0;
      fprintf(stderr, "  %-30s %8zu %12.6f %12.6f %12.6f %12.6f\n",
              names_[i], counts_[i], totals_[i], mean, mins_[i], maxs_[i]);
      total += totals_[i];
    }
    fprintf(stderr, "  %-30s %8s %12.6f\n", "TOTAL", "", total);
    fprintf(stderr, "\n");
  }

  void ExportCSV(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) return;
    fprintf(fp, "phase,calls,total_sec,mean_sec,min_sec,max_sec\n");
    for (int i = 0; i < n_; i++) {
      double mean = counts_[i] > 0 ? totals_[i] / counts_[i] : 0.0;
      fprintf(fp, "%s,%zu,%.9f,%.9f,%.9f,%.9f\n",
              names_[i], counts_[i], totals_[i], mean, mins_[i], maxs_[i]);
    }
    fclose(fp);
  }

  bool HasDuration(const std::string& tag) {
    return Find(tag) >= 0;
  }

  void Reset() { n_ = 0; }

  size_t ActivePhaseCount() {
    size_t c = 0;
    for (int i = 0; i < n_; i++) if (active_[i]) c++;
    return c;
  }

  void DumpActivePhases() {
    bool any = false;
    for (int i = 0; i < n_; i++) {
      if (active_[i]) {
        if (!any) { fprintf(stderr, "[AJB_BP][Overlap] Active:"); any = true; }
        fprintf(stderr, " [%s]", names_[i]);
      }
    }
    if (any) fprintf(stderr, "\n");
  }

 private:
  TimeDurations() : n_(0) {}
  ~TimeDurations() = default;

  static const int MAX_TAGS = 256;
  int n_;
  char names_[MAX_TAGS][64];
  double totals_[MAX_TAGS];
  double mins_[MAX_TAGS];
  double maxs_[MAX_TAGS];
  size_t counts_[MAX_TAGS];
  bool active_[MAX_TAGS];
  std::chrono::time_point<std::chrono::high_resolution_clock> starts_[MAX_TAGS];

  int Find(const std::string& tag) {
    for (int i = 0; i < n_; i++) {
      if (strncmp(names_[i], tag.c_str(), 63) == 0) return i;
    }
    return -1;
  }

  int FindOrCreate(const std::string& tag) {
    int idx = Find(tag);
    if (idx >= 0) return idx;
    if (n_ >= MAX_TAGS) return 0;  // overflow safety
    idx = n_++;
    strncpy(names_[idx], tag.c_str(), 63);
    names_[idx][63] = '\0';
    totals_[idx] = 0.0;
    mins_[idx] = 1e30;
    maxs_[idx] = 0.0;
    counts_[idx] = 0;
    active_[idx] = false;
    return idx;
  }
};

// RAII scope timer with NVTX
struct TimeScope {
  explicit TimeScope(const std::string& tag) : tag_(tag) {
    TimeDurations::Get().Tic(tag_);
    nvtxRangePushA(tag_.c_str());
  }
  ~TimeScope() {
    nvtxRangePop();
    TimeDurations::Get().Toc(tag_);
  }
  std::string tag_;
};

// 便捷宏
#define AJB_TIME_SCOPE(name) TimeScope _ajb_ts_##__LINE__(name)
#define AJB_TIC(name) TimeDurations::Get().Tic(name)
#define AJB_TOC(name) TimeDurations::Get().Toc(name)
