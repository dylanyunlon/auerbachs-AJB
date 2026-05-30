#pragma once

#include <algorithm>
#include <iostream>
#include <string>

#include <termcolor/termcolor.hpp>

#include "pinned_vector.cuh"

template <typename T>
bool IsSortedPrintFailures(const PinnedVector<T>& vector) {
  bool is_sorted = true;
  for (size_t i = 1; i < vector.size(); ++i) {
    if (vector[i - 1] > vector[i]) {
      const size_t begin = std::max<size_t>(0, i - 5);
      const size_t end = std::min<size_t>(i + 5, vector.size() - 1);

      std::cout << "[ERROR] IsSortedPrintFailures: ";
      for (size_t j = begin; j < end; ++j) {
        if (i == j) {
          std::cout << termcolor::red;
        }
        std::cout << vector[j];
        if (i == j) {
          std::cout << termcolor::reset;
        }
        std::cout << ", ";
      }
      std::cout << "i = " << termcolor::red << i << termcolor::reset << std::endl;
      is_sorted = false;
    }
  }
  return is_sorted;
}

template <typename T>
void PrintVector(const std::string& name, const PinnedVector<T>& vector) {
  std::cout << name << ": ";
  for (size_t i = 0; i < vector.size(); ++i) {
    std::cout << vector[i];
    if (i < vector.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << std::endl;
}

// =============================================================================
// AJB Debug Extensions — breakpoint-style state dumps for development
// =============================================================================

#include <chrono>
#include <cstdio>
#include <map>
#include <vector>

// ---------------------------------------------------------------------------
// AJB_BREAKPOINT: conditional print that dumps file, line, and a message.
// Use like: AJB_BREAKPOINT("partition done, n_r=%zu", n_r);
// When AJB_DEBUG is defined, this pauses and waits for ENTER (interactive).
// Without AJB_DEBUG, it just prints.
// ---------------------------------------------------------------------------
#ifdef AJB_DEBUG
#define AJB_BREAKPOINT(fmt, ...)                                             \
  do {                                                                       \
    fprintf(stderr,                                                          \
            "\n=== AJB BREAKPOINT === %s:%d [%s]\n  " fmt "\n"               \
            "  Press ENTER to continue...\n",                                \
            __FILE__, __LINE__, __func__, ##__VA_ARGS__);                    \
    getchar();                                                               \
  } while (0)
#else
#define AJB_BREAKPOINT(fmt, ...)                                             \
  do {                                                                       \
    fprintf(stderr,                                                          \
            "[AJB] %s:%d [%s] " fmt "\n",                                    \
            __FILE__, __LINE__, __func__, ##__VA_ARGS__);                    \
  } while (0)
#endif

// ---------------------------------------------------------------------------
// AJB_DUMP_ARRAY: print the first/last N elements of a raw array
// ---------------------------------------------------------------------------
template <typename T>
void AJBDumpArray(const char* name, const T* data, size_t size,
                  size_t head = 5, size_t tail = 5) {
  printf("[AJB_DUMP] %s (size=%zu): [", name, size);
  size_t show_head = std::min(head, size);
  for (size_t i = 0; i < show_head; ++i) {
    printf("%s%s", std::to_string(data[i]).c_str(),
           (i + 1 < show_head) ? ", " : "");
  }
  if (size > head + tail) {
    printf(" ... ");
  }
  if (size > head) {
    size_t start = std::max(head, size - tail);
    for (size_t i = start; i < size; ++i) {
      printf("%s%s", std::to_string(data[i]).c_str(),
             (i + 1 < size) ? ", " : "");
    }
  }
  printf("]\n");
}

#define AJB_DUMP_VEC(vec) AJBDumpArray(#vec, (vec).data(), (vec).size())

// ---------------------------------------------------------------------------
// AJB_TIMER: scoped timer that prints elapsed time on destruction
// ---------------------------------------------------------------------------
struct AJBTimer {
  const char* label;
  std::chrono::high_resolution_clock::time_point start;

  explicit AJBTimer(const char* lbl) : label(lbl) {
    start = std::chrono::high_resolution_clock::now();
    printf("[AJB_TIMER] >>> %s started\n", label);
  }

  ~AJBTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("[AJB_TIMER] <<< %s finished in %.6f s\n", label, elapsed);
  }
};

// ---------------------------------------------------------------------------
// Memory usage reporter (CPU side)
// ---------------------------------------------------------------------------
inline void AJBReportMemory(const char* label = "MemReport") {
  FILE* f = fopen("/proc/self/status", "r");
  if (!f) {
    printf("[%s] Cannot read /proc/self/status\n", label);
    return;
  }

  char line[256];
  size_t vm_rss = 0, vm_size = 0;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "VmRSS:", 6) == 0)  sscanf(line + 6, "%zu", &vm_rss);
    if (strncmp(line, "VmSize:", 7) == 0)  sscanf(line + 7, "%zu", &vm_size);
  }
  fclose(f);

  printf("[%s] VmRSS=%zu kB (%.2f MB) | VmSize=%zu kB (%.2f GB)\n",
         label, vm_rss, vm_rss / 1024.0, vm_size, vm_size / 1048576.0);
}

// ---------------------------------------------------------------------------
// GPU memory reporter (per-device)
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
inline void AJBReportGPUMemory(int device_id, const char* label = "GPUMem") {
  cudaSetDevice(device_id);
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  printf("[%s] GPU %d: free=%.2f GB / total=%.2f GB (%.1f%% used)\n",
         label, device_id,
         free_mem / 1e9, total_mem / 1e9,
         100.0 * (1.0 - static_cast<double>(free_mem) / total_mem));
}
#else
inline void AJBReportGPUMemory(int device_id, const char* label = "GPUMem") {
  printf("[%s] GPU %d: (no CUDA runtime)\n", label, device_id);
}
#endif

// ---------------------------------------------------------------------------
// Cumulative event counter — track how often something happens
// ---------------------------------------------------------------------------
class AJBEventCounter {
 public:
  void Count(const std::string& event) { counts_[event]++; }

  void PrintAll(const char* label = "EventCounts") const {
    printf("[%s]\n", label);
    for (const auto& [name, count] : counts_) {
      printf("  %-40s : %zu\n", name.c_str(), count);
    }
  }

  size_t Get(const std::string& event) const {
    auto it = counts_.find(event);
    return (it != counts_.end()) ? it->second : 0;
  }

 private:
  std::map<std::string, size_t> counts_;
};

// Global event counter (opt-in via AJB_ENABLE_COUNTERS)
#ifdef AJB_ENABLE_COUNTERS
inline AJBEventCounter& GetGlobalCounter() {
  static AJBEventCounter counter;
  return counter;
}
#define AJB_COUNT(event) GetGlobalCounter().Count(event)
#define AJB_PRINT_COUNTS() GetGlobalCounter().PrintAll("GlobalEventCounts")
#else
#define AJB_COUNT(event) ((void)0)
#define AJB_PRINT_COUNTS() ((void)0)
#endif
