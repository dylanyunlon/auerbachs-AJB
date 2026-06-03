#pragma once
// ==========================================================================
// ajb_debug_infra.cuh — AJB unified runtime diagnostics infrastructure
//
// Provides structured state snapshots, breakpoint-style dump macros, and
// timing instrumentation that works on both CPU and GPU paths.
//
// NOT fprintf noise. Each macro captures structured data you can:
//   1. grep in logs:  grep '\[AJB_' run.log
//   2. set gdb watchpoints on: watch ajb_bp_counter
//   3. parse programmatically: scripts/debug/parse_ajb_trace.py
// ==========================================================================

#include <cstdio>
#include <cstdint>
#include <chrono>

// Global breakpoint counter — set a gdb watchpoint on this to break on
// any AJB diagnostic event: (gdb) watch ajb_bp_counter
static thread_local uint64_t ajb_bp_counter = 0;

// AJB_SNAP: print a structured state snapshot and bump the bp counter.
// Format: [AJB_SNAP][file:line][tag] key=val key=val ...
#define AJB_SNAP(tag, fmt, ...) do { \
    ++ajb_bp_counter; \
    fprintf(stderr, "[AJB_SNAP][%s:%d][%s] " fmt "\n", \
            __FILE__, __LINE__, tag, ##__VA_ARGS__); \
} while(0)

// AJB_TIMER_BEGIN / AJB_TIMER_END: bracket a section with wall-clock timing.
// Usage:
//   AJB_TIMER_BEGIN(my_section);
//   ... code ...
//   AJB_TIMER_END(my_section, "sort phase");
#define AJB_TIMER_BEGIN(name) \
    auto ajb_timer_##name = std::chrono::steady_clock::now()

#define AJB_TIMER_END(name, tag) do { \
    auto ajb_timer_end_##name = std::chrono::steady_clock::now(); \
    double ajb_ms_##name = std::chrono::duration<double, std::milli>( \
        ajb_timer_end_##name - ajb_timer_##name).count(); \
    AJB_SNAP(tag, "elapsed=%.3fms", ajb_ms_##name); \
} while(0)

// AJB_ARRAY_STATS: print min/max/sum of a numeric array (CPU side).
// Useful for dumping histogram distributions, key ranges, etc.
template<typename T>
static inline void ajb_array_stats(const char* tag, const T* arr, size_t n,
                                    FILE* out = stderr) {
    if (n == 0) { fprintf(out, "[AJB_ARRAY][%s] empty\n", tag); return; }
    T mn = arr[0], mx = arr[0];
    double sum = 0;
    size_t zeros = 0;
    for (size_t i = 0; i < n; ++i) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
        sum += static_cast<double>(arr[i]);
        if (arr[i] == 0) zeros++;
    }
    fprintf(out, "[AJB_ARRAY][%s] n=%zu min=%lld max=%lld sum=%.0f zeros=%zu "
            "mean=%.2f\n", tag, n, (long long)mn, (long long)mx, sum, zeros,
            sum / n);
}

// AJB_VALIDATE_SORTED: strided sortedness check with early abort.
// Returns true if sorted. Prints first violation.
template<typename T>
static inline bool ajb_validate_sorted(const char* tag, const T* keys,
                                        size_t n, size_t stride = 1000) {
    if (n < 2) return true;
    size_t violations = 0, first_v = 0, checked = 0;
    for (size_t i = 1; i < n; i += stride) {
        checked++;
        if (keys[i] < keys[i - 1]) {
            if (violations == 0) first_v = i;
            if (++violations >= 10) break;
        }
    }
    fprintf(stderr, "[AJB_VALIDATE][%s] n=%zu stride=%zu checked=%zu "
            "violations=%zu %s\n", tag, n, stride, checked, violations,
            violations == 0 ? "PASS" : "FAIL");
    if (violations > 0)
        fprintf(stderr, "[AJB_VALIDATE][%s] first@%zu: [%zu]=%lld > [%zu]=%lld\n",
                tag, first_v, first_v-1, (long long)keys[first_v-1],
                first_v, (long long)keys[first_v]);
    return violations == 0;
}
