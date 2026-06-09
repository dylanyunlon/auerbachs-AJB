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

// ==========================================================================
// AjbTraceRing — lock-free single-producer circular buffer for trace events.
//
// Records the last N trace events with minimal overhead. When the ring is
// full, old entries are silently overwritten. This lets you do a "flight
// recorder" style dump of the last N events on crash or assertion failure.
// ==========================================================================
struct AjbTraceEntry {
    uint64_t sequence;         // monotonic event counter
    uint64_t timestamp_us;     // microseconds since epoch
    char tag[32];              // e.g. "Transfer", "Sort", "Join"
    char message[128];         // structured payload
};

class AjbTraceRing {
  public:
    explicit AjbTraceRing(size_t capacity = 1024)
        : ring_(capacity), capacity_(capacity), head_(0) {}

    // Record one event (single-producer, safe without locks)
    void Record(const char* tag, const char* fmt, ...) {
        size_t slot = head_ % capacity_;
        AjbTraceEntry& e = ring_[slot];
        e.sequence = head_++;
        auto now = std::chrono::steady_clock::now();
        e.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
        snprintf(e.tag, sizeof(e.tag), "%s", tag);

        va_list args;
        va_start(args, fmt);
        vsnprintf(e.message, sizeof(e.message), fmt, args);
        va_end(args);
    }

    // Dump the last `count` entries to stderr (newest first)
    void Dump(size_t count = 50) const {
        size_t start = (head_ > count) ? head_ - count : 0;
        fprintf(stderr, "[AjbTraceRing] dumping %zu/%zu events (head=%zu):\n",
                std::min(count, head_), head_, head_);
        for (size_t i = start; i < head_; ++i) {
            const AjbTraceEntry& e = ring_[i % capacity_];
            fprintf(stderr, "  #%zu [%s] %s\n", e.sequence, e.tag, e.message);
        }
    }

    size_t Size() const { return std::min(head_, capacity_); }
    size_t TotalRecorded() const { return head_; }

  private:
    std::vector<AjbTraceEntry> ring_;
    size_t capacity_;
    size_t head_;  // single-producer: no atomics needed
};

// Global trace ring (for crash dumps)
static AjbTraceRing g_ajb_trace_ring(2048);

#define AJB_TRACE(tag, fmt, ...) \
    g_ajb_trace_ring.Record(tag, fmt, ##__VA_ARGS__)

// ==========================================================================
// AjbPhaseTimer — hierarchical timer with a stack of nested phases.
//
// Usage:
//   AjbPhaseTimer timer;
//   timer.Enter("sort");
//     timer.Enter("radix_pass_1");
//     timer.Leave();  // radix_pass_1
//     timer.Enter("merge_pass");
//     timer.Leave();  // merge_pass
//   timer.Leave();  // sort
//   timer.PrintTree();  // shows indented timing tree
// ==========================================================================
class AjbPhaseTimer {
  public:
    struct PhaseRecord {
        const char* name;
        int depth;
        double start_ms;
        double end_ms;
        double elapsed_ms() const { return end_ms - start_ms; }
    };

    AjbPhaseTimer() : base_(std::chrono::steady_clock::now()), depth_(0) {}

    void Enter(const char* name) {
        double now_ms = ElapsedMs();
        records_.push_back({name, depth_, now_ms, 0.0});
        stack_.push_back(records_.size() - 1);
        depth_++;

        AJB_TRACE("PhaseEnter", "depth=%d name=%s", depth_ - 1, name);
    }

    void Leave() {
        if (stack_.empty()) return;
        double now_ms = ElapsedMs();
        size_t idx = stack_.back();
        stack_.pop_back();
        records_[idx].end_ms = now_ms;
        depth_--;

        fprintf(stderr, "[AJB_TIMER][Phase] %*s%s: %.3f ms\n",
                records_[idx].depth * 2, "", records_[idx].name,
                records_[idx].elapsed_ms());
        AJB_TRACE("PhaseLeave", "%s %.3fms", records_[idx].name,
                  records_[idx].elapsed_ms());
    }

    void PrintTree() const {
        fprintf(stderr, "\n[AJB_TIMER] === Phase Tree ===\n");
        double total = records_.empty() ? 0.0 : records_[0].elapsed_ms();
        for (const auto& r : records_) {
            double pct = (total > 0) ? 100.0 * r.elapsed_ms() / total : 0.0;
            fprintf(stderr, "[AJB_TIMER] %*s%-24s %8.3f ms (%5.1f%%)\n",
                    r.depth * 2, "", r.name, r.elapsed_ms(), pct);
        }
        fprintf(stderr, "[AJB_TIMER] === End Phase Tree ===\n\n");
    }

    const std::vector<PhaseRecord>& Records() const { return records_; }

  private:
    double ElapsedMs() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - base_).count();
    }

    std::chrono::steady_clock::time_point base_;
    std::vector<PhaseRecord> records_;
    std::vector<size_t> stack_;  // indices into records_
    int depth_;
};

// ==========================================================================
// AjbMemoryWatermark — track peak RSS and per-phase memory deltas.
//
// Reads /proc/self/status on Linux for VmRSS. On non-Linux, returns 0.
// ==========================================================================
class AjbMemoryWatermark {
  public:
    AjbMemoryWatermark() : peak_kb_(0), baseline_kb_(0) {
        baseline_kb_ = CurrentRSSKB();
        peak_kb_ = baseline_kb_;
    }

    // Check current RSS and update peak if new high
    size_t Check(const char* label = nullptr) {
        size_t now = CurrentRSSKB();
        bool new_peak = (now > peak_kb_);
        if (new_peak) peak_kb_ = now;

        if (label && new_peak) {
            fprintf(stderr, "[AJB_MEM][Watermark] %s: RSS=%zu KB (NEW PEAK, "
                    "delta=+%zu KB from baseline)\n",
                    label, now, now - baseline_kb_);
            AJB_TRACE("MemPeak", "%s rss=%zuKB peak=%zuKB", label, now, peak_kb_);
        }
        return now;
    }

    size_t PeakKB() const { return peak_kb_; }
    size_t BaselineKB() const { return baseline_kb_; }
    size_t DeltaKB() const { return (peak_kb_ > baseline_kb_) ? peak_kb_ - baseline_kb_ : 0; }

    void PrintSummary() const {
        fprintf(stderr, "[AJB_MEM] baseline=%zu KB peak=%zu KB delta=%zu KB\n",
                baseline_kb_, peak_kb_, DeltaKB());
    }

  private:
    static size_t CurrentRSSKB() {
#ifdef __linux__
        FILE* fp = fopen("/proc/self/status", "r");
        if (!fp) return 0;
        char buf[256];
        size_t rss = 0;
        while (fgets(buf, sizeof(buf), fp)) {
            if (strncmp(buf, "VmRSS:", 6) == 0) {
                // "VmRSS:\t   12345 kB\n"
                const char* p = buf + 6;
                while (*p == ' ' || *p == '\t') ++p;
                rss = strtoull(p, nullptr, 10);
                break;
            }
        }
        fclose(fp);
        return rss;
#else
        return 0;
#endif
    }

    size_t peak_kb_;
    size_t baseline_kb_;
};

// Global memory watermark instance
static AjbMemoryWatermark g_ajb_mem_watermark;

#define AJB_MEM_CHECK(label) g_ajb_mem_watermark.Check(label)
#define AJB_MEM_SUMMARY() g_ajb_mem_watermark.PrintSummary()

