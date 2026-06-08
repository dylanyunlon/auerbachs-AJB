#pragma once
#include "common/ajb_debug_infra.cuh"

#include <vector>
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <numeric>

#include "error_utilities.cuh"

class StreamPool {
 public:
  StreamPool() = default;

  ~StreamPool() {
    PrintUtilizationReport();
    Free();
  }

  void Initialize(size_t num_streams, int device_id = 0) {
    device_id_ = device_id;
#ifdef DEBUG_BUILD
    printf("[StreamPool] Initialize with %lu streams on dev=%d.\n", num_streams, device_id);
#endif
    if (streams_.size() < num_streams) {
      Free();

      // Query device priority range for priority stream separation
      int lo_pri = 0, hi_pri = 0;
      CheckCudaError(cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri));
      // hi_pri is numerically lower = higher priority

      // Allocate priority (transfer) streams: first priority_count_ streams
      priority_count_ = (num_streams >= 4) ? (num_streams / 3) : 1;
      if (priority_count_ < 1) priority_count_ = 1;
      size_t normal_count = num_streams - priority_count_;

      streams_.resize(num_streams);
      acquire_count_.assign(num_streams, 0);
      release_count_.assign(num_streams, 0);
      cumulative_idle_ns_.assign(num_streams, 0);

      // Create priority (transfer) streams with high priority
      for (size_t i = 0; i < priority_count_; ++i) {
        CheckCudaError(cudaStreamCreateWithPriority(&streams_[i], cudaStreamNonBlocking, hi_pri));
      }
      // Create normal (compute) streams with default priority
      for (size_t i = priority_count_; i < num_streams; ++i) {
        CheckCudaError(cudaStreamCreateWithPriority(&streams_[i], cudaStreamNonBlocking, lo_pri));
      }

      // Create events for idle-time measurement per stream
      acquire_events_.resize(num_streams);
      release_events_.resize(num_streams);
      has_pending_release_.assign(num_streams, false);
      for (size_t i = 0; i < num_streams; ++i) {
        CheckCudaError(cudaEventCreateWithFlags(&acquire_events_[i], cudaEventDefault));
        CheckCudaError(cudaEventCreateWithFlags(&release_events_[i], cudaEventDefault));
      }

      capacity_ = num_streams;
      lifetime_start_ = std::chrono::steady_clock::now();
    }
  }

  void Free() {
    for (size_t i = 0; i < acquire_events_.size(); ++i) {
      cudaEventDestroy(acquire_events_[i]);
      cudaEventDestroy(release_events_[i]);
    }
    acquire_events_.clear();
    release_events_.clear();
    has_pending_release_.clear();

    for (cudaStream_t& stream : streams_) {
      CheckCudaError(cudaStreamDestroy(stream));
    }
    streams_.clear();
    streams_.shrink_to_fit();
    acquire_count_.clear();
    release_count_.clear();
    cumulative_idle_ns_.clear();
    capacity_ = 0;
    priority_count_ = 0;
  }

  // Acquire a stream by index with event timestamp and utilization tracking
  cudaStream_t GetStream(size_t index) {
    assert(index < streams_.size() && "StreamPool: index out of bounds");

    // If there was a previous release event, measure idle gap
    if (has_pending_release_[index]) {
      float idle_ms = 0.0f;
      cudaError_t err = cudaEventElapsedTime(&idle_ms, release_events_[index], acquire_events_[index]);
      if (err == cudaSuccess && idle_ms > 0.0f) {
        cumulative_idle_ns_[index] += static_cast<uint64_t>(idle_ms * 1e6);
      }
      has_pending_release_[index] = false;
    }

    // Record acquire event on this stream
    CheckCudaError(cudaEventRecord(acquire_events_[index], streams_[index]));
    acquire_count_[index]++;
    return streams_[index];
  }

  // Return a stream after use — records release event for idle tracking
  void ReturnStream(size_t index) {
    assert(index < streams_.size() && "StreamPool: return index out of bounds");
    CheckCudaError(cudaEventRecord(release_events_[index], streams_[index]));
    has_pending_release_[index] = true;
    release_count_[index]++;
  }

  // Get a high-priority stream for transfer operations
  cudaStream_t GetTransferStream() {
    assert(priority_count_ > 0 && "StreamPool: no priority streams");
    // Pick the least-acquired among priority streams
    size_t best = 0;
    for (size_t i = 1; i < priority_count_; ++i) {
      if (acquire_count_[i] < acquire_count_[best]) best = i;
    }
    return GetStream(best);
  }

  // Get a normal-priority stream for compute operations
  cudaStream_t GetComputeStream() {
    assert(priority_count_ < streams_.size() && "StreamPool: no compute streams");
    size_t best = priority_count_;
    for (size_t i = priority_count_ + 1; i < streams_.size(); ++i) {
      if (acquire_count_[i] < acquire_count_[best]) best = i;
    }
    return GetStream(best);
  }

  // round-robin: pick least-acquired stream across all pools
  cudaStream_t GetLeastUsed() {
    assert(!streams_.empty() && "StreamPool: no streams allocated");
    size_t min_idx = 0;
    size_t min_val = acquire_count_[0];
    for (size_t i = 1; i < acquire_count_.size(); i++) {
      if (acquire_count_[i] < min_val) {
        min_val = acquire_count_[i];
        min_idx = i;
      }
    }
    return GetStream(min_idx);
  }

  size_t Size() const { return streams_.size(); }
  size_t PriorityCount() const { return priority_count_; }
  size_t ComputeCount() const { return streams_.size() - priority_count_; }

  // Print utilization statistics at destruction or on demand
  void PrintUtilizationReport(int override_dev = -1) const {
    if (streams_.empty()) return;
    int dev = (override_dev >= 0) ? override_dev : device_id_;

    size_t total_acq = 0, total_rel = 0, peak_acq = 0;
    for (size_t i = 0; i < streams_.size(); ++i) {
      total_acq += acquire_count_[i];
      total_rel += release_count_[i];
      if (acquire_count_[i] > peak_acq) peak_acq = acquire_count_[i];
    }
    double avg_acq = (double)total_acq / streams_.size();
    double imbalance = avg_acq > 0.0 ? (double)peak_acq / avg_acq : 0.0;

    // Compute total idle fraction across all streams
    uint64_t total_idle_ns = 0;
    for (auto ns : cumulative_idle_ns_) total_idle_ns += ns;

    auto wall_elapsed = std::chrono::steady_clock::now() - lifetime_start_;
    uint64_t wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_elapsed).count();
    uint64_t total_wall_ns = wall_ns * streams_.size();
    double idle_pct = (total_wall_ns > 0) ? 100.0 * total_idle_ns / total_wall_ns : 0.0;

    fprintf(stderr, "[AJB_BP][StreamPool] dev=%d streams=%zu(pri=%zu,compute=%zu) "
            "acquires=%zu releases=%zu avg_acq=%.1f peak=%zu imbalance=%.2f idle_pct=%.1f%%\n",
            dev, streams_.size(), priority_count_, streams_.size() - priority_count_,
            total_acq, total_rel, avg_acq, peak_acq, imbalance, idle_pct);

    // Per-stream detail
    for (size_t i = 0; i < streams_.size(); ++i) {
      const char* pool_label = (i < priority_count_) ? "transfer" : "compute";
      double stream_idle_pct = (wall_ns > 0) ? 100.0 * cumulative_idle_ns_[i] / wall_ns : 0.0;
      fprintf(stderr, "[AJB_BP][StreamPool]   stream[%zu](%s) acq=%zu rel=%zu idle=%.1f%%\n",
              i, pool_label, acquire_count_[i], release_count_[i], stream_idle_pct);
    }
  }

  // Legacy compat
  void DumpUsage(int device_id = 0) const {
    PrintUtilizationReport(device_id);
  }

 private:
  std::vector<cudaStream_t> streams_;
  std::vector<size_t> acquire_count_;
  std::vector<size_t> release_count_;
  size_t capacity_ = 0;
  size_t priority_count_ = 0;  // number of high-priority (transfer) streams
  int device_id_ = 0;

  // Event-based idle time tracking
  std::vector<cudaEvent_t> acquire_events_;
  std::vector<cudaEvent_t> release_events_;
  std::vector<bool> has_pending_release_;
  std::vector<uint64_t> cumulative_idle_ns_;
  std::chrono::steady_clock::time_point lifetime_start_;
};
