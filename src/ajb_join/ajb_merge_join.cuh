#pragma once
// =============================================================================
// ajb_merge_join.cuh — AJB Adaptive Merge Join Pipeline
//
// Adapts the upstream MergeJoin() from merge_join/merge_join.cuh by inserting
// tier-aware transfer scheduling into the chunk-group loop. The core merge-path
// partitioning and GPU kernel dispatch are RETAINED (~80%); the new ~20% is:
//
//  1. TierTransferScheduler integration (ShouldTransfer / RecordTransfer)
//  2. Per-chunk-group debug dump of partition state (partition sizes,
//     boundary vectors, imbalance metrics)
//  3. Staleness tracking for merge-path boundaries (how stale are the
//     boundaries when we skip a transfer?)
//  4. Comprehensive timing breakdown per chunk group
//
// Build: include this instead of merge_join.cuh in the AJB benchmark.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

#include "common/device_allocator.cuh"
#include "common/error_utilities.cuh"
#include "common/host_allocator.cuh"
#include "common/math_utilities.cuh"
#include "common/parallel_algorithms.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/stream_pool.cuh"

// Upstream merge-join internals (kernel dispatch, partition logic)
#include "merge_join/constants.cuh"
#include "merge_join/join_result.cuh"
#include "merge_join/kernels.cuh"
#include "merge_join/merge_join.cuh"

// AJB-specific
#include "ajb_join/tier_transfer_scheduler.cuh"

// ---------------------------------------------------------------------------
// Per-device partition state — what AJB tracks beyond the upstream join
// ---------------------------------------------------------------------------
template <typename T>
struct DevicePartitionState {
  size_t device_id;
  size_t n_r_local;         // local R partition size (tuples)
  size_t n_s_local;         // local S partition size (tuples)
  size_t boundary_version;  // how many times the boundary vector was synced
  size_t partition_version; // how many times the build partition was synced
  double imbalance;         // |local_tuples - ideal_tuples| / ideal_tuples

  void DebugPrint() const {
    printf("  [DeviceState] GPU %zu | R_local=%zu S_local=%zu | "
           "boundary_v=%zu partition_v=%zu | imbalance=%.4f\n",
           device_id, n_r_local, n_s_local,
           boundary_version, partition_version, imbalance);
  }
};

// ---------------------------------------------------------------------------
// Chunk-group debug snapshot — printed/logged every N chunk groups
// ---------------------------------------------------------------------------
template <typename T>
struct ChunkGroupSnapshot {
  size_t chunk_group_index;
  double elapsed_seconds;
  std::vector<DevicePartitionState<T>> device_states;
  bool transferred_x;
  bool transferred_u;
  bool transferred_v;
  size_t cumulative_slow_bytes;
  size_t cumulative_fast_bytes;

  void DebugPrint() const {
    printf("\n--- Chunk Group %zu (t=%.4f s) ---\n",
           chunk_group_index, elapsed_seconds);
    printf("  Transfers this round: x=%s u=%s v=%s\n",
           transferred_x ? "YES" : "skip",
           transferred_u ? "YES" : "skip",
           transferred_v ? "YES" : "skip");
    printf("  Cumulative bytes: slow=%.2f MB fast=%.2f MB\n",
           cumulative_slow_bytes / 1e6, cumulative_fast_bytes / 1e6);
    for (const auto& ds : device_states) {
      ds.DebugPrint();
    }
    printf("---\n\n");
  }
};

// ---------------------------------------------------------------------------
// AJB adaptive merge join — the main entry point
//
// This replaces MergeJoin() from the upstream. The signature is intentionally
// wider: it takes a TierTransferScheduler and returns both the JoinResult
// and a vector of chunk-group snapshots for post-hoc analysis.
// ---------------------------------------------------------------------------
template <typename T, typename V>
JoinResult<T> AJBMergeJoin(
    PinnedVector<T>& keys_r, PinnedVector<V>& values_r,
    PinnedVector<T>& keys_s, PinnedVector<V>& values_s,
    const std::vector<int>& gpus,
    std::vector<DeviceAllocator>& device_allocators,
    std::vector<StreamPool>& stream_pools,
    TierTransferScheduler& scheduler,
    const bool materialize,
    std::vector<ChunkGroupSnapshot<T>>* snapshots = nullptr,
    size_t debug_print_interval = 10) {

  const size_t device_count = gpus.size();

  printf("\n");
  printf("######################################################################\n");
  printf("#  AJBMergeJoin — Bandwidth-Tier-Adaptive Join Pipeline              #\n");
  printf("######################################################################\n");
  printf("  R tuples:     %zu\n", keys_r.size());
  printf("  S tuples:     %zu\n", keys_s.size());
  printf("  Devices:      %zu\n", device_count);
  printf("  Materialize:  %s\n", materialize ? "yes" : "no");
  printf("  Debug interval: every %zu chunk groups\n", debug_print_interval);
  scheduler.GetCadence().DebugPrint("  Active Cadence");
  printf("######################################################################\n\n");

  // --- Phase 0: Device initialization (retained from upstream) ---
  for (size_t g = 0; g < device_count; ++g) {
    #ifdef __CUDACC__
    CheckCudaError(cudaSetDevice(gpus[g]));
    constexpr double kDeviceMemoryUtilization = 0.98;
    size_t free_global_memory, total_global_memory;
    CheckCudaError(cudaMemGetInfo(&free_global_memory, &total_global_memory));
    device_allocators[g].Initialize(kDeviceMemoryUtilization * free_global_memory);
    stream_pools[g].Initialize(kNumJoinStreams);
    printf("  [Init] GPU %d: %.2f GB free / %.2f GB total\n",
           gpus[g],
           free_global_memory / 1e9,
           total_global_memory / 1e9);
    #else
    printf("  [Init] GPU %d: (no CUDA runtime — CPU-only mode)\n", gpus[g]);
    #endif
  }

  auto pipeline_start = std::chrono::high_resolution_clock::now();
  TimeScope time_scope("join_phase");

  // --- Phase 1: Merge-path partitioning across devices (upstream logic) ---
  const size_t per_device_length =
      (keys_r.size() + keys_s.size() + device_count - 1) / device_count;

  std::vector<JoinResult<T>> device_results(device_count);
  std::vector<longlong2> starts(device_count);
  std::vector<longlong2> ends(device_count);

  printf("  [Partition] per_device_length=%zu\n", per_device_length);

  for (size_t i_device = 0; i_device < device_count; ++i_device) {
    if (i_device == 0) {
      starts[i_device] = {0, 0};
    } else {
      starts[i_device] = ends[i_device - 1];
    }

    const long long diagonal = (i_device + 1) * per_device_length;
    if (diagonal >= static_cast<long long>(keys_r.size() + keys_s.size())) {
      ends[i_device].x = keys_r.size();
      ends[i_device].y = keys_s.size();
    } else {
      // Merge-path partitioning (upstream algorithm, unchanged)
      #ifdef __CUDACC__
      ends[i_device].x = mgpu::merge_path<mgpu::bounds_t::bounds_upper>(
          keys_r.data(), static_cast<long long>(keys_r.size()),
          keys_s.data(), static_cast<long long>(keys_s.size()),
          diagonal, mgpu::less_t<T>());
      ends[i_device].y = diagonal - ends[i_device].x;
      #else
      // CPU fallback: simple diagonal split
      ends[i_device].x = std::min<long long>(diagonal, keys_r.size());
      ends[i_device].y = diagonal - ends[i_device].x;
      #endif
    }

    printf("  [Partition] Device %zu: R[%lld..%lld) S[%lld..%lld) "
           "= %lld R-tuples + %lld S-tuples\n",
           i_device,
           starts[i_device].x, ends[i_device].x,
           starts[i_device].y, ends[i_device].y,
           ends[i_device].x - starts[i_device].x,
           ends[i_device].y - starts[i_device].y);
  }

  // Handle cross-boundary key ranges (upstream utility, unchanged)
  JoinResult<T> answer;
  HandleLongKeyRanges(keys_r.data(), keys_s.data(),
                      keys_r.size(), keys_s.size(),
                      {0, 0}, starts.data(), ends.data(),
                      device_count, answer, materialize);

  printf("  [LongKeyRanges] Pre-join matches from boundary handling: %zu\n",
         answer.count_);

  // --- Phase 2: AJB chunk-group loop with tier-aware transfers ---
  //
  // This is where AJB diverges from upstream. Instead of a single
  // #pragma omp parallel for, we iterate chunk groups explicitly and
  // interleave transfer decisions via the scheduler.
  //
  // The number of "chunk groups" is determined by the per-device workload
  // divided by chunk_size. For the merge join, we treat each device's
  // iteration count as one chunk group.
  // ---------------------------------------------------------------------------

  const size_t total_chunk_groups = device_count;  // simplified; real AJB subdivides further
  printf("\n  [AJBLoop] Starting %zu chunk groups with adaptive transfer...\n\n",
         total_chunk_groups);

  for (size_t t = 0; t < total_chunk_groups; ++t) {
    auto cg_start = std::chrono::high_resolution_clock::now();

    // --- Transfer decisions ---
    bool do_x = scheduler.ShouldTransfer(
        TransferEvent::StructureKind::kBuildPartition, t);
    bool do_u = scheduler.ShouldTransfer(
        TransferEvent::StructureKind::kMergePathBoundary, t);
    bool do_v = scheduler.ShouldTransfer(
        TransferEvent::StructureKind::kMaterializationBuffer, t);

    // --- Simulate cross-device transfers ---
    // In a real CUDA deployment, these would be cudaMemcpyPeerAsync calls
    // routed through the appropriate tier. Here we measure the overhead
    // and record it for the figure_data_emitter.

    const long long n_r_device = ends[t].x - starts[t].x;
    const long long n_s_device = ends[t].y - starts[t].y;

    if (do_x && t > 0) {
      // Build partition transfer: large, goes over slow tier if cross-island
      size_t partition_bytes = (n_r_device + n_s_device) * sizeof(T);
      auto tx_start = std::chrono::high_resolution_clock::now();
      // In real code: cudaMemcpyPeerAsync(...)
      auto tx_end = std::chrono::high_resolution_clock::now();
      double tx_dur = std::chrono::duration<double>(tx_end - tx_start).count();

      scheduler.RecordTransfer(
          TransferEvent::StructureKind::kBuildPartition,
          gpus[t > 0 ? t - 1 : 0], gpus[t], t,
          partition_bytes, tx_dur);
    }

    if (do_u && t > 0) {
      // Boundary vector transfer: small (just offsets), can use fast tier
      size_t boundary_bytes = 2 * sizeof(longlong2);  // start + end
      auto tx_start = std::chrono::high_resolution_clock::now();
      auto tx_end = std::chrono::high_resolution_clock::now();
      double tx_dur = std::chrono::duration<double>(tx_end - tx_start).count();

      scheduler.RecordTransfer(
          TransferEvent::StructureKind::kMergePathBoundary,
          gpus[t > 0 ? t - 1 : 0], gpus[t], t,
          boundary_bytes, tx_dur);
    }

    if (do_v) {
      // Materialization buffer: streaming results, always transferred
      size_t mat_bytes = n_r_device * sizeof(T);  // approximate
      auto tx_start = std::chrono::high_resolution_clock::now();
      auto tx_end = std::chrono::high_resolution_clock::now();
      double tx_dur = std::chrono::duration<double>(tx_end - tx_start).count();

      scheduler.RecordTransfer(
          TransferEvent::StructureKind::kMaterializationBuffer,
          gpus[t], -1, t,  // to host
          mat_bytes, tx_dur);
    }

    // --- Dispatch per-device join (upstream logic, unchanged) ---
    if (n_r_device > 0 && n_s_device > 0) {
      #ifdef __CUDACC__
      device_results[t] = GlobalJoin(
          keys_r.data() + starts[t].x,
          keys_s.data() + starts[t].y,
          n_r_device, n_s_device,
          starts[t],
          gpus[t],
          device_allocators[t],
          stream_pools[t],
          materialize);
      #else
      // CPU fallback: brute-force nested-loop join for testing
      size_t local_count = 0;
      for (long long r = starts[t].x; r < ends[t].x; ++r) {
        for (long long s = starts[t].y; s < ends[t].y; ++s) {
          if (keys_r[r] == keys_s[s]) local_count++;
        }
      }
      device_results[t] = JoinResult<T>(local_count);
      #endif
    }

    auto cg_end = std::chrono::high_resolution_clock::now();
    double cg_duration = std::chrono::duration<double>(cg_end - cg_start).count();

    // --- Build debug snapshot ---
    if (snapshots || (t % debug_print_interval == 0)) {
      double elapsed = std::chrono::duration<double>(cg_end - pipeline_start).count();

      // Compute per-device imbalance
      double ideal_per_device = static_cast<double>(keys_r.size() + keys_s.size()) / device_count;

      ChunkGroupSnapshot<T> snap;
      snap.chunk_group_index = t;
      snap.elapsed_seconds = elapsed;
      snap.transferred_x = do_x;
      snap.transferred_u = do_u;
      snap.transferred_v = do_v;
      snap.cumulative_slow_bytes = scheduler.GetSlowTierBytes();
      snap.cumulative_fast_bytes = scheduler.GetFastTierBytes();

      for (size_t d = 0; d < device_count; ++d) {
        double local_total = (ends[d].x - starts[d].x) + (ends[d].y - starts[d].y);
        DevicePartitionState<T> ds;
        ds.device_id = d;
        ds.n_r_local = ends[d].x - starts[d].x;
        ds.n_s_local = ends[d].y - starts[d].y;
        ds.boundary_version = (do_u || t == 0) ? t + 1 : t;
        ds.partition_version = (do_x || t == 0) ? t + 1 : t;
        ds.imbalance = (ideal_per_device > 0)
            ? std::fabs(local_total - ideal_per_device) / ideal_per_device
            : 0.0;
        snap.device_states.push_back(ds);
      }

      if (t % debug_print_interval == 0) {
        snap.DebugPrint();
        printf("  Chunk group %zu completed in %.6f s "
               "(join_count_so_far=%zu)\n",
               t, cg_duration,
               answer.count_ + device_results[t].count_);
      }

      if (snapshots) {
        snapshots->push_back(snap);
      }
    }
  }

  // --- Phase 3: Aggregate results (upstream logic, unchanged) ---
  for (size_t i_device = 0; i_device < device_count; ++i_device) {
    answer.count_ += device_results[i_device].count_;
  }

  if (materialize) {
    answer.items_.reserve(answer.count_);
    for (size_t i_device = 0; i_device < device_count; ++i_device) {
      for (auto& item : device_results[i_device].items_) {
        answer.items_.emplace_back(std::move(item));
      }
      device_results[i_device].items_.clear();
      device_results[i_device].items_.shrink_to_fit();
    }
  }

  auto pipeline_end = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration<double>(pipeline_end - pipeline_start).count();

  printf("\n");
  printf("######################################################################\n");
  printf("#  AJBMergeJoin COMPLETE                                             #\n");
  printf("######################################################################\n");
  printf("  Total matches:    %zu\n", answer.count_);
  printf("  Pipeline time:    %.6f s\n", total_time);
  printf("  Materialized:     %s (%zu items)\n",
         materialize ? "yes" : "no", answer.items_.size());
  printf("######################################################################\n");

  // Print the transfer summary
  scheduler.PrintSummary();

  return answer;
}
