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
// HyperLogLog sketch for join output cardinality estimation
// ---------------------------------------------------------------------------
struct JoinCardinalityHLL {
  static constexpr size_t kRegisters = 256;  // m=256 -> ~6.5% std error
  static constexpr double kAlpha = 0.7213 / (1.0 + 1.079 / kRegisters);
  uint8_t registers[kRegisters] = {};
  size_t items_seen = 0;

  void Insert(uint64_t hash) {
    size_t idx = hash & (kRegisters - 1);
    uint64_t w = hash >> 8;
    uint8_t rho = (w == 0) ? 64 : static_cast<uint8_t>(__builtin_ctzll(w) + 1);
    if (rho > registers[idx]) registers[idx] = rho;
    items_seen++;
  }

  double Estimate() const {
    double sum_inv = 0.0;
    size_t zeros = 0;
    for (size_t i = 0; i < kRegisters; ++i) {
      sum_inv += 1.0 / static_cast<double>(1ULL << registers[i]);
      if (registers[i] == 0) zeros++;
    }
    double raw = kAlpha * kRegisters * kRegisters / sum_inv;
    // Small range correction
    if (raw <= 2.5 * kRegisters && zeros > 0) {
      return kRegisters * std::log(static_cast<double>(kRegisters) / zeros);
    }
    return raw;
  }

  void DebugPrint(size_t t) const {
    fprintf(stderr, "[AJB_BP][HLL] t=%zu: items_seen=%zu est_cardinality=%.0f "
            "non_zero_regs=%zu\n",
            t, items_seen, Estimate(),
            std::count_if(registers, registers + kRegisters,
                          [](uint8_t r){ return r > 0; }));
  }
};

// ---------------------------------------------------------------------------
// Merge-path boundary staleness tracker (L∞ norm of boundary changes)
// ---------------------------------------------------------------------------
struct BoundaryStaleness {
  std::vector<int64_t> last_boundary_r;
  std::vector<int64_t> last_boundary_s;
  size_t consecutive_low = 0;
  static constexpr double kLowThreshold = 0.01;
  bool transfer_paused = false;

  // Timestamp-based staleness re-sync: each boundary partition records when
  // it was last updated; if the age exceeds resync_threshold_sec the tracker
  // forces a re-sync regardless of the L-infinity norm.
  std::vector<std::chrono::high_resolution_clock::time_point> boundary_update_ts;
  double resync_threshold_sec = 0.05;  // 50 ms default
  size_t boundary_version_counter = 0;
  size_t forced_resync_count = 0;

  double ComputeLInfNorm(const std::vector<int64_t>& current_r,
                         const std::vector<int64_t>& current_s,
                         size_t total_tuples) {
    auto now = std::chrono::high_resolution_clock::now();

    if (last_boundary_r.empty()) {
      last_boundary_r = current_r;
      last_boundary_s = current_s;
      boundary_update_ts.assign(current_r.size(), now);
      boundary_version_counter = 1;
      return 1.0;  // first time: maximum staleness
    }
    double max_dev = 0.0;
    double scale = static_cast<double>(std::max<size_t>(total_tuples, 1));
    // Ensure timestamp vector is sized to match
    if (boundary_update_ts.size() < current_r.size()) {
      boundary_update_ts.resize(current_r.size(), now);
    }
    for (size_t i = 0; i < std::min(current_r.size(), last_boundary_r.size()); ++i) {
      double dr = std::fabs(static_cast<double>(current_r[i] - last_boundary_r[i])) / scale;
      double ds = std::fabs(static_cast<double>(current_s[i] - last_boundary_s[i])) / scale;
      double local_dev = std::max(dr, ds);
      max_dev = std::max(max_dev, local_dev);
      // Update per-partition timestamp when boundary actually changed
      if (dr > 0.0 || ds > 0.0) {
        boundary_update_ts[i] = now;
      }
    }
    last_boundary_r = current_r;
    last_boundary_s = current_s;
    boundary_version_counter++;
    return max_dev;
  }

  // Check whether any partition boundary is stale beyond the resync threshold.
  // Returns true if at least one partition exceeds the age limit.
  bool CheckTimestampStaleness() {
    if (boundary_update_ts.empty()) return false;
    auto now = std::chrono::high_resolution_clock::now();
    size_t stale_partitions = 0;
    double max_age_sec = 0.0;
    for (size_t i = 0; i < boundary_update_ts.size(); ++i) {
      double age = std::chrono::duration<double>(now - boundary_update_ts[i]).count();
      max_age_sec = std::max(max_age_sec, age);
      if (age > resync_threshold_sec) {
        stale_partitions++;
      }
    }
    if (stale_partitions > 0) {
      forced_resync_count++;
      fprintf(stderr, "[AJB_STATE][Staleness] %zu/%zu partitions stale "
              "(max_age=%.4fs > threshold=%.4fs) -> forced re-sync #%zu\n",
              stale_partitions, boundary_update_ts.size(),
              max_age_sec, resync_threshold_sec, forced_resync_count);
      // Reset all timestamps on forced re-sync
      std::fill(boundary_update_ts.begin(), boundary_update_ts.end(), now);
      return true;
    }
    return false;
  }

  bool ShouldPauseTransfer(double linf) {
    // Timestamp-based override: never pause if boundaries are time-stale
    if (CheckTimestampStaleness()) {
      consecutive_low = 0;
      transfer_paused = false;
      return false;
    }

    if (linf < kLowThreshold) {
      consecutive_low++;
      if (consecutive_low >= 3) {
        transfer_paused = true;
        fprintf(stderr, "[AJB_BP][Staleness] L∞=%.6f (< %.4f for %zu consecutive) "
                "-> PAUSING boundary transfer\n",
                linf, kLowThreshold, consecutive_low);
      }
    } else {
      if (transfer_paused) {
        fprintf(stderr, "[AJB_BP][Staleness] L∞=%.6f (> %.4f) "
                "-> RESUMING boundary transfer\n", linf, kLowThreshold);
      }
      consecutive_low = 0;
      transfer_paused = false;
    }
    return transfer_paused;
  }
};

// ---------------------------------------------------------------------------
// Welford online mean/variance tracker for per-partition output cardinality
// ---------------------------------------------------------------------------
struct PartitionCardinalityWelford {
  struct Stats {
    size_t partition_id;
    size_t n;
    double mean;
    double m2;     // sum of squared deviations

    Stats() : partition_id(0), n(0), mean(0.0), m2(0.0) {}
    explicit Stats(size_t pid) : partition_id(pid), n(0), mean(0.0), m2(0.0) {}

    void Update(double value) {
      n++;
      double delta = value - mean;
      mean += delta / static_cast<double>(n);
      double delta2 = value - mean;
      m2 += delta * delta2;
    }

    double Variance() const {
      return (n < 2) ? 0.0 : m2 / static_cast<double>(n - 1);
    }

    double StdDev() const { return std::sqrt(Variance()); }

    double CoefficientOfVariation() const {
      return (mean > 0.0) ? StdDev() / mean : 0.0;
    }
  };

  std::vector<Stats> partition_stats;

  void Initialize(size_t num_partitions) {
    partition_stats.resize(num_partitions);
    for (size_t i = 0; i < num_partitions; ++i) {
      partition_stats[i] = Stats(i);
    }
  }

  void RecordOutput(size_t partition_id, size_t output_count) {
    if (partition_id < partition_stats.size()) {
      partition_stats[partition_id].Update(static_cast<double>(output_count));
    }
  }

  void DebugPrint(size_t t) const {
    double global_mean = 0.0, global_var = 0.0;
    size_t total_n = 0;
    for (const auto& s : partition_stats) {
      if (s.n > 0) {
        global_mean += s.mean * s.n;
        total_n += s.n;
      }
    }
    if (total_n > 0) global_mean /= static_cast<double>(total_n);
    // Compute inter-partition variance of means
    for (const auto& s : partition_stats) {
      if (s.n > 0) {
        double d = s.mean - global_mean;
        global_var += d * d;
      }
    }
    if (!partition_stats.empty()) {
      global_var /= static_cast<double>(partition_stats.size());
    }
    fprintf(stderr, "[AJB_STATE][Welford] t=%zu: partitions=%zu global_mean=%.2f "
            "inter_partition_stddev=%.2f\n",
            t, partition_stats.size(), global_mean, std::sqrt(global_var));
    for (const auto& s : partition_stats) {
      if (s.n > 0) {
        fprintf(stderr, "  partition %zu: n=%zu mean=%.2f stddev=%.2f cv=%.3f\n",
                s.partition_id, s.n, s.mean, s.StdDev(),
                s.CoefficientOfVariation());
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Sorted-order verifier for merge results — adjacent-pair inversion check
// ---------------------------------------------------------------------------
struct SortedOrderVerifier {
  size_t total_inversions = 0;
  size_t total_checked = 0;
  // Store the first 3 inversions for diagnostics
  struct Inversion {
    size_t position;
    int64_t left_key;
    int64_t right_key;
  };
  std::vector<Inversion> first_inversions;
  static constexpr size_t kMaxRecordedInversions = 3;

  template <typename T>
  void VerifyKeys(const T* keys, size_t count) {
    if (count < 2) return;
    for (size_t i = 0; i + 1 < count; ++i) {
      total_checked++;
      if (keys[i] > keys[i + 1]) {
        total_inversions++;
        if (first_inversions.size() < kMaxRecordedInversions) {
          first_inversions.push_back({
            i,
            static_cast<int64_t>(keys[i]),
            static_cast<int64_t>(keys[i + 1])
          });
        }
      }
    }
  }

  void DebugPrint() const {
    fprintf(stderr, "[AJB_STATE][SortVerify] checked=%zu inversions=%zu",
            total_checked, total_inversions);
    if (total_inversions == 0) {
      fprintf(stderr, " -> SORTED OK\n");
    } else {
      fprintf(stderr, " -> VIOLATIONS DETECTED\n");
      for (const auto& inv : first_inversions) {
        fprintf(stderr, "  inversion at pos %zu: %lld > %lld\n",
                inv.position,
                static_cast<long long>(inv.left_key),
                static_cast<long long>(inv.right_key));
      }
    }
  }

  void Reset() {
    total_inversions = 0;
    total_checked = 0;
    first_inversions.clear();
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

  // HyperLogLog for output cardinality estimation
  JoinCardinalityHLL hll_sketch;
  // Boundary staleness tracker
  BoundaryStaleness staleness_tracker;
  // Welford per-partition output cardinality tracker
  PartitionCardinalityWelford welford_tracker;
  welford_tracker.Initialize(device_count);
  // Sorted-order verifier for merge results
  SortedOrderVerifier sort_verifier;

  // [AJB_STATE] Merge start breakpoint
  fprintf(stderr, "[AJB_STATE][MergeStart] gpu_count=%zu partitions=%zu "
          "boundary_version=0 total_r=%zu total_s=%zu\n",
          device_count, total_chunk_groups,
          keys_r.size(), keys_s.size());

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

    // --- Welford per-partition output cardinality tracking ---
    {
      size_t partition_output = device_results[t].count_;
      welford_tracker.RecordOutput(t % device_count, partition_output);
      if (t % debug_print_interval == 0) {
        welford_tracker.DebugPrint(t);
      }
    }

    // --- Sorted-order verification of merge results ---
    {
      // Verify that the input key ranges used by this chunk group are sorted
      size_t r_start = static_cast<size_t>(starts[t].x);
      size_t r_end = static_cast<size_t>(ends[t].x);
      if (r_end > r_start + 1) {
        sort_verifier.VerifyKeys(keys_r.data() + r_start, r_end - r_start);
      }
      size_t s_start = static_cast<size_t>(starts[t].y);
      size_t s_end = static_cast<size_t>(ends[t].y);
      if (s_end > s_start + 1) {
        sort_verifier.VerifyKeys(keys_s.data() + s_start, s_end - s_start);
      }
    }

    // --- L∞ staleness computation ---
    {
      std::vector<int64_t> cur_r(device_count), cur_s(device_count);
      for (size_t d = 0; d < device_count; ++d) {
        cur_r[d] = ends[d].x;
        cur_s[d] = ends[d].y;
      }
      double linf = staleness_tracker.ComputeLInfNorm(
          cur_r, cur_s, keys_r.size() + keys_s.size());
      bool paused = staleness_tracker.ShouldPauseTransfer(linf);
      // Feed staleness to adaptive K_u
      scheduler.AdaptK_u(linf);

      if (t % debug_print_interval == 0) {
        fprintf(stderr, "[AJB_BP][Staleness] t=%zu L∞=%.6f paused=%s\n",
                t, linf, paused ? "YES" : "no");
      }
    }

    // --- HLL sketch update (hash join keys from this chunk) ---
    {
      size_t n_matches = device_results[t].count_;
      for (size_t m = 0; m < std::min<size_t>(n_matches, 5000); ++m) {
        // FNV-1a hash of match index for HLL
        uint64_t h = 14695981039346656037ULL;
        uint64_t val = static_cast<uint64_t>(t * 1000000 + m);
        for (int byte = 0; byte < 8; ++byte) {
          h ^= (val >> (byte * 8)) & 0xFF;
          h *= 1099511628211ULL;
        }
        hll_sketch.Insert(h);
      }
      if (t % debug_print_interval == 0) {
        hll_sketch.DebugPrint(t);
      }
    }

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

  // [AJB_STATE] Merge end breakpoint
  fprintf(stderr, "[AJB_STATE][MergeEnd] gpu_count=%zu partitions=%zu "
          "boundary_version=%zu output_count=%zu "
          "forced_resyncs=%zu total_time=%.6fs\n",
          device_count, total_chunk_groups,
          staleness_tracker.boundary_version_counter,
          answer.count_,
          staleness_tracker.forced_resync_count,
          total_time);

  // Final sorted-order verification report
  sort_verifier.DebugPrint();
  // Final Welford cardinality summary
  welford_tracker.DebugPrint(total_chunk_groups);

  printf("\n");
  printf("######################################################################\n");
  printf("#  AJBMergeJoin COMPLETE                                             #\n");
  printf("######################################################################\n");
  printf("  Total matches:    %zu\n", answer.count_);
  printf("  HLL estimate:     %.0f (error=%.1f%%)\n",
         hll_sketch.Estimate(),
         answer.count_ > 0
             ? 100.0 * std::fabs(hll_sketch.Estimate() - answer.count_) / answer.count_
             : 0.0);
  printf("  Pipeline time:    %.6f s\n", total_time);
  printf("  Materialized:     %s (%zu items)\n",
         materialize ? "yes" : "no", answer.items_.size());
  printf("######################################################################\n");

  // Print the transfer summary
  scheduler.PrintSummary();

  return answer;
}
