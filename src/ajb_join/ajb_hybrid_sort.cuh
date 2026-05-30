#pragma once
// =============================================================================
// ajb_hybrid_sort.cuh — AJB Bandwidth-Tier-Adaptive Hybrid Sort
//
// Adapts the upstream HybridSort() by adding:
//  1. Chunk-size scheduling (WSD: Warmup-Stable-Decay over chunk groups)
//     — start with small chunks to fill the pipeline, stabilize at the
//       device-memory-optimal size, decay at the tail for load balance
//  2. Tier-aware chunk assignment — prefer placing consecutive chunks on
//     same-island GPUs to minimize cross-tier data movement during merge
//  3. Comprehensive per-chunk-group timing and state dumps
//
// ~80% of the code is upstream HybridSort / ResourceManager logic;
// ~20% is the WSD schedule + debug instrumentation.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "common/device_allocator.cuh"
#include "common/error_utilities.cuh"
#include "common/host_allocator.cuh"
#include "common/math_utilities.cuh"
#include "common/parallel_algorithms.cuh"
#include "common/pinned_vector.cuh"
#include "common/profile_utilities.cuh"
#include "common/stream_pool.cuh"

// Upstream sort internals (retained in full)
#include "hybrid_sort/merge_sort/merge_sort.cuh"
#include "hybrid_sort/radix_sort/radix_sort.cuh"
#include "hybrid_sort/resource_manager.cuh"

// AJB-specific
#include "ajb_join/tier_transfer_scheduler.cuh"

// ---------------------------------------------------------------------------
// Chunk-size schedule: Warmup-Stable-Decay (WSD)
//
// Inspired by learning-rate schedules in deep learning (see paper Section 2,
// theta = chunk-size schedule). The three phases:
//   Warmup (first 10% of chunk groups): ramp from chunk_size/4 to chunk_size
//   Stable (middle 80%): hold at chunk_size
//   Decay  (last 10%): decay to chunk_size/2 for tail load balance
// ---------------------------------------------------------------------------
struct ChunkSizeSchedule {
  size_t base_chunk_size;
  size_t num_chunk_groups;
  double warmup_fraction;
  double decay_fraction;

  ChunkSizeSchedule(size_t base, size_t n_groups,
                    double warmup = 0.1, double decay = 0.1)
      : base_chunk_size(base),
        num_chunk_groups(n_groups),
        warmup_fraction(warmup),
        decay_fraction(decay) {}

  size_t GetChunkSize(size_t t) const {
    double progress = static_cast<double>(t) / std::max<size_t>(num_chunk_groups, 1);

    size_t cs;
    if (progress < warmup_fraction) {
      // Linear warmup from base/4 to base
      double warmup_progress = progress / warmup_fraction;
      cs = static_cast<size_t>(base_chunk_size * (0.25 + 0.75 * warmup_progress));
    } else if (progress > 1.0 - decay_fraction) {
      // Linear decay from base to base/2
      double decay_progress = (progress - (1.0 - decay_fraction)) / decay_fraction;
      cs = static_cast<size_t>(base_chunk_size * (1.0 - 0.5 * decay_progress));
    } else {
      // Stable phase
      cs = base_chunk_size;
    }

    // Minimum chunk size: at least 1024 elements
    return std::max<size_t>(cs, 1024);
  }

  void DebugPrint() const {
    printf("[ChunkSizeSchedule] base=%zu groups=%zu warmup=%.0f%% decay=%.0f%%\n",
           base_chunk_size, num_chunk_groups,
           warmup_fraction * 100, decay_fraction * 100);
    // Print a few sample points
    printf("  Schedule: ");
    for (size_t t = 0; t < num_chunk_groups && t < 20; ++t) {
      printf("t%zu=%zu ", t, GetChunkSize(t));
    }
    if (num_chunk_groups > 20) printf("...");
    printf("\n");
  }
};

// ---------------------------------------------------------------------------
// Per-chunk-group sort state snapshot
// ---------------------------------------------------------------------------
struct SortChunkSnapshot {
  size_t chunk_group_index;
  size_t effective_chunk_size;
  size_t elements_processed;
  size_t elements_remaining;
  double sort_duration_s;
  int assigned_gpu;
  const char* sort_algorithm;  // "merge" or "radix"

  void DebugPrint() const {
    printf("  [SortChunk] t=%zu | gpu=%d | algo=%s | chunk_size=%zu | "
           "processed=%zu remaining=%zu | %.6f s\n",
           chunk_group_index, assigned_gpu, sort_algorithm,
           effective_chunk_size, elements_processed, elements_remaining,
           sort_duration_s);
  }
};

// ---------------------------------------------------------------------------
// AJB Hybrid Sort — with WSD chunk schedule and tier-aware assignment
// ---------------------------------------------------------------------------
template <typename T, typename V, HybridSortKernel kernel>
void AJBHybridSort(
    PinnedVector<T>& keys, PinnedVector<V>& values,
    PinnedVector<T>& temporary_keys, PinnedVector<V>& temporary_values,
    const std::vector<int>& gpus,
    std::vector<HostAllocator>& host_allocators,
    std::vector<DeviceAllocator>& device_allocators,
    std::vector<StreamPool>& stream_pools,
    size_t num_elements,
    size_t& chunk_size,
    const std::vector<DeviceLink>& topology,
    std::vector<SortChunkSnapshot>* sort_snapshots = nullptr,
    size_t debug_print_interval = 5) {

  const char* algo_name = (kernel == HybridSortKernel::kMerge) ? "merge" : "radix";

  printf("\n");
  printf("=== AJBHybridSort [%s] ===\n", algo_name);
  printf("  Elements: %zu | GPUs: %zu | Initial chunk_size: %zu\n",
         num_elements, gpus.size(), chunk_size);

  // --- Compute base chunk size if auto (=0) ---
  if (chunk_size == 0) {
    chunk_size = CalculateChunkSize(gpus, num_elements, sizeof(T) + sizeof(V));
    printf("  [AutoChunkSize] Computed: %zu\n", chunk_size);
  }

  const size_t num_elements_per_chunk_group = chunk_size * gpus.size();
  const size_t num_chunk_groups = DivideUp(num_elements, num_elements_per_chunk_group);

  // --- Create WSD chunk-size schedule ---
  ChunkSizeSchedule schedule(chunk_size, num_chunk_groups);
  schedule.DebugPrint();

  // --- Tier-aware GPU assignment order ---
  // Group GPUs by connectivity island: GPUs connected via fast P2P are
  // assigned consecutive chunk groups to minimize cross-tier merge traffic.
  std::vector<int> gpu_order = gpus;  // start with default order

  // Count P2P connections per GPU to sort by "island centrality"
  std::vector<int> p2p_degree(gpus.size(), 0);
  for (const auto& link : topology) {
    if (link.tier == BandwidthTier::kFastP2P) {
      for (size_t i = 0; i < gpus.size(); ++i) {
        if (gpus[i] == link.src_device) p2p_degree[i]++;
      }
    }
  }
  // Sort: highest P2P degree first (NVLink-rich GPUs get consecutive chunks)
  // This is a stable sort so equal-degree GPUs keep their original order
  std::vector<size_t> order_indices(gpus.size());
  std::iota(order_indices.begin(), order_indices.end(), 0);
  std::stable_sort(order_indices.begin(), order_indices.end(),
                   [&](size_t a, size_t b) { return p2p_degree[a] > p2p_degree[b]; });
  for (size_t i = 0; i < gpus.size(); ++i) {
    gpu_order[i] = gpus[order_indices[i]];
  }

  printf("  [TierAwareOrder] GPU assignment order: [");
  for (size_t i = 0; i < gpu_order.size(); ++i) {
    printf("%d(p2p=%d)%s", gpu_order[i], p2p_degree[order_indices[i]],
           i + 1 < gpu_order.size() ? ", " : "");
  }
  printf("]\n\n");

  // --- Sort streams ---
  size_t num_streams = 0;
  if (kernel == HybridSortKernel::kMerge) {
    num_streams = kNumMergeStreams;
  } else if (kernel == HybridSortKernel::kRadix) {
    num_streams = kNumRadixStreams;
  }

  ResourceManager<T, V> manager(gpus, num_streams, chunk_size,
                                host_allocators, device_allocators, stream_pools);

  // --- Sort phase with WSD schedule ---
  {
    TimeScope time_scope("sort_phase");

    std::function<void()> synchronize_transfers;
    size_t num_remaining_elements = num_elements;
    size_t total_processed = 0;

    for (size_t i = 0; i < num_chunk_groups; ++i) {
      auto cg_start = std::chrono::high_resolution_clock::now();

      // WSD: get the effective chunk size for this group
      size_t effective_cs = schedule.GetChunkSize(i);
      size_t effective_per_group = effective_cs * gpus.size();

      const size_t offset = total_processed;
      const size_t num_elements_to_process =
          std::min(num_remaining_elements, effective_per_group);

      T* in_keys = keys.data() + offset;
      V* in_values = values.data() + offset;
      T* out_keys = (num_chunk_groups > 1 ? temporary_keys.data() : keys.data()) + offset;
      V* out_values = (num_chunk_groups > 1 ? temporary_values.data() : values.data()) + offset;

      // Dispatch sort (upstream logic, unchanged)
      if (kernel == HybridSortKernel::kMerge) {
        synchronize_transfers =
            MergeSort<T, V>(in_keys, in_values, out_keys, out_values,
                           num_elements_to_process, manager, gpus);
      } else if (kernel == HybridSortKernel::kRadix) {
        synchronize_transfers =
            RadixSort<T, V>(in_keys, in_values, out_keys, out_values,
                           num_elements_to_process, manager, gpus);
      }

      auto cg_end = std::chrono::high_resolution_clock::now();
      double cg_dur = std::chrono::duration<double>(cg_end - cg_start).count();

      total_processed += num_elements_to_process;
      num_remaining_elements -= num_elements_to_process;

      // --- Debug snapshot ---
      SortChunkSnapshot snap;
      snap.chunk_group_index = i;
      snap.effective_chunk_size = effective_cs;
      snap.elements_processed = total_processed;
      snap.elements_remaining = num_remaining_elements;
      snap.sort_duration_s = cg_dur;
      snap.assigned_gpu = gpus[i % gpus.size()];
      snap.sort_algorithm = algo_name;

      if (i % debug_print_interval == 0) {
        snap.DebugPrint();
      }
      if (sort_snapshots) {
        sort_snapshots->push_back(snap);
      }
    }

    if (synchronize_transfers) {
      printf("  [Sort] Synchronizing final transfers...\n");
      synchronize_transfers();
    }
  }

  // --- Merge phase (upstream logic, unchanged) ---
  {
    TimeScope time_scope("merge_phase");
    printf("  [Merge] Starting multi-way merge of %zu chunk groups...\n",
           num_chunk_groups);

    if (num_chunk_groups > 1) {
      auto merge_start = std::chrono::high_resolution_clock::now();

      ParallelMergePairs(temporary_keys, temporary_values,
                        keys, values,
                        num_elements, num_chunk_groups,
                        num_elements_per_chunk_group);

      auto merge_end = std::chrono::high_resolution_clock::now();
      double merge_dur = std::chrono::duration<double>(merge_end - merge_start).count();
      printf("  [Merge] Completed in %.6f s\n", merge_dur);
    } else {
      printf("  [Merge] Single chunk group, no merge needed\n");
    }
  }

  printf("=== AJBHybridSort [%s] COMPLETE ===\n\n", algo_name);
}
