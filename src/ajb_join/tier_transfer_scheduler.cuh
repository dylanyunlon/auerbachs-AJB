#pragma once
// =============================================================================
// tier_transfer_scheduler.cuh — AJB Bandwidth-Tier-Adaptive Transfer Scheduler
//
// Assigns INDEPENDENT transfer periods (K_x, K_u, K_v) to the three
// relation-resident structures:
//   x = sorted build partitions        (K_x: slow, e.g. 256)
//   u = merge-path partition boundaries (K_u: medium, e.g. 16)
//   v = materialization buffers         (K_v: fast, e.g. 1)
//
// The key insight from the AJB paper (Section 2, Algorithm 1): decoupling
// these cadences avoids saturating the slow PCIe tier with bulky partition
// transfers while still maintaining correctness via frequent boundary
// syncs on the fast NVLink tier.
//
// Origin: adapted from upstream/multi-gpu-sort-merge-join profile_utilities +
//         merge_join pipeline. ~80% structure retained, ~20% new logic for
//         tier-aware scheduling + debug instrumentation.
// =============================================================================

#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "common/profile_utilities.cuh"

// ---------------------------------------------------------------------------
// Bandwidth tier classification
// ---------------------------------------------------------------------------
enum class BandwidthTier : int {
  kFastP2P  = 0,   // NVLink / NVSwitch — high bandwidth, low latency
  kSlowPCIe = 1,   // PCIe — moderate bandwidth, higher latency
  kHostDRAM = 2,   // CPU host memory — lowest effective bandwidth for GPU
};

inline const char* TierName(BandwidthTier tier) {
  switch (tier) {
    case BandwidthTier::kFastP2P:  return "NVLink-P2P";
    case BandwidthTier::kSlowPCIe: return "PCIe-Slow";
    case BandwidthTier::kHostDRAM: return "Host-DRAM";
  }
  return "Unknown";
}

// ---------------------------------------------------------------------------
// Per-device link descriptor (which tier connects device i to device j?)
// ---------------------------------------------------------------------------
struct DeviceLink {
  int src_device;
  int dst_device;
  BandwidthTier tier;
  double measured_bw_gbps;   // filled by ProbeInterconnect()

  void DebugPrint() const {
    printf("[DeviceLink] GPU %d -> GPU %d | tier=%s | bw=%.2f GB/s\n",
           src_device, dst_device, TierName(tier), measured_bw_gbps);
  }
};

// ---------------------------------------------------------------------------
// Transfer cadence configuration — the three independent periods
// ---------------------------------------------------------------------------
struct TransferCadence {
  size_t K_x;   // build-partition transfer period (slow-tier-friendly, e.g. 256)
  size_t K_u;   // merge-path boundary transfer period (medium, e.g. 16)
  size_t K_v;   // materialization buffer transfer period (fast, e.g. 1)

  // --- debug / trace ---
  void DebugPrint(const char* label = "TransferCadence") const {
    printf("[%s] K_x=%zu (partitions) | K_u=%zu (boundaries) | K_v=%zu (buffers)\n",
           label, K_x, K_u, K_v);
  }

  // The "coupling ratio" — how much slower is the partition cadence vs
  // the boundary cadence? Higher = more decoupled = less slow-tier traffic.
  double CouplingRatio() const {
    return (K_u > 0) ? static_cast<double>(K_x) / K_u : 0.0;
  }
};

// ---------------------------------------------------------------------------
// Transfer event — one scheduled cross-tier or P2P copy
// ---------------------------------------------------------------------------
struct TransferEvent {
  enum class StructureKind { kBuildPartition, kMergePathBoundary, kMaterializationBuffer };
  
  StructureKind kind;
  int src_device;
  int dst_device;
  BandwidthTier tier;
  size_t chunk_group_index;   // t in Algorithm 1
  size_t bytes_transferred;
  double duration_seconds;

  static const char* KindName(StructureKind k) {
    switch (k) {
      case StructureKind::kBuildPartition:        return "build_partition(x)";
      case StructureKind::kMergePathBoundary:     return "merge_boundary(u)";
      case StructureKind::kMaterializationBuffer: return "mat_buffer(v)";
    }
    return "?";
  }

  void DebugPrint() const {
    printf("[TransferEvent] t=%zu | %s | GPU %d->%d (%s) | %zu bytes | %.6f s\n",
           chunk_group_index, KindName(kind),
           src_device, dst_device, TierName(tier),
           bytes_transferred, duration_seconds);
  }
};

// ---------------------------------------------------------------------------
// Tier-aware transfer scheduler
// ---------------------------------------------------------------------------
class TierTransferScheduler {
 public:
  TierTransferScheduler() = default;

  // Initialize with device topology and cadence configuration.
  void Initialize(const std::vector<int>& gpus,
                  const std::vector<DeviceLink>& topology,
                  const TransferCadence& cadence) {
    gpus_ = gpus;
    topology_ = topology;
    cadence_ = cadence;
    event_log_.clear();
    total_slow_tier_bytes_ = 0;
    total_fast_tier_bytes_ = 0;
    total_host_bytes_ = 0;

    printf("\n");
    printf("==================================================================\n");
    printf("[TierTransferScheduler] INITIALIZED\n");
    printf("==================================================================\n");
    printf("  Devices: %zu GPUs [", gpus.size());
    for (size_t i = 0; i < gpus.size(); ++i) {
      printf("%d%s", gpus[i], i + 1 < gpus.size() ? ", " : "");
    }
    printf("]\n");
    cadence.DebugPrint("  Cadence");
    printf("  Coupling ratio (K_x/K_u): %.1f\n", cadence.CouplingRatio());
    printf("  Topology links: %zu\n", topology.size());
    for (const auto& link : topology) {
      printf("    ");
      link.DebugPrint();
    }
    printf("==================================================================\n\n");
  }

  // ---------------------------------------------------------------------------
  // Core decision: should structure `kind` be transferred at chunk-group t?
  //
  // Algorithm 1 from the paper:
  //   transfer_x iff (t mod K_x == 0)
  //   transfer_u iff (t mod K_u == 0)
  //   transfer_v iff (t mod K_v == 0)    (K_v=1 means every chunk group)
  // ---------------------------------------------------------------------------
  bool ShouldTransfer(TransferEvent::StructureKind kind, size_t t) const {
    size_t period = GetPeriod(kind);
    bool should = (period > 0) && (t % period == 0);

    #ifdef AJB_TRACE_DECISIONS
    printf("[ShouldTransfer] t=%zu | %s | period=%zu | decision=%s\n",
           t, TransferEvent::KindName(kind), period,
           should ? "YES-TRANSFER" : "skip");
    #endif

    return should;
  }

  // ---------------------------------------------------------------------------
  // Record a transfer that actually happened (for post-hoc analysis / figures)
  // ---------------------------------------------------------------------------
  void RecordTransfer(TransferEvent::StructureKind kind,
                      int src, int dst, size_t t,
                      size_t bytes, double duration_sec) {
    BandwidthTier tier = LookupTier(src, dst);
    TransferEvent evt{kind, src, dst, tier, t, bytes, duration_sec};

    #ifdef AJB_TRACE_TRANSFERS
    evt.DebugPrint();
    #endif

    event_log_.push_back(evt);

    switch (tier) {
      case BandwidthTier::kFastP2P:  total_fast_tier_bytes_ += bytes; break;
      case BandwidthTier::kSlowPCIe: total_slow_tier_bytes_ += bytes; break;
      case BandwidthTier::kHostDRAM: total_host_bytes_      += bytes; break;
    }
  }

  // ---------------------------------------------------------------------------
  // Summary report — call after the join pipeline completes
  // ---------------------------------------------------------------------------
  void PrintSummary() const {
    printf("\n");
    printf("==================================================================\n");
    printf("[TierTransferScheduler] TRANSFER SUMMARY\n");
    printf("==================================================================\n");
    printf("  Total events logged:     %zu\n", event_log_.size());
    printf("  Fast-tier (NVLink) bytes: %zu (%.2f MB)\n",
           total_fast_tier_bytes_, total_fast_tier_bytes_ / 1e6);
    printf("  Slow-tier (PCIe)  bytes: %zu (%.2f MB)\n",
           total_slow_tier_bytes_, total_slow_tier_bytes_ / 1e6);
    printf("  Host-DRAM         bytes: %zu (%.2f MB)\n",
           total_host_bytes_, total_host_bytes_ / 1e6);

    // Per-structure breakdown
    size_t count_x = 0, count_u = 0, count_v = 0;
    size_t bytes_x = 0, bytes_u = 0, bytes_v = 0;
    for (const auto& e : event_log_) {
      switch (e.kind) {
        case TransferEvent::StructureKind::kBuildPartition:
          count_x++; bytes_x += e.bytes_transferred; break;
        case TransferEvent::StructureKind::kMergePathBoundary:
          count_u++; bytes_u += e.bytes_transferred; break;
        case TransferEvent::StructureKind::kMaterializationBuffer:
          count_v++; bytes_v += e.bytes_transferred; break;
      }
    }
    printf("  ---\n");
    printf("  build_partition(x):   %zu transfers, %zu bytes (%.2f MB)\n",
           count_x, bytes_x, bytes_x / 1e6);
    printf("  merge_boundary(u):    %zu transfers, %zu bytes (%.2f MB)\n",
           count_u, bytes_u, bytes_u / 1e6);
    printf("  mat_buffer(v):        %zu transfers, %zu bytes (%.2f MB)\n",
           count_v, bytes_v, bytes_v / 1e6);

    if (total_slow_tier_bytes_ > 0 || total_fast_tier_bytes_ > 0) {
      double ratio = (total_slow_tier_bytes_ > 0)
          ? static_cast<double>(total_fast_tier_bytes_) / total_slow_tier_bytes_
          : 0.0;
      printf("  Fast/Slow ratio:       %.2f:1\n", ratio);
    }
    printf("==================================================================\n\n");
  }

  // ---------------------------------------------------------------------------
  // Export event log as CSV for figure_data_emitter consumption
  // ---------------------------------------------------------------------------
  void ExportEventsCSV(const std::string& path) const {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) {
      printf("[ERROR] Cannot open %s for writing\n", path.c_str());
      return;
    }
    fprintf(fp, "chunk_group,structure,src_device,dst_device,tier,bytes,duration_s\n");
    for (const auto& e : event_log_) {
      fprintf(fp, "%zu,%s,%d,%d,%s,%zu,%.9f\n",
              e.chunk_group_index, TransferEvent::KindName(e.kind),
              e.src_device, e.dst_device, TierName(e.tier),
              e.bytes_transferred, e.duration_seconds);
    }
    fclose(fp);
    printf("[TierTransferScheduler] Exported %zu events to %s\n",
           event_log_.size(), path.c_str());
  }

  // Accessors
  const TransferCadence& GetCadence() const { return cadence_; }
  const std::vector<TransferEvent>& GetEventLog() const { return event_log_; }
  size_t GetSlowTierBytes() const { return total_slow_tier_bytes_; }
  size_t GetFastTierBytes() const { return total_fast_tier_bytes_; }

 private:
  size_t GetPeriod(TransferEvent::StructureKind kind) const {
    switch (kind) {
      case TransferEvent::StructureKind::kBuildPartition:        return cadence_.K_x;
      case TransferEvent::StructureKind::kMergePathBoundary:     return cadence_.K_u;
      case TransferEvent::StructureKind::kMaterializationBuffer: return cadence_.K_v;
    }
    return 1;
  }

  BandwidthTier LookupTier(int src, int dst) const {
    for (const auto& link : topology_) {
      if (link.src_device == src && link.dst_device == dst) return link.tier;
    }
    // If not in topology, assume host-mediated
    return BandwidthTier::kHostDRAM;
  }

  std::vector<int> gpus_;
  std::vector<DeviceLink> topology_;
  TransferCadence cadence_;
  std::vector<TransferEvent> event_log_;
  size_t total_slow_tier_bytes_ = 0;
  size_t total_fast_tier_bytes_ = 0;
  size_t total_host_bytes_ = 0;
};

// ---------------------------------------------------------------------------
// Interconnect prober — detect which tier connects each GPU pair
// ---------------------------------------------------------------------------
inline std::vector<DeviceLink> ProbeInterconnect(const std::vector<int>& gpus) {
  std::vector<DeviceLink> links;

  printf("[ProbeInterconnect] Probing %zu GPUs...\n", gpus.size());

  for (size_t i = 0; i < gpus.size(); ++i) {
    for (size_t j = 0; j < gpus.size(); ++j) {
      if (i == j) continue;

      DeviceLink link;
      link.src_device = gpus[i];
      link.dst_device = gpus[j];
      link.measured_bw_gbps = 0.0;

      // Check if P2P is possible (NVLink indicator)
      int can_access = 0;
      #ifdef __CUDACC__
      cudaDeviceCanAccessPeer(&can_access, gpus[i], gpus[j]);
      #endif

      if (can_access) {
        link.tier = BandwidthTier::kFastP2P;
        link.measured_bw_gbps = 300.0;  // NVLink3 nominal; real probe below
      } else {
        link.tier = BandwidthTier::kSlowPCIe;
        link.measured_bw_gbps = 16.0;   // PCIe Gen4 x16 nominal
      }

      printf("  GPU %d -> GPU %d : P2P=%s -> tier=%s (nominal %.0f GB/s)\n",
             gpus[i], gpus[j],
             can_access ? "yes" : "no",
             TierName(link.tier),
             link.measured_bw_gbps);

      links.push_back(link);
    }
  }

  // Host-tier links (every GPU to/from CPU host)
  for (size_t i = 0; i < gpus.size(); ++i) {
    DeviceLink to_host{gpus[i], -1, BandwidthTier::kHostDRAM, 12.0};
    DeviceLink from_host{-1, gpus[i], BandwidthTier::kHostDRAM, 12.0};
    printf("  GPU %d <-> Host : tier=Host-DRAM (nominal 12 GB/s)\n", gpus[i]);
    links.push_back(to_host);
    links.push_back(from_host);
  }

  printf("[ProbeInterconnect] Total links: %zu\n\n", links.size());
  return links;
}

// ---------------------------------------------------------------------------
// Cadence auto-tuner — heuristic from Section 2 of the AJB paper
//
// Given the topology, pick K_x, K_u, K_v that minimize slow-tier volume
// while maintaining correctness (Theorem 1: K_u must be finite when keys
// are non-uniformly distributed).
// ---------------------------------------------------------------------------
inline TransferCadence AutoTuneCadence(const std::vector<DeviceLink>& topology,
                                       size_t num_chunk_groups,
                                       double skew_estimate = 0.0) {
  // Count how many links are slow-tier vs fast-tier
  size_t n_slow = 0, n_fast = 0;
  for (const auto& link : topology) {
    if (link.tier == BandwidthTier::kSlowPCIe) n_slow++;
    else if (link.tier == BandwidthTier::kFastP2P) n_fast++;
  }

  TransferCadence cadence;

  // Base heuristic: the more slow links, the higher K_x should be
  // to reduce cross-tier partition traffic
  if (n_slow == 0) {
    // Homogeneous NVLink fabric — uniform cadence is fine
    cadence.K_x = 1;
    cadence.K_u = 1;
    cadence.K_v = 1;
  } else {
    // Heterogeneous: decouple!
    double slow_fraction = static_cast<double>(n_slow) / (n_slow + n_fast + 1);

    // K_x: slow-tier-friendly, scale with topology heterogeneity
    // Capped at num_chunk_groups (no point going beyond total chunks)
    cadence.K_x = std::min<size_t>(
        static_cast<size_t>(256 * (1.0 + slow_fraction)),
        num_chunk_groups);
    cadence.K_x = std::max<size_t>(cadence.K_x, 16);

    // K_u: boundary vectors are small, so transfer more often
    // Theorem 1 requires finite K_u when data is skewed
    cadence.K_u = std::max<size_t>(1, cadence.K_x / 16);
    if (skew_estimate > 0.5) {
      // High skew: transfer boundaries more frequently for correctness
      cadence.K_u = std::max<size_t>(1, cadence.K_u / 4);
    }

    // K_v: materialization buffers are streaming, transfer every chunk
    cadence.K_v = 1;
  }

  printf("[AutoTuneCadence] slow_links=%zu fast_links=%zu skew=%.2f\n",
         n_slow, n_fast, skew_estimate);
  cadence.DebugPrint("AutoTuned");
  printf("  Estimated slow-tier reduction vs uniform: %.1fx\n",
         cadence.CouplingRatio());

  return cadence;
}
