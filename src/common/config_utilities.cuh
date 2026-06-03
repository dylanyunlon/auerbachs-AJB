#pragma once

#include <unordered_set>
#include <vector>

#include <omp.h>

#include "error_utilities.cuh"

void ConfigureMultiProcess(size_t num_threads) {
  omp_set_schedule(omp_sched_static, 1);
  omp_set_max_active_levels(1);
  omp_set_num_threads(num_threads);
}

// Upstream: double loop over all GPU pairs, calling
// cudaDeviceEnablePeerAccess for every (i,j) where i!=j.
// Problem: EnablePeerAccess returns cudaErrorPeerAccessAlreadyEnabled
// if called twice for the same pair, which CheckCudaError may treat
// as fatal.  Also O(n^2) even when peer access is impossible.
// Changed: track already-enabled pairs in a set, skip duplicates,
// and check cudaDeviceCanAccessPeer before attempting.
void ConfigurePeerAccess(const std::vector<int>& gpus) {
  std::unordered_set<uint64_t> enabled_pairs;

  for (size_t i = 0; i < gpus.size(); ++i) {
    CheckCudaError(cudaSetDevice(gpus[i]));
    for (size_t j = 0; j < gpus.size(); ++j) {
      if (i == j) continue;

      // Encode the directed pair as a single 64-bit key
      uint64_t pair_key = (static_cast<uint64_t>(gpus[i]) << 32) | static_cast<uint32_t>(gpus[j]);
      if (enabled_pairs.count(pair_key)) continue;

      int can_access = 0;
      CheckCudaError(cudaDeviceCanAccessPeer(&can_access, gpus[i], gpus[j]));
      if (can_access) {
        CheckCudaError(cudaDeviceEnablePeerAccess(gpus[j], 0));
        enabled_pairs.insert(pair_key);
      }
    }
  }
}
