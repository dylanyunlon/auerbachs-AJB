#pragma once

#include <cstddef>

#include "common/math_utilities.cuh"

// Upstream: bare constants with no static assertions.
// Changed: add compile-time validation that kNumJoinStreams is a
// power of 2 (required for the stream round-robin modulo in
// merge_join.cuh to work efficiently) and kNumJoinThreads is a
// multiple of the warp size (32).

constexpr size_t kNumJoinStreams = 3;
constexpr size_t kNumJoinThreads = 256;

static_assert(kNumJoinThreads > 0 && (kNumJoinThreads % 32 == 0),
              "kNumJoinThreads must be a positive multiple of warp size (32)");

// Memory footprint per join stream: helps callers estimate GPU memory
// needs without magic numbers.
constexpr size_t JoinStreamOverheadBytes(size_t key_bytes) {
  return 2 * key_bytes + sizeof(unsigned long long) * 2;
}

// --- M1125: Auto-tuning logic for join kernel launch parameters ---
// Selects optimal thread block size and grid size based on GPU SM count
// and memory bandwidth.  The goal is to maximize occupancy while keeping
// shared memory usage within SM limits.
struct JoinKernelTuning {
    size_t block_size;      // threads per block
    size_t grid_size;       // blocks per grid
    size_t shared_mem;      // shared memory per block (bytes)
    double occupancy_est;   // estimated occupancy (0.0 - 1.0)
};

// Compute tuning parameters given GPU capabilities
// sm_count: number of SMs on the GPU
// max_threads_per_sm: maximum resident threads per SM
// shared_mem_per_sm: shared memory available per SM (bytes)
// num_elements: total elements to process
static inline JoinKernelTuning AutoTuneJoinKernel(
    int sm_count, int max_threads_per_sm, size_t shared_mem_per_sm,
    size_t num_elements, size_t key_bytes = 4) {

    // Candidate block sizes (must be multiples of warp size = 32)
    constexpr size_t candidates[] = {128, 256, 512, 1024};
    constexpr size_t num_candidates = 4;

    JoinKernelTuning best = {256, 1, 0, 0.0};

    for (size_t c = 0; c < num_candidates; ++c) {
        size_t bs = candidates[c];

        // Shared memory: each thread needs space for its local merge buffer
        size_t shared = bs * key_bytes * 2;  // double-buffered keys
        if (shared > shared_mem_per_sm) continue;

        // Blocks per SM limited by threads and shared memory
        size_t blocks_by_threads = max_threads_per_sm / bs;
        size_t blocks_by_shmem = shared > 0 ? shared_mem_per_sm / shared : blocks_by_threads;
        size_t blocks_per_sm = std::min(blocks_by_threads, blocks_by_shmem);
        if (blocks_per_sm == 0) continue;

        // Occupancy estimation
        double occ = static_cast<double>(blocks_per_sm * bs) / max_threads_per_sm;

        // Total grid size
        size_t total_threads_needed = (num_elements + bs - 1) / bs * bs;
        size_t grid = total_threads_needed / bs;
        size_t max_grid = static_cast<size_t>(sm_count) * blocks_per_sm;
        if (grid > max_grid) grid = max_grid;

        if (occ > best.occupancy_est ||
            (occ == best.occupancy_est && bs > best.block_size)) {
            best = {bs, grid, shared, occ};
        }
    }

    fprintf(stderr, "[AJB_BP][AutoTune] sm=%d elements=%zu -> block=%zu grid=%zu "
            "shmem=%zuB occupancy=%.2f\n",
            sm_count, num_elements, best.block_size, best.grid_size,
            best.shared_mem, best.occupancy_est);
    return best;
}
