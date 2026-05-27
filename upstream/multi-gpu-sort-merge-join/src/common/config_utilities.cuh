#pragma once

#include <vector>

#include <omp.h>

#include "error_utilities.cuh"

void ConfigureMultiProcess(size_t num_threads) {
  omp_set_schedule(omp_sched_static, 1);
  omp_set_max_active_levels(1);
  omp_set_num_threads(num_threads);
}

void ConfigurePeerAccess(const std::vector<int>& gpus) {
  for (size_t i = 0; i < gpus.size(); ++i) {
    CheckCudaError(cudaSetDevice(gpus[i]));
    for (size_t j = 0; j < gpus.size(); ++j) {
      if (i != j) {
        CheckCudaError(cudaDeviceEnablePeerAccess(gpus[j], 0));
      }
    }
  }
}
