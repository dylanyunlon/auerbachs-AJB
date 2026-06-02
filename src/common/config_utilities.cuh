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

// [AJB] config dump: 在benchmark启动时打印全部配置参数
#include <cstdio>
static inline void ajb_dump_config(const char* config_str, const char* tag) {
    fprintf(stderr, "[AJB_STATE][Config] %s: %s\n", tag, config_str);
}
