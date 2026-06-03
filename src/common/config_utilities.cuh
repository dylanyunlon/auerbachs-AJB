#pragma once

#include <vector>
#include <unordered_set>

#include <omp.h>

#include "error_utilities.cuh"

// Upstream: 固定omp_sched_static + chunk=1.
// AJB改写: 根据线程数动态选schedule策略.
// 线程数>GPU数*2时用dynamic(减少尾延迟), 否则保持static.
void ConfigureMultiProcess(size_t num_threads, size_t num_gpus = 0) {
  if (num_gpus > 0 && num_threads > num_gpus * 2) {
    omp_set_schedule(omp_sched_dynamic, 1);
  } else {
    omp_set_schedule(omp_sched_static, 1);
  }
  omp_set_max_active_levels(1);
  omp_set_num_threads(num_threads);

  // 断点: 配置完成后打印实际生效的OMP参数
  omp_sched_t kind; int chunk;
  omp_get_schedule(&kind, &chunk);
  const char* sched_name = (kind == omp_sched_static) ? "static" :
                           (kind == omp_sched_dynamic) ? "dynamic" : "other";
  fprintf(stderr, "[DEBUG][ConfigMP] threads=%zu gpus=%zu schedule=%s chunk=%d\n",
          num_threads, num_gpus, sched_name, chunk);
}

// Upstream: O(n^2)双重循环, 对每对GPU无条件enable peer access.
// AJB改写:
//   1. 先query CanAccessPeer, 跳过不支持P2P的pair(避免CUDA error)
//   2. 用set记录已启用的pair, 跳过重复enable(idempotent但浪费时间)
//   3. 打印P2P拓扑矩阵 — 在多机环境下这是第一个要看的调试信息
void ConfigurePeerAccess(const std::vector<int>& gpus) {
  std::unordered_set<int64_t> enabled_pairs;

  // 先打印P2P可达性矩阵
  fprintf(stderr, "[DEBUG][PeerAccess] topology (%zuGPUs):\n       ", gpus.size());
  for (size_t j = 0; j < gpus.size(); ++j) fprintf(stderr, " GPU%-3d", gpus[j]);
  fprintf(stderr, "\n");

  for (size_t i = 0; i < gpus.size(); ++i) {
    CheckCudaError(cudaSetDevice(gpus[i]));
    fprintf(stderr, "  GPU%-3d", gpus[i]);

    for (size_t j = 0; j < gpus.size(); ++j) {
      if (i == j) { fprintf(stderr, "  self "); continue; }

      int can_access = 0;
      cudaDeviceCanAccessPeer(&can_access, gpus[i], gpus[j]);

      if (!can_access) {
        fprintf(stderr, "  no   ");
        continue;
      }

      // 用 (i*1000+j) 作key避免重复enable
      int64_t pair_key = static_cast<int64_t>(gpus[i]) * 1000 + gpus[j];
      if (enabled_pairs.count(pair_key)) {
        fprintf(stderr, "  dup  ");
        continue;
      }

      cudaError_t err = cudaDeviceEnablePeerAccess(gpus[j], 0);
      if (err == cudaSuccess) {
        enabled_pairs.insert(pair_key);
        fprintf(stderr, "  yes  ");
      } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();  // 清除error状态
        enabled_pairs.insert(pair_key);
        fprintf(stderr, "  dup  ");
      } else {
        fprintf(stderr, "  ERR  ");
      }
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "[DEBUG][PeerAccess] enabled %zu directional P2P links\n", enabled_pairs.size());
}
