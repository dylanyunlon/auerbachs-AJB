#pragma once

#include <iostream>

#ifdef DEBUG_BUILD
#define CheckCudaLaunchError()                                                                                       \
  {                                                                                                                  \
    cudaError_t error_code = cudaGetLastError();                                                                     \
    if (error_code != cudaSuccess) {                                                                                 \
      const char* error_name = cudaGetErrorName(error_code);                                                         \
      std::cout << "[ERROR] " << __FILE__ << " " << __LINE__ << error_code << " " << error_name << " " << std::endl; \
    }                                                                                                                \
  }
#else
#define CheckCudaLaunchError()
#endif

#ifdef DEBUG_BUILD
#define CheckCudaError(instruction)                                                                                  \
  {                                                                                                                  \
    cudaError_t error_code = (instruction);                                                                          \
    if (error_code != cudaSuccess) {                                                                                 \
      const char* error_name = cudaGetErrorName(error_code);                                                         \
      std::cout << "[ERROR] " << __FILE__ << " " << __LINE__ << error_code << " " << error_name << " " << std::endl; \
    }                                                                                                                \
  }
#else
#define CheckCudaError(instruction) instruction
#endif

// [AJB] 增强版CUDA错误检查: 带文件名+行号+上下文tag
#include <cstdio>
#define AJB_CUDA_CHECK(call, tag) do {                                      \
    cudaError_t err = (call);                                               \
    if(err != cudaSuccess){                                                 \
        fprintf(stderr, "[AJB_FAIL][CUDA] %s @ %s:%d: %s (%d)\n",          \
                tag, __FILE__, __LINE__, cudaGetErrorString(err), (int)err); \
    }                                                                       \
} while(0)

// [AJB] 批量检查: 在pipeline的每个阶段末尾调用
static inline void ajb_cuda_sync_check(const char* phase) {
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
        fprintf(stderr, "[AJB_FAIL][CUDA] sync after %s: %s\n", phase, cudaGetErrorString(err));
    else
        fprintf(stderr, "[AJB_TRACE][CUDA] sync OK after %s\n", phase);
}
