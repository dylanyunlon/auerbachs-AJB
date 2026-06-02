#pragma once

#include <iostream>

#ifdef DEBUG_BUILD
#define CheckCudaLaunchError()                                                                                       \
  {                                                                                                                  \
    cudaError_t error_code = cudaGetLastError();                                                                     \
    if (error_code != cudaSuccess) {                                                                                 \
      const char* error_name = cudaGetErrorName(error_code);                                                         \
      std::cout << "[ERROR] " << __FILE__ << " " << __LINE__ << error_code << " " << error_name << " " << std::endl; \
      fprintf(stderr, "[AJB_FAIL] CUDA launch @ %s:%d code=%d %s\n", __FILE__, __LINE__, error_code, error_name);   \
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
