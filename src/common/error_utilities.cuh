#pragma once

#include <iostream>
#include <stdexcept>

// Upstream: CheckCudaError in release mode expands to just the bare
// instruction with no error check at all — silent failures.
// Changed: release mode still evaluates the return code and throws
// on error.  The cost is one branch per CUDA call, which is negligible
// compared to the kernel launch / memcpy latency.

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
// Release mode: still check the error code, throw on failure.
// Upstream just discarded the return value entirely.
#define CheckCudaError(instruction)                    \
  {                                                    \
    cudaError_t _err = (instruction);                  \
    if (_err != cudaSuccess) {                         \
      throw std::runtime_error(                        \
          std::string("CUDA error: ") +                \
          cudaGetErrorString(_err));                    \
    }                                                  \
  }
#endif
