cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(multi-gpu-sort-merge-join LANGUAGES CUDA CXX)

find_package(OpenMP REQUIRED)

message("-- Building for ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp")
set(CMAKE_CXX_STANDARD 20)

set(CUDA_ARCH_LIST 70 80 90)

add_subdirectory(third_party/thrust)

set(WITH_OPEN_MP OMP)
set(CUDA_NVTX_LIBRARY nvToolsExt)
set(THRUST_REQUIRED_SYSTEMS CUDA ${WITH_OPEN_MP})

thrust_create_target(ThrustCUDA HOST ${WITH_OPEN_MP} DEVICE CUDA)

add_executable(cpu_merge_benchmark "src/cpu_merge_benchmark.cu")
target_link_libraries(cpu_merge_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET cpu_merge_benchmark PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
target_include_directories(cpu_merge_benchmark SYSTEM PRIVATE
  src/
  third_party/
  third_party/cxxopts/include
  third_party/parallel/include
  third_party/termcolor/include
)

add_executable(cpu_sort_benchmark "src/cpu_sort_benchmark.cu")
target_link_libraries(cpu_sort_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET cpu_sort_benchmark PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
target_include_directories(cpu_sort_benchmark SYSTEM PRIVATE
  src/
  third_party/
  third_party/cxxopts/include
  third_party/parallel/include
  third_party/termcolor/include
)

add_executable(gpu_merge_benchmark "src/gpu_merge_benchmark.cu")
target_link_libraries(gpu_merge_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
target_compile_options(gpu_merge_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-extended-lambda>)
set_property(TARGET gpu_merge_benchmark PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
target_include_directories(gpu_merge_benchmark SYSTEM PRIVATE
  src/
  third_party/
  third_party/cxxopts/include
  third_party/moderngpu/src
  third_party/termcolor/include
)

add_executable(gpu_sort_benchmark "src/gpu_sort_benchmark.cu")
target_link_libraries(gpu_sort_benchmark PUBLIC Thrust::Thrust ThrustCUDA)
target_compile_options(gpu_sort_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-extended-lambda>)
set_property(TARGET gpu_sort_benchmark PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
target_include_directories(gpu_sort_benchmark SYSTEM PRIVATE
  src/
  third_party/
  third_party/cxxopts/include
  third_party/moderngpu/src
  third_party/termcolor/include
)

add_executable(sort_benchmark "src/sort_benchmark.cu")
target_link_libraries(sort_benchmark PUBLIC Thrust::Thrust ThrustCUDA ${CUDA_NVTX_LIBRARY})
target_compile_options(sort_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -Xptxas --disable-warnings>)
target_compile_options(sort_benchmark PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;-extended-lambda;-DDEBUG_BUILD>")
target_compile_options(sort_benchmark PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-extended-lambda>")
set_property(TARGET sort_benchmark PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
target_include_directories(sort_benchmark PUBLIC
  src/
  third_party/
  third_party/csvparser/single_include
  third_party/cxxopts/include
  third_party/moderngpu/src
  third_party/parallel/include
  third_party/tabulate/include
  third_party/termcolor/include
)
target_include_directories(sort_benchmark SYSTEM PRIVATE)

add_executable(join_benchmark "src/join_benchmark.cu")
target_link_libraries(join_benchmark PUBLIC Thrust::Thrust ThrustCUDA ${CUDA_NVTX_LIBRARY})
target_compile_options(join_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -Xptxas --disable-warnings>)
target_compile_options(join_benchmark PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;-extended-lambda;-DDEBUG_BUILD>")
target_compile_options(join_benchmark PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-extended-lambda>")
set_property(TARGET join_benchmark PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
target_include_directories(join_benchmark PUBLIC
  src/
  third_party/
  third_party/csvparser/single_include
  third_party/cxxopts/include
  third_party/moderngpu/src
  third_party/parallel/include
  third_party/tabulate/include
  third_party/termcolor/include
)
target_include_directories(join_benchmark SYSTEM PRIVATE)
