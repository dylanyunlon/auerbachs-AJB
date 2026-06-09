cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(auerbachs-ajb LANGUAGES CUDA CXX)

find_package(OpenMP REQUIRED)

message("-- Building for ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp")
set(CMAKE_CXX_STANDARD 20)

# CUDA architectures: override via cmake -DCMAKE_CUDA_ARCHITECTURES="86;90"
# Default: sm_70+80 (safe for CUDA 11.x); ags1: use "86;90" with CUDA 12+
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CUDA_ARCH_LIST 70 80)
else()
  string(REPLACE ";" " " CUDA_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")
endif()

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
target_compile_options(sort_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream;per-thread;-Xptxas;--disable-warnings>)
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
target_compile_options(join_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream;per-thread;-Xptxas;--disable-warnings>)
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

# =============================================================================
# AJB targets — Adaptive Bandwidth-tier Join
# =============================================================================

# --- ajb_benchmark: main AJB benchmark (requires CUDA) ---
add_executable(ajb_benchmark "src/ajb_benchmark.cu")
target_link_libraries(ajb_benchmark PUBLIC Thrust::Thrust ThrustCUDA ${CUDA_NVTX_LIBRARY})
target_compile_options(ajb_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream;per-thread;-Xptxas;--disable-warnings>)
target_compile_options(ajb_benchmark PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;-extended-lambda;-DDEBUG_BUILD;-DAJB_DEBUG;-DAJB_TRACE_DECISIONS;-DAJB_TRACE_TRANSFERS>")
target_compile_options(ajb_benchmark PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-extended-lambda;-DAJB_TRACE_DECISIONS>")
set_property(TARGET ajb_benchmark PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
target_include_directories(ajb_benchmark PUBLIC
  src/
  src/ajb_join/
  src/joinrenum/
  third_party/
  third_party/csvparser/single_include
  third_party/cxxopts/include
  third_party/moderngpu/src
  third_party/parallel/include
  third_party/tabulate/include
  third_party/termcolor/include
)
target_include_directories(ajb_benchmark SYSTEM PRIVATE)

# =============================================================================
# AJB JoinREnum test/tool targets (CPU-only, no CUDA required)
#
# These wrap the upstream joinrenum test programs with AJB debug
# instrumentation: scoped timers, structured state dumps, memory
# reporters, and pass/fail verdicts.
#
# Build all:  cmake --build . --target ajb_joinrenum_tests
# Build one:  cmake --build . --target ajb_test_bucket_pool
# =============================================================================

find_library(GLPK_LIBRARY glpk)

if(GLPK_LIBRARY)
  message("-- GLPK found: ${GLPK_LIBRARY}  → building AJB joinrenum tests")

  set(AJB_RENUM_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src/joinrenum
  )

  # =================================================================
  # M1231: unified test/tool targets — no _full/_upstream/_demo ghosts
  # Each test/tool exists exactly once on disk, built once.
  # =================================================================

  # --- Tests that need GLPK (joinrenum core) ---
  set(AJB_RENUM_TESTS_GLPK
    test_bucket_pool
    test_count_oracle
    test_enumerator
    test_index
    test_join_tree
    test_rr_access_tree
    test_renum_baseline
    test_sample_baseline
  )

  foreach(test_name ${AJB_RENUM_TESTS_GLPK})
    add_executable(ajb_${test_name} "src/joinrenum/tests/${test_name}.cpp")
    target_include_directories(ajb_${test_name} PRIVATE ${AJB_RENUM_INCLUDE_DIRS})
    target_link_libraries(ajb_${test_name} PRIVATE ${GLPK_LIBRARY})
    target_compile_features(ajb_${test_name} PRIVATE cxx_std_17)
    set_target_properties(ajb_${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  endforeach()

  # --- Tests that do NOT need GLPK ---
  set(AJB_RENUM_TESTS_NOGLPK
    test_unordered_map
    test_join_baseline
    test_join_triangle
  )

  foreach(test_name ${AJB_RENUM_TESTS_NOGLPK})
    add_executable(ajb_${test_name} "src/joinrenum/tests/${test_name}.cpp")
    target_include_directories(ajb_${test_name} PRIVATE ${AJB_RENUM_INCLUDE_DIRS})
    target_compile_features(ajb_${test_name} PRIVATE cxx_std_17)
    set_target_properties(ajb_${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  endforeach()

  # --- Tool targets ---
  set(AJB_RENUM_TOOLS
    gen_co_data
    run_bpt
    upper_bound
    wash_data
  )

  foreach(tool_name ${AJB_RENUM_TOOLS})
    add_executable(ajb_${tool_name} "src/joinrenum/tools/${tool_name}.cpp")
    target_include_directories(ajb_${tool_name} PRIVATE ${AJB_RENUM_INCLUDE_DIRS})
    target_compile_features(ajb_${tool_name} PRIVATE cxx_std_17)
    if(${tool_name} STREQUAL "gen_co_data")
      target_link_libraries(ajb_${tool_name} PRIVATE ${GLPK_LIBRARY})
    endif()
    set_target_properties(ajb_${tool_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  endforeach()

  # --- Convenience umbrella targets ---
  add_custom_target(ajb_joinrenum_tests)
  add_custom_target(ajb_joinrenum_all)

  foreach(test_name ${AJB_RENUM_TESTS_GLPK} ${AJB_RENUM_TESTS_NOGLPK})
    add_dependencies(ajb_joinrenum_tests ajb_${test_name})
    add_dependencies(ajb_joinrenum_all ajb_${test_name})
  endforeach()
  foreach(tool_name ${AJB_RENUM_TOOLS})
    add_dependencies(ajb_joinrenum_all ajb_${tool_name})
  endforeach()

else()
  message("-- GLPK not found — skipping AJB joinrenum tests (install libglpk-dev)")
endif()
