cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(auerbachs-ajb LANGUAGES CUDA CXX)

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

# =============================================================================
# AJB targets — Adaptive Bandwidth-tier Join
# =============================================================================

# --- ajb_benchmark: main AJB benchmark (requires CUDA) ---
add_executable(ajb_benchmark "src/ajb_benchmark.cu")
target_link_libraries(ajb_benchmark PUBLIC Thrust::Thrust ThrustCUDA ${CUDA_NVTX_LIBRARY})
target_compile_options(ajb_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -Xptxas --disable-warnings>)
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

  # --- test targets (original AJB-adapted, Claude-8) ---
  set(AJB_RENUM_TESTS
    test_bucket_pool
    test_count_oracle
    test_enumerator
    test_index
    test_join_tree
    test_rr_access_tree
    test_unordered_map
  )

  foreach(test_name ${AJB_RENUM_TESTS})
    add_executable(ajb_${test_name} "src/joinrenum/tests/${test_name}.cpp")
    target_include_directories(ajb_${test_name} PRIVATE ${AJB_RENUM_INCLUDE_DIRS})
    target_link_libraries(ajb_${test_name} PRIVATE ${GLPK_LIBRARY})
    target_compile_features(ajb_${test_name} PRIVATE cxx_std_17)
    set_target_properties(ajb_${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  endforeach()

  # --- _full test targets (Claude-9: upstream logic + 20% AJB trace/debug) ---
  #
  # These preserve the complete upstream test logic and add structured
  # [AJB_TRACE] / [AJB_STATE] / [AJB_MEM] / [AJB_TIMER] output for
  # breakpoint-style debugging.  Build one: cmake --build . --target ajb_test_index_full
  set(AJB_RENUM_TESTS_FULL
    test_enumerator_full
    test_index_full
    test_join_tree_full
    test_rr_access_tree_full
    test_count_oracle_full
    test_bucket_pool_full
    test_unordered_map_full
  )

  # Tests that need GLPK
  set(AJB_FULL_NEEDS_GLPK
    test_enumerator_full
    test_index_full
    test_join_tree_full
    test_rr_access_tree_full
    test_count_oracle_full
    test_bucket_pool_full
  )

  foreach(test_name ${AJB_RENUM_TESTS_FULL})
    add_executable(ajb_${test_name} "src/joinrenum/tests/${test_name}.cpp")
    target_include_directories(ajb_${test_name} PRIVATE ${AJB_RENUM_INCLUDE_DIRS})
    target_compile_features(ajb_${test_name} PRIVATE cxx_std_17)
    set_target_properties(ajb_${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    # Link GLPK only where needed
    list(FIND AJB_FULL_NEEDS_GLPK ${test_name} _glpk_idx)
    if(NOT _glpk_idx EQUAL -1)
      target_link_libraries(ajb_${test_name} PRIVATE ${GLPK_LIBRARY})
    endif()
  endforeach()

  # --- test_join_baseline (CPU triangle join benchmark, no GLPK) ---
  add_executable(ajb_test_join_baseline "src/joinrenum/tests/test_join_baseline.cpp")
  target_include_directories(ajb_test_join_baseline PRIVATE ${AJB_RENUM_INCLUDE_DIRS})
  target_compile_features(ajb_test_join_baseline PRIVATE cxx_std_17)
  set_target_properties(ajb_test_join_baseline PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

  # --- tool targets (original AJB-adapted, Claude-8) ---
  set(AJB_RENUM_TOOLS
    gen_co_data
    run_bpt
    upper_bound_demo
    wash_data
  )

  foreach(tool_name ${AJB_RENUM_TOOLS})
    add_executable(ajb_${tool_name} "src/joinrenum/tools/${tool_name}.cpp")
    target_include_directories(ajb_${tool_name} PRIVATE ${AJB_RENUM_INCLUDE_DIRS})
    target_compile_features(ajb_${tool_name} PRIVATE cxx_std_17)
    # Only link glpk for tools that need it (gen_co_data uses RangeTree)
    if(${tool_name} STREQUAL "gen_co_data")
      target_link_libraries(ajb_${tool_name} PRIVATE ${GLPK_LIBRARY})
    endif()
    set_target_properties(ajb_${tool_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  endforeach()

  # --- _full tool targets (Claude-9: upstream logic + 20% AJB diagnostics) ---
  set(AJB_RENUM_TOOLS_FULL
    gen_co_data_full
    run_bpt_full
    upper_bound_full
    wash_data_full
  )

  foreach(tool_name ${AJB_RENUM_TOOLS_FULL})
    add_executable(ajb_${tool_name} "src/joinrenum/tools/${tool_name}.cpp")
    target_include_directories(ajb_${tool_name} PRIVATE ${AJB_RENUM_INCLUDE_DIRS})
    target_compile_features(ajb_${tool_name} PRIVATE cxx_std_17)
    if(${tool_name} STREQUAL "gen_co_data_full")
      target_link_libraries(ajb_${tool_name} PRIVATE ${GLPK_LIBRARY})
    endif()
    set_target_properties(ajb_${tool_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  endforeach()

  # --- _upstream test targets (Claude-10: upstream files with algorithm-level changes) ---
  set(AJB_RENUM_TESTS_UPSTREAM
    test_count_oracle_upstream
    test_index_upstream
    test_join_tree_upstream
    test_enumerator_upstream
    test_rr_access_tree_upstream
    test_bucket_pool_upstream
    test_unordered_map_upstream
  )
  # These need GLPK (same as _full except test_unordered_map)
  set(AJB_UPSTREAM_NEEDS_GLPK
    test_count_oracle_upstream
    test_index_upstream
    test_join_tree_upstream
    test_enumerator_upstream
    test_rr_access_tree_upstream
    test_bucket_pool_upstream
  )

  foreach(test_name ${AJB_RENUM_TESTS_UPSTREAM})
    add_executable(ajb_${test_name} "src/joinrenum/tests/${test_name}.cpp")
    target_include_directories(ajb_${test_name} PRIVATE ${AJB_RENUM_INCLUDE_DIRS})
    target_compile_features(ajb_${test_name} PRIVATE cxx_std_17)
    set_target_properties(ajb_${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    list(FIND AJB_UPSTREAM_NEEDS_GLPK ${test_name} _glpk_idx)
    if(NOT _glpk_idx EQUAL -1)
      target_link_libraries(ajb_${test_name} PRIVATE ${GLPK_LIBRARY})
    endif()
  endforeach()

  # --- _upstream tool targets ---
  set(AJB_RENUM_TOOLS_UPSTREAM
    gen_co_data_upstream
    run_bpt_upstream
    upper_bound_upstream
    wash_data_upstream
  )

  foreach(tool_name ${AJB_RENUM_TOOLS_UPSTREAM})
    add_executable(ajb_${tool_name} "src/joinrenum/tools/${tool_name}.cpp")
    target_include_directories(ajb_${tool_name} PRIVATE ${AJB_RENUM_INCLUDE_DIRS})
    target_compile_features(ajb_${tool_name} PRIVATE cxx_std_17)
    if(${tool_name} STREQUAL "gen_co_data_upstream")
      target_link_libraries(ajb_${tool_name} PRIVATE ${GLPK_LIBRARY})
    endif()
    set_target_properties(ajb_${tool_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
  endforeach()

  # Convenience targets
  add_custom_target(ajb_joinrenum_tests)       # original tests
  add_custom_target(ajb_joinrenum_tests_full)   # _full tests + baseline
  add_custom_target(ajb_joinrenum_tests_upstream)  # _upstream tests
  add_custom_target(ajb_joinrenum_all)          # everything

  foreach(test_name ${AJB_RENUM_TESTS})
    add_dependencies(ajb_joinrenum_tests ajb_${test_name})
    add_dependencies(ajb_joinrenum_all ajb_${test_name})
  endforeach()
  foreach(tool_name ${AJB_RENUM_TOOLS})
    add_dependencies(ajb_joinrenum_tests ajb_${tool_name})
    add_dependencies(ajb_joinrenum_all ajb_${tool_name})
  endforeach()

  foreach(test_name ${AJB_RENUM_TESTS_FULL})
    add_dependencies(ajb_joinrenum_tests_full ajb_${test_name})
    add_dependencies(ajb_joinrenum_all ajb_${test_name})
  endforeach()
  add_dependencies(ajb_joinrenum_tests_full ajb_test_join_baseline)
  add_dependencies(ajb_joinrenum_all ajb_test_join_baseline)
  foreach(tool_name ${AJB_RENUM_TOOLS_FULL})
    add_dependencies(ajb_joinrenum_tests_full ajb_${tool_name})
    add_dependencies(ajb_joinrenum_all ajb_${tool_name})
  endforeach()

  foreach(test_name ${AJB_RENUM_TESTS_UPSTREAM})
    add_dependencies(ajb_joinrenum_tests_upstream ajb_${test_name})
    add_dependencies(ajb_joinrenum_all ajb_${test_name})
  endforeach()
  foreach(tool_name ${AJB_RENUM_TOOLS_UPSTREAM})
    add_dependencies(ajb_joinrenum_tests_upstream ajb_${tool_name})
    add_dependencies(ajb_joinrenum_all ajb_${tool_name})
  endforeach()

else()
  message("-- GLPK not found — skipping AJB joinrenum tests (install libglpk-dev)")
endif()
