# metal/ - strata-poly on Apple Silicon (Metal).  Its own CMake project, macOS only, never part of the CUDA / HIP builds.
#
# strata-poly's host sources from ../src, UNCHANGED: include/metal_compat goes first on the include path and gives them
# the CUDA runtime names on Metal, as include/strata/hip_compat does on HIP, over the Metal device runtime
# src/gpu/runtime.cpp (metal-cpp) and strata-poly's Metal kernels in src/poly/.
#
#   cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release -DMETAL_CPP_DIR=... -DSTRATA_GGML_DIR=... \
#         -DSTRATA_METAL_POLY=ON
#   ninja -C build-metal strata-poly
cmake_minimum_required(VERSION 3.24)
project(strata_metal LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "metal/ builds on macOS only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_OBJCXX_FLAGS_RELEASE "-O2")

include(CTest)

get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(S ${STRATA_ROOT}/src)

# ---- the device runtime ----------------------------------------------------------------------------------------------
set(METAL_CPP_DIR "" CACHE PATH "metal-cpp headers (https://developer.apple.com/metal/cpp/), e.g. metal-cpp_26")
if(NOT EXISTS "${METAL_CPP_DIR}/Metal/Metal.hpp")
  message(STATUS "strata metal: METAL_CPP_DIR is not set: the device runtime and the engine are not built")
  return()
endif()

add_library(strata_gpu_metal STATIC src/gpu/runtime.cpp)
target_include_directories(strata_gpu_metal BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/metal_compat
                           ${CMAKE_CURRENT_SOURCE_DIR}/include ${STRATA_ROOT}/include)
target_include_directories(strata_gpu_metal SYSTEM PUBLIC ${METAL_CPP_DIR})
target_compile_definitions(strata_gpu_metal PUBLIC STRATA_GPU_METAL=1)
target_link_libraries(strata_gpu_metal PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")

# the runtime end to end, with the shared src/core/graph.cpp built unchanged
add_library(strata_core_metal STATIC ${S}/core/graph.cpp)
target_link_libraries(strata_core_metal PUBLIC strata_gpu_metal)
add_executable(runtime_hello tests/runtime_hello.cpp)
target_link_libraries(runtime_hello PRIVATE strata_core_metal)
if(BUILD_TESTING)
  add_test(NAME runtime_hello COMMAND runtime_hello)
  set_tests_properties(runtime_hello PROPERTIES LABELS "metal;gpu;runtime"
                       RESOURCE_LOCK metal_gpu TIMEOUT 120)
endif()

# ---- strata-poly ------------------------------------------------------------------------------------------------------
option(STRATA_METAL_POLY "Build strata-poly for Metal and its checks" OFF)
if(NOT STRATA_METAL_POLY)
  return()
endif()

# ggml (the GGUF block formats' type traits and ggml-cpu, for the CPU experts) from a llama.cpp checkout at the commit
# the PC build pins (CMakeLists.txt: 3cf03257).
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout at the pinned commit")
if(NOT EXISTS "${STRATA_GGML_DIR}/ggml/CMakeLists.txt")
  message(FATAL_ERROR "STRATA_GGML_DIR='${STRATA_GGML_DIR}' is not a llama.cpp checkout")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

# strata-poly's Metal kernels (compiled at run time from the .metal sources).  Full-model validation is pending.
add_library(strata_poly_metal_kernels STATIC src/poly/basic.cpp src/poly/attention.cpp src/poly/recurrent.cpp
            src/poly/quantized.cpp src/poly/unported.cpp)
target_include_directories(strata_poly_metal_kernels PRIVATE ${S}/poly ${STRATA_GGML_DIR}/ggml/include)
target_compile_definitions(strata_poly_metal_kernels PRIVATE
  STRATA_POLY_BASIC_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/basic.metal"
  STRATA_POLY_ATTENTION_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/attention.metal"
  STRATA_POLY_FLASH_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/flash.metal"
  STRATA_POLY_RECURRENT_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/recurrent.metal"
  STRATA_POLY_GGML_SOURCE="${STRATA_GGML_DIR}/ggml/src"
  STRATA_POLY_QUANT_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/quantized.metal"
  STRATA_POLY_MM_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/mm.metal")
target_link_libraries(strata_poly_metal_kernels PUBLIC strata_gpu_metal ggml-base)
add_executable(poly_basic_parity poly_tests/basic_parity.cpp)
target_link_libraries(poly_basic_parity PRIVATE strata_poly_metal_kernels)
set_source_files_properties(${STRATA_ROOT}/src/poly/route_test.cu PROPERTIES LANGUAGE CXX)
add_executable(poly_route_metal_test ${STRATA_ROOT}/src/poly/route_test.cu)
target_link_libraries(poly_route_metal_test PRIVATE strata_poly_metal_kernels)
if(BUILD_TESTING)
  add_test(NAME poly_basic_parity COMMAND poly_basic_parity)
  set_tests_properties(poly_basic_parity PROPERTIES LABELS "metal;gpu;poly"
                      RESOURCE_LOCK metal_gpu TIMEOUT 120)
  add_test(NAME poly_route_metal_test COMMAND poly_route_metal_test)
  set_tests_properties(poly_route_metal_test PROPERTIES LABELS "metal;gpu;poly"
                      RESOURCE_LOCK metal_gpu TIMEOUT 120)
endif()

# The engine keeps the shared line protocol, expert cache, CPU experts, batching, conversations and speculative
# decoding.
add_executable(strata-poly ${S}/poly/main.cpp ${S}/poly/engine.cpp ${S}/poly/qwen35.cpp
  ${S}/poly/gemma4.cpp ${S}/poly/cpu_experts.cpp ${S}/poly/spill.cpp ${S}/platform/memory.cpp
  ${S}/core/conversation_memory.cpp ${S}/spec/draft_policy.cpp ${S}/spec/suffix_drafter.cpp)
target_include_directories(strata-poly PRIVATE ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src)
target_link_libraries(strata-poly PRIVATE strata_poly_metal_kernels ggml-cpu ggml-base)
target_compile_definitions(strata-poly PRIVATE STRATA_VERSION="0.1.45-metal-poly")
set_source_files_properties(${S}/poly/norm_q8_test.cu PROPERTIES LANGUAGE CXX)
add_executable(poly_norm_q8_metal_test ${S}/poly/norm_q8_test.cu)
target_link_libraries(poly_norm_q8_metal_test PRIVATE strata_poly_metal_kernels)
if(BUILD_TESTING)
  add_test(NAME poly_norm_q8_metal_test COMMAND poly_norm_q8_metal_test)
  set_tests_properties(poly_norm_q8_metal_test PROPERTIES LABELS "metal;gpu;poly" RESOURCE_LOCK metal_gpu TIMEOUT 120)
endif()

# Mmq::mul's time on a prompt batch's shapes, run by hand
add_executable(poly_mm_bench poly_tests/mm_bench.cpp)
target_include_directories(poly_mm_bench PRIVATE ${S}/poly)
target_link_libraries(poly_mm_bench PRIVATE strata_poly_metal_kernels)
foreach(t attention quantized recurrent operations)
  add_executable(poly_${t}_parity poly_tests/${t}_parity.cpp)
  target_include_directories(poly_${t}_parity PRIVATE ${S}/poly)
  target_link_libraries(poly_${t}_parity PRIVATE strata_poly_metal_kernels)
  if(BUILD_TESTING)
    add_test(NAME poly_${t}_parity COMMAND poly_${t}_parity)
    set_tests_properties(poly_${t}_parity PROPERTIES LABELS "metal;gpu;poly" RESOURCE_LOCK metal_gpu TIMEOUT 120)
  endif()
endforeach()

if(BUILD_TESTING)
  add_executable(poly_tiny_model_fixture poly_tests/tiny_model_fixture.cpp)
  target_link_libraries(poly_tiny_model_fixture PRIVATE ggml-base)
  foreach(mode graphs launches cpu_experts)
    set(extra)
    if(mode STREQUAL "launches")
      set(extra --no-graphs)
    elseif(mode STREQUAL "cpu_experts")
      set(extra --no-gpu-experts)
    endif()
    add_test(NAME poly_tiny_qwen_${mode} COMMAND strata-poly --model ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe.gguf
      --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 --max-new 4 --max-context 128
      --prompt-batch 32 --threads 2 --no-draft ${extra})
    set_tests_properties(poly_tiny_qwen_${mode} PROPERTIES LABELS "metal;gpu;poly;synthetic"
      RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_qwen35moe)
  endforeach()

  add_executable(poly_memory_test poly_tests/memory_test.cpp ${S}/core/conversation_memory.cpp)
  target_include_directories(poly_memory_test PRIVATE ${STRATA_ROOT}/include)
  add_test(NAME poly_memory_test COMMAND poly_memory_test)
  set_tests_properties(poly_memory_test PROPERTIES LABELS "metal;poly;cpu")
  add_executable(poly_conversation_memory_test ${S}/core/conversation_memory_test.cpp ${S}/core/conversation_memory.cpp)
  target_include_directories(poly_conversation_memory_test PRIVATE ${STRATA_ROOT}/include)
  add_test(NAME poly_conversation_memory_test COMMAND poly_conversation_memory_test)
  set_tests_properties(poly_conversation_memory_test PROPERTIES LABELS "metal;poly;cpu")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  foreach(arch qwen35moe gemma4)
    add_test(NAME poly_tiny_${arch}_fixture COMMAND poly_tiny_model_fixture ${CMAKE_CURRENT_BINARY_DIR}/tiny-${arch}.gguf ${arch})
    set_tests_properties(poly_tiny_${arch}_fixture PROPERTIES LABELS "metal;poly;synthetic" FIXTURES_SETUP tiny_${arch})
    add_test(NAME poly_tiny_${arch}_batch COMMAND strata-poly --model ${CMAKE_CURRENT_BINARY_DIR}/tiny-${arch}.gguf
      --tokens 1,2,3,4,5 --max-new 8 --max-context 128 --prompt-batch 32 --threads 2 --no-draft --batch 2 --check-batch 2)
    set_tests_properties(poly_tiny_${arch}_batch PROPERTIES LABELS "metal;gpu;poly;synthetic"
      RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_${arch})
    add_test(NAME poly_tiny_${arch}_protocol COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py
      $<TARGET_FILE:strata-poly> ${CMAKE_CURRENT_BINARY_DIR}/tiny-${arch}.gguf)
    set_tests_properties(poly_tiny_${arch}_protocol PROPERTIES LABELS "metal;gpu;poly;synthetic"
      RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_${arch})
  endforeach()
  add_test(NAME poly_tiny_qwen_q8 COMMAND strata-poly --model ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe.gguf
    --tokens 1,2,3,4,5 --max-new 8 --max-context 128 --prompt-batch 32 --threads 2 --no-draft --batch 2 --check-batch 2 --kv q8_0)
  set_tests_properties(poly_tiny_qwen_q8 PROPERTIES LABELS "metal;gpu;poly;synthetic"
    RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_qwen35moe)
  # each sequence's draft block guesses in the batch, and the answers are still each one's alone
  add_test(NAME poly_tiny_qwen_batch_mtp COMMAND strata-poly --model ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe.gguf
    --tokens 1,2,3,4,5 --max-new 8 --max-context 128 --prompt-batch 32 --threads 2 --window 2 --batch 2 --check-batch 2
    --batch-mtp)
  set_tests_properties(poly_tiny_qwen_batch_mtp PROPERTIES LABELS "metal;gpu;poly;synthetic"
    RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_qwen35moe)
  add_test(NAME poly_tiny_qwen_draft COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py
    $<TARGET_FILE:strata-poly> ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe.gguf --window 4)
  set_tests_properties(poly_tiny_qwen_draft PROPERTIES LABELS "metal;gpu;poly;synthetic"
    RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_qwen35moe)

  target_compile_definitions(strata_gpu_metal PRIVATE STRATA_METAL_TESTING=1)
  add_executable(poly_runtime_safety poly_tests/runtime_safety.cpp)
  target_compile_definitions(poly_runtime_safety PRIVATE STRATA_METAL_TESTING=1)
  target_link_libraries(poly_runtime_safety PRIVATE strata_gpu_metal)
  add_test(NAME poly_runtime_failure COMMAND poly_runtime_safety failure)
  set_tests_properties(poly_runtime_failure PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 30 LABELS "metal;gpu;poly")
  add_test(NAME poly_runtime_safety COMMAND poly_runtime_safety)
  set_tests_properties(poly_runtime_safety PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 60 LABELS "metal;gpu;poly")

  get_target_property(poly_engine_test_sources strata-poly SOURCES)
  list(REMOVE_ITEM poly_engine_test_sources ${S}/poly/main.cpp)
  add_library(poly_engine_test_host STATIC ${poly_engine_test_sources})
  target_include_directories(poly_engine_test_host PUBLIC ${S}/poly ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src)
  target_link_libraries(poly_engine_test_host PUBLIC strata_poly_metal_kernels ggml-cpu ggml-base)
  add_executable(poly_memory_growth poly_tests/memory_growth.cpp)
  target_compile_definitions(poly_memory_growth PRIVATE STRATA_METAL_TESTING=1)
  target_link_libraries(poly_memory_growth PRIVATE poly_engine_test_host)
  add_test(NAME poly_memory_growth COMMAND poly_memory_growth)
  set_tests_properties(poly_memory_growth PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 60 LABELS "metal;gpu;poly")
  foreach(arch qwen35moe gemma4)
    add_test(NAME poly_balanced_${arch}_protocol COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py $<TARGET_FILE:strata-poly>
      ${CMAKE_CURRENT_BINARY_DIR}/tiny-${arch}.gguf --metal-optimize --metal-memory balanced --no-draft)
    set_tests_properties(poly_balanced_${arch}_protocol PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 120
      LABELS "metal;gpu;poly;synthetic" FIXTURES_REQUIRED tiny_${arch})
  endforeach()
  add_test(NAME poly_balanced_qwen_q8_protocol COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py $<TARGET_FILE:strata-poly>
    ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe.gguf --metal-optimize --metal-memory balanced --kv q8_0)
  set_tests_properties(poly_balanced_qwen_q8_protocol PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 120
    LABELS "metal;gpu;poly;synthetic" FIXTURES_REQUIRED tiny_qwen35moe)

  add_test(NAME poly_tiny_qwen_fp16_fixture COMMAND poly_tiny_model_fixture
    ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe-fp16.gguf qwen35moe-fp16)
  set_tests_properties(poly_tiny_qwen_fp16_fixture PROPERTIES LABELS "metal;poly;synthetic"
    FIXTURES_SETUP tiny_qwen_fp16)
  add_test(NAME poly_metal_defaults COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/defaults_test.py $<TARGET_FILE:strata-poly> ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(poly_metal_defaults PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 180
    LABELS "metal;gpu;poly;synthetic"
    FIXTURES_REQUIRED "tiny_qwen35moe;tiny_gemma4;tiny_qwen_fp16")
  add_test(NAME poly_legacy_qwen_protocol COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py $<TARGET_FILE:strata-poly>
    ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe.gguf --no-metal-optimize --metal-memory legacy --kv fp16)
  set_tests_properties(poly_legacy_qwen_protocol PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 120
    LABELS "metal;gpu;poly;synthetic" FIXTURES_REQUIRED tiny_qwen35moe)
endif()
