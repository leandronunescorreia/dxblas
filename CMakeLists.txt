cmake_minimum_required(VERSION 3.20)

project(dxBLAS_D3D11
    VERSION 0.2.0
    LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "dxBLAS D3D11 currently requires Windows.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(dxBLAS_BUILD_EXAMPLE "Build the step-by-step example" ON)
option(dxBLAS_ENABLE_D3D11_DEBUG "Define D3D11 debug support for the library" OFF)

add_library(dxblas
    src/dxblas.cpp
    src/d3d11_backend.cpp
    src/backend/arena_texture.cpp
    src/backend/arena_texture.hpp
    src/backend/d3d11_shader_manager.cpp
    src/backend/d3d11_shader_manager.hpp
    src/shaders/sum.hlsl
    src/shaders/mul.hlsl
    src/shaders/gemm.hlsl)

set_source_files_properties(
    src/shaders/sum.hlsl
    src/shaders/mul.hlsl
    src/shaders/gemm.hlsl
    PROPERTIES HEADER_FILE_ONLY TRUE)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

target_include_directories(dxblas
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_compile_definitions(dxblas
    PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX)

target_link_libraries(dxblas
    PUBLIC
        d3d11
        dxgi
        d3dcompiler)

if(MSVC)
    target_compile_options(dxblas PRIVATE /W4 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(dxblas PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(dxBLAS_BUILD_EXAMPLE)
    add_executable(dxblas_first_step
        examples/fist_step.cpp)

    target_link_libraries(dxblas_first_step
        PRIVATE dxblas)

    target_compile_definitions(dxblas_first_step
        PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX)

    add_executable(dxblas_big_array
        examples/big_array.cpp)

    target_link_libraries(dxblas_big_array
        PRIVATE dxblas)

    target_compile_definitions(dxblas_big_array
        PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX)

    add_executable(dxblas_benchmark_2048
        examples/benchmark_2048.cpp)

    target_link_libraries(dxblas_benchmark_2048
        PRIVATE dxblas)

    target_compile_definitions(dxblas_benchmark_2048
        PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX)

    add_executable(dxblas_typed_operations
        examples/typed_operations.cpp)

    target_link_libraries(dxblas_typed_operations
        PRIVATE dxblas)

    target_compile_definitions(dxblas_typed_operations
        PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX)

    add_executable(dxblas_gpumem_roundtrip examples/gpumem_roundtrip.cpp)
    target_link_libraries(dxblas_gpumem_roundtrip PRIVATE dxblas)
    target_compile_definitions(dxblas_gpumem_roundtrip
        PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

    add_executable(dxblas_gpumem_sum examples/gpumem_sum.cpp)
    target_link_libraries(dxblas_gpumem_sum PRIVATE dxblas)
    target_compile_definitions(dxblas_gpumem_sum
        PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

    add_executable(dxblas_copy_tests examples/copy_tests.cpp)
    target_link_libraries(dxblas_copy_tests PRIVATE dxblas)
    target_compile_definitions(dxblas_copy_tests
        PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

    add_executable(dxblas_copy_timing examples/copy_timing.cpp)
    target_link_libraries(dxblas_copy_timing PRIVATE dxblas)
    target_include_directories(dxblas_copy_timing PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(dxblas_copy_timing
        PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

    add_executable(dxblas_gemm examples/dxblas_gemm.cpp)
    target_link_libraries(dxblas_gemm PRIVATE dxblas)
    target_compile_definitions(dxblas_gemm PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

    add_executable(dxblas_matrix_multiply_4096 examples/matrix_multiply_4096.cpp)
    target_link_libraries(dxblas_matrix_multiply_4096 PRIVATE dxblas)
    target_compile_definitions(dxblas_matrix_multiply_4096 PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

    # Also deploy editable HLSL beside the executable. Shader loading has an
    # embedded fallback, so this is for development and runtime overrides.
    add_custom_command(TARGET dxblas_gemm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
            "$<TARGET_FILE_DIR:dxblas_gemm>/shaders"
        VERBATIM)
endif()
