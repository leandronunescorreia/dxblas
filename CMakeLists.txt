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
    src/backend/d3d11_shader_manager.cpp
    src/backend/d3d11_shader_manager.hpp
    src/shaders/sum.hlsl
    src/shaders/mul.hlsl)

set_source_files_properties(
    src/shaders/sum.hlsl
    src/shaders/mul.hlsl
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
endif()
