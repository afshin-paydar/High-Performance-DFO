cmake_minimum_required(VERSION 3.18)
project(DFO_GPU LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Auto-detect CUDA architecture (for CMake >= 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Main executable
add_executable(dfo_gpu
    src/dfo_gpu.cu
)

# Link CUDA runtime and curand
target_link_libraries(dfo_gpu
    CUDA::curand
    CUDA::cudart
)

# Optional: Build with different architectures
option(BUILD_ALL_ARCHS "Build for multiple GPU architectures" OFF)
if(BUILD_ALL_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
endif()

# Tests
enable_testing()

add_test(NAME test_standard_sphere
    COMMAND dfo_gpu standard sphere 100 30 500)

add_test(NAME test_udfo1500_sphere
    COMMAND dfo_gpu udfo1500 sphere 100 30 500)

add_test(NAME test_udfoz5_rastrigin
    COMMAND dfo_gpu udfoz5 rastrigin 100 30 500)

# Installation
install(TARGETS dfo_gpu
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/dfo_gpu
)

# CPack for packaging
include(CPack)
set(CPACK_PACKAGE_NAME "DFO-GPU")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION "High-Performance GPU DFO/uDFO Optimizer")
