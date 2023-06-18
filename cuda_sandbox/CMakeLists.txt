# ---[ Check cmake version.
cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/lib64)



# ---[ Project specification.
project(cuda_sandbox LANGUAGES CXX CUDA)

include(GNUInstallDirs)

# ##########################################
# cusolver_examples build mode
# ##########################################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# cusolver_examples building flags
# ##########################################

# Global CXX/CUDA flags

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

#                                       needed to pass lambdas to kernels
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  --expt-extended-lambda; # --opt-level=4 
#)

SET(CUDA_PROPAGATE_HOST_FLAGS ON)
SET(CUDA_SEPARABLE_COMPILATION OFF)
SET(CUDA_VERBOSE_BUILD ON)

# Debug options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -g -lineinfo")

# Google benchmark flag/options
set(BENCHMARK_FORMAT csv)
set(BENCHMARK_OUT "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark_2023/results.csv")
set(BENCHMARK_OUT_FORMAT csv)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(benchmark REQUIRED)

# ##########################################
# cusolver_examples target
# ##########################################
include_directories(include)


#MATRIX COPY Executable
add_executable(${PROJECT_NAME}_matrix_copy
		src/copy/cublas_copy_example.cu)
target_include_directories(${PROJECT_NAME}_matrix_copy
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_matrix_copy
    PUBLIC
        cublas
)

#MATRIX MULTIPLICATION Executable
add_executable(${PROJECT_NAME}_matrix_multiplication
		src/matrix_multiplication/example_matrix_multiplication.cu)
target_include_directories(${PROJECT_NAME}_matrix_copy
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_matrix_multiplication
    PUBLIC
        cublas
        Eigen3::Eigen
)


#MATRIX INVERSE Executable
add_executable(${PROJECT_NAME}_matrix_inverse 
		src/getrf/cusolver_getrf_example.cu
)
target_include_directories(${PROJECT_NAME}_matrix_inverse
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_matrix_inverse
    PUBLIC
        cusolver
        cublas
        Eigen3::Eigen
)


#MAIN Executable
add_executable(${PROJECT_NAME}_main 
        src/main/main.cu
        include/chebyshev_differentiation.h
        include/utilities.h
        include/spectral_integration_utilities.h
        include/lie_algebra_utilities.h
        include/globals.h
        include/getCusolverErrorString.h
)
target_include_directories(${PROJECT_NAME}_main
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_main
    PUBLIC
        cusolver
        cublas
        Eigen3::Eigen
)

#MAIN_KERNEL Executable
#MAIN Executable
add_executable(${PROJECT_NAME}_main_kernel
        src/main/main_kernel.cu
        include/chebyshev_differentiation.h
        include/utilities.h
        include/spectral_integration_utilities.h
        include/lie_algebra_utilities.h
        include/globals.h
        include/getCusolverErrorString.h
)
target_include_directories(${PROJECT_NAME}_main_kernel
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_main_kernel
    PUBLIC
        cusolver
        cublas
        Eigen3::Eigen
)


#MATMUL BENCHMARK Executable
add_executable(${PROJECT_NAME}_matMul_benchmark
        src/benchmark_2023/matMul_benchmark.cu
        include/utilities.h
        include/getCusolverErrorString.h
        include/benchmark_csv_exporter.h
)
target_include_directories(${PROJECT_NAME}_matMul_benchmark
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_matMul_benchmark
    PUBLIC
        cusolver
        cublas
        Eigen3::Eigen
        benchmark::benchmark
)
# Add custom command to export benchmark results as CSV
add_custom_command(TARGET ${PROJECT_NAME}_matMul_benchmark POST_BUILD
    COMMAND ${PROJECT_NAME}_matMul_benchmark --benchmark_out_format=csv --benchmark_out="matMul_benchmark_result.csv"
    # ${PROJECT_NAME}_matMul_benchmark --benchmark_format=csv
    COMMENT "Exporting benchmark results to result.csv"
)

#MATLU BENCHMARK Executable
add_executable(${PROJECT_NAME}_matLU_benchmark
        src/benchmark_2023/matLU_benchmark.cu
        include/utilities.h
        include/getCusolverErrorString.h
        include/benchmark_csv_exporter.h
)
target_include_directories(${PROJECT_NAME}_matLU_benchmark
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_matLU_benchmark
    PUBLIC
        cusolver
        cublas
        Eigen3::Eigen
        benchmark::benchmark
)
# Add custom command to export benchmark results as CSV
add_custom_command(TARGET ${PROJECT_NAME}_matLU_benchmark POST_BUILD
    COMMAND ${PROJECT_NAME}_matLU_benchmark --benchmark_out_format=csv --benchmark_out="matLU_benchmark_result.csv"
    # ${PROJECT_NAME}_matMul_benchmark --benchmark_format=csv
    COMMENT "Exporting benchmark results to result.csv"
)


#MATINVBENCHMARK Executable
add_executable(${PROJECT_NAME}_matSys_benchmark
        src/benchmark_2023/matSys_benchmark.cu
        include/utilities.h
        include/getCusolverErrorString.h
        include/benchmark_csv_exporter.h
)
target_include_directories(${PROJECT_NAME}_matSys_benchmark
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_matSys_benchmark
    PUBLIC
        cusolver
        cublas
        Eigen3::Eigen
        benchmark::benchmark
)
# Add custom command to export benchmark results as CSV
add_custom_command(TARGET ${PROJECT_NAME}_matSys_benchmark POST_BUILD
    COMMAND ${PROJECT_NAME}_matSys_benchmark --benchmark_out_format=csv --benchmark_out="matSys_benchmark_result.csv"
    # ${PROJECT_NAME}_matMul_benchmark --benchmark_format=csv
    COMMENT "Exporting benchmark results to result.csv"
)


#MAIN BENCHMARK CPU Executable
add_executable(${PROJECT_NAME}_main_benchmark_CPU
    src/benchmark_2023/main_benchmark_CPU.cpp
    include/chebyshev_differentiation.h
    include/spectral_integration_utilities.h
    include/benchmark_csv_exporter.h
)
target_link_libraries(${PROJECT_NAME}_main_benchmark_CPU
    PUBLIC
        Eigen3::Eigen
        benchmark::benchmark
)
# Add custom command to export benchmark results as CSV
add_custom_command(TARGET ${PROJECT_NAME}_main_benchmark_CPU POST_BUILD
    COMMAND ${PROJECT_NAME}_main_benchmark_CPU --benchmark_out_format=csv --benchmark_out="main_benchmark_CPU_result.csv"
    # ${PROJECT_NAME}_matMul_benchmark --benchmark_format=csv
    COMMENT "Exporting benchmark results to result.csv"
)


#MAIN BENCHMARK GPU Executable
add_executable(${PROJECT_NAME}_main_benchmark_GPU
        src/benchmark_2023/main_benchmark_GPU.cu
        include/chebyshev_differentiation.h
        include/utilities.h
        include/spectral_integration_utilities.h
        include/lie_algebra_utilities.h
        include/globals.h
        include/getCusolverErrorString.h
)
target_include_directories(${PROJECT_NAME}_main_benchmark_GPU 
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_main_benchmark_GPU
    PUBLIC
        cusolver
        cublas
        Eigen3::Eigen
        benchmark::benchmark
        )
# Add custom command to export benchmark results as CSV
add_custom_command(TARGET ${PROJECT_NAME}_main_benchmark_GPU POST_BUILD
    COMMAND ${PROJECT_NAME}_main_benchmark_GPU --benchmark_out_format=csv --benchmark_out="main_benchmark_GPU_result.csv"
    # ${PROJECT_NAME}_matMul_benchmark --benchmark_format=csv
    COMMENT "Exporting benchmark results to result.csv"
)