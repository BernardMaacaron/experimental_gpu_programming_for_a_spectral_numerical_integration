#include <iostream>
#include <benchmark/benchmark.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

//  CUDA Basic Linear Algebra 
#include <cublas_v2.h>

#include "utilities.h"  //  Utilities for error handling (must be after cublas or cusolver)
#include "getCusolverErrorString.h"
#include "benchmark_csv_exporter.h"


#include <Eigen/Dense>



void benchmarkMatMul_CPU(::benchmark::State &t_state)
{
    const unsigned int dim = t_state.range(0);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim);

    t_state.counters = {
      {"dim: ", dim},
    };


    while(t_state.KeepRunning()){
        Eigen::MatrixXd A_inv = A.inverse();
    }
    //exportBenchmarkResultsToCSV(benchmark1_name + ".csv", .name(), .iterations(), t_state.real_time(), t_state.cpu_time());
};


void benchmarkMatMul_GPU(::benchmark::State &t_state)
{
    int dim = t_state.range(0);

    t_state.counters = {
      {"dim: ", dim},
    };

    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;

    CUBLAS_CHECK(
        cublasCreate(&cublasH)
    );

    CUSOLVER_CHECK(
        cusolverDnCreate(&cusolverH)
    );

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim);

    const auto size_of_double = sizeof(double);
    
    const int rows_A = A.rows();
    const int cols_A = A.cols();
    const int ld_A = rows_A;

    const int rows_A_inv = rows_A;
    const int cols_A_inv = cols_A;
    const int ld_A_inv = rows_A_inv;

    double*  d_A = nullptr;
    double*  d_A_inv = nullptr;


    // Compute the memory occupation
    const auto size_of_A_in_bytes = size_of_double * A.size();
    const auto size_of_A_inv_in_bytes = size_of_A_in_bytes;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), size_of_A_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A_inv), size_of_A_inv_in_bytes)
    );

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_A, A.data(), size_of_A_in_bytes, cudaMemcpyHostToDevice)
    );

    // Create an array to store the error code of cublasDgetrfBatched
    int devInfoArray[1];

    for (auto _ : t_state) {
        auto start = std::chrono::high_resolution_clock::now();

        cublasStatus_t status = cublasDgetrf(cublasH, dim, d_A, ld_A, NULL, d_A_inv, ld_A_inv, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Matrix inversion failed.\n");
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        t_state.SetIterationTime(elapsed_seconds.count());
    }

    //exportBenchmarkResultsToCSV(benchmark2_name + ".csv", t_state.name(), t_state.iterations(), t_state.real_time(), t_state.cpu_time());
   
    CUDA_CHECK(
        cudaFree(d_A)
    );
    CUBLAS_CHECK(
        cublasDestroy(cublasH)
    );
    CUSOLVER_CHECK(
        cusolverDnDestroy(cusolverH)
    );
    CUDA_CHECK(
        cudaDeviceReset()
    );
};



int main(int argc, char *argv[])
{

    const unsigned int repetitions = 20;

    std::vector<unsigned int> matrix_dim = {20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500};

    const std::string benchmark1_name = "Matrix Multiplication CPU";
    const std::string benchmark2_name = "Matrix Multiplication GPU";


    for(const auto dim : matrix_dim)
        ::benchmark::RegisterBenchmark(benchmark1_name.c_str(), benchmarkMatMul_CPU)->Arg(dim)->Repetitions(repetitions)->Unit(::benchmark::kMicrosecond);
        
    ::benchmark::RegisterBenchmark(benchmark1_name.c_str(), [](::benchmark::State &t_state){
        for(auto _ : t_state)
            int a = 0;
    });

    for(const auto dim : matrix_dim)
        ::benchmark::RegisterBenchmark(benchmark2_name.c_str(), benchmarkMatMul_GPU)->Arg(dim)->Repetitions(repetitions)->Unit(::benchmark::kMicrosecond);

    ::benchmark::RegisterBenchmark(benchmark2_name.c_str(), [](::benchmark::State &t_state){
        for(auto _ : t_state)
            int a = 0;
    });
            



    ::benchmark::Initialize(&argc, argv);


    ::benchmark::RunSpecifiedBenchmarks();


    return 0;
}
