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
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(dim, dim);

    t_state.counters = {
      {"dim: ", dim},
    };


    while(t_state.KeepRunning()){
        A = A*B;
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
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(dim, dim);

    const auto size_of_double = sizeof(double);
        
    double alpha_cublas = 1.0;
    double beta_cublas = 0.0;


    const int rows_A = A.rows();
    const int cols_B = B.cols();
    const int cols_A = A.cols();
    const int ld_A = rows_A;
    const int ld_B = B.rows();

    double*  d_A = nullptr;
    double*  d_B = nullptr;
    double*  d_b = nullptr;

    // Compute the memory occupation (I commented out the memory occupation for res in the following.)
    const auto size_of_A_in_bytes = size_of_double * A.size();
    const auto size_of_B_in_bytes = size_of_double * B.size();
    const auto size_of_b_in_bytes = size_of_double * B.size();

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), size_of_A_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B), size_of_B_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes)
    );

    // //Template
    // for (auto _ : t_state) {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     //insert benchmark code here
    //     auto end = std::chrono::high_resolution_clock::now();

    //     auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    //     t_state.SetIterationTime(elapsed_seconds.count());
    // }

    for (auto _ : t_state) {
        auto start = std::chrono::high_resolution_clock::now();
        CUBLAS_CHECK(
            cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_A, cols_B, cols_A, &alpha_cublas, d_A, ld_A, d_B, ld_B, &beta_cublas, d_b, ld_A)
        );
        CUDA_CHECK(
            cudaMemcpy(A.data(), d_b, size_of_b_in_bytes, cudaMemcpyDeviceToHost)
        );
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        t_state.SetIterationTime(elapsed_seconds.count());
    }

    //exportBenchmarkResultsToCSV(benchmark2_name + ".csv", t_state.name(), t_state.iterations(), t_state.real_time(), t_state.cpu_time());
   
    CUDA_CHECK(
        cudaFree(d_A)
    );
    CUDA_CHECK(
        cudaFree(d_B)
    );
    CUDA_CHECK(
        cudaFree(d_b)
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

    const unsigned int repetitions = 1;

    std::vector<unsigned int> matrix_dim = {5, 10, 15, 20};

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
