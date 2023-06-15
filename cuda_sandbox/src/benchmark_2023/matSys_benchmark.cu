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
#include <Eigen/LU>



void benchmarkMatMul_CPU(::benchmark::State &t_state)
{
    const unsigned int dim = t_state.range(0);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(dim, 1);


    t_state.counters = {
      {"dim: ", dim},
    };


    while(t_state.KeepRunning()){
        Eigen::MatrixXd x = A.lu().solve(b);
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
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(dim, 1);

    const auto size_of_double = sizeof(double);
        
    double alpha_cublas = 1.0;
    double beta_cublas = 0.0;


    const int rows_A = A.rows();
    const int cols_A = A.cols();
    const int ld_A = rows_A;
    const int rows_b = b.rows();
    const int cols_b = b.cols();
    const int ld_b = cols_b;

    // LU factorization variables
    int info = 0;
    int lwork = 0;

    double*  d_A = nullptr;
    double*  d_b = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    CUDA_CHECK(
        cusolverDnDgetrf_bufferSize(cusolverH, rows_A, cols_A, d_A, ld_A, &lwork);
    );

    // Compute the memory occupation (I commented out the memory occupation for res in the following.)
    const auto size_of_A_in_bytes = size_of_double * A.size();
    const auto size_of_b_in_bytes = size_of_double * b.size();
    

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), size_of_A_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork)
    );

        //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_A, A.data(), size_of_A_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_b, b.data(), size_of_b_in_bytes, cudaMemcpyHostToDevice)
    );
    

    for (auto _ : t_state) {
        auto start = std::chrono::high_resolution_clock::now();

        CUSOLVER_CHECK(
            cusolverDnDgetrf(cusolverH, rows_A, cols_A, d_A, ld_A, d_work, NULL, d_info)
        );

        CUSOLVER_CHECK(
            cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_A, 1, d_A, ld_A, NULL, d_b, ld_b, d_info)
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

    const unsigned int repetitions = 20;

    std::vector<unsigned int> matrix_dim = {20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500};

    const std::string benchmark1_name = "System Solver CPU";
    const std::string benchmark2_name = "System Solver GPU";


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
