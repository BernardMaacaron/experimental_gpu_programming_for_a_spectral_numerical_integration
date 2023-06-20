#include <cstdio>
#include <cstdlib>
#include <vector>

#include <fstream>
#include <cmath>

#include <cublas_v2.h>
// // #include <cuda_runtime.h>

#include "spectral_integration_utilities.h"
#include "chebyshev_differentiation.h"
#include "lie_algebra_utilities.h"
#include "utilities.h"  //  Utilities for error handling (must be after cublas or cusolver)
#include "globals.h"
#include "getCusolverErrorString.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <benchmark/benchmark.h>


//GLOBAL VARIABLES ARE DEFINED IN globals.h

// Used to build Q_stack
__global__ void computeCMatrixKernel(const double* d_K_stack, const double* D_NN, double* C_NN) {

    int i = threadIdx.x;

    #pragma region Compute_C_NN
    if (i < number_of_Chebyshev_points-1) {
        int row = 0;
        int col = 1;
        int row_index = row * (number_of_Chebyshev_points - 1) + i;
        int col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] + 0.5*d_K_stack[3*i+0];


        row = 0;
        col = 2;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] + 0.5*d_K_stack[3*i+1];

        row = 0;
        col = 3;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] + 0.5*d_K_stack[3*i+2];

        row = 1;
        col = 0;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] - 0.5*d_K_stack[3*i+0];

        row = 1;
        col = 2;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] - 0.5*d_K_stack[3*i+2];

        row = 1;
        col = 3;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] + 0.5*d_K_stack[3*i+1];

        row = 2;
        col = 0;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] - 0.5*d_K_stack[3*i+1];

        row = 2;
        col = 1;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] + 0.5*d_K_stack[3*i+2];

        row = 2;
        col = 3;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] - 0.5*d_K_stack[3*i+0];

        row = 3;
        col = 0;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] - 0.5*d_K_stack[3*i+2];

        row = 3;
        col = 1;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] - 0.5*d_K_stack[3*i+1];

        row = 3;
        col = 2;
        row_index = row * (number_of_Chebyshev_points - 1) + i;
        col_index = col * (number_of_Chebyshev_points - 1) + i;
        C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] + 0.5*d_K_stack[3*i+0];
    }
    #pragma endregion

}

Eigen::VectorXd integrateQuaternions()
{       
    #pragma region BENCHMARK1
    ::benchmark::RegisterBenchmark("Quaternions computeCMatrixKernel:", [&](::benchmark::State &t_state){

    t_state.counters = {
        {"na", na},
        {"ne", ne},
        {"Cheb pts", number_of_Chebyshev_points}
    };

    // INITIALISATION
    #pragma region K_stack

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void**)&d_Phi_stack, size_of_Phi_stack_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_qe, size_of_qe_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_K_stack, size_of_K_stack_in_bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_Phi_stack, Phi_stack.data(), size_of_Phi_stack_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qe, qe.data(), size_of_qe_in_bytes, cudaMemcpyHostToDevice));

    // Define stride values
    int ld_Phi_stack = na; // Leading dimension of matrix A
    int inc_qe = 1; // Increment for x
    int inc_K_stack = 1; // Increment for y
    int stride_Phi_stack = na * (na * ne); // Stride between matrices in A
    int stride_qe = 0; // Stride between vectors in x
    int stride_K_stack = na; // Stride between vectors in y

    // Perform gemvStridedBatched operation
    double alpha_cublas = 1.0; // Scalar alpha
    double beta_cublas = 0.0; // Scalar beta
    CUBLAS_CHECK(cublasDgemvStridedBatched(cublasH, CUBLAS_OP_N, na, na*ne, &alpha_cublas, d_Phi_stack, ld_Phi_stack, stride_Phi_stack,
                                            d_qe, inc_qe, stride_qe, &beta_cublas, d_K_stack, inc_K_stack, stride_K_stack, number_of_Chebyshev_points));
    
    #pragma endregion

    // Vectors definitions
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN_F);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN_F);

    Eigen::MatrixXd C_NN = D_NN;

    Eigen::MatrixXd q_init(4,1);
    q_init << 1, 0, 0, 0;

    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(quaternion_problem_dimension,1);

    // Dimension definition
    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;  

    const int rows_q_init = q_init.rows();
    const int cols_q_init = q_init.cols();
    const int ld_q_init = rows_q_init;

    const int rows_b = b.rows();
    const int cols_b = b.cols();
    const int ld_b = rows_b;

    const int rows_res = b.rows();
    const int cols_res = b.cols();
    
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_Q_stack = rows_C_NN;
    const int cols_Q_stack = cols_b;

    Eigen::MatrixXd Q_stack_CUDA(rows_Q_stack, cols_Q_stack); //What we want to calculate

    // LU factorization variables
    int info = 0;
    int lwork = 0;
    
    // Compute the memory occupation 
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_q_init_in_bytes = size_of_double * q_init.size();
    const auto size_of_b_in_bytes = size_of_double * b.size();
    const auto size_of_res_in_bytes = size_of_double * rows_res * cols_res;
    const auto size_of_Q_stack_in_bytes = size_of_double * rows_Q_stack * cols_Q_stack;
    const auto size_of_D_NN_in_bytes = D_NN.size() * size_of_double;
    
    // Create Pointers for computeCMatrixKernel
    double* d_D_NN = nullptr;
    double* d_C_NN = nullptr;
    // Create Pointers
    double* d_D_IN = nullptr;
    double* d_q_init = nullptr;
    double* d_b = nullptr;
    double* d_res = nullptr;
    double* d_Q_stack = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;


    // Allocate the memory for computeCMatrixKernel
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_qe), size_of_qe_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_NN), size_of_D_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_D_NN_in_bytes)); // same size of D_NN
    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_q_init), size_of_q_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q_stack), size_of_Q_stack_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_Q_stack, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
    };
    // Allocate the memory for LU factorization workspace
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork));

    //  Copy the data for computeCMatrixKernel
    CUDA_CHECK(cudaMemcpy(d_D_NN, D_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice));
    //  Copy the data
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_init, q_init.data(), size_of_q_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), size_of_b_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));

    //  BENCHMARK
    for (auto _ : t_state) {
        auto start = std::chrono::high_resolution_clock::now();
        // TO BENCHMARK: START
        // Launch kernel with one block
        computeCMatrixKernel<<<1, number_of_Chebyshev_points-1>>>(d_K_stack, d_D_NN, d_C_NN);

        // TO BENCHMARK: END
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        t_state.SetIterationTime(elapsed_seconds.count());
    }

    // Computing b = -D_IN*q_init + b
    alpha_cublas = -1.0;
    beta_cublas = 1.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_q_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_q_init, ld_q_init, &beta_cublas, d_b, ld_b));

    // LU factorization
    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

    // Solving the final system
    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_b, ld_b, d_info));

    // Memory Copy
    CUDA_CHECK(cudaMemcpy(Q_stack_CUDA.data(), d_b, size_of_b_in_bytes, cudaMemcpyDeviceToHost));

    // FREEING MEMORY
    CUDA_CHECK(cudaFree(d_D_NN));
    CUDA_CHECK(cudaFree(d_D_IN));
    CUDA_CHECK(cudaFree(d_C_NN));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_q_init));
    CUDA_CHECK(cudaFree(d_Q_stack));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_work));

    })->Repetitions(20)->Unit(::benchmark::kMicrosecond);

    #pragma endregion

    #pragma region BENCHMARK2
    ::benchmark::RegisterBenchmark("Quaternions cublasDgemm:", [&](::benchmark::State &t_state){

    t_state.counters = {
        {"na", na},
        {"ne", ne},
        {"Cheb pts", number_of_Chebyshev_points}
    };

    // INITIALISATION
    #pragma region K_stack

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void**)&d_Phi_stack, size_of_Phi_stack_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_qe, size_of_qe_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_K_stack, size_of_K_stack_in_bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_Phi_stack, Phi_stack.data(), size_of_Phi_stack_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qe, qe.data(), size_of_qe_in_bytes, cudaMemcpyHostToDevice));

    // Define stride values
    int ld_Phi_stack = na; // Leading dimension of matrix A
    int inc_qe = 1; // Increment for x
    int inc_K_stack = 1; // Increment for y
    int stride_Phi_stack = na * (na * ne); // Stride between matrices in A
    int stride_qe = 0; // Stride between vectors in x
    int stride_K_stack = na; // Stride between vectors in y

    // Perform gemvStridedBatched operation
    double alpha_cublas = 1.0; // Scalar alpha
    double beta_cublas = 0.0; // Scalar beta
    CUBLAS_CHECK(cublasDgemvStridedBatched(cublasH, CUBLAS_OP_N, na, na*ne, &alpha_cublas, d_Phi_stack, ld_Phi_stack, stride_Phi_stack,
                                            d_qe, inc_qe, stride_qe, &beta_cublas, d_K_stack, inc_K_stack, stride_K_stack, number_of_Chebyshev_points));
    
    #pragma endregion

    // Vectors definitions
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN_F);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN_F);

    Eigen::MatrixXd C_NN = D_NN;

    Eigen::MatrixXd q_init(4,1);
    q_init << 1, 0, 0, 0;

    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(quaternion_problem_dimension,1);

    // Dimension definition
    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;  

    const int rows_q_init = q_init.rows();
    const int cols_q_init = q_init.cols();
    const int ld_q_init = rows_q_init;

    const int rows_b = b.rows();
    const int cols_b = b.cols();
    const int ld_b = rows_b;

    const int rows_res = b.rows();
    const int cols_res = b.cols();
    
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_Q_stack = rows_C_NN;
    const int cols_Q_stack = cols_b;

    Eigen::MatrixXd Q_stack_CUDA(rows_Q_stack, cols_Q_stack); //What we want to calculate

    // LU factorization variables
    int info = 0;
    int lwork = 0;
    
    // Compute the memory occupation 
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_q_init_in_bytes = size_of_double * q_init.size();
    const auto size_of_b_in_bytes = size_of_double * b.size();
    const auto size_of_res_in_bytes = size_of_double * rows_res * cols_res;
    const auto size_of_Q_stack_in_bytes = size_of_double * rows_Q_stack * cols_Q_stack;
    const auto size_of_D_NN_in_bytes = D_NN.size() * size_of_double;
    
    // Create Pointers for computeCMatrixKernel
    double* d_D_NN = nullptr;
    double* d_C_NN = nullptr;
    // Create Pointers
    double* d_D_IN = nullptr;
    double* d_q_init = nullptr;
    double* d_b = nullptr;
    double* d_res = nullptr;
    double* d_Q_stack = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;


    // Allocate the memory for computeCMatrixKernel
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_qe), size_of_qe_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_NN), size_of_D_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_D_NN_in_bytes)); // same size of D_NN
    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_q_init), size_of_q_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q_stack), size_of_Q_stack_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_Q_stack, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
    };
    // Allocate the memory for LU factorization workspace
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork));

    //  Copy the data for computeCMatrixKernel
    CUDA_CHECK(cudaMemcpy(d_D_NN, D_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice));
    //  Copy the data
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_init, q_init.data(), size_of_q_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), size_of_b_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel with one block        
    computeCMatrixKernel<<<1, number_of_Chebyshev_points-1>>>(d_K_stack, d_D_NN, d_C_NN);

    //  BENCHMARK
    for (auto _ : t_state) {
        auto start = std::chrono::high_resolution_clock::now();
        // TO BENCHMARK: START

        // Computing b = -D_IN*q_init + b
        alpha_cublas = -1.0;
        beta_cublas = 1.0;
        CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_q_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_q_init, ld_q_init, &beta_cublas, d_b, ld_b));

        // TO BENCHMARK: END
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        t_state.SetIterationTime(elapsed_seconds.count());
    }


    // LU factorization
    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

    // Solving the final system
    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_b, ld_b, d_info));

    // Memory Copy
    CUDA_CHECK(cudaMemcpy(Q_stack_CUDA.data(), d_b, size_of_b_in_bytes, cudaMemcpyDeviceToHost));

    // FREEING MEMORY
    CUDA_CHECK(cudaFree(d_D_NN));
    CUDA_CHECK(cudaFree(d_D_IN));
    CUDA_CHECK(cudaFree(d_C_NN));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_q_init));
    CUDA_CHECK(cudaFree(d_Q_stack));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_work));

    })->Repetitions(20)->Unit(::benchmark::kMicrosecond);

    #pragma endregion

    #pragma region BENCHMARK3
    ::benchmark::RegisterBenchmark("Quaternions cusolverDnDgetrf:", [&](::benchmark::State &t_state){

    t_state.counters = {
        {"na", na},
        {"ne", ne},
        {"Cheb pts", number_of_Chebyshev_points}
    };

    // INITIALISATION
    #pragma region K_stack

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void**)&d_Phi_stack, size_of_Phi_stack_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_qe, size_of_qe_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_K_stack, size_of_K_stack_in_bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_Phi_stack, Phi_stack.data(), size_of_Phi_stack_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qe, qe.data(), size_of_qe_in_bytes, cudaMemcpyHostToDevice));

    // Define stride values
    int ld_Phi_stack = na; // Leading dimension of matrix A
    int inc_qe = 1; // Increment for x
    int inc_K_stack = 1; // Increment for y
    int stride_Phi_stack = na * (na * ne); // Stride between matrices in A
    int stride_qe = 0; // Stride between vectors in x
    int stride_K_stack = na; // Stride between vectors in y

    // Perform gemvStridedBatched operation
    double alpha_cublas = 1.0; // Scalar alpha
    double beta_cublas = 0.0; // Scalar beta
    CUBLAS_CHECK(cublasDgemvStridedBatched(cublasH, CUBLAS_OP_N, na, na*ne, &alpha_cublas, d_Phi_stack, ld_Phi_stack, stride_Phi_stack,
                                            d_qe, inc_qe, stride_qe, &beta_cublas, d_K_stack, inc_K_stack, stride_K_stack, number_of_Chebyshev_points));
    
    #pragma endregion

    // Vectors definitions
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN_F);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN_F);

    Eigen::MatrixXd C_NN = D_NN;

    Eigen::MatrixXd q_init(4,1);
    q_init << 1, 0, 0, 0;

    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(quaternion_problem_dimension,1);

    // Dimension definition
    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;  

    const int rows_q_init = q_init.rows();
    const int cols_q_init = q_init.cols();
    const int ld_q_init = rows_q_init;

    const int rows_b = b.rows();
    const int cols_b = b.cols();
    const int ld_b = rows_b;

    const int rows_res = b.rows();
    const int cols_res = b.cols();
    
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_Q_stack = rows_C_NN;
    const int cols_Q_stack = cols_b;

    Eigen::MatrixXd Q_stack_CUDA(rows_Q_stack, cols_Q_stack); //What we want to calculate

    // LU factorization variables
    int info = 0;
    int lwork = 0;
    
    // Compute the memory occupation 
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_q_init_in_bytes = size_of_double * q_init.size();
    const auto size_of_b_in_bytes = size_of_double * b.size();
    const auto size_of_res_in_bytes = size_of_double * rows_res * cols_res;
    const auto size_of_Q_stack_in_bytes = size_of_double * rows_Q_stack * cols_Q_stack;
    const auto size_of_D_NN_in_bytes = D_NN.size() * size_of_double;
    
    // Create Pointers for computeCMatrixKernel
    double* d_D_NN = nullptr;
    double* d_C_NN = nullptr;
    // Create Pointers
    double* d_D_IN = nullptr;
    double* d_q_init = nullptr;
    double* d_b = nullptr;
    double* d_res = nullptr;
    double* d_Q_stack = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;


    // Allocate the memory for computeCMatrixKernel
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_qe), size_of_qe_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_NN), size_of_D_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_D_NN_in_bytes)); // same size of D_NN
    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_q_init), size_of_q_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q_stack), size_of_Q_stack_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_Q_stack, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
    };
    // Allocate the memory for LU factorization workspace
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork));

    //  Copy the data for computeCMatrixKernel
    CUDA_CHECK(cudaMemcpy(d_D_NN, D_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice));
    //  Copy the data
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_init, q_init.data(), size_of_q_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), size_of_b_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel with one block        
    computeCMatrixKernel<<<1, number_of_Chebyshev_points-1>>>(d_K_stack, d_D_NN, d_C_NN);

    // Computing b = -D_IN*q_init + b
    alpha_cublas = -1.0;
    beta_cublas = 1.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_q_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_q_init, ld_q_init, &beta_cublas, d_b, ld_b));

    //  BENCHMARK
    for (auto _ : t_state) {
        auto start = std::chrono::high_resolution_clock::now();
        // TO BENCHMARK: START

        // LU factorization
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

        // TO BENCHMARK: END
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        t_state.SetIterationTime(elapsed_seconds.count());
    }

    // Solving the final system
    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_b, ld_b, d_info));

    // Memory Copy
    CUDA_CHECK(cudaMemcpy(Q_stack_CUDA.data(), d_b, size_of_b_in_bytes, cudaMemcpyDeviceToHost));

    // FREEING MEMORY
    CUDA_CHECK(cudaFree(d_D_NN));
    CUDA_CHECK(cudaFree(d_D_IN));
    CUDA_CHECK(cudaFree(d_C_NN));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_q_init));
    CUDA_CHECK(cudaFree(d_Q_stack));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_work));

    })->Repetitions(20)->Unit(::benchmark::kMicrosecond);

    #pragma endregion

    #pragma region BENCHMARK4
    ::benchmark::RegisterBenchmark("Quaternions cusolverDnDgetrs:", [&](::benchmark::State &t_state){

    t_state.counters = {
        {"na", na},
        {"ne", ne},
        {"Cheb pts", number_of_Chebyshev_points}
    };

    // INITIALISATION
    #pragma region K_stack

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void**)&d_Phi_stack, size_of_Phi_stack_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_qe, size_of_qe_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_K_stack, size_of_K_stack_in_bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_Phi_stack, Phi_stack.data(), size_of_Phi_stack_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qe, qe.data(), size_of_qe_in_bytes, cudaMemcpyHostToDevice));

    // Define stride values
    int ld_Phi_stack = na; // Leading dimension of matrix A
    int inc_qe = 1; // Increment for x
    int inc_K_stack = 1; // Increment for y
    int stride_Phi_stack = na * (na * ne); // Stride between matrices in A
    int stride_qe = 0; // Stride between vectors in x
    int stride_K_stack = na; // Stride between vectors in y

    // Perform gemvStridedBatched operation
    double alpha_cublas = 1.0; // Scalar alpha
    double beta_cublas = 0.0; // Scalar beta
    CUBLAS_CHECK(cublasDgemvStridedBatched(cublasH, CUBLAS_OP_N, na, na*ne, &alpha_cublas, d_Phi_stack, ld_Phi_stack, stride_Phi_stack,
                                            d_qe, inc_qe, stride_qe, &beta_cublas, d_K_stack, inc_K_stack, stride_K_stack, number_of_Chebyshev_points));
    
    #pragma endregion

    // Vectors definitions
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN_F);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN_F);

    Eigen::MatrixXd C_NN = D_NN;

    Eigen::MatrixXd q_init(4,1);
    q_init << 1, 0, 0, 0;

    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(quaternion_problem_dimension,1);

    // Dimension definition
    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;  

    const int rows_q_init = q_init.rows();
    const int cols_q_init = q_init.cols();
    const int ld_q_init = rows_q_init;

    const int rows_b = b.rows();
    const int cols_b = b.cols();
    const int ld_b = rows_b;

    const int rows_res = b.rows();
    const int cols_res = b.cols();
    
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_Q_stack = rows_C_NN;
    const int cols_Q_stack = cols_b;

    Eigen::MatrixXd Q_stack_CUDA(rows_Q_stack, cols_Q_stack); //What we want to calculate

    // LU factorization variables
    int info = 0;
    int lwork = 0;
    
    // Compute the memory occupation 
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_q_init_in_bytes = size_of_double * q_init.size();
    const auto size_of_b_in_bytes = size_of_double * b.size();
    const auto size_of_res_in_bytes = size_of_double * rows_res * cols_res;
    const auto size_of_Q_stack_in_bytes = size_of_double * rows_Q_stack * cols_Q_stack;
    const auto size_of_D_NN_in_bytes = D_NN.size() * size_of_double;
    
    // Create Pointers for computeCMatrixKernel
    double* d_D_NN = nullptr;
    double* d_C_NN = nullptr;
    // Create Pointers
    double* d_D_IN = nullptr;
    double* d_q_init = nullptr;
    double* d_b = nullptr;
    double* d_res = nullptr;
    double* d_Q_stack = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;


    // Allocate the memory for computeCMatrixKernel
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_qe), size_of_qe_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_NN), size_of_D_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_D_NN_in_bytes)); // same size of D_NN
    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_q_init), size_of_q_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q_stack), size_of_Q_stack_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_Q_stack, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
    };
    // Allocate the memory for LU factorization workspace
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork));

    //  Copy the data for computeCMatrixKernel
    CUDA_CHECK(cudaMemcpy(d_D_NN, D_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice));
    //  Copy the data
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_init, q_init.data(), size_of_q_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), size_of_b_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel with one block        
    computeCMatrixKernel<<<1, number_of_Chebyshev_points-1>>>(d_K_stack, d_D_NN, d_C_NN);

    // Computing b = -D_IN*q_init + b
    alpha_cublas = -1.0;
    beta_cublas = 1.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_q_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_q_init, ld_q_init, &beta_cublas, d_b, ld_b));

    // LU factorization
    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

    //  BENCHMARK
    for (auto _ : t_state) {
        auto start = std::chrono::high_resolution_clock::now();
        // TO BENCHMARK: START

        // Solving the final system
        CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_b, ld_b, d_info));

        // TO BENCHMARK: END
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        t_state.SetIterationTime(elapsed_seconds.count());
    }

    // Memory Copy
    CUDA_CHECK(cudaMemcpy(Q_stack_CUDA.data(), d_b, size_of_b_in_bytes, cudaMemcpyDeviceToHost));

    // FREEING MEMORY
    CUDA_CHECK(cudaFree(d_D_NN));
    CUDA_CHECK(cudaFree(d_D_IN));
    CUDA_CHECK(cudaFree(d_C_NN));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_q_init));
    CUDA_CHECK(cudaFree(d_Q_stack));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_work));

    })->Repetitions(20)->Unit(::benchmark::kMicrosecond);

    #pragma endregion

    #pragma region OUTBENCH
    #pragma region K_stack

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void**)&d_Phi_stack, size_of_Phi_stack_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_qe, size_of_qe_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_K_stack, size_of_K_stack_in_bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_Phi_stack, Phi_stack.data(), size_of_Phi_stack_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qe, qe.data(), size_of_qe_in_bytes, cudaMemcpyHostToDevice));

    // Define stride values
    int ld_Phi_stack = na; // Leading dimension of matrix A
    int inc_qe = 1; // Increment for x
    int inc_K_stack = 1; // Increment for y
    int stride_Phi_stack = na * (na * ne); // Stride between matrices in A
    int stride_qe = 0; // Stride between vectors in x
    int stride_K_stack = na; // Stride between vectors in y

    // Perform gemvStridedBatched operation
    double alpha_cublas = 1.0; // Scalar alpha
    double beta_cublas = 0.0; // Scalar beta
    CUBLAS_CHECK(cublasDgemvStridedBatched(cublasH, CUBLAS_OP_N, na, na*ne, &alpha_cublas, d_Phi_stack, ld_Phi_stack, stride_Phi_stack,
                                            d_qe, inc_qe, stride_qe, &beta_cublas, d_K_stack, inc_K_stack, stride_K_stack, number_of_Chebyshev_points));
    
    #pragma endregion

    // Vectors definitions
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN_F);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN_F);

    Eigen::MatrixXd C_NN = D_NN;

    Eigen::MatrixXd q_init(4,1);
    q_init << 1, 0, 0, 0;

    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(quaternion_problem_dimension,1);

    // Dimension definition
    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;  

    const int rows_q_init = q_init.rows();
    const int cols_q_init = q_init.cols();
    const int ld_q_init = rows_q_init;

    const int rows_b = b.rows();
    const int cols_b = b.cols();
    const int ld_b = rows_b;

    const int rows_res = b.rows();
    const int cols_res = b.cols();
    
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_Q_stack = rows_C_NN;
    const int cols_Q_stack = cols_b;

    Eigen::MatrixXd Q_stack_CUDA(rows_Q_stack, cols_Q_stack); //What we want to calculate

    // LU factorization variables
    int info = 0;
    int lwork = 0;
    
    // Compute the memory occupation 
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_q_init_in_bytes = size_of_double * q_init.size();
    const auto size_of_b_in_bytes = size_of_double * b.size();
    const auto size_of_res_in_bytes = size_of_double * rows_res * cols_res;
    const auto size_of_Q_stack_in_bytes = size_of_double * rows_Q_stack * cols_Q_stack;
    const auto size_of_D_NN_in_bytes = D_NN.size() * size_of_double;
    
    // Create Pointers for computeCMatrixKernel
    double* d_D_NN = nullptr;
    double* d_C_NN = nullptr;
    // Create Pointers
    double* d_D_IN = nullptr;
    double* d_q_init = nullptr;
    double* d_b = nullptr;
    double* d_res = nullptr;
    double* d_Q_stack = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;


    // Allocate the memory for computeCMatrixKernel
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_qe), size_of_qe_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_NN), size_of_D_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_D_NN_in_bytes)); // same size of D_NN
    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_q_init), size_of_q_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q_stack), size_of_Q_stack_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_Q_stack, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
    };
    // Allocate the memory for LU factorization workspace
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork));

    //  Copy the data for computeCMatrixKernel
    CUDA_CHECK(cudaMemcpy(d_D_NN, D_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice));
    //  Copy the data
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_init, q_init.data(), size_of_q_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), size_of_b_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));

    computeCMatrixKernel<<<1, number_of_Chebyshev_points-1>>>(d_K_stack, d_D_NN, d_C_NN);

    // Computing b = -D_IN*q_init + b
    alpha_cublas = -1.0;
    beta_cublas = 1.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_q_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_q_init, ld_q_init, &beta_cublas, d_b, ld_b));

    // LU factorization
    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

    // Solving the final system
    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_b, ld_b, d_info));

    // Memory Copy
    CUDA_CHECK(cudaMemcpy(Q_stack_CUDA.data(), d_b, size_of_b_in_bytes, cudaMemcpyDeviceToHost));

    // FREEING MEMORY
    CUDA_CHECK(cudaFree(d_D_NN));
    CUDA_CHECK(cudaFree(d_D_IN));
    CUDA_CHECK(cudaFree(d_C_NN));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_q_init));
    CUDA_CHECK(cudaFree(d_Q_stack));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_work));

    return Q_stack_CUDA;

    #pragma endregion

}




int main(int argc, char *argv[])
{
    //  cuda blas api need CUBLAS_CHECK
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    //  Here we give some value for the strain
    qe <<   0,
            0,
            0,
            1.2877691307032,
           -1.63807499160786,
            0.437406679142598,
            0,
            0,
            0;
    // qe.setZero();
    

    // Here we give some value for the strain rate
    for (unsigned int i = 0; i < number_of_Chebyshev_points; ++i) {
        Phi_stack.block<na, ne>(i*na, 0) =  Phi<na, ne>(Chebyshev_points[i]);
    }

    const auto Q_stack_CUDA = integrateQuaternions();
    // std::cout << "Quaternion Integration : \n" << Q_stack_CUDA << "\n" << std::endl;
    
    // const auto r_stack_CUDA = integratePosition(Q_stack_CUDA);
    // // std::cout << "Position Integration : \n" << r_stack_CUDA << "\n" << std::endl;

    // const auto N_stack_CUDA = integrateInternalForces(Q_stack_CUDA);
    // // std::cout << "Internal Forces Integration : \n" << N_stack_CUDA << "\n" << std::endl;

    // const auto C_stack_CUDA = integrateInternalCouples(N_stack_CUDA);
    // // std::cout << "Internal Couples Integration : \n" << C_stack_CUDA << "\n" << std::endl;
    
    // const auto Lambda_stack_CUDA = buildLambda(C_stack_CUDA, N_stack_CUDA);
    // //std::cout << "Lambda_stack : \n" << Lambda_stack_CUDA << "\n" << std::endl;

    // const auto Qa_stack_CUDA = integrateGeneralisedForces(Lambda_stack_CUDA);
    // // std::cout << "Generalized Forces Integration : \n" << Qa_stack_CUDA << std::endl;

    // Benchmark initialization6
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

    // Destroy the handle
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}