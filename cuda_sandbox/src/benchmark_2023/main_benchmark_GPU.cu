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

// Used to build Q_stack
Eigen::MatrixXd computeCMatrix(const Eigen::VectorXd &t_qe, const Eigen::MatrixXd &D_NN)
{
    Eigen::MatrixXd C_NN = D_NN;
    //  Define the Chebyshev points on the unit circle
    const auto Chebyshev_points = ComputeChebyshevPoints<number_of_Chebyshev_points>();

    Eigen::Vector3d K;
    Eigen::MatrixXd Z_at_chebyshev_point(quaternion_state_dimension, quaternion_state_dimension);
    Eigen::MatrixXd A_at_chebyshev_point(quaternion_state_dimension, quaternion_state_dimension);

    for(unsigned int i=0; i<Chebyshev_points.size()-1; i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(Chebyshev_points[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        Z_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                  K(0),     0,   K(2),  -K(1),
                                  K(1), -K(2),      0,   K(0),
                                  K(2),  K(1),  -K(0),      0;

        A_at_chebyshev_point = 0.5*Z_at_chebyshev_point;


        for (unsigned int row = 0; row < quaternion_state_dimension; ++row) {
            for (unsigned int col = 0; col < quaternion_state_dimension; ++col) {
                int row_index = row*(number_of_Chebyshev_points-1)+i;
                int col_index = col*(number_of_Chebyshev_points-1)+i;
                C_NN(row_index, col_index) = D_NN(row_index, col_index) - A_at_chebyshev_point(row, col);
            }
        }

    }

    return C_NN;
}

Eigen::VectorXd integrateQuaternions()
{   
    // ==================================== BENCHMARK ==================================== 

    ::benchmark::RegisterBenchmark("Integrate Quaternions:", [&](::benchmark::State &t_state){

    t_state.counters = {
        {"na", na},
        {"ne", ne},
        {"Cheb pts", number_of_Chebyshev_points}
    };

    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //FORWARD INTEGRATION:
    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN_F = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);
    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN_F = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);


    // INITIALISATION
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN_F);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN_F);

    Eigen::MatrixXd C_NN = Eigen::MatrixXd::Zero(quaternion_problem_dimension, quaternion_problem_dimension);

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

    // LU factorization variables
    int info = 0;
    int lwork = 0;

    // Create Pointers
    double* d_D_IN = nullptr;
    double* d_q_init = nullptr;
    double* d_b = nullptr;
    double* d_res = nullptr;
    double* d_Q_stack = nullptr;
    double* d_C_NN = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    // Compute the memory occupation 
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_q_init_in_bytes = size_of_double * q_init.size();
    const auto size_of_b_in_bytes = size_of_double * b.size();
    const auto size_of_res_in_bytes = size_of_double * rows_res * cols_res;
    const auto size_of_Q_stack_in_bytes = size_of_double * rows_Q_stack * cols_Q_stack;
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_q_init), size_of_q_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q_stack), size_of_Q_stack_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_init, q_init.data(), size_of_q_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), size_of_b_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_Q_stack, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
    };
    // Allocate the memory for LU factorization workspace
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork));

    //What we want to calculate
    Eigen::MatrixXd Q_stack_CUDA(rows_Q_stack, cols_Q_stack);

    // Computing b = -D_IN*q_init + b
    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;

    //  BENCHMARK
        for (auto _ : t_state) {
            auto start = std::chrono::high_resolution_clock::now();
            //INSERT BENCHMARK CODE HERE:
            C_NN =  computeCMatrix(qe, D_NN);
            
            CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice));

            CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_q_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_q_init, ld_q_init, &beta_cublas, d_b, ld_b));

            CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

            CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_b, ld_b, d_info));

            //END OF BENCHMARK
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

            t_state.SetIterationTime(elapsed_seconds.count());
        }

        //FREEING MEMORY
        CUDA_CHECK(cudaFree(d_D_IN));
        CUDA_CHECK(cudaFree(d_C_NN));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_q_init));
        CUDA_CHECK(cudaFree(d_Q_stack));
        CUDA_CHECK(cudaFree(d_res));
        CUDA_CHECK(cudaFree(d_work));

    })->Repetitions(20)->Unit(::benchmark::kMicrosecond);

    // ==================================== STANDARD ====================================
    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //FORWARD INTEGRATION:
    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN_F = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);
    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN_F = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);

    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN_F);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN_F);

    Eigen::MatrixXd C_NN =  computeCMatrix(qe, D_NN);

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

    // LU factorization variables
    int info = 0;
    int lwork = 0;

    // Create Pointers
    double* d_D_IN = nullptr;
    double* d_q_init = nullptr;
    double* d_b = nullptr;
    double* d_res = nullptr;
    double* d_Q_stack = nullptr;
    double* d_C_NN = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    // Compute the memory occupation 
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_q_init_in_bytes = size_of_double * q_init.size();
    const auto size_of_b_in_bytes = size_of_double * b.size();
    const auto size_of_res_in_bytes = size_of_double * rows_res * cols_res;
    const auto size_of_Q_stack_in_bytes = size_of_double * rows_Q_stack * cols_Q_stack;
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_q_init), size_of_q_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q_stack), size_of_Q_stack_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_init, q_init.data(), size_of_q_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), size_of_b_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_Q_stack, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
    };
    // Allocate the memory for LU factorization workspace
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork)
    );

    //What we want to calculate
    Eigen::MatrixXd Q_stack_CUDA(rows_Q_stack, cols_Q_stack);

    // Computing b = -D_IN*q_init + b
    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;

    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_q_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_q_init, ld_q_init, &beta_cublas, d_b, ld_b));

    // LU factorization
    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

    // Solving the final system
    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_b, ld_b, d_info));

    // Memory Copy
    CUDA_CHECK(cudaMemcpy(Q_stack_CUDA.data(), d_b, size_of_b_in_bytes, cudaMemcpyDeviceToHost));

    //FREEING MEMORY
    CUDA_CHECK(cudaFree(d_D_IN));
    CUDA_CHECK(cudaFree(d_C_NN));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_q_init));
    CUDA_CHECK(cudaFree(d_Q_stack));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_work));


    return Q_stack_CUDA;
}



// Used to build r_stack
Eigen::MatrixXd updatePositionb(Eigen::MatrixXd t_Q_stack_CUDA) { 

    Eigen::Matrix<double, number_of_Chebyshev_points-1, position_dimension> b;

    Eigen::Quaterniond q;

    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {

        q = { t_Q_stack_CUDA(i),
              t_Q_stack_CUDA(i  +  (number_of_Chebyshev_points-1)),
              t_Q_stack_CUDA(i + 2*(number_of_Chebyshev_points-1)),
              t_Q_stack_CUDA(i + 3*(number_of_Chebyshev_points-1)) };

        b.block<1,3>(i, 0) = (q.toRotationMatrix()*Eigen::Vector3d(1, 0, 0)).transpose();

    }
    return b;
}

Eigen::MatrixXd integratePosition(Eigen::MatrixXd t_Q_stack_CUDA)
{   
    // ==================================== BENCHMARK ==================================== 

    ::benchmark::RegisterBenchmark("Integrate Position:", [&](::benchmark::State &t_state){
    
    t_state.counters = {
        {"na", na},
        {"ne", ne},
        {"Cheb pts", number_of_Chebyshev_points}
    };

    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //FORWARD INTEGRATION:
    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN_F = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);
    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN_F = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);


    // INITIALISATION
    Eigen::Vector3d r_init;
    r_init << 0,
              0,
              0;

    //  This matrix remains constant so we can pre invert
    Eigen::MatrixXd Dn_NN_inv = Dn_NN_F.inverse();

    Eigen::MatrixXd ivp(number_of_Chebyshev_points-1, position_dimension);
    for(unsigned int i=0; i<ivp.rows(); i++)
        ivp.row(i) = Dn_IN_F(i, 0) * r_init.transpose();

    Eigen::MatrixXd b_NN = Eigen::MatrixXd::Zero(number_of_Chebyshev_points-1, position_dimension);

    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(number_of_Chebyshev_points-1, position_dimension);

    // Define dimensions
    const int rows_Dn_NN_inv = Dn_NN_inv.rows();
    const int cols_Dn_NN_inv = Dn_NN_inv.cols();
    const int ld_Dn_NN_inv = rows_Dn_NN_inv;  

    const int rows_res = res.rows();
    const int cols_res = res.cols();
    const int ld_res = rows_res;

    const int rows_r_stack = rows_Dn_NN_inv;
    const int cols_r_stack = cols_res;
    const int ld_r_stack = rows_r_stack;

    // Create Pointers
    double* d_Dn_NN_inv = nullptr;
    double* d_res = nullptr;
    double* d_r_stack = nullptr;

    // Compute the memory occupation
    const auto size_of_Dn_NN_inv_in_bytes = size_of_double * Dn_NN_inv.size();
    const auto size_of_res_in_bytes = size_of_double * res.size();
    const auto size_of_r_stack_in_bytes = size_of_double * rows_r_stack * cols_r_stack;

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Dn_NN_inv), size_of_Dn_NN_inv_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_r_stack), size_of_r_stack_in_bytes));
    
    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_Dn_NN_inv, Dn_NN_inv.data(), size_of_Dn_NN_inv_in_bytes, cudaMemcpyHostToDevice));

    // What we want to calculate 
    Eigen::MatrixXd r_stack_CUDA(rows_r_stack, cols_r_stack);

    // Compute r_stack = Dn_NN_inv*res
    double alpha_cublas = 1.0;
    double beta_cublas = 0.0;
    //  BENCHMARK
        for (auto _ : t_state) {
            auto start = std::chrono::high_resolution_clock::now();
            //INSERT BENCHMARK CODE HERE:
            b_NN = updatePositionb(t_Q_stack_CUDA);
            
            res = b_NN - ivp;

            CUDA_CHECK(cudaMemcpy(d_res, res.data(), size_of_res_in_bytes, cudaMemcpyHostToDevice));

            CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_Dn_NN_inv, cols_res, cols_Dn_NN_inv, &alpha_cublas, d_Dn_NN_inv, ld_Dn_NN_inv, d_res, ld_res, &beta_cublas, d_r_stack, ld_r_stack));
            //END OF BENCHMARK
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

            t_state.SetIterationTime(elapsed_seconds.count());
        }

        //FREEING MEMORY
        CUDA_CHECK(cudaFree(d_Dn_NN_inv));
        CUDA_CHECK(cudaFree(d_r_stack));
        CUDA_CHECK(cudaFree(d_res));
    })->Repetitions(20)->Unit(::benchmark::kMicrosecond);

    // ==================================== STANDARD ====================================

    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //FORWARD INTEGRATION:
    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN_F = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);
    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN_F = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);


    Eigen::Vector3d r_init;
    r_init << 0,
              0,
              0;

    //  This matrix remains constant so we can pre invert
    Eigen::MatrixXd Dn_NN_inv = Dn_NN_F.inverse();

    Eigen::MatrixXd ivp(number_of_Chebyshev_points-1, position_dimension);
    for(unsigned int i=0; i<ivp.rows(); i++)
        ivp.row(i) = Dn_IN_F(i, 0) * r_init.transpose();
    
    Eigen::MatrixXd b_NN = Eigen::MatrixXd::Zero(number_of_Chebyshev_points-1, position_dimension);
    b_NN = updatePositionb(t_Q_stack_CUDA);

    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(number_of_Chebyshev_points-1, position_dimension);
    res = b_NN - ivp;

    // Define dimensions
    const int rows_Dn_NN_inv = Dn_NN_inv.rows();
    const int cols_Dn_NN_inv = Dn_NN_inv.cols();
    const int ld_Dn_NN_inv = rows_Dn_NN_inv;  

    const int rows_res = res.rows();
    const int cols_res = res.cols();
    const int ld_res = rows_res;

    const int rows_r_stack = rows_Dn_NN_inv;
    const int cols_r_stack = cols_res;
    const int ld_r_stack = rows_r_stack;

    // Create Pointers
    double* d_Dn_NN_inv = nullptr;
    double* d_res = nullptr;
    double* d_r_stack = nullptr;

    // Compute the memory occupation
    const auto size_of_Dn_NN_inv_in_bytes = size_of_double * Dn_NN_inv.size();
    const auto size_of_res_in_bytes = size_of_double * res.size();
    const auto size_of_r_stack_in_bytes = size_of_double * rows_r_stack * cols_r_stack;

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Dn_NN_inv), size_of_Dn_NN_inv_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_r_stack), size_of_r_stack_in_bytes));
    
    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_Dn_NN_inv, Dn_NN_inv.data(), size_of_Dn_NN_inv_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res, res.data(), size_of_res_in_bytes, cudaMemcpyHostToDevice));

    // What we want to calculate 
    Eigen::MatrixXd r_stack_CUDA(rows_r_stack, cols_r_stack);

    // Compute r_stack = Dn_NN_inv*res
    double alpha_cublas = 1.0;
    double beta_cublas = 0.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_Dn_NN_inv, cols_res, cols_Dn_NN_inv, &alpha_cublas, d_Dn_NN_inv, ld_Dn_NN_inv, d_res, ld_res, &beta_cublas, d_r_stack, ld_r_stack));

    CUDA_CHECK(cudaMemcpy(r_stack_CUDA.data(), d_r_stack, size_of_r_stack_in_bytes, cudaMemcpyDeviceToHost));

    //FREEING MEMORY
    CUDA_CHECK(cudaFree(d_Dn_NN_inv));
    CUDA_CHECK(cudaFree(d_r_stack));
    CUDA_CHECK(cudaFree(d_res));

    return r_stack_CUDA;
}



// Used to build Lambda_stack:
Eigen::MatrixXd updateCMatrix(const Eigen::VectorXd &t_qe, const Eigen::MatrixXd &D_NN)
{
    Eigen::MatrixXd C_NN = D_NN;

    //  Define the Chebyshev points on the unit circle
    const auto Chebyshev_points = ComputeChebyshevPoints<number_of_Chebyshev_points>();

    Eigen::Vector3d K;
    Eigen::MatrixXd A_at_chebyshev_point(lambda_dimension/2, lambda_dimension/2);

    for(unsigned int i=0; i<Chebyshev_points.size()-1; i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(Chebyshev_points[i])*t_qe;

        //  Build Skew Symmetric K matrix (K_hat)
        Eigen::Matrix3d K_hat = skew(K);
        A_at_chebyshev_point = K_hat.transpose();

        for (unsigned int row = 0; row < lambda_dimension/2; ++row) {
            for (unsigned int col = 0; col < lambda_dimension/2; ++col) {
                int row_index = row*(number_of_Chebyshev_points-1)+i;
                int col_index = col*(number_of_Chebyshev_points-1)+i;
                C_NN(row_index, col_index) = D_NN(row_index, col_index) - A_at_chebyshev_point(row, col);
            }
        }

    }

    return C_NN;

}

Eigen::VectorXd computeNbar (Eigen::MatrixXd t_Q_stack_CUDA) 
{
    // Variables definition to include gravity (Nbar)
    const double g = 9.81; // m/s^2
    const double radius = 0.001; // m
    const double A = M_PI*radius*radius;
    const double rho = 7800; // kg/m^3

    Eigen::VectorXd Fg(lambda_dimension/2);
    Fg << 0, 0, -A*g*rho;
    
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    Eigen::VectorXd Nbar = Eigen::VectorXd::Zero(lambda_dimension/2);
    Eigen::VectorXd Nbar_stack = Eigen::VectorXd::Zero((lambda_dimension/2)*(number_of_Chebyshev_points-1));

    // to fix
    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {

        Eigen::Quaterniond Qbar(t_Q_stack_CUDA(i),
              t_Q_stack_CUDA(i  +  (number_of_Chebyshev_points-1)),
              t_Q_stack_CUDA(i + 2*(number_of_Chebyshev_points-1)),
              t_Q_stack_CUDA(i + 3*(number_of_Chebyshev_points-1)));

        
        R = Qbar.toRotationMatrix();
        Nbar = R.transpose()*Fg;

        Nbar_stack(i) = Nbar.x();
        Nbar_stack(i  +  (number_of_Chebyshev_points-1)) = Nbar.y();
        Nbar_stack(i + 2*(number_of_Chebyshev_points-1)) = Nbar.z();

    }

    return Nbar_stack;

}

Eigen::MatrixXd integrateInternalForces(Eigen::MatrixXd t_Q_stack_CUDA)
{   
    // ==================================== BENCHMARK ==================================== 
    ::benchmark::RegisterBenchmark("Integrate Internal Forces:", [&](::benchmark::State &t_state){
    
    t_state.counters = {
        {"na", na},
        {"ne", ne},
        {"Cheb pts", number_of_Chebyshev_points}
    };

    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();
    //BACKWARD INTEGRATION:
    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN_B = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);
    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN_B = Dn.block<number_of_Chebyshev_points-1, 1>(1, 0);


    // INITIALISATION
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_NN_B);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_IN_B);

    Eigen::MatrixXd C_NN = Eigen::MatrixXd::Zero((number_of_Chebyshev_points-1)*(lambda_dimension/2), (number_of_Chebyshev_points-1)*(lambda_dimension/2));

    Eigen::VectorXd N_init(lambda_dimension/2);
    N_init << 1, 0, 0;

    Eigen::MatrixXd beta = Eigen::MatrixXd::Zero((number_of_Chebyshev_points-1)*(lambda_dimension/2), 1);

    //Definition of matrices dimensions.
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;

    const int rows_N_init = N_init.rows();
    const int cols_N_init = N_init.cols();
    const int ld_N_init = rows_N_init;

    const int rows_beta = beta.rows();
    const int cols_beta = beta.cols();
    const int ld_beta = rows_beta;

    const int rows_N_stack = rows_beta;
    const int cols_N_stack = cols_beta;
    
    int info = 0;
    int lwork = 0;

    // Create Pointers
    double* d_C_NN = nullptr;
    double* d_D_IN = nullptr;
    double* d_N_init = nullptr;
    double* d_beta = nullptr;
    double* d_N_stack = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    // Compute the memory occupation
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_N_init_in_bytes = size_of_double * N_init.size();
    const auto size_of_beta_in_bytes = size_of_double * beta.size();
    const auto size_of_N_stack_in_bytes = size_of_double * rows_N_stack * cols_N_stack;

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_N_init), size_of_N_init_in_bytes)
);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_beta), size_of_beta_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_N_stack), size_of_N_stack_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N_init, N_init.data(), size_of_N_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, cols_C_NN, cols_C_NN, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
        // Handle or debug the error appropriately
    };

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork));


    //What we want to calculate
    Eigen::MatrixXd N_stack_CUDA(rows_N_stack, cols_N_stack);

    // res = -D_IN*N_init + beta
    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;    
    //  BENCHMARK
        for (auto _ : t_state) {
            auto start = std::chrono::high_resolution_clock::now();
            //INSERT BENCHMARK CODE HERE:
            C_NN =  updateCMatrix(qe, D_NN);

            beta = -computeNbar(t_Q_stack_CUDA);

            CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), size_of_beta_in_bytes, cudaMemcpyHostToDevice));

            CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_N_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_N_init, ld_N_init, &beta_cublas, d_beta, ld_beta));

            // LU factorization
            CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, cols_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));
            // Solving the final system
            CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, cols_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_beta, ld_beta, d_info));
            //END OF BENCHMARK
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

            t_state.SetIterationTime(elapsed_seconds.count());
        }

        //FREEING MEMORY
        CUDA_CHECK(cudaFree(d_beta));
        CUDA_CHECK(cudaFree(d_C_NN));
        CUDA_CHECK(cudaFree(d_D_IN));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_N_init));
        CUDA_CHECK(cudaFree(d_N_stack));
        CUDA_CHECK(cudaFree(d_work));

    })->Repetitions(20)->Unit(::benchmark::kMicrosecond);
    
    // ==================================== STANDARD ====================================

        //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();
    //BACKWARD INTEGRATION:
    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN_B = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);
    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN_B = Dn.block<number_of_Chebyshev_points-1, 1>(1, 0);

    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_NN_B);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_IN_B);

    Eigen::MatrixXd C_NN =  updateCMatrix(qe, D_NN);

    Eigen::VectorXd N_init(lambda_dimension/2);
    N_init << 1, 0, 0;

    Eigen::MatrixXd beta = -computeNbar(t_Q_stack_CUDA);

    //Definition of matrices dimensions.
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;

    const int rows_N_init = N_init.rows();
    const int cols_N_init = N_init.cols();
    const int ld_N_init = rows_N_init;

    const int rows_beta = beta.rows();
    const int cols_beta = beta.cols();
    const int ld_beta = rows_beta;

    const int rows_N_stack = rows_beta;
    const int cols_N_stack = cols_beta;
    
    int info = 0;
    int lwork = 0;

    // Create Pointers
    double* d_C_NN = nullptr;
    double* d_D_IN = nullptr;
    double* d_N_init = nullptr;
    double* d_beta = nullptr;
    double* d_N_stack = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    // Compute the memory occupation
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_N_init_in_bytes = size_of_double * N_init.size();
    const auto size_of_beta_in_bytes = size_of_double * beta.size();
    const auto size_of_N_stack_in_bytes = size_of_double * rows_N_stack * cols_N_stack;

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_N_init), size_of_N_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_beta), size_of_beta_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_N_stack), size_of_N_stack_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N_init, N_init.data(), size_of_N_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), size_of_beta_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, cols_C_NN, cols_C_NN, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
        // Handle or debug the error appropriately
    };

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork));


    //What we want to calculate
    Eigen::MatrixXd N_stack_CUDA(rows_N_stack, cols_N_stack);

    // res = -D_IN*N_init + beta
    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_N_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_N_init, ld_N_init, &beta_cublas, d_beta, ld_beta));

    // LU factorization
    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, cols_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

    // Solving the final system
    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, cols_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_beta, ld_beta, d_info));

    CUDA_CHECK(cudaMemcpy(N_stack_CUDA.data(), d_beta, size_of_beta_in_bytes, cudaMemcpyDeviceToHost));

    //FREEING MEMORY
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_C_NN));
    CUDA_CHECK(cudaFree(d_D_IN));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_N_init));
    CUDA_CHECK(cudaFree(d_N_stack));
    CUDA_CHECK(cudaFree(d_work));


    return N_stack_CUDA;
}

Eigen::MatrixXd updateCouplesb(Eigen::MatrixXd t_N_stack_CUDA) {

    Eigen::MatrixXd beta((lambda_dimension/2)*(number_of_Chebyshev_points-1), 1); // Dimension: 45x1

    Eigen::VectorXd Gamma(lambda_dimension/2);
    Gamma << 1, 0, 0;

    //  TODO: Update it to work with any possible C_bar
    //  Building C_bar
    const Eigen::Vector3d C_bar = Eigen::Vector3d::Zero();

    Eigen::Vector3d N;

    Eigen::Vector3d b;


    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {

        N << t_N_stack_CUDA(i),
             t_N_stack_CUDA(i  +  (number_of_Chebyshev_points-1)),
             t_N_stack_CUDA(i + 2*(number_of_Chebyshev_points-1));


        b = skew(Gamma).transpose()*N-C_bar;

        beta(i) = b(0);
        beta(i  +  (number_of_Chebyshev_points-1)) = b(1);
        beta(i + 2*(number_of_Chebyshev_points-1)) = b(2);

    }


    return beta;
}

Eigen::MatrixXd integrateInternalCouples(Eigen::MatrixXd t_N_stack_CUDA)
{
    // ==================================== BENCHMARK ==================================== 
    ::benchmark::RegisterBenchmark("Integrate Internal Couples:", [&](::benchmark::State &t_state){

    t_state.counters = {
        {"na", na},
        {"ne", ne},
        {"Cheb pts", number_of_Chebyshev_points}
    };

    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();
    //BACKWARD INTEGRATION:
    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN_B = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);
    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN_B = Dn.block<number_of_Chebyshev_points-1, 1>(1, 0);

    // INITIALISATION
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_NN_B); // Dimension: 45x45
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_IN_B); // Dimension: 45x3

    Eigen::MatrixXd C_NN = Eigen::MatrixXd::Zero((number_of_Chebyshev_points-1)*(lambda_dimension/2), (number_of_Chebyshev_points-1)*(lambda_dimension/2));
    
    Eigen::MatrixXd beta_NN = Eigen::MatrixXd::Zero((number_of_Chebyshev_points-1)*(lambda_dimension/2), 1);

    Eigen::VectorXd C_init(lambda_dimension/2);
    C_init << 1, 0, 0;

    //What we want to calculate
    Eigen::MatrixXd C_stack_CUDA(t_N_stack_CUDA.rows(), t_N_stack_CUDA.cols());

    //Definition of matrices dimensions.
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;

    const int rows_C_init = C_init.rows();
    const int cols_C_init = C_init.cols();
    const int ld_C_init = rows_C_init;

    const int rows_beta_NN = beta_NN.rows();
    const int cols_beta_NN = beta_NN.cols();
    const int ld_beta_NN = rows_beta_NN;
    
    int info = 0;
    int lwork = 0;

    // Create Pointers
    double* d_C_NN = nullptr;
    double* d_D_IN = nullptr;
    double* d_C_init = nullptr;
    double* d_beta_NN = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    // Compute the memory occupation
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_C_init_in_bytes = size_of_double * C_init.size();
    const auto size_of_beta_NN_in_bytes = size_of_double * beta_NN.size();
    const auto size_of_N_stack_in_bytes = size_of_double * t_N_stack_CUDA.size();

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_init), size_of_C_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_beta_NN), size_of_beta_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_init, C_init.data(), size_of_C_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
        // Handle or debug the error appropriately
    };

    //Has to be after cusolverDnDgetrf_bufferSize as lwork is only computed then.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork));

    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;

    //  BENCHMARK
        for (auto _ : t_state) {
            auto start = std::chrono::high_resolution_clock::now();
            //INSERT BENCHMARK CODE HERE:
            C_NN =  updateCMatrix(qe, D_NN);

            beta_NN = updateCouplesb(t_N_stack_CUDA);

            CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_beta_NN, beta_NN.data(), size_of_beta_NN_in_bytes, cudaMemcpyHostToDevice));

            CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_C_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_C_init, ld_C_init, &beta_cublas, d_beta_NN, ld_beta_NN));

            // LU factorization
            CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

            // Solving the final system
            CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_beta_NN, ld_beta_NN, d_info));

            //END OF BENCHMARK
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

            t_state.SetIterationTime(elapsed_seconds.count());
        }

        //FREEING MEMORY
        CUDA_CHECK(cudaFree(d_beta_NN));
        CUDA_CHECK(cudaFree(d_C_init));
        CUDA_CHECK(cudaFree(d_C_NN));
        CUDA_CHECK(cudaFree(d_D_IN));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_work));

    })->Repetitions(20)->Unit(::benchmark::kMicrosecond);

    // ==================================== STANDARD ====================================
    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();
    //BACKWARD INTEGRATION:
    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN_B = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);
    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN_B = Dn.block<number_of_Chebyshev_points-1, 1>(1, 0);


    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_NN_B); // Dimension: 45x45
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_IN_B); // Dimension: 45x3

    Eigen::MatrixXd C_NN =  updateCMatrix(qe, D_NN);
    
    Eigen::MatrixXd beta_NN = updateCouplesb(t_N_stack_CUDA);

    Eigen::VectorXd C_init(lambda_dimension/2);
    C_init << 1, 0, 0;

    //What we want to calculate
    Eigen::MatrixXd C_stack_CUDA(t_N_stack_CUDA.rows(), t_N_stack_CUDA.cols());

    //Definition of matrices dimensions.
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;

    const int rows_C_init = C_init.rows();
    const int cols_C_init = C_init.cols();
    const int ld_C_init = rows_C_init;

    const int rows_beta_NN = beta_NN.rows();
    const int cols_beta_NN = beta_NN.cols();
    const int ld_beta_NN = rows_beta_NN;
    
    int info = 0;
    int lwork = 0;

    // Create Pointers
    double* d_C_NN = nullptr;
    double* d_D_IN = nullptr;
    double* d_C_init = nullptr;
    double* d_beta_NN = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    // Compute the memory occupation
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_C_init_in_bytes = size_of_double * C_init.size();
    const auto size_of_beta_NN_in_bytes = size_of_double * beta_NN.size();
    const auto size_of_N_stack_in_bytes = size_of_double * t_N_stack_CUDA.size();

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_init), size_of_C_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_beta_NN), size_of_beta_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_init, C_init.data(), size_of_C_init_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta_NN, beta_NN.data(), size_of_beta_NN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
        // Handle or debug the error appropriately
    };

    //Has to be after cusolverDnDgetrf_bufferSize as lwork is only computed then.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork));

    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;
    // res = -D_IN*C_init + beta_NN
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_C_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_C_init, ld_C_init, &beta_cublas, d_beta_NN, ld_beta_NN));

    // LU factorization
    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

    // Solving the final system
    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_beta_NN, ld_beta_NN, d_info));

    CUDA_CHECK(cudaMemcpy(C_stack_CUDA.data(), d_beta_NN, size_of_beta_NN_in_bytes, cudaMemcpyDeviceToHost));

    //FREEING MEMORY
    CUDA_CHECK(cudaFree(d_beta_NN));
    CUDA_CHECK(cudaFree(d_C_init));
    CUDA_CHECK(cudaFree(d_C_NN));
    CUDA_CHECK(cudaFree(d_D_IN));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    return C_stack_CUDA;
}

Eigen::MatrixXd buildLambda(Eigen::MatrixXd t_C_stack_CUDA, Eigen::MatrixXd t_N_stack_CUDA)
{
    Eigen::Vector3d C;
    Eigen::Vector3d N;

    Eigen::VectorXd lambda(lambda_dimension);

    Eigen::MatrixXd Lambda_stack(lambda_dimension*(number_of_Chebyshev_points-1), 1);

    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {

        N << t_N_stack_CUDA(i),
             t_N_stack_CUDA(i  +  (number_of_Chebyshev_points-1)),
             t_N_stack_CUDA(i + 2*(number_of_Chebyshev_points-1));

        C << t_C_stack_CUDA(i),
             t_C_stack_CUDA(i  +  (number_of_Chebyshev_points-1)),
             t_C_stack_CUDA(i + 2*(number_of_Chebyshev_points-1));

        lambda << C, N;

        Lambda_stack.block<6,1>(i*lambda_dimension,0) = lambda;
    }

    return Lambda_stack;
}



// Used to build Qa_stack
Eigen::MatrixXd updateQad_vector_b(Eigen::MatrixXd t_Lambda_stack)
{
    //  Define the Chebyshev points on the unit circle
    const auto Chebyshev_points = ComputeChebyshevPoints<number_of_Chebyshev_points>();

    Eigen::MatrixXd B_NN(number_of_Chebyshev_points-1, Qa_dimension);

    Eigen::VectorXd b(Qa_dimension);

    Eigen::MatrixXd B(6, 3);
    B.block(0, 0, 3, 3).setIdentity();
    B.block(3, 0, 3, 3).setZero();

    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {

        // NOTE: Lambda_stack is already built without the first cheb. pt. however we need to index the Chebyshev_points[1] as our first cheb. pt (PORCA PUTTANA)
        b =  -Phi<na,ne>(Chebyshev_points[i+1]).transpose()*B.transpose()*t_Lambda_stack.block<lambda_dimension,1>(lambda_dimension*i,0);

        B_NN.block<1,Qa_dimension>(i, 0) = b.transpose();
    }
    return B_NN;
}

Eigen::MatrixXd integrateGeneralisedForces(Eigen::MatrixXd t_Lambda_stack)
{   
    // ==================================== BENCHMARK ==================================== 
    ::benchmark::RegisterBenchmark("Integrate Generalized Forces:", [&](::benchmark::State &t_state){
    
    t_state.counters = {
        {"na", na},
        {"ne", ne},
        {"Cheb pts", number_of_Chebyshev_points}
    };

    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();
    //BACKWARD INTEGRATION:
    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN_B = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);
    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN_B = Dn.block<number_of_Chebyshev_points-1, 1>(1, 0);


    // INITIALISATION
    Eigen::Vector3d Qa_init;
    Qa_init << 0,
               0,
               0;

    // Dn_NN is constant so we can pre-invert
    Eigen::MatrixXd Dn_NN_inv = Dn_NN_B.inverse();

    Eigen::MatrixXd B_NN = Eigen::MatrixXd::Zero(number_of_Chebyshev_points-1, Qa_dimension);
    
    //Definition of matrices dimensions.
    const int rows_B_NN = B_NN.rows();
    const int cols_B_NN = B_NN.cols();
    const int ld_B_NN = rows_B_NN;

    const int rows_Dn_NN_inv = Dn_NN_inv.rows();
    const int cols_Dn_NN_inv = Dn_NN_inv.cols();
    const int ld_Dn_NN_inv = rows_Dn_NN_inv;

    const int rows_Qa_stack = rows_Dn_NN_inv;
    const int cols_Qa_stack = cols_B_NN;
    const int ld_Qa_stack = rows_Qa_stack;

    // Create Pointers
    double* d_B_NN = nullptr;
    double* d_Dn_NN_inv = nullptr;    
    double* d_Qa_stack = nullptr;

    // Compute the memory occupation
    const auto size_of_B_NN_in_bytes = size_of_double * B_NN.size();
    const auto size_of_Dn_NN_inv_in_bytes = size_of_double * Dn_NN_inv.size();
    const auto size_of_Qa_stack_in_bytes = size_of_double * rows_Qa_stack * cols_Qa_stack;

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B_NN), size_of_B_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Dn_NN_inv), size_of_Dn_NN_inv_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Qa_stack), size_of_Qa_stack_in_bytes));

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_Dn_NN_inv, Dn_NN_inv.data(), size_of_Dn_NN_inv_in_bytes, cudaMemcpyHostToDevice));

    // Variable to check the result
    Eigen::MatrixXd Qa_stack_CUDA(rows_Qa_stack, cols_Qa_stack);

    // Compute Qa_stack = Dn_NN_inv*B_NN
    double alpha_cublas = 1.0;
    double beta_cublas = 0.0;
    //  BENCHMARK
        for (auto _ : t_state) {
            auto start = std::chrono::high_resolution_clock::now();
            //INSERT BENCHMARK CODE HERE:
            B_NN = updateQad_vector_b(t_Lambda_stack);

            CUDA_CHECK(cudaMemcpy(d_B_NN, B_NN.data(), size_of_B_NN_in_bytes, cudaMemcpyHostToDevice));
            
            CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_Dn_NN_inv, cols_B_NN, cols_Dn_NN_inv, &alpha_cublas, d_Dn_NN_inv, ld_Dn_NN_inv, d_B_NN, ld_B_NN, &beta_cublas, d_Qa_stack, ld_Qa_stack));
            //END OF BENCHMARK
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

            t_state.SetIterationTime(elapsed_seconds.count());
        }

        //FREEING MEMORY
        CUDA_CHECK(cudaFree(d_B_NN));
        CUDA_CHECK(cudaFree(d_Qa_stack));
        CUDA_CHECK(cudaFree(d_Dn_NN_inv));

    })->Repetitions(20)->Unit(::benchmark::kMicrosecond);

    // ==================================== STANDARD ====================================

    // Qa_stack = B_NN*Dn_NN_inv
    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();
    //BACKWARD INTEGRATION:
    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN_B = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);
    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN_B = Dn.block<number_of_Chebyshev_points-1, 1>(1, 0);

    Eigen::Vector3d Qa_init;
    Qa_init << 0,
               0,
               0;

    // Dn_NN is constant so we can pre-invert
    Eigen::MatrixXd Dn_NN_inv = Dn_NN_B.inverse();

    Eigen::MatrixXd B_NN = updateQad_vector_b(t_Lambda_stack);
    
    //Definition of matrices dimensions.
    const int rows_B_NN = B_NN.rows();
    const int cols_B_NN = B_NN.cols();
    const int ld_B_NN = rows_B_NN;

    const int rows_Dn_NN_inv = Dn_NN_inv.rows();
    const int cols_Dn_NN_inv = Dn_NN_inv.cols();
    const int ld_Dn_NN_inv = rows_Dn_NN_inv;

    const int rows_Qa_stack = rows_Dn_NN_inv;
    const int cols_Qa_stack = cols_B_NN;
    const int ld_Qa_stack = rows_Qa_stack;

    // Create Pointers
    double* d_B_NN = nullptr;
    double* d_Dn_NN_inv = nullptr;    
    double* d_Qa_stack = nullptr;

    // Compute the memory occupation
    const auto size_of_B_NN_in_bytes = size_of_double * B_NN.size();
    const auto size_of_Dn_NN_inv_in_bytes = size_of_double * Dn_NN_inv.size();
    const auto size_of_Qa_stack_in_bytes = size_of_double * rows_Qa_stack * cols_Qa_stack;

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B_NN), size_of_B_NN_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Dn_NN_inv), size_of_Dn_NN_inv_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Qa_stack), size_of_Qa_stack_in_bytes));

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_B_NN, B_NN.data(), size_of_B_NN_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Dn_NN_inv, Dn_NN_inv.data(), size_of_Dn_NN_inv_in_bytes, cudaMemcpyHostToDevice));


    // Variable to check the result
    Eigen::MatrixXd Qa_stack_CUDA(rows_Qa_stack, cols_Qa_stack);

    // Compute Qa_stack = Dn_NN_inv*B_NN
    double alpha_cublas = 1.0;
    double beta_cublas = 0.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_Dn_NN_inv, cols_B_NN, cols_Dn_NN_inv, &alpha_cublas, d_Dn_NN_inv, ld_Dn_NN_inv, d_B_NN, ld_B_NN, &beta_cublas, d_Qa_stack, ld_Qa_stack));

    CUDA_CHECK(cudaMemcpy(Qa_stack_CUDA.data(), d_Qa_stack, size_of_Qa_stack_in_bytes, cudaMemcpyDeviceToHost));

    //FREEING MEMORY
    CUDA_CHECK(cudaFree(d_B_NN));
    CUDA_CHECK(cudaFree(d_Qa_stack));
    CUDA_CHECK(cudaFree(d_Dn_NN_inv));

    return Qa_stack_CUDA;
}






int main(int argc, char *argv[])
{
    // CUDA initialization 
    CUBLAS_CHECK(
        cublasCreate(&cublasH)
    );

    CUSOLVER_CHECK(
        cusolverDnCreate(&cusolverH)
    );


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
    qe.setZero();
    

    const auto Q_stack_CUDA = integrateQuaternions();
    // std::cout << "Quaternion Integration : \n" << Q_stack_CUDA << std::endl;
    
    const auto r_stack_CUDA = integratePosition(Q_stack_CUDA);
    // std::cout << "Position Integration : \n" << r_stack_CUDA << std::endl;

    const auto N_stack_CUDA = integrateInternalForces(Q_stack_CUDA);
    // std::cout << "Internal Forces Integration : \n" << N_stack_CUDA << "\n" << std::endl;

    const auto C_stack_CUDA = integrateInternalCouples(N_stack_CUDA);
    // std::cout << "Internal Couples Integration : \n" << C_stack_CUDA << "\n" << std::endl;

    const auto Lambda_stack_CUDA = buildLambda(C_stack_CUDA, N_stack_CUDA);
    //std::cout << "Lambda_stack : \n" << Lambda_stack_CUDA << "\n" << std::endl;

    const auto Qa_stack_CUDA = integrateGeneralisedForces(Lambda_stack_CUDA);
    // std::cout << "Generalized Forces Integration : \n" << Qa_stack_CUDA << std::endl;

    // Benchmark initialization6
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    
    // Destry cuda objects
        CUBLAS_CHECK(
        cublasDestroy(cublasH)
    );

    CUSOLVER_CHECK(
        cusolverDnDestroy(cusolverH)
    );

    CUDA_CHECK(
        cudaDeviceReset()
    );

    return 0;
}
