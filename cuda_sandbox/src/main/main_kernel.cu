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


// CUDA specific variables
const auto size_of_double = sizeof(double);
cusolverDnHandle_t cusolverH = NULL;
cublasHandle_t cublasH = NULL;


static const unsigned int number_of_Chebyshev_points = 16;

static const unsigned int quaternion_state_dimension = 4;
static const unsigned int position_dimension = 3;
static const unsigned int quaternion_problem_dimension = quaternion_state_dimension * (number_of_Chebyshev_points-1);

static const unsigned int lambda_dimension = 6;

static const unsigned int Qa_dimension = 9;


// Defining qe in the CPU and its GPU parameters
Eigen::Matrix<double, ne*na, 1> qe;
double* d_qe = nullptr;
int size_of_qe_in_bytes = ne * na * size_of_double;


//  Obtain the Chebyshev differentiation matrix
const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

//FORWARD INTEGRATION:
//  Extract D_NN from the differentiation matrix (for the spectral integration)
const Eigen::MatrixXd Dn_NN_F = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);
//  Extract D_IN (for the propagation of initial conditions)
const Eigen::MatrixXd Dn_IN_F = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);

//BACKWARD INTEGRATION:
//  Extract D_NN from the differentiation matrix (for the spectral integration)
const Eigen::MatrixXd Dn_NN_B = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);
//  Extract D_IN (for the propagation of initial conditions)
const Eigen::MatrixXd Dn_IN_B = Dn.block<number_of_Chebyshev_points-1, 1>(1, 0);


// Define the Chebyshev points on the unit circle
const auto Chebyshev_points = ComputeChebyshevPoints<number_of_Chebyshev_points>();
Eigen::MatrixXd Phi_stack = Eigen::MatrixXd::Zero(na*number_of_Chebyshev_points, na*ne);


// K_stack parameters for GPU
double* d_K_stack = nullptr;
int size_of_K_stack_in_bytes = na * number_of_Chebyshev_points * size_of_double;






// Function Definitions

__global__ void computeCMatrixKernel(const double* d_K_stack, const double* D_NN, double* C_NN) {

    int i = threadIdx.x;


    // Extract the curvature from the strain and compute A_at_chebyshev_point
    // Z_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
    //                             K(0),     0,   K(2),  -K(1),
    //                             K(1), -K(2),      0,   K(0),
    //                             K(2),  K(1),  -K(0),      0;

    // A_at_chebyshev_point = 0.5 * Z_at_chebyshev_point;

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

    // for (unsigned int row = 0; row < quaternion_state_dimension; ++row) {
    //     for (unsigned int col = 0; col < quaternion_state_dimension; ++col) {
    //         int row_index = row * (number_of_Chebyshev_points - 1) + i;
    //         int col_index = col * (number_of_Chebyshev_points - 1) + i;
    //         C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] - A_at_chebyshev_point(row, col);
    //     }
    // }
}

/*
Eigen::MatrixXd computeCMatrix(const Eigen::VectorXd &t_qe, const Eigen::MatrixXd &D_NN)
{
    Eigen::MatrixXd C_NN = D_NN;

    // Compute the memory occupation 
    const auto size_of_t_qe_in_bytes = t_qe.size()*size_of_double;
    const auto size_of_D_NN_in_bytes = D_NN.size() * size_of_double;
    const auto size_of_C_NN_in_bytes = C_NN.size() * size_of_double;
    
    // Create Pointers
    double* d_t_qe;
    double* d_D_NN;
    double* d_C_NN;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_t_qe), size_of_t_qe_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>&d_D_NN, size_of_D_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>&d_C_NN, size_of_C_NN_in_bytes)
    );

    //  Copy the data
    CUDA_CHECK(
        cudaMemcpy(d_t_qe, t_qe.data(), size_of_t_qe_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_D_NN, D_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice)
    );

    // Launch kernel with one block
    int threadsPerBlock = number_of_Chebyshev_points;
    computeCMatrixKernel<<<1, threadsPerBlock>>>(d_t_qe, d_D_NN, d_C_NN);

    // Copy result back 
    CUDA_CHECK(
        cudaMemcpy(C_NN.data(), d_C_NN, size_of_C_NN_in_bytes, cudaMemcpyDeviceToHost);
    );

    // Free the memory
    CUDA_CHECK(
        cudaFree(t_qe)
    );
    CUDA_CHECK(
        cudaFree(D_NN)
    );
    CUDA_CHECK(
        cudaFree(C_NN)
    );

    return C_NN;
}
*/

// Used to build Q_stack
Eigen::VectorXd integrateQuaternions()
{
    #pragma region K_stack

    // Allocate memory on the device
    double* d_Phi_stack = nullptr;

    int size_of_Phi_stack_in_bytes = (na * number_of_Chebyshev_points) * (na * ne) * size_of_double;


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


    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN_F);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN_F);


    //Compute C_NN
    Eigen::MatrixXd C_NN = D_NN;
    
    // Compute the memory occupation 
    const auto size_of_D_NN_in_bytes = D_NN.size() * size_of_double;
    
    // Create Pointers
    double* d_D_NN = nullptr;
    double* d_C_NN = nullptr;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_qe), size_of_qe_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_D_NN), size_of_D_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_D_NN_in_bytes) //Same size as D_NN
    );

    //  Copy the data
    CUDA_CHECK(
        cudaMemcpy(d_D_NN, D_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_C_NN, C_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice)
    );

    // Launch kernel with one block
    computeCMatrixKernel<<<1, number_of_Chebyshev_points>>>(d_K_stack, d_D_NN, d_C_NN);

    // Free the memory
    CUDA_CHECK(
        cudaFree(d_qe)
    );
    CUDA_CHECK(
        cudaFree(d_D_NN)
    );

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
    //double* d_C_NN = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    // Compute the memory occupation 
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_q_init_in_bytes = size_of_double * q_init.size();
    const auto size_of_b_in_bytes = size_of_double * b.size();
    const auto size_of_res_in_bytes = size_of_double * rows_res * cols_res;
    const auto size_of_Q_stack_in_bytes = size_of_double * rows_Q_stack * cols_Q_stack;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_q_init), size_of_q_init_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Q_stack), size_of_Q_stack_in_bytes)
    );
    // CUDA_CHECK(
    //     cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes)
    // );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int))
    );

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_q_init, q_init.data(), size_of_q_init_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_b, b.data(), size_of_b_in_bytes, cudaMemcpyHostToDevice)
    );
    // CUDA_CHECK(
    //     cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice)
    // );
    CUDA_CHECK(
        cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice)
    );

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
    alpha_cublas = -1.0;
    beta_cublas = 1.0;


    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_q_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_q_init, ld_q_init, &beta_cublas, d_b, ld_b)
    );

    // LU factorization
    CUSOLVER_CHECK(
        cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info)
    );

    // Solving the final system
    CUSOLVER_CHECK(
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_b, ld_b, d_info)
    );

    // Memory Copy
    CUDA_CHECK(
        cudaMemcpy(Q_stack_CUDA.data(), d_b, size_of_b_in_bytes, cudaMemcpyDeviceToHost)
    );

    //FREEING MEMORY
    CUDA_CHECK(
        cudaFree(d_D_IN)
    );
    CUDA_CHECK(
        cudaFree(d_C_NN)
    );
    CUDA_CHECK(
        cudaFree(d_b)
    );
    CUDA_CHECK(
        cudaFree(d_info)
    );
    CUDA_CHECK(
        cudaFree(d_q_init)
    );
    CUDA_CHECK(
        cudaFree(d_Q_stack)
    );
    CUDA_CHECK(
        cudaFree(d_res)
    );
    CUDA_CHECK(
        cudaFree(d_work)
    );


    return Q_stack_CUDA;
}






// Used to build r_stack

__device__ void quaternionToRotationMatrix(const double* q, double* R) {
    double q0 = q[0];
    double q1 = q[1];
    double q2 = q[2];
    double q3 = q[3];

    R[0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
    R[1] = 2.0 * (q1 * q2 - q0 * q3);
    R[2] = 2.0 * (q1 * q3 + q0 * q2);
    R[3] = 2.0 * (q1 * q2 + q0 * q3);
    R[4] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3;
    R[5] = 2.0 * (q2 * q3 - q0 * q1);
    R[6] = 2.0 * (q1 * q3 - q0 * q2);
    R[7] = 2.0 * (q2 * q3 + q0 * q1);
    R[8] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
}

__global__ void updatePositionbKernel(const double* t_Q_stack_CUDA, double* t_b){
    
    int i = threadIdx.x;

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b(t_b, number_of_Chebyshev_points-1, position_dimension);

    if (i < number_of_Chebyshev_points-1) {
        double q[4] = { t_Q_stack_CUDA[i], 
                        t_Q_stack_CUDA[i + (number_of_Chebyshev_points-1)],
                        t_Q_stack_CUDA[i + 2*(number_of_Chebyshev_points-1)], 
                        t_Q_stack_CUDA[i + 3*(number_of_Chebyshev_points-1)]
                        };

        double R[9];

        quaternionToRotationMatrix(q, R);

        b.block<1, 3>(i, 0) = (Eigen::Map<Eigen::MatrixXd>(R, 3, 3) * Eigen::Vector3d(1, 0, 0)).transpose();
    }
}

// Eigen::MatrixXd computeIvp(Eigen::MatrixXd t_Dn_IN_F, Eigen::Vector3d t_r_init)
// {
//     Eigen::MatrixXd ivp(number_of_Chebyshev_points-1, position_dimension);
//     for(unsigned int i=0; i<(number_of_Chebyshev_points-1); i++)
//     ivp.row(i) = Dn_IN_F(i, 0) * r_init.transpose();

//     return ivp;
// }

__global__ void computeIvpKernel(const double* t_Dn_IN_F_data, const double* t_r_init_data, double* ivp_data) {
    int i = threadIdx.x;

    Eigen::Map<const Eigen::MatrixXd> t_Dn_IN_F(t_Dn_IN_F_data, number_of_Chebyshev_points-1, 1); // the 1 at the end is because we only need one (the first) col of Dn_IN_F
    Eigen::Map<const Eigen::Vector3d> t_r_init(t_r_init_data);
    Eigen::Map<Eigen::MatrixXd> ivp(ivp_data, number_of_Chebyshev_points-1, position_dimension);

    if (i < number_of_Chebyshev_points-1) {
        ivp.row(i) = t_Dn_IN_F(i, 0) * t_r_init.transpose();
    }
}

Eigen::MatrixXd integratePosition(Eigen::MatrixXd t_Q_stack_CUDA)
{   
    Eigen::Vector3d r_init;
    r_init << 0,
              0,
              0;

    //  This matrix remains constant so we can pre invert
    Eigen::MatrixXd Dn_NN_inv = Dn_NN_F.inverse();

    Eigen::Matrix<double, number_of_Chebyshev_points-1, position_dimension> b; //used in updatePositionbKernel
    
    Eigen::MatrixXd ivp(number_of_Chebyshev_points-1, position_dimension);

    Eigen::MatrixXd res(number_of_Chebyshev_points-1, position_dimension);

    // Compute the memory occupation
    const auto size_of_Q_stack_CUDA_in_bytes = t_Q_stack_CUDA.size() * size_of_double;
    const auto size_of_b_in_bytes = b.size() * size_of_double;
    const auto size_of_res_in_bytes = res.size() * size_of_double;
    const auto size_of_r_init_in_bytes = r_init.size() * size_of_double;
    const auto size_of_ivp_in_bytes = ivp.size() * size_of_double;
    const auto size_of_Dn_IN_F_in_bytes = Dn_IN_F.size() * size_of_double;

    // Create Pointers
    double* d_Q_stack_CUDA;
    double* d_b;
    double* d_r_init;
    double* d_ivp;
    double* d_Dn_IN_F;

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q_stack_CUDA), size_of_Q_stack_CUDA_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_r_init), size_of_r_init_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ivp), size_of_ivp_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Dn_IN_F), size_of_Dn_IN_F_in_bytes));

    //  Copy the data
    CUDA_CHECK(
        cudaMemcpy(d_Q_stack_CUDA, t_Q_stack_CUDA.data(), size_of_Q_stack_CUDA_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_r_init, r_init.data(), size_of_r_init_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_Dn_IN_F, Dn_NN_F.data(), size_of_Dn_IN_F_in_bytes, cudaMemcpyHostToDevice)
    );

    // Launch the kernel for b: the result of the kernel is stored into d_b
    updatePositionbKernel<<<1, number_of_Chebyshev_points-1>>>(d_Q_stack_CUDA, d_b);
    
    // Launch the kernel for ivp: the result of the kernel is stored into d_b
    computeIvpKernel<<<1, number_of_Chebyshev_points-1>>>(d_Dn_IN_F, d_r_init, d_ivp);

    // Before we had b_NN = updatePositionb and thn res = B_NN -ivp so we have to do the same somehow 

    // Now the operation i have to perform is res = -d_ivp+d_b into the GPU

    // Dimensions
    const int ld_b = b.rows();
    const int rows_ivp = ivp.rows();
    const int cols_ivp = ivp.cols();
    const int ld_ivp = rows_ivp;

    // Computing b = -ivp + b       
    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;

    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_ivp, cols_ivp, cols_ivp, &alpha_cublas, d_ivp, ld_ivp, nullptr, ld_ivp, &beta_cublas, d_b, ld_b)
    );

    // This passage is for sure not necessary but right now it's the fastes thing. If everything work we will fix it.
    CUDA_CHECK(
        cudaMemcpy(res.data(), d_b, size_of_res_in_bytes, cudaMemcpyDeviceToHost)
    );

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
    const auto size_of_r_stack_in_bytes = size_of_double * rows_r_stack * cols_r_stack;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Dn_NN_inv), size_of_Dn_NN_inv_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes)
    );
        CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_r_stack), size_of_r_stack_in_bytes)
    );
    
    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_Dn_NN_inv, Dn_NN_inv.data(), size_of_Dn_NN_inv_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_res, res.data(), size_of_res_in_bytes, cudaMemcpyHostToDevice)
    );

    // What we want to calculate 
    Eigen::MatrixXd r_stack_CUDA(rows_r_stack, cols_r_stack);

    // Compute r_stack = Dn_NN_inv*res
    alpha_cublas = 1.0;
    beta_cublas = 0.0;
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_Dn_NN_inv, cols_res, cols_Dn_NN_inv, &alpha_cublas, d_Dn_NN_inv, ld_Dn_NN_inv, d_res, ld_res, &beta_cublas, d_r_stack, ld_r_stack)
    );

    CUDA_CHECK(
        cudaMemcpy(r_stack_CUDA.data(), d_r_stack, size_of_r_stack_in_bytes, cudaMemcpyDeviceToHost));

    //FREEING MEMORY
    CUDA_CHECK(
        cudaFree(d_Dn_NN_inv)
    );
    CUDA_CHECK(
        cudaFree(d_r_stack)
    );
    CUDA_CHECK(
        cudaFree(d_res)
    );

    return r_stack_CUDA;
}








// Used to build Lambda_stack:
__global__ void updateCMatrixKernel(const double* d_K_stack, const double* D_NN, double* C_NN) {

    int i = threadIdx.x;

    // // Transpose of the K_hat matrix is equivalent to the K_hat matrix itself multiplied by -1

    // v_hat <<  0   ,  -t_v(2),   t_v(1),
    //         t_v(2),     0   ,  -t_v(0),
    //        -t_v(1),   t_v(0),     0   ;

    // v_hat TRANSPOSE <<  0   ,  t_v(2),   -t_v(1),
    //                  -t_v(2),     0   ,   t_v(0),
    //                  t_v(1),   -t_v(0),     0   ;


    // C_NN(row_index, col_index) = D_NN(row_index, col_index) - K_hat.transpose()(row, col) becomes
    // C_NN(row_index, col_index) = D_NN(row_index, col_index) + K_hat(row, col)

    // for (unsigned int row = 0; row < lambda_dimension/2; ++row) {
    //     for (unsigned int col = 0; col < lambda_dimension/2; ++col) {
    //         C_NN(row_index, col_index) = D_NN(row_index, col_index) + K_hat(row, col);
    //     }
    // }

    #pragma region Compute_C_NN
    if (i < number_of_Chebyshev_points-1) {
    int row = 0;
    int col = 1;
    int row_index = row * (number_of_Chebyshev_points - 1) + i;
    int col_index = col * (number_of_Chebyshev_points - 1) + i;
    C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] - d_K_stack[3*i+2];


    row = 0;
    col = 2;
    row_index = row * (number_of_Chebyshev_points - 1) + i;
    col_index = col * (number_of_Chebyshev_points - 1) + i;
    C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] + d_K_stack[3*i+1];


    row = 1;
    col = 0;
    row_index = row * (number_of_Chebyshev_points - 1) + i;
    col_index = col * (number_of_Chebyshev_points - 1) + i;
    C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] + d_K_stack[3*i+2];

    row = 1;
    col = 2;
    row_index = row * (number_of_Chebyshev_points - 1) + i;
    col_index = col * (number_of_Chebyshev_points - 1) + i;
    C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] - d_K_stack[3*i+0];

    row = 2;
    col = 0;
    row_index = row * (number_of_Chebyshev_points - 1) + i;
    col_index = col * (number_of_Chebyshev_points - 1) + i;
    C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] - d_K_stack[3*i+1];

    row = 2;
    col = 1;
    row_index = row * (number_of_Chebyshev_points - 1) + i;
    col_index = col * (number_of_Chebyshev_points - 1) + i;
    C_NN[row_index * quaternion_state_dimension + col_index] = D_NN[row_index * quaternion_state_dimension + col_index] + d_K_stack[3*i+0];
    }
    #pragma endregion
}


// Eigen::MatrixXd updateCMatrix(const Eigen::VectorXd &t_qe, const Eigen::MatrixXd &D_NN)
// {
//     Eigen::MatrixXd C_NN = D_NN;
//     //  Define the Chebyshev points on the unit circle
//     const auto Chebyshev_points = ComputeChebyshevPoints<number_of_Chebyshev_points>();

//     Eigen::Vector3d K;
//     Eigen::MatrixXd A_at_chebyshev_point(lambda_dimension/2, lambda_dimension/2);

//     for(unsigned int i=0; i<Chebyshev_points.size()-1; i++){

//         //  Extract the curvature from the strain
//         K = Phi<na, ne>(Chebyshev_points[i])*t_qe;

//         //  Build Skew Symmetric K matrix (K_hat)
//         Eigen::Matrix3d K_hat = skew(K);
//         A_at_chebyshev_point = K_hat.transpose();

//         for (unsigned int row = 0; row < lambda_dimension/2; ++row) {
//             for (unsigned int col = 0; col < lambda_dimension/2; ++col) {
//                 int row_index = row*(number_of_Chebyshev_points-1)+i;
//                 int col_index = col*(number_of_Chebyshev_points-1)+i;
//                 C_NN(row_index, col_index) = D_NN(row_index, col_index) - A_at_chebyshev_point(row, col);
//             }
//         }

//     }

//     return C_NN;

// }

__global__ void computeNbarKernel(const double* t_Q_stack_CUDA_data, double* Nbar_stack_data) {
int i = threadIdx.x;

    const double g = 9.81; // m/s^2
    const double radius = 0.001; // m
    const double A = M_PI * radius * radius;
    const double rho = 7800; // kg/m^3

    Eigen::Map<const Eigen::VectorXd> t_Q_stack_CUDA(t_Q_stack_CUDA_data, (number_of_Chebyshev_points - 1) * 4);
    Eigen::Map<Eigen::VectorXd> Nbar_stack(Nbar_stack_data, (number_of_Chebyshev_points - 1) * 3);

    if (i < number_of_Chebyshev_points - 1) {
        Eigen::Quaterniond Qbar(t_Q_stack_CUDA(i), t_Q_stack_CUDA(i + (number_of_Chebyshev_points - 1)),
                                t_Q_stack_CUDA(i + 2 * (number_of_Chebyshev_points - 1)),
                                t_Q_stack_CUDA(i + 3 * (number_of_Chebyshev_points - 1)));

        double R[9];
        quaternionToRotationMatrix(Qbar.coeffs().data(), R); //Qbar.coeffs().data() returns a pointer to the raw data of the quaternion coefficients of Qbar

        double Fg[3] = {0, 0, -A * g * rho};
        double Nbar[3];

        // Compute the memory occupation
        const auto size_of_R_in_bytes = 9 * size_of_double;
        const auto size_of_Fg_in_bytes = 3 * size_of_double;
        const auto size_of_Nbar_in_bytes = 3 * size_of_double;

        // Create Pointers
        double* d_R;
        double* d_Fg;
        double* d_Nbar;

        // Memory allocation
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), size_of_R_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Fg), size_of_Fg_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Nbar), size_of_Nbar_in_bytes));

        CUDA_CHECK(cudaMemcpy(d_R, R, size_of_R_in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Fg, Fg, size_of_Fg_in_bytes, cudaMemcpyHostToDevice));

        // cublasDgemm to perform Nbar = R.transpose()*Fg
        double alpha_cublas = 1.0;
        double beta_cublas = 0.0;
        CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 3, 1, 3, &alpha_cublas, d_R, 3, d_Fg, 1, &beta_cublas, d_Nbar, 1));

        CUDA_CHECK(cudaMemcpy(Nbar, d_Nbar, 3 * sizeof(double), cudaMemcpyDeviceToHost));

        Nbar_stack(i) = Nbar[0];
        Nbar_stack(i + (number_of_Chebyshev_points - 1)) = Nbar[1];
        Nbar_stack(i + 2 * (number_of_Chebyshev_points - 1)) = Nbar[2];

        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_Fg));
        CUDA_CHECK(cudaFree(d_Nbar));
    }
}

Eigen::MatrixXd integrateInternalForces(Eigen::MatrixXd t_Q_stack_CUDA)
{   
    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_NN_B);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_IN_B);

    //Compute C_NN
    Eigen::MatrixXd C_NN = D_NN;
    
    // Compute the memory occupation 
    const auto size_of_D_NN_in_bytes = D_NN.size() * size_of_double;
    
    // Create Pointers
    double* d_D_NN = nullptr;
    double* d_C_NN = nullptr;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_qe), size_of_qe_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_D_NN), size_of_D_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_D_NN_in_bytes) //Same size as D_NN
    );

    //  Copy the data
    CUDA_CHECK(
        cudaMemcpy(d_D_NN, D_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_C_NN, C_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice)
    );

    // Launch kernel with one block
    updateCMatrixKernel<<<1, number_of_Chebyshev_points>>>(d_K_stack, d_D_NN, d_C_NN);
    // Eigen::MatrixXd C_NN =  updateCMatrix(qe, D_NN);

    Eigen::VectorXd N_init(lambda_dimension/2);
    N_init << 1, 0, 0;

    Eigen::MatrixXd beta = Eigen::MatrixXd::Zero((lambda_dimension/2)*(number_of_Chebyshev_points-1),1);

    // Create Pointers
    double* d_Q_stack_CUDA = nullptr;
    double* d_Nbar_stack = nullptr;

    // Compute the memory occupation
    const auto size_of_Q_stack_CUDA_in_bytes = size_of_double * t_Q_stack_CUDA.size();
    const auto size_of_Nbar_stack_in_bytes = size_of_double * beta.size(); // Same dimension of beta (beta = -Nbar)

    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Q_stack_CUDA), size_of_Q_stack_CUDA_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Nbar_stack), size_of_Nbar_stack_in_bytes));

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_Q_stack_CUDA, t_Q_stack_CUDA.data(), size_of_Q_stack_CUDA_in_bytes, cudaMemcpyHostToDevice));

    // Launch the kernel: computeNbarKernel
    computeNbarKernel<<<1, number_of_Chebyshev_points - 1>>>(d_Q_stack_CUDA, d_Nbar_stack);

    //  Copy the data
    CUDA_CHECK(cudaMemcpy(beta.data(), d_Nbar_stack, size_of_Nbar_stack_in_bytes, cudaMemcpyDeviceToHost));

    //Eigen::MatrixXd beta = -computeNbar(t_Q_stack_CUDA);

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
    // double* d_C_NN = nullptr;
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
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_N_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_N_init, ld_N_init, &beta_cublas, d_beta, ld_beta)
    );

    // LU factorization
    CUSOLVER_CHECK(
        cusolverDnDgetrf(cusolverH, cols_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

    // Solving the final system
    CUSOLVER_CHECK(
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, cols_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_beta, ld_beta, d_info));

    CUDA_CHECK(
        cudaMemcpy(N_stack_CUDA.data(), d_beta, size_of_beta_in_bytes, cudaMemcpyDeviceToHost));

    //FREEING MEMORY
    CUDA_CHECK(
        cudaFree(d_beta)
    );
    CUDA_CHECK(
        cudaFree(d_C_NN)
    );
    CUDA_CHECK(
        cudaFree(d_D_IN)
    );
    CUDA_CHECK(
        cudaFree(d_info)
    );
    CUDA_CHECK(
        cudaFree(d_N_init)
    );
    CUDA_CHECK(
        cudaFree(d_N_stack)
    );
    CUDA_CHECK(
        cudaFree(d_work)
    );


    return N_stack_CUDA;
}

__global__ void updateCouplesbKernel(const double* t_N_stack_CUDA, double* d_beta) {
    int idx = threadIdx.x;

    if (idx < number_of_Chebyshev_points - 1) {
        Eigen::VectorXd Gamma(lambda_dimension / 2);
        Gamma << 1, 0, 0;

        // Construct the skew-symmetric matrix manually
        Eigen::Matrix3d skewGamma;
        skewGamma << 0, -Gamma(2), Gamma(1),
                     Gamma(2), 0, -Gamma(0),
                    -Gamma(1), Gamma(0), 0;

        const Eigen::Vector3d C_bar = Eigen::Vector3d::Zero();
        Eigen::Vector3d N;

        int offset = idx * lambda_dimension / 2;

        for (int i = 0; i < lambda_dimension / 2; ++i) {
            N(i) = t_N_stack_CUDA[offset + i];
        }

        // Dimensions definition
        const int rows_skewGamma = 3;
        const int cols_skewGamma = 3;        
        const int ld_skewGamma = 3;

        const int rows_N = N.rows();
        const int cols_N = N.cols();
        const int ld_N = rows_N;

        const int rows_C_bar = C_bar.rows();
        const int cols_C_bar = C_bar.cols();
        const int ld_C_bar = rows_N;

        const int ld_beta = (lambda_dimension/2)*(number_of_Chebyshev_points-1);
        
        // Create Pointers
        double* d_skewGamma = nullptr;
        double* d_N = nullptr;
        double* d_C_bar = nullptr;
        double* d_beta = nullptr;

        // Compute the memory occupation
        const auto size_of_skewGamma_in_bytes = size_of_double * rows_skewGamma*cols_skewGamma;
        const auto size_of_N_in_bytes = size_of_double * N.size();
        const auto size_of_C_bar_in_bytes = size_of_double * C_bar.size();

        // Allocate the memory
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_skewGamma), size_of_skewGamma_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_N), size_of_N_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_bar), size_of_C_bar_in_bytes));

        //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
        CUDA_CHECK(cudaMemcpy(d_skewGamma, skewGamma.data(), size_of_skewGamma_in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_N, N.data(), size_of_N_in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_bar, C_bar.data(), size_of_C_bar_in_bytes, cudaMemcpyHostToDevice));

        // Perform b = skewGamma.transpose() * N - C_bar
        const double alpha_cublas = 1.0;
        const double beta_cublas = -1.0;
        CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, rows_skewGamma, cols_N, cols_skewGamma, &alpha_cublas, d_skewGamma, ld_skewGamma, d_N, ld_N, &beta_cublas, d_beta, ld_beta));

        CUDA_CHECK(cudaMemcpy(beta + offset, d_beta, 3 * sizeof(double), cudaMemcpyDeviceToHost)); // I hope this works

        cudaFree(d_skewGamma);
        cudaFree(d_N);
        cudaFree(d_beta);
    }
}

Eigen::MatrixXd integrateInternalCouples(Eigen::MatrixXd t_N_stack_CUDA)
{
    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_NN_B);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_IN_B);

    //Compute C_NN
    Eigen::MatrixXd C_NN = D_NN;
    
    // Compute the memory occupation 
    const auto size_of_D_NN_in_bytes = D_NN.size() * size_of_double;
    
    // Create Pointers
    double* d_D_NN = nullptr;
    double* d_C_NN = nullptr;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_qe), size_of_qe_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_D_NN), size_of_D_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_D_NN_in_bytes) //Same size as D_NN
    );

    //  Copy the data
    CUDA_CHECK(
        cudaMemcpy(d_D_NN, D_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_C_NN, C_NN.data(), size_of_D_NN_in_bytes, cudaMemcpyHostToDevice)
    );

    // Launch kernel with one block
    updateCMatrixKernel<<<1, number_of_Chebyshev_points>>>(d_K_stack, d_D_NN, d_C_NN);
    
    Eigen::MatrixXd beta_NN = Eigen::MatrixXd::Zero((lambda_dimension/2)*(number_of_Chebyshev_points-1), 1);

    // Create Pointers
    double* d_N_stack_CUDA = nullptr;
    double* d_beta_NN = nullptr;

    // Compute the memory occupation
    const auto size_of_N_stack_CUDA_in_bytes = size_of_double * t_N_stack_CUDA.size();
    const auto size_of_Nbar_stack_in_bytes = size_of_double * beta_NN.size(); // Same dimension of beta (beta = -Nbar)
    const auto size_of_beta_NN_in_bytes = size_of_double * beta_NN.size();


    // Allocate the memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_N_stack_CUDA), size_of_N_stack_CUDA_in_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_beta_NN), size_of_beta_NN_in_bytes));

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(cudaMemcpy(d_N_stack_CUDA, t_N_stack_CUDA.data(), size_of_N_stack_CUDA_in_bytes, cudaMemcpyHostToDevice));

    // Launch the kernel: computeNbarKernel
    updateCouplesbKernel<<<1, number_of_Chebyshev_points - 1>>>(d_N_stack_CUDA, d_beta_NN);

    //  Copy the data
    CUDA_CHECK(cudaMemcpy(beta_NN.data(), d_beta_NN, size_of_beta_NN_in_bytes, cudaMemcpyDeviceToHost));

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
    // double* d_C_NN = nullptr;
    double* d_D_IN = nullptr;
    double* d_C_init = nullptr;
    // double* d_beta_NN = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    // Compute the memory occupation
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_C_init_in_bytes = size_of_double * C_init.size();
    // const auto size_of_beta_NN_in_bytes = size_of_double * beta_NN.size();
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
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork)
    );

    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;
    // res = -D_IN*C_init + beta_NN
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_C_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_C_init, ld_C_init, &beta_cublas, d_beta_NN, ld_beta_NN)
    );

    // LU factorization
    CUSOLVER_CHECK(
        cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info)
    );

    // Solving the final system
    CUSOLVER_CHECK(
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_beta_NN, ld_beta_NN, d_info)
    );

    CUDA_CHECK(
        cudaMemcpy(C_stack_CUDA.data(), d_beta_NN, size_of_beta_NN_in_bytes, cudaMemcpyDeviceToHost)
    );

    //FREEING MEMORY
    CUDA_CHECK(
        cudaFree(d_beta_NN)
    );
    CUDA_CHECK(
        cudaFree(d_C_init)
    );
    CUDA_CHECK(
        cudaFree(d_C_NN)
    );
    CUDA_CHECK(
        cudaFree(d_D_IN)
    );
    CUDA_CHECK(
        cudaFree(d_info)
    );
    CUDA_CHECK(
        cudaFree(d_work)
    );

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
    // Qa_stack = B_NN*Dn_NN_inv
    Eigen::Vector3d Qa_init;
    Qa_init << 0,
               0,
               0;

    Eigen::MatrixXd B_NN(number_of_Chebyshev_points-1, Qa_dimension);

    // Dn_NN is constant so we can pre-invert
    Eigen::MatrixXd Dn_NN_inv = Dn_NN_B.inverse();

    B_NN = updateQad_vector_b(t_Lambda_stack);
    
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
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B_NN), size_of_B_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Dn_NN_inv), size_of_Dn_NN_inv_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Qa_stack), size_of_Qa_stack_in_bytes)
    );

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_B_NN, B_NN.data(), size_of_B_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_Dn_NN_inv, Dn_NN_inv.data(), size_of_Dn_NN_inv_in_bytes, cudaMemcpyHostToDevice)
    );


    // Variable to check the result
    Eigen::MatrixXd Qa_stack_CUDA(rows_Qa_stack, cols_Qa_stack);

    // Compute Qa_stack = Dn_NN_inv*B_NN
    double alpha_cublas = 1.0;
    double beta_cublas = 0.0;
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_Dn_NN_inv, cols_B_NN, cols_Dn_NN_inv, &alpha_cublas, d_Dn_NN_inv, ld_Dn_NN_inv, d_B_NN, ld_B_NN, &beta_cublas, d_Qa_stack, ld_Qa_stack)
    );

    CUDA_CHECK(
        cudaMemcpy(Qa_stack_CUDA.data(), d_Qa_stack, size_of_Qa_stack_in_bytes, cudaMemcpyDeviceToHost));

    //FREEING MEMORY
    CUDA_CHECK(
        cudaFree(d_B_NN)
    );
    CUDA_CHECK(
        cudaFree(d_Qa_stack)
    );
    CUDA_CHECK(
        cudaFree(d_Dn_NN_inv)
    );

    return Qa_stack_CUDA;
}






int main(int argc, char *argv[])
{
/* step 1: create cublas handle, bind a stream 

    Explaination:

    The handler is an object which is used to manage the api in its threads and eventually thrown errors


    Then there are the streams. But we don't need to know what they are.
    We do not use them for now.

    If you are interested:

    Streams define the flow of data when copying.
    Imagine: We have 100 units of data (whatever it is) to copy from one place to another.
    Memory is already allocated. 
    Normal way: copy data 1, then data 2, than data 3, ..... untill the end.

    With streams, we copy data in parallel. It boils down to this.
    Here you can find a more detailed and clear explaination (with figures)

    Look only at the first 6 slides

    https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf

*/
    //  cuda blas api need CUBLAS_CHECK
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
    // qe.setZero();
    

    // Here we give some value for the strain rate
    for (unsigned int i = 0; i < number_of_Chebyshev_points; ++i) {
        Phi_stack.block<na, ne>(i*na, 0) =  Phi<na, ne>(Chebyshev_points[i]);
    }

    const auto Q_stack_CUDA = integrateQuaternions();
    std::cout << "Quaternion Integration : \n" << Q_stack_CUDA << std::endl;
    
    // const auto r_stack_CUDA = integratePosition(Q_stack_CUDA);
    // std::cout << "Position Integration : \n" << r_stack_CUDA << std::endl;

    // const auto N_stack_CUDA = integrateInternalForces(Q_stack_CUDA);
    // std::cout << "Internal Forces Integration : \n" << N_stack_CUDA << "\n" << std::endl;

    // const auto C_stack_CUDA = integrateInternalCouples(N_stack_CUDA);
    // std::cout << "Internal Couples Integration : \n" << C_stack_CUDA << "\n" << std::endl;

    // std::cout << "Internal Forces MATRIX : \n" << toMatrix(N_stack_CUDA, number_of_Chebyshev_points) << "\n" << std::endl;
    // std::cout << "Internal Couples MATRIX : \n" << toMatrix(C_stack_CUDA, number_of_Chebyshev_points) << "\n" << std::endl;
    
    // const auto Lambda_stack_CUDA = buildLambda(C_stack_CUDA, N_stack_CUDA);
    // //std::cout << "Lambda_stack : \n" << Lambda_stack_CUDA << "\n" << std::endl;

    // const auto Qa_stack_CUDA = integrateGeneralisedForces(Lambda_stack_CUDA);
    // std::cout << "Generalized Forces Integration : \n" << Qa_stack_CUDA << std::endl;

    /*
    Destry cuda objects
    */
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
