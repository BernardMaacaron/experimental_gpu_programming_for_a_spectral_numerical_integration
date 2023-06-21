#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

//  CUDA Basic Linear Algebra 
#include <cublas_v2.h>

#include "utilities.h"  //  Utilities for error handling (must be after cublas or cusolver)
#include "getCusolverErrorString.h"


#include <Eigen/Dense>

// Function to check and print cuBLAS errors
void checkCublasError(cublasStatus_t status, const char* errorMessage) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << errorMessage << " Error code: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}



int main(int argc, char *argv[])
{
    cublasHandle_t cublasH = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));


        // Define the Chebyshev points on the unit circle
    Eigen::MatrixXd A(4,2);
    A << 1, 2,
        3, 4,
        5, 6,
        7, 8;


    Eigen::VectorXd X(4);
    X<< 1,
        1;


    int m = 2;
    int n = 2;

    double* d_A = nullptr;
    int size_of_A_in_bytes = A.size()*sizeof(double);

    double* d_X = nullptr;
    int size_of_X_in_bytes = X.size()*sizeof(double);


    // K_stack parameters for GPU
    double* d_Y = nullptr;
    int size_of_Y_in_bytes = A.rows()*sizeof(double);

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_of_A_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_X, size_of_X_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, size_of_Y_in_bytes));

    // double* d_y_array[2];
    // cudaMalloc((void**)&d_y_array[0], m * sizeof(double));
    // cudaMalloc((void**)&d_y_array[1], m * sizeof(double));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), size_of_A_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, X.data(), size_of_X_in_bytes, cudaMemcpyHostToDevice));
    

    // Define stride values
    int ld_A = 4; // Leading dimension of matrix A
    // int ld_X = 2;
    // int ld_Y = 4;
    int inc_X = 1; // Increment for x
    int inc_Y = 1; // Increment for y
    long long int stride_A = 2; // Stride between matrices in A
    long long int stride_X = 0; // Stride between vectors in x
    long long int stride_Y = 2; // Stride between vectors in y

    // Perform gemvStridedBatched operation
    double alpha_cublas = 1.0; // Scalar alpha
    double beta_cublas = 0.0; // Scalar beta
    cublasStatus_t status = cublasDgemvStridedBatched(cublasH, CUBLAS_OP_N, m, n, &alpha_cublas, d_A, ld_A, stride_A,
                                            d_X, inc_X, stride_X, &beta_cublas, d_Y, inc_Y, stride_Y, 2);


    // cublasStatus_t status = cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, n, &alpha_cublas, d_A, ld_A, stride_A,
    //                                         d_X, ld_X, stride_X, &beta_cublas, d_Y, ld_Y, stride_Y, 2);


    // Check for errors
    checkCublasError(status, "cuBLAS matrix multiplication failed.");


    Eigen::VectorXd Y = Eigen::VectorXd::Zero(4);
    CUDA_CHECK(cudaMemcpy(Y.data(), d_Y, size_of_Y_in_bytes, cudaMemcpyDeviceToHost));
    std::cout<<"Y = \n" <<Y<<std::endl;
    
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));

        // Destroy the handle
    CUBLAS_CHECK(
        cublasDestroy(cublasH)
    );

    CUDA_CHECK(
        cudaDeviceReset()
    );

    return 0;
}
                                        