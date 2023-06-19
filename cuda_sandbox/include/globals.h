#ifndef GLOBALS_H
#define GLOBALS_H

#include <Eigen/Dense>

constexpr unsigned int na = 3;  //  Kirkhoff rod
constexpr unsigned int ne = 3;  // dimesion of qe
constexpr unsigned int num_ch_nodes = 5;

static const unsigned int number_of_Chebyshev_points = 16;



static const unsigned int quaternion_state_dimension = 4;
static const unsigned int position_dimension = 3;
static const unsigned int quaternion_problem_dimension = quaternion_state_dimension * (number_of_Chebyshev_points-1);

static const unsigned int lambda_dimension = 6;

static const unsigned int Qa_dimension = 9;


// CUDA specific variables
cusolverDnHandle_t cusolverH = NULL;
cublasHandle_t cublasH = NULL;
const auto size_of_double = sizeof(double);


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
double* d_Phi_stack = nullptr;
int size_of_Phi_stack_in_bytes = (na * number_of_Chebyshev_points) * (na * ne) * size_of_double;


// K_stack parameters for GPU
double* d_K_stack = nullptr;
int size_of_K_stack_in_bytes = na * number_of_Chebyshev_points * size_of_double;

#endif