#ifndef GLOBALS_H
#define GLOBALS_H

#include <Eigen/Dense> 
#include <algorithm>

//Imported from Andrea's Code - src/include/CROSP/polynomial_representation/polynomial_representation.hpp

//  A vector representing the allowed deformation of the rod (default kirkhoff rod)
    std::array<bool, 6> m_admitted_deformations {true, true, true, false, false, false };


    // //  The number of allowed deformation (asserted from the vector)
    // static const unsigned int na { [&]()->unsigned int{ return std::count(m_admitted_deformations.begin(),
    //                                                                 m_admitted_deformations.end(), true);}() };

    // //  Number of elastic modes per allowed deformation
    // std::vector<unsigned int> m_number_of_modes_stack { default_number_of_modes };
    // unsigned int m_total_number_of_modes { static_cast<unsigned int>(std::accumulate(m_number_of_modes_stack.begin(),
    //                                                                                     m_number_of_modes_stack.end(), 0)) };

    // //  The polynomial base used to discretize the strain field
    // PolynomialBase m_polynomial_base { legendre_polynomial_base };

    //  The matrix mapping the allowed strain into the full strain field
    Eigen::MatrixXd B()
    {
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);

        std::vector<int> indexes;
        std::for_each(m_admitted_deformations.begin(),
                    m_admitted_deformations.end(),
                    [&indexes, index=0](const bool dof)mutable{   if(dof == true)
                                                                        indexes.push_back(index);
                                                                    index++;});
        const Eigen::MatrixXd map = I(Eigen::all, indexes);
        return map;
    };

//Andrea's code - END

constexpr unsigned int na = 3;  //  Kirkhoff rod
constexpr unsigned int ne = 3;  // dimesion of qe
// constexpr unsigned int num_ch_nodes = 5;

static const unsigned int number_of_Chebyshev_points = 16;



static const unsigned int quaternion_state_dimension = 4;
static const unsigned int position_dimension = 3;
static const unsigned int quaternion_problem_dimension = quaternion_state_dimension * (number_of_Chebyshev_points-1);

static const unsigned int lambda_dimension = 6;
static const unsigned int half_lambda_problem_dimension = lambda_dimension * (number_of_Chebyshev_points-1) *0.5;

static const unsigned int Qa_dimension = 9;


// CUDA specific variables
cusolverDnHandle_t cusolverH = NULL;
cublasHandle_t cublasH = NULL;
const auto size_of_double = sizeof(double);


// Defining qe in the CPU and its GPU parameters
// Eigen::Matrix<double, ne*na, 1> qe;

Eigen::VectorXd qe = Eigen::VectorXd::Zero(ne*na);
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
int size_of_K_stack_in_bytes = 3 * number_of_Chebyshev_points * size_of_double;

#endif