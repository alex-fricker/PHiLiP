#ifndef __NN_LSPG_ROM_TESTING__
#define __NN_LSPG_ROM_TESTING__

#include "tests.h"
#include "parameters/all_parameters.h"
#include "reduced_order/rom_snapshots.hpp"
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
class NNLSPGROMTesting: public TestsBase
{
public:
    /// Constructor
    NNLSPGROMTesting(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~NNLSPGROMTesting() {};

    /// Run the test
    int run_test() const;

    int run_iteration(
        const ProperOrthogonalDecomposition::ROMSnapshots<dim, nstate> &testing_matrix,
        const Parameters::AllParameters *const test_parameters) const;

private:
    const dealii::ParameterHandler &parameter_handler;  ///< Parameter handler for storing the .prm file being ran
    Eigen::MatrixXd testing_parameters;  ///< Parameters for testing the network
    Eigen::MatrixXd testing_snapshots;  ///< Snapshot for testing the network

    /// Outputs the ROM solution to a csv file to view in paraview
    void output_rom_solution_to_csv(
        const Eigen::MatrixXd &rom_solutions,
        const Eigen::MatrixXd &eval_parameters, 
        const PHiLiP::Parameters::AllParameters *const all_parameters,
        const std::string &filename_suffix) const;
};

}  ///< Tests namespace
}  ///< PHiLiP namespace

#endif
