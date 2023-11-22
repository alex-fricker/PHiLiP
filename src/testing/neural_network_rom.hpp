#ifndef __NEURAL_NETWORK_ROM__
#define __NEURAL_NETWORK_ROM__

#include "tests.h"
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
class NeuralNetworkROM: public TestsBase
{
public:
    /// Constructor
    NeuralNetworkROM(const PHiLiP::Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~NeuralNetworkROM() {};

    /// Run the test
    int run_test() const;

private:
    const dealii::ParameterHandler &parameter_handler;  ///< Parameter handler for storing the .prm file being ran
    Eigen::MatrixXd testing_parameters;  ///< Parameters for testing the network
    Eigen::MatrixXd testing_snapshots;  ///< Snapshot for testing the network

    /// Outputs the ROM solution to a csv file to view in paraview
    void output_nnrom_solution_to_csv(
        const Eigen::MatrixXd &rom_solutions,
        const Eigen::MatrixXd &eval_parameters, 
        const PHiLiP::Parameters::AllParameters *const all_parameters) const;
};

}  ///< Tests namespace
}  ///< PHiLiP namespace

#endif
