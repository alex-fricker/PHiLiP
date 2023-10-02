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
};

}  ///< Tests namespace
}  ///< PHiLiP namespace

#endif
