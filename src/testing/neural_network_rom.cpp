#include "neural_network_rom.hpp"
#include <iostream>
#include <filesystem>
#include "parameters/all_parameters.h"
#include <eigen/Eigen/Dense>
#include "reduced_order/rom_snapshots.hpp"
#include "/usr/include/python3.8/Python.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
NeuralNetworkROM<dim, nstate>::NeuralNetworkROM(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{

}

template <int dim, int nstate>
int NeuralNetworkROM<dim, nstate>::run_test() const
{
    this->pcout << "Starting neural network reduced order model test"
                << std::endl;

    ProperOrthogonalDecomposition::ROMSnapshots<dim, nstate> snapshot_matrix(
        all_parameters,
        parameter_handler);
    snapshot_matrix.build_snapshot_matrix(all_parameters->reduced_order_param.num_halton);
    snapshot_matrix.write_snapshot_data_to_file();

    // Run the python script for the neural network ROM


    return 0;
}

#if PHILIP_DIM==1
        template class NeuralNetworkROM<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class NeuralNetworkROM<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}  ///< Tests namespace
}  ///< PHiLiP namespace  
