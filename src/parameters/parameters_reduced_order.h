#ifndef __PARAMETERS_REDUCED_ORDER_H__
#define __PARAMETERS_REDUCED_ORDER_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to reduced-order model
class ReducedOrderModelParam
{
public:

    /// Tolerance for POD adaptation
    double adaptation_tolerance;

    /// Path to search for snapshots or saved POD basis
    std::string path_to_search;

    /// Tolerance of the reduced-order nonlinear residual
    double reduced_residual_tolerance;

    /// Number of Halton sequence points to add to initial snapshot set
    int num_halton;

    /// Recomputation parameter for adaptive sampling algorithm
    int recomputation_coefficient;

    /// Names of parameters
    std::vector<std::string> parameter_names;

    /// Minimum value of parameters
    std::vector<double> parameter_min_values;

    /// Maximum value of parameters
    std::vector<double> parameter_max_values;

    /// Save the csv/vkt file (depends on snapshot type) of each snapshot
    bool save_snapshot;

    /// Neural network ROM parameters ///
    int num_pod_modes;  ///< Number of POD modes to use
    int architecture;  ///< Neural network architecture to use
    int epochs;  ///< Number of epochs for training
    double learning_rate;  ///< Learning rate for training
    int training_batch_size;  ///< Batch size for training
    double weight_decay;  ///< Adds a penalty to the L2 norm of the weights in the loss function
    bool recompute_training_snapshot_matrix;  ///< Recompute or use existing training matrix
    int num_evaluation_points;  ///< Number of points to test the rom at
    std::string snapshot_type;   ///< Type of data to build the snapshot matrix with
    std::string snapshot_distribution;   ///< Distribution of points for snapshot generation  

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);

};

} // Parameters namespace
} // PHiLiP namespace
#endif
