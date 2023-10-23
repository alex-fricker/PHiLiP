#ifndef _ROM_SNAPSHOTS__
#define _ROM_SNAPSHOTS__

#include <eigen/Eigen/Dense>
#include "parameters/all_parameters.h"
#include <deal.II/numerics/vector_tools.h>
#include "pod_basis_online.h"
#include <string>
#include "../flow_solver/flow_solver.h"


namespace PHiLiP {
namespace ProperOrthogonalDecomposition{

template <int dim, int nstate>
class ROMSnapshots
{
public:
    /// Constructor
    ROMSnapshots(const Parameters::AllParameters *const parameters_input,
                 const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~ROMSnapshots() {};

    /// Method to build the snapshot matrix
    void build_snapshot_matrix(const int n_snapshots);

    /// Update parameters for a new snapshot or FOM solution
    Parameters::AllParameters reinit_parameters(
        const Eigen::RowVectorXd &new_parameter) const;

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> solve_snapshot_FOM(
        const Eigen::RowVectorXd& parameter) const;

    /// POD basis
    std::shared_ptr<ProperOrthogonalDecomposition::OnlinePOD<dim>> pod;

    /// Function to write the snapshot matrix and associated snapshot points to files
    std::vector<std::string> write_snapshot_data_to_file(std::string const &save_name) const;

    /// Gets the snapshot names
    std::vector<std::string> get_pathnames(std::string const &save_name) const;

    /// Get the a Halton sequence of parameters
    Eigen::MatrixXd get_halton_points(const int &n_points);

private:
    const Parameters::AllParameters *const all_parameters;  ///< Pointer to all parameters
    const dealii::ParameterHandler &parameter_handler;   ///< Dummy parameter handler because flowsolver requires it
    int n_snapshots;  ///< Number of snapshot to be created
    int n_params;  ///< Number of parameters
    Eigen::MatrixXd snapshot_points;  ///< Parameters where each snapshot will be evaluated
    Eigen::RowVectorXd snapshots_residual_L2_norm;  ///< L2 norm of each snapshot in the matrix
    const int mpi_rank; ///< MPI rank.
    dealii::ConditionalOStream pcout;  ///< Used as std::cout, but only prints if mpi_rank == 0
    const MPI_Comm mpi_communicator; ///< MPI communicator.

    /// Selects the points in the parameters space to evaluate the snapshots at using a halton sequence
    void generate_snapshot_points_halton();

    /// Returns the pressure in each cell
    std::vector<double> get_cell_pressures(
        const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver,
        const bool only_boundary_cells,
        const bool export_pressure_vtu,
        const std::string filename = "pressure.vtu") const;
};
}  // POD namespace
}  // PHiLiP namespace

#endif
