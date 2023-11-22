#ifndef _ROM_SNAPSHOTS__
#define _ROM_SNAPSHOTS__

#include "../flow_solver/flow_solver.h"
#include "parameters/all_parameters.h"
#include "pod_basis_online.h"

#include <eigen/Eigen/Dense>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out_faces.h>

#include <string>




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
    void build_snapshot_matrix(
        const int n_snapshots, 
        const bool save_snapshot_vector,
        const bool set_domain_extremes);

    /// Update parameters for a new snapshot or FOM solution
    Parameters::AllParameters reinit_parameters(
        const Eigen::RowVectorXd &new_parameter) const;

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> solve_snapshot_FOM(
        const Eigen::RowVectorXd& parameter) const;

    /// Function to write the snapshot matrix and associated snapshot points to files
    std::vector<std::string> write_snapshot_data_to_file(std::string const &save_name) const;

    /// Gets the snapshot names
    std::vector<std::string> get_pathnames(std::string const &save_name) const;

    /// Get the a Halton sequence of parameters
    Eigen::MatrixXd get_snapshot_points(const int &n_points, const bool set_domain_extremes);

    /// Returns the pressure at the face quadrature nodes on faces at the boundary
    dealii::LinearAlgebra::distributed::Vector<double> get_boundary_face_pressures(
        const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver) const;

    /// Returns the pressure at the volume quadrature nodes
    dealii::LinearAlgebra::distributed::Vector<double> get_cell_volume_pressures(
        const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver) const;

    /// Output the quad node locations and pressure values to a csv file
    void output_solution_to_csv(
        const std::string &filename,
        const int &num_quads_global,
        const std::vector<std::vector<double>> &quad_locations_local,
        const std::vector<double> &cell_pressures_local) const;

    Eigen::MatrixXd snapshot_points;  ///< Parameters where each snapshot will be evaluated
    std::shared_ptr<ProperOrthogonalDecomposition::OnlinePOD<dim>> pod;  ///< POD basis
    int n_snapshots;  ///< Number of snapshot to be created
    int n_params;  ///< Number of parameters

private:
    const Parameters::AllParameters *const all_parameters;  ///< Pointer to all parameters
    const dealii::ParameterHandler &parameter_handler;   ///< Dummy parameter handler because flowsolver requires it
    Eigen::RowVectorXd snapshots_residual_L2_norm;  ///< L2 norm of each snapshot in the matrix
    const int mpi_rank; ///< MPI rank.
    dealii::ConditionalOStream pcout;  ///< Used as std::cout, but only prints if mpi_rank == 0
    const MPI_Comm mpi_communicator;  ///< MPI communicator.
    std::shared_ptr<DGBase<dim,double>> dummy_dg;  ///< dummy DG for reading snapshot data from file

    /// Selects the points in the parameters space to evaluate the snapshots at using a halton sequence
    void generate_snapshot_points_halton(const bool set_extremes);

    /// Selects the points in the parameter space to evaluate the snapshots at using a linear distribution
    void generate_snapshot_points_linear();

    /// Computes pressure at quadrature node q
    double compute_pressure_at_q(
        const std::array<double ,nstate> &conservative_soln) const;

    /// Computes pressure coefficicent at quadrature node q
    double compute_pressure_coeff_at_q(
        const std::array<double ,nstate> &conservative_soln) const;

    /// Output the quadrature node locations to file (for post-processing)
    void output_quad_locations_to_file(
        const int &num_quads_global,
        const std::vector<std::vector<double>> &quad_locations_local) const;

    /// Build a dealii distributed vector
    dealii::LinearAlgebra::distributed::Vector<double> build_distributed_vector(
        const int &num_quads_global,
        const std::vector<double> &cell_pressures_local) const;


};

}  // POD namespace
}  // PHiLiP namespace

#endif
