#include "rom_snapshots.hpp"
#include "eigen/Eigen/Dense"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "halton.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "physics/euler.h"
#include <deal.II/lac/vector_operation.h>
#include <iostream>
#include <filesystem>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition{


template <int dim, int nstate>
ROMSnapshots<dim, nstate>::ROMSnapshots(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : all_parameters(parameters_input)
    , parameter_handler(parameter_handler_input)
    , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank==0)
{
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver =
        FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(
            this->all_parameters, 
            parameter_handler);

    const bool compute_dRdW = true;
    flow_solver->dg->assemble_residual(compute_dRdW);
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix =
        std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();

    system_matrix->copy_from(flow_solver->dg->system_matrix);
    pod = std::make_shared<ProperOrthogonalDecomposition::OnlinePOD<dim>>(system_matrix);

    this-> n_params = this->all_parameters->reduced_order_param.parameter_names.size();
}

template <int dim, int nstate>
std::vector<std::string> ROMSnapshots<dim, nstate>::get_pathnames(std::string const &save_name) const
{
    std::string snapshots_path = save_name + "_matrix.txt";
    std::string parameters_path = save_name + "_parameters.txt";
    std::string residuals_path = save_name + "_residuals.txt";
    return std::vector<std::string> {snapshots_path, parameters_path, residuals_path};
}

template <int dim, int nstate>
std::vector<std::string> ROMSnapshots<dim, nstate>::write_snapshot_data_to_file(std::string const &save_name) const
{
    std::vector<std::string> pathnames = get_pathnames(save_name);

    std::ofstream snapshots_matrix_file(pathnames[0]);
    unsigned int precision = 16;
    pod->dealiiSnapshotMatrix.print_formatted(snapshots_matrix_file, precision);
    snapshots_matrix_file.close();

    const static Eigen::IOFormat csv_format(Eigen::FullPrecision, Eigen::DontAlignCols, "  ", "\n");
    std::ofstream snapshots_points_file(pathnames[1]);
    if (snapshots_points_file.is_open())
    {
        snapshots_points_file << snapshot_points.format(csv_format);
        snapshots_points_file.close();
    }

    std::ofstream snapshots_residual_file(pathnames[2]);
    if (snapshots_residual_file.is_open())
    {
        snapshots_residual_file << snapshots_residual_L2_norm.format(csv_format);
        snapshots_residual_file.close();
    }
    return pathnames;
}

template <int dim, int nstate>
void ROMSnapshots<dim, nstate>::build_snapshot_matrix(const int n_snapshots)
{
    this->n_snapshots = n_snapshots;
    this->pcout << "\nBuilding snapshot matrix for "
                << n_snapshots
                << " snapshots."
                << std::endl;

    snapshots_residual_L2_norm.resize(n_snapshots);
    generate_snapshot_points_halton();
    for (int i = 0; i < n_snapshots; i++)
    {   
        Eigen::RowVectorXd snapshot_params = snapshot_points.col(i).transpose().segment(0, n_params);
        this->pcout << "\n###################################\nSolving FOM snapshot number " 
                    << i + 1 
                    << " of " 
                    << n_snapshots 
                    << "\n###################################\n"
                    <<std::endl;
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> snapshot = solve_snapshot_FOM(snapshot_params);
        double residual_L2_norm = snapshot->dg->get_residual_l2norm();
        if (residual_L2_norm < all_parameters->ode_solver_param.nonlinear_steady_residual_tolerance)
        {
            if (all_parameters->reduced_order_param.snapshot_type == "dg_solution")
            {
                pod->addSnapshot(snapshot->dg->solution);
                snapshots_residual_L2_norm(i) = snapshot->dg->get_residual_l2norm();
            } 
            else if (all_parameters->reduced_order_param.snapshot_type == "pressure")
            {
                std::vector<double> cell_pressures_vector = get_cell_pressures(snapshot);
                dealii::LinearAlgebra::distributed::Vector<double> cell_pressures_dealii(cell_pressures_vector.size());
                for (size_t i = 0; i < cell_pressures_vector.size(); i++)
                {
                    cell_pressures_dealii(i) = cell_pressures_vector[i];
                }
                pod->addSnapshot(cell_pressures_dealii);
                snapshots_residual_L2_norm(i) = snapshot->dg->get_residual_l2norm();
            }
        }
        else
        {
            snapshots_residual_L2_norm(i) = -1;
            this->pcout << "Snapshot number " << i
                        << " did not converge and is omitted from the snapshot matrix.\n" 
                        << std::endl;
        }
    }
    pod->computeBasis();
}

template <int dim, int nstate>
std::vector<double> ROMSnapshots<dim, nstate>::get_cell_pressures(
    const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver) const
{
    Physics::Euler<dim,nstate,double> euler_physics_double = Physics::Euler<dim, nstate, double>(
        all_parameters,
        all_parameters->euler_param.ref_length,
        all_parameters->euler_param.gamma_gas,
        all_parameters->euler_param.mach_inf,
        all_parameters->euler_param.angle_of_attack,
        all_parameters->euler_param.side_slip_angle);

    int overintegrate = 2;
    dealii::QGauss<dim> quad_extra(flow_solver->dg->max_degree+1+overintegrate);
    const dealii::Mapping<dim> &mapping = (*(flow_solver->dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_extra(mapping, 
        flow_solver->dg->fe_collection[all_parameters->flow_solver_param.poly_degree], 
        quad_extra, 
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    std::array<double,nstate> soln_at_q;
    std::vector<double> cell_pressures;

    for (auto cell = flow_solver->dg->dof_handler.begin_active(); cell!=flow_solver->dg->dof_handler.end(); ++cell)
    {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad)
        {
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof)
            {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += flow_solver->dg->solution[dofs_indices[idof]] * 
                    fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            cell_pressures.push_back(euler_physics_double.compute_pressure(soln_at_q));
        }
    }
    return cell_pressures;
}

template <int dim, int nstate>
std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>>
ROMSnapshots<dim, nstate>::solve_snapshot_FOM(const Eigen::RowVectorXd& parameter) const
{
    Parameters::AllParameters params = reinit_parameters(parameter);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver =
        FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);

    // Solve implicit solution
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::implicit_solver;
    flow_solver->ode_solver =
        PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(
            ode_solver_type,
            flow_solver->dg);

    flow_solver->ode_solver->allocate_ode_system();
    flow_solver->run();
    return flow_solver;
}

template <int dim, int nstate>
Eigen::MatrixXd ROMSnapshots<dim, nstate>::get_halton_points(const int &n_points)
{
    this->n_snapshots = n_points;
    generate_snapshot_points_halton();
    return snapshot_points;
}

template <int dim, int nstate>
void ROMSnapshots<dim, nstate>::generate_snapshot_points_halton()
{
    snapshot_points.resize(n_params, n_snapshots);
    const double pi = atan(1.0) * 4.0;

    double *halton_seq_value = nullptr;
    for (int i = 1; i <= n_snapshots; i++)
    {
        halton_seq_value = ProperOrthogonalDecomposition::halton(i, n_params);

        for (int j = 0; j < n_params; j++)
        {
            snapshot_points(j, i-1) =
                halton_seq_value[j] * (this->all_parameters->reduced_order_param.parameter_max_values[j] - 
                                        this->all_parameters->reduced_order_param.parameter_min_values[j]) +
                this->all_parameters->reduced_order_param.parameter_min_values[j];

            if (this->all_parameters-> reduced_order_param.parameter_names[j] == "alpha")
            {
                snapshot_points(j, i-1) *= pi / 180;  // Convert parameter to radians
            }
        }
    }
    delete [] halton_seq_value;
}

template <int dim, int nstate>
Parameters::AllParameters ROMSnapshots<dim, nstate>::reinit_parameters(
    const Eigen::RowVectorXd &new_parameter) const
{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters updated_parameters = *(this->all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;

    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot)
    {
        if (this->all_parameters->reduced_order_param.parameter_names.size() == 1)
        {
            if (this->all_parameters->reduced_order_param.parameter_names[0] == "rewienski_a")
            {
                updated_parameters.burgers_param.rewienski_a = new_parameter(0);
            }
            else if (this->all_parameters->reduced_order_param.parameter_names[0] == "rewienski_b")
            {
                updated_parameters.burgers_param.rewienski_b = new_parameter(0);
            }
        }
        else if(this->all_parameters->reduced_order_param.parameter_names.size() == 1)
        {
            updated_parameters.burgers_param.rewienski_a = new_parameter(0);
            updated_parameters.burgers_param.rewienski_b = new_parameter(1);
        }
        else {
            pcout << "Too many parameters specified for Burgers Rewienski snapshot case." << std::endl;
        }
    }
    else if (flow_type == FlowCaseEnum::naca0012)
    {
        if (this->all_parameters->reduced_order_param.parameter_names.size() == 1)
        {
            if (this->all_parameters->reduced_order_param.parameter_names[0] == "mach")
            {
                updated_parameters.euler_param.mach_inf = new_parameter(0);
            }
            else if (this->all_parameters->reduced_order_param.parameter_names[0] == "alpha")
            {
                updated_parameters.euler_param.angle_of_attack = new_parameter(0); //radians!
            }
        }
        else if (this->all_parameters->reduced_order_param.parameter_names.size() == 2)
        {
            updated_parameters.euler_param.mach_inf = new_parameter(0);
            updated_parameters.euler_param.angle_of_attack = new_parameter(1); //radians!
        }
        else 
        {
            pcout << "Too many parameters specified for NACA0012 case." << std::endl;
        }
    }
    else if (flow_type == FlowCaseEnum::gaussian_bump)
    {
        if(this->all_parameters->reduced_order_param.parameter_names.size() == 1)
        {
            if(this->all_parameters->reduced_order_param.parameter_names[0] == "mach")
            {
                updated_parameters.euler_param.mach_inf = new_parameter(0);
            }
        }
        else 
        {
            pcout << "Too many parameters specified for Gaussian bump case." << std::endl;
        }
    }
    else
    {
        pcout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
        std::abort();
    }
    return updated_parameters;
}


#if PHILIP_DIM==1
        template class ROMSnapshots<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class ROMSnapshots<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}  // POD Namespace
}  // PHiLiP Namespace
