#include "rom_snapshots.hpp"
#include "eigen/Eigen/Dense"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "halton.h"
#include <deal.II/lac/vector_operation.h>
#include <iostream>
#include <filesystem>

namespace PHiLiP {
namespace ProperOrthogonalDecomposition{


template <int dim, int nstate>
ROMSnapshots<dim, nstate>::ROMSnapshots(const Parameters::AllParameters *const parameters_input,
                                        const dealii::ParameterHandler &parameter_handler_input)
    : all_parameters(parameters_input)
    , parameter_handler(parameter_handler_input)
    , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank==0)
{
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(all_parameters, parameter_handler);
    const bool compute_dRdW = true;
    flow_solver->dg->assemble_residual(compute_dRdW);
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
    system_matrix->copy_from(flow_solver->dg->system_matrix);
    pod = std::make_shared<ProperOrthogonalDecomposition::OnlinePOD<dim>>(system_matrix);
}

template <int dim, int nstate>
void ROMSnapshots<dim, nstate>::write_snapshot_data_to_file() const
{
    std::ofstream snapshot_matrix_file(std::to_string(n_snapshots) + "_snapshot_matrix.txt");
    unsigned int precision = 16;
    pod->dealiiSnapshotMatrix.print_formatted(snapshot_matrix_file, precision);
    snapshot_matrix_file.close();

    const static Eigen::IOFormat csv_format(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream snapshots_points_file(std::to_string(n_snapshots) + "_snapshot_points.txt");
    if (snapshots_points_file.is_open())
    {
        snapshots_points_file << snapshot_points.format(csv_format);
        snapshots_points_file.close();
    }
}

template <int dim, int nstate>
void ROMSnapshots<dim, nstate>::build_snapshot_matrix(const unsigned int n_snapshots)
{
    this->n_snapshots = n_snapshots;
    snapshot_points = generate_snapshot_points_halton();

    for (unsigned int i=0; i < n_snapshots; i++)
    {
        Eigen::RowVectorXd snapshot_params = snapshot_points.col(i).transpose();
        dealii::LinearAlgebra::distributed::Vector<double> snapshot = solve_snapshot_FOM(snapshot_params);
        pod->addSnapshot(snapshot);
    }
    pod->computeBasis();
}

template <int dim, int nstate>
dealii::LinearAlgebra::distributed::Vector<double> ROMSnapshots<dim, nstate>::solve_snapshot_FOM(const Eigen::RowVectorXd& parameter) const{
    this->pcout << "Solving FOM at " << parameter << std::endl;
    Parameters::AllParameters params = reinit_parameters(parameter);

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&params, parameter_handler);

    // Solve implicit solution
    auto ode_solver_type = Parameters::ODESolverParam::ODESolverEnum::implicit_solver;
    flow_solver->ode_solver =  PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(ode_solver_type, flow_solver->dg);
    flow_solver->ode_solver->allocate_ode_system();
    flow_solver->run();

    this->pcout << "Done solving FOM." << std::endl;
    return flow_solver->dg->solution;
}

template <int dim, int nstate>
Eigen::MatrixXd ROMSnapshots<dim, nstate>::generate_snapshot_points_halton()
{
    int n_params = all_parameters->reduced_order_param.parameter_names.size();
    Eigen::MatrixXd snapshot_points(n_params, n_snapshots);

    double *halton_seq_value = nullptr;
    for (unsigned int i=0; i < n_snapshots; i++)
    {
        halton_seq_value = ProperOrthogonalDecomposition::halton(i, n_params);

        for (int j=0; j < n_params; j++)
        {
            snapshot_points(i, j) = (
                halton_seq_value[j] * (all_parameters->reduced_order_param.parameter_min_values[j] - 
                                        all_parameters->reduced_order_param.parameter_max_values[j]) +
                all_parameters->reduced_order_param.parameter_min_values[j]
            );
        }
    }
    delete [] halton_seq_value;

    return snapshot_points;
}

template <int dim, int nstate>
Parameters::AllParameters ROMSnapshots<dim, nstate>::reinit_parameters(const Eigen::RowVectorXd &new_parameter) const
{
    // Copy all parameters
    PHiLiP::Parameters::AllParameters updated_parameters = *(this->all_parameters);

    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_parameters->flow_solver_param.flow_case_type;

    if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot)
    {
        if (all_parameters->reduced_order_param.parameter_names.size() == 1)
        {
            if (all_parameters->reduced_order_param.parameter_names[0] == "rewienski_a")
            {
                updated_parameters.burgers_param.rewienski_a = new_parameter(0);
            }
            else if (all_parameters->reduced_order_param.parameter_names[0] == "rewienski_b")
            {
                updated_parameters.burgers_param.rewienski_b = new_parameter(0);
            }
        }
        else if(all_parameters->reduced_order_param.parameter_names.size() == 1)
        {
            updated_parameters.burgers_param.rewienski_a = new_parameter(0);
            updated_parameters.burgers_param.rewienski_b = new_parameter(1);
        }
        else {
            this->pcout << "Too many parameters specified for Burgers Rewienski snapshot case." << std::endl;
        }
    }
    else if (flow_type == FlowCaseEnum::naca0012)
    {
        if (all_parameters->reduced_order_param.parameter_names.size() == 1)
        {
            if (all_parameters->reduced_order_param.parameter_names[0] == "mach")
            {
                updated_parameters.euler_param.mach_inf = new_parameter(0);
            }
            else if (all_parameters->reduced_order_param.parameter_names[0] == "alpha")
            {
                updated_parameters.euler_param.angle_of_attack = new_parameter(0); //radians!
            }
        }
        else if (all_parameters->reduced_order_param.parameter_names.size() == 2)
        {
            updated_parameters.euler_param.mach_inf = new_parameter(0);
            updated_parameters.euler_param.angle_of_attack = new_parameter(1); //radians!
        }
        else 
        {
            this->pcout << "Too many parameters specified for NACA0012 case." << std::endl;
        }
    }
    else if (flow_type == FlowCaseEnum::gaussian_bump)
    {
        if(all_parameters->reduced_order_param.parameter_names.size() == 1)
        {
            if(all_parameters->reduced_order_param.parameter_names[0] == "mach")
            {
                updated_parameters.euler_param.mach_inf = new_parameter(0);
            }
        }
        else 
        {
            this->pcout << "Too many parameters specified for Gaussian bump case." << std::endl;
        }
    }
    else
    {
        this->pcout << "Invalid flow case. You probably forgot to specify a flow case in the prm file." << std::endl;
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
