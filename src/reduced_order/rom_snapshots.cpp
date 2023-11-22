#include "rom_snapshots.hpp"
#include "eigen/Eigen/Dense"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "halton.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "physics/euler.h"
#include "post_processor/physics_post_processor.h"
#include "post_processor/data_out_euler_faces.hpp"
#include "pod_basis_offline.h"

#include <deal.II/lac/vector_operation.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/fe/fe_dgq.h>

//#include <deal.II/fe/mapping_q1.h> // Might need mapping_q
#include <deal.II/fe/mapping_q.h> // Might need mapping_q
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/mapping_manifold.h>
#include <deal.II/fe/mapping_fe_field.h>

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <EpetraExt_Transpose_RowMatrix.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <typeinfo>

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
    , mpi_communicator(MPI_COMM_WORLD)
{
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver =
        FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(
            this->all_parameters, 
            parameter_handler);
    dummy_dg = flow_solver->dg;

    const bool compute_dRdW = true;
    flow_solver->dg->assemble_residual(compute_dRdW);
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix =
        std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();

    system_matrix->copy_from(dummy_dg->system_matrix);
    pod = std::make_shared<ProperOrthogonalDecomposition::OnlinePOD<dim>>(system_matrix);

    this-> n_params = this->all_parameters->reduced_order_param.parameter_names.size();
}

template <int dim, int nstate>
void ROMSnapshots<dim, nstate>::build_snapshot_matrix(
    const int n_snapshots, 
    const bool save_snapshot_vector,
    const bool set_domain_extremes)
{
    pcout << "Building snapshot matrix...\n" << std::endl;

    if (all_parameters->reduced_order_param.snapshot_distribution == "halton")
    {
        this->n_snapshots = n_snapshots;
        generate_snapshot_points_halton(set_domain_extremes);
    }
    else if (all_parameters->reduced_order_param.snapshot_distribution == "linear")
    {
        this->n_snapshots = n_snapshots * n_params;
        generate_snapshot_points_linear();
    }
    else
    {
        pcout << "Invalid snapshot points distribution selected." << std::endl;
        std::abort();
    }

    snapshots_residual_L2_norm.resize(n_snapshots);

    for (int snapshot_i = 0; snapshot_i < n_snapshots; snapshot_i++)
    {   
        this->pcout << "\n###################################\nSolving FOM snapshot number " 
                    << snapshot_i + 1 
                    << " of " 
                    << n_snapshots 
                    << "\n###################################\n"
                    <<std::endl;

        Eigen::RowVectorXd snapshot_params = snapshot_points.col(snapshot_i).transpose();
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> snapshot = solve_snapshot_FOM(snapshot_params);
        double residual_L2_norm = snapshot->dg->get_residual_l2norm();

        if (residual_L2_norm < all_parameters->ode_solver_param.nonlinear_steady_residual_tolerance)
        {
            snapshots_residual_L2_norm(snapshot_i) = snapshot->dg->get_residual_l2norm();
            std::string snapshot_type = all_parameters->reduced_order_param.snapshot_type;

            if (snapshot_type == "dg_solution")
            {
                pod->addSnapshot(snapshot->dg->solution);
                if (all_parameters->reduced_order_param.save_snapshot)
                {
                    snapshot->dg->output_results_vtk(snapshot->ode_solver->current_iteration);
                }
            } 
            else if (snapshot_type == "volume_pressure" || snapshot_type == "surface_pressure")
            {
                std::string filename;
                for (int i = 0; i < n_params ; i++) { filename += std::to_string(snapshot_params(i)) + "_"; }
                filename += snapshot_type;
                dealii::LinearAlgebra::distributed::Vector<double> pressures;

                if (snapshot_type == "surface_pressure") 
                { 
                    pressures = get_boundary_face_pressures(snapshot);
                }
                else if (snapshot_type == "volume_pressure")
                {
                    pressures = get_cell_volume_pressures(snapshot);
                }

                pod->addSnapshot(pressures);

                if (save_snapshot_vector)
                {
                    dealii::LinearAlgebra::ReadWriteVector<double> write_solution(snapshot->dg->solution.size());
                    write_solution.import(snapshot->dg->solution, dealii::VectorOperation::values::insert);
                    if (mpi_rank == 0)
                    {
                        std::string fn;
                        for (int i = 0; i < n_params; ++i) { fn += std::to_string(snapshot_params(i)) + "_"; }
                        fn += "solution_snapshot.txt";
                        std::ofstream out_file(fn);
                        for (unsigned int i = 0; i < snapshot->dg->solution.size(); ++i)
                        {
                            out_file << "  " << std::setprecision(17) << write_solution(i) << "\n";
                        }
                        out_file.close();
                    }
                }
            }   
            else
            {
                pcout << "Invalid snapshot type selected." << std::endl;
            }
        }
        else
        {
            snapshots_residual_L2_norm(snapshot_i) = -1;
            this->pcout << "Snapshot number " << snapshot_i
                        << " did not converge and is omitted from the snapshot matrix.\n" 
                        << std::endl;
        }
    }
    pod->computeBasis();
    pcout << "Done building snapshot matrix.\n" << std::endl;
}

template <int dim, int nstate>
dealii::LinearAlgebra::distributed::Vector<double> ROMSnapshots<dim, nstate>::get_boundary_face_pressures(
    const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver) const
{
    const dealii::Mapping<dim> &mapping = (*(flow_solver->dg->high_order_grid->mapping_fe_field));
    dealii::FEFaceValues<dim, dim> fe_face_values(
        mapping, 
        flow_solver->dg->fe_collection[all_parameters->flow_solver_param.poly_degree], 
        flow_solver->dg->face_quadrature_collection[all_parameters->flow_solver_param.poly_degree],
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    std::vector<dealii::types::global_dof_index> dofs_indices (fe_face_values.dofs_per_cell);

    const unsigned int n_quad_pts = fe_face_values.n_quadrature_points;
    std::array<double,nstate> soln_at_q;
    std::vector<double> cell_pressures;
    std::vector<std::vector<double>> quad_node_locations;
    int num_face_quads_local = 0;

    for (auto cell = flow_solver->dg->dof_handler.begin_active(); cell != flow_solver->dg->dof_handler.end(); ++cell)
    {
        if (!cell->is_locally_owned() || !cell->at_boundary()) { continue; }

        for (const auto &face_id : dealii::GeometryInfo<dim>::face_indices())
        {
            if (cell->face(face_id)->boundary_id() == 1001)
            {
                fe_face_values.reinit(cell, face_id);
                cell->get_dof_indices(dofs_indices);

                for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad)
                {   
                    ++num_face_quads_local;
                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                    for (unsigned int idof=0; idof<fe_face_values.dofs_per_cell; ++idof)
                    {
                        const unsigned int istate = fe_face_values.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += flow_solver->dg->solution[dofs_indices[idof]] * 
                            fe_face_values.shape_value_component(idof, iquad, istate);
                    }
                    double pressure = compute_pressure_at_q(soln_at_q);
                    // double pressure = compute_pressure_coeff_at_q(soln_at_q);
                    cell_pressures.push_back(pressure);

                    dealii::Point<dim> quad_pt = fe_face_values.quadrature_point(iquad);
                    std::vector<double> quad_pt_loc(dim);
                    for (int i = 0; i < dim; i++) { quad_pt_loc[i] = quad_pt[i]; }
                    quad_node_locations.push_back(quad_pt_loc);
                }
            }
        }
    }
    MPI_Barrier(mpi_communicator);

    int num_face_quads_global = 0;
    MPI_Allreduce(&num_face_quads_local, &num_face_quads_global, 1, MPI_INT, MPI_SUM, mpi_communicator);

    dealii::LinearAlgebra::distributed::Vector<double> cell_pressures_dealii = 
        build_distributed_vector(num_face_quads_global, cell_pressures);

    output_quad_locations_to_file(num_face_quads_global, quad_node_locations);

    if (all_parameters->reduced_order_param.save_snapshot)
    {
        const double pi = atan(1.0) * 4.0;
        double AoA = flow_solver->dg->all_parameters->euler_param.angle_of_attack * 180 / pi;
        std::string fn = all_parameters->solution_vtk_files_directory_name + "/mach_" 
            + std::to_string(flow_solver->dg->all_parameters->euler_param.mach_inf)
            + "_aoa_" + std::to_string(AoA) + "_fom_" 
            + all_parameters->reduced_order_param.snapshot_type + "_solution.csv";
        output_solution_to_csv(fn, num_face_quads_global, quad_node_locations, cell_pressures);
    }

    return cell_pressures_dealii;
}

template <int dim, int nstate>
dealii::LinearAlgebra::distributed::Vector<double> ROMSnapshots<dim, nstate>::get_cell_volume_pressures(
    const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver) const
{
    const dealii::Mapping<dim> &mapping = (*(flow_solver->dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim, dim> fe_values(
        mapping, 
        flow_solver->dg->fe_collection[all_parameters->flow_solver_param.poly_degree], 
        flow_solver->dg->volume_quadrature_collection[all_parameters->flow_solver_param.poly_degree],
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values.dofs_per_cell);

    const unsigned int n_quad_pts = fe_values.n_quadrature_points;
    std::array<double,nstate> soln_at_q;
    std::vector<double> cell_pressures;
    std::vector<std::vector<double>> quad_node_locations;
    int num_quads_local = 0;

    for (auto cell = flow_solver->dg->dof_handler.begin_active(); cell != flow_solver->dg->dof_handler.end(); ++cell)
    {
        if (!cell->is_locally_owned()) { continue; }
        fe_values.reinit(cell);
        cell->get_dof_indices(dofs_indices);

        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad)
        {   
            ++num_quads_local;
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
            for (unsigned int idof=0; idof<fe_values.dofs_per_cell; ++idof)
            {
                const unsigned int istate = fe_values.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += flow_solver->dg->solution[dofs_indices[idof]] * 
                    fe_values.shape_value_component(idof, iquad, istate);
            }
            double pressure = compute_pressure_at_q(soln_at_q);
            // double pressure = compute_pressure_coeff_at_q(soln_at_q);
            cell_pressures.push_back(pressure);

            dealii::Point<dim> quad_pt = fe_values.quadrature_point(iquad);
            std::vector<double> quad_pt_loc(dim);
            for (int i = 0; i < dim; i++) { quad_pt_loc[i] = quad_pt[i]; }
            quad_node_locations.push_back(quad_pt_loc);
        }
    }
    MPI_Barrier(mpi_communicator);
    
    int num_quads_global = 0;
    MPI_Allreduce(&num_quads_local, &num_quads_global, 1, MPI_INT, MPI_SUM, mpi_communicator);

    dealii::LinearAlgebra::distributed::Vector<double> cell_pressures_dealii = 
        build_distributed_vector(num_quads_global, cell_pressures);

    output_quad_locations_to_file(num_quads_global, quad_node_locations);

    if (all_parameters->reduced_order_param.save_snapshot)
    {
        const double pi = atan(1.0) * 4.0;
        double AoA = flow_solver->dg->all_parameters->euler_param.angle_of_attack * 180 / pi;
        std::string fn = all_parameters->solution_vtk_files_directory_name + "/mach_" 
            + std::to_string(flow_solver->dg->all_parameters->euler_param.mach_inf)
            + "_aoa_" + std::to_string(AoA) + "_fom_" 
            + all_parameters->reduced_order_param.snapshot_type + "_solution.csv";
        output_solution_to_csv(fn, num_quads_global, quad_node_locations, cell_pressures);
    }

    return cell_pressures_dealii;
}

template <int dim, int nstate>
double ROMSnapshots<dim, nstate>::compute_pressure_coeff_at_q(
    const std::array<double ,nstate> &conservative_soln) const
{
    const double density = conservative_soln[0];
    // const double tot_energy = conservative_soln[nstate-1];

    dealii::Tensor<1, dim, double> vel;
    for (int d = 0; d < dim; d++) { vel[d] = conservative_soln[d+1] / density; }

    double vel2 = 0.0;
    for (int d = 0; d < dim; d++) 
    { 
        vel2 = vel2 + vel[d] * vel[d]; 
    }

    // double gamm1 = all_parameters->euler_param.gamma_gas - 1.0;
    // double pressure = gamm1 * (tot_energy - 0.5 * density * vel2);    

    // double Minf = all_parameters->euler_param.mach_inf;
    // double gamma_gas = all_parameters->euler_param.gamma_gas;
    // double pressure_inf = 1.0 / (gamma_gas * Minf * Minf);
    // double cp = (pressure - pressure_inf) / 0.5 ;
    double cp = 1 - vel2;

    return cp;
}

template <int dim, int nstate>
double ROMSnapshots<dim, nstate>::compute_pressure_at_q(
    const std::array<double ,nstate> &conservative_soln) const
{
    const double density = conservative_soln[0];
    const double tot_energy = conservative_soln[nstate-1];

    dealii::Tensor<1, dim, double> vel;
    for (int d = 0; d < dim; d++) { vel[d] = conservative_soln[d+1] / density; }

    double vel2 = 0.0;
    for (int d = 0; d < dim; d++) 
    { 
        vel2 = vel2 + vel[d] * vel[d]; 
    }

    double gamm1 = all_parameters->euler_param.gamma_gas - 1.0;
    double pressure = gamm1 * (tot_energy - 0.5 * density * vel2);    

    return pressure;
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
void ROMSnapshots<dim, nstate>::generate_snapshot_points_halton(const bool set_extremes)
{
    if (set_extremes)
    {
        snapshot_points.resize(n_params, n_snapshots);
        const double pi = atan(1.0) * 4.0;

        // Set snapshot points on extremes of the domain
        for (int j = 0; j < n_params; j++)
        {
            snapshot_points(j, 0) = this->all_parameters->reduced_order_param.parameter_min_values[j];
            snapshot_points(j, n_snapshots-1) = this->all_parameters->reduced_order_param.parameter_max_values[j];
            if (this->all_parameters-> reduced_order_param.parameter_names[j] == "alpha")
            {
                snapshot_points(j, 0) *= pi / 180;  // Convert parameter to radians
                snapshot_points(j, n_snapshots-1) *= pi / 180;
            }
        }

        // Use a halton sequence to fill in the remainder of the domain
        double *halton_seq_value = nullptr;
        for (int i = 1; i < n_snapshots - 1; i++)
        {
            halton_seq_value = ProperOrthogonalDecomposition::halton(i, n_params);

            for (int j = 0; j < n_params; j++)
            {
                snapshot_points(j, i) =
                    halton_seq_value[j] * (this->all_parameters->reduced_order_param.parameter_max_values[j] - 
                                            this->all_parameters->reduced_order_param.parameter_min_values[j]) +
                    this->all_parameters->reduced_order_param.parameter_min_values[j];

                if (this->all_parameters-> reduced_order_param.parameter_names[j] == "alpha")
                {
                    snapshot_points(j, i) *= pi / 180;  // Convert parameter to radians
                }
            }
        }
        delete [] halton_seq_value;
    }
    else
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
}

template <int dim, int nstate>
void ROMSnapshots<dim, nstate>::generate_snapshot_points_linear()
{
    const double pi = atan(1.0) * 4.0;
    snapshot_points.resize(n_params, n_snapshots);

    for (int j = 0; j < n_params; j++)
    {
        double start = this->all_parameters->reduced_order_param.parameter_min_values[j];
        double end = this->all_parameters->reduced_order_param.parameter_max_values[j];

        double step = (end - start ) / (n_snapshots - 1);
        for (int i = 0; i < n_snapshots; i++)
        {
            snapshot_points(j, i) = start + i * step;
            pcout << this->all_parameters->reduced_order_param.parameter_names[j] << ": " << snapshot_points(i, j) << std::endl;

            if (this->all_parameters->reduced_order_param.parameter_names[j] == "alpha")
            {
                snapshot_points(j, i) *= pi / 180;  // Convert parameter to radians
            }
        }
    }
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
Eigen::MatrixXd ROMSnapshots<dim, nstate>::get_snapshot_points(const int &n_points, const bool set_domain_extremes)
{
    this->n_snapshots = n_points;

    if (all_parameters->reduced_order_param.snapshot_distribution == "halton")
    {
        generate_snapshot_points_halton(set_domain_extremes);
    }
    else if (all_parameters->reduced_order_param.snapshot_distribution == "linear")
    {
        generate_snapshot_points_linear();
    }
    else
    {
        pcout << "Invalid snapshot points distribution selected." << std::endl;
        std::abort();
    }

    return snapshot_points;
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
void ROMSnapshots<dim, nstate>::output_quad_locations_to_file(
    const int &num_quads_global,
    const std::vector<std::vector<double>> &quad_locations_local) const
{
    std::vector<std::vector<std::vector<double>>> quad_locations_global = 
        dealii::Utilities::MPI::all_gather(mpi_communicator, quad_locations_local);

    dealii::LAPACKFullMatrix<double> data_output_matrix(num_quads_global, dim);
    int row = 0;
    for (unsigned int proc = 0; proc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); proc++)
    {
        std::vector<std::vector<double>> quad_locations_local = quad_locations_global[proc];

        if (quad_locations_local.empty()) { continue; }

        for (size_t m = 0; m < quad_locations_local.size(); m++)
        {
            for (int n = 0; n < dim; n++)
            {
                data_output_matrix.set(row, n, quad_locations_local[m][n]);
            }
            ++row;
        }
    }
    std::string fn = all_parameters->solution_vtk_files_directory_name + "/point_locations.txt";
    std::ofstream quad_point_locations(fn);
    unsigned int precision = 16;
    data_output_matrix.print_formatted(quad_point_locations, precision);
    quad_point_locations.close();
}

template <int dim, int nstate>
dealii::LinearAlgebra::distributed::Vector<double> ROMSnapshots<dim, nstate>::build_distributed_vector(
    const int &num_quads_global,
    const std::vector<double> &cell_pressures_local) const
{
    std::vector<std::vector<double>> cell_pressures_global = 
        dealii::Utilities::MPI::all_gather(mpi_communicator, cell_pressures_local);

    dealii::LinearAlgebra::distributed::Vector<double> cell_pressures_dealii(num_quads_global);
    int idx = 0;
    for (std::vector<double> pressure_vector_local : cell_pressures_global)
    {
        for (size_t i = 0; i < pressure_vector_local.size(); i++)
        {
            cell_pressures_dealii(idx) = pressure_vector_local[i];
            ++idx;
        }
    }
    cell_pressures_dealii.update_ghost_values();

    return cell_pressures_dealii;
}

template <int dim, int nstate>
void ROMSnapshots<dim, nstate>::output_solution_to_csv(
    const std::string &filename,
    const int &num_quads_global,
    const std::vector<std::vector<double>> &quad_locations_local,
    const std::vector<double> &cell_pressures_local) const
{
    std::vector<std::vector<double>> cell_pressures_global = 
        dealii::Utilities::MPI::all_gather(mpi_communicator, cell_pressures_local);

    std::vector<std::vector<std::vector<double>>> quad_locations_global = 
        dealii::Utilities::MPI::all_gather(mpi_communicator, quad_locations_local);

    Eigen::MatrixXd solution(num_quads_global, dim+1);
    int row = 0;
    for (unsigned int proc = 0; proc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); proc++)
    {
        std::vector<std::vector<double>> quad_locations_local = quad_locations_global[proc];
        std::vector<double> pressure_vector_local = cell_pressures_global[proc];

        if (quad_locations_local.empty()) { continue; }

        for (size_t m = 0; m < quad_locations_local.size(); m++)
        {
            for (int n = 0; n < dim; n++)
            {
                solution(row, n) = quad_locations_local[m][n];
            }
            solution(row, dim) = pressure_vector_local[m];
            ++row;
        }
    }

    const static Eigen::IOFormat csv_format(Eigen::FullPrecision, Eigen::DontAlignCols, ",", "\n");
    std::ofstream solution_out(filename);
    if (solution_out.is_open())
    {
        solution_out << solution.format(csv_format);
        solution_out.close();
    }
}

#if PHILIP_DIM!=1
    template class ROMSnapshots<PHILIP_DIM, PHILIP_DIM+2>;
    // template class DataOutAirfoilSurface<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}  // POD Namespace
}  // PHiLiP Namespace
