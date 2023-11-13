#include "rom_snapshots.hpp"
#include "eigen/Eigen/Dense"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "halton.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "physics/euler.h"
#include "post_processor/physics_post_processor.h"
#include "post_processor/data_out_euler_faces.hpp"

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

    const bool compute_dRdW = true;
    flow_solver->dg->assemble_residual(compute_dRdW);
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> system_matrix =
        std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();

    system_matrix->copy_from(flow_solver->dg->system_matrix);
    pod = std::make_shared<ProperOrthogonalDecomposition::OnlinePOD<dim>>(system_matrix);

    this-> n_params = this->all_parameters->reduced_order_param.parameter_names.size();
}

template <int dim, int nstate>
void ROMSnapshots<dim, nstate>::build_snapshot_matrix(const int n_snapshots)
{
    this->n_snapshots = n_snapshots;

    snapshots_residual_L2_norm.resize(n_snapshots);
    generate_snapshot_points_halton();
    for (int snapshot_i = 0; snapshot_i < n_snapshots; snapshot_i++)
    {   
        Eigen::RowVectorXd snapshot_params = snapshot_points.col(snapshot_i).transpose().segment(0, n_params);
        this->pcout << "\n###################################\nSolving FOM snapshot number " 
                    << snapshot_i + 1 
                    << " of " 
                    << n_snapshots 
                    << "\n###################################\n"
                    <<std::endl;
        std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> snapshot = solve_snapshot_FOM(snapshot_params);
        double residual_L2_norm = snapshot->dg->get_residual_l2norm();
        if (residual_L2_norm < all_parameters->ode_solver_param.nonlinear_steady_residual_tolerance)
        {
            snapshots_residual_L2_norm(snapshot_i) = snapshot->dg->get_residual_l2norm();
            std::string snapshot_type = all_parameters->reduced_order_param.snapshot_type;

            if (snapshot_type == "dg_solution")
            {
                pod->addSnapshot(snapshot->dg->solution);
                if (all_parameters->reduced_order_param.save_snapshot_vtk)
                {
                    snapshot->dg->output_results_vtk(snapshot->ode_solver->current_iteration);
                }
            } 
            else if (snapshot_type == "volume_pressure" || snapshot_type == "surface_pressure")
            {
                std::string filename;
                for (int i = 0; i < n_params ; i++) { filename += std::to_string(snapshot_params(i)) + "_"; }
                filename += snapshot_type;
                // std::vector<double> pressures_vector;
                dealii::LinearAlgebra::distributed::Vector<double> pressures;

                if (snapshot_type == "surface_pressure") 
                { 
                    // pressures_vector = get_boundary_face_pressures(snapshot, filename);
                    pressures = get_boundary_face_pressures(snapshot, filename);
                }
                // else if (snapshot_type == "volume_pressure")
                // {
                //     // pressures_vector = get_cell_volume_pressures(snapshot, filename);
                //     pressures = get_cell_volume_pressures(snapshot, filename);
                // }

                // dealii::LinearAlgebra::distributed::Vector<double> pressures_dealii(
                //     pressures_vector.size());
                // for (size_t i = 0; i < pressures_vector.size(); i++)
                // {
                //     pressures_dealii(i) = pressures_vector[i];
                // }

                // pod->addSnapshot(pressures_dealii);
                pod->addSnapshot(pressures);
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
}

template <int dim, int nstate>
dealii::LinearAlgebra::distributed::Vector<double> ROMSnapshots<dim, nstate>::get_boundary_face_pressures(
    const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver,
    const std::string &filename) const
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

    std::vector<std::vector<double>> cell_pressures_global = 
        dealii::Utilities::MPI::all_gather(mpi_communicator, cell_pressures);
    std::vector<std::vector<std::vector<double>>> quad_node_locations_global = 
        dealii::Utilities::MPI::all_gather(mpi_communicator, quad_node_locations);

    dealii::LinearAlgebra::distributed::Vector<double> cell_pressures_dealii(num_face_quads_global);
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

    dealii::LAPACKFullMatrix<double> data_output_matrix(num_face_quads_global, dim + 1);
    int row = 0;
    for (unsigned int proc = 0; proc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); proc++)
    {
        std::vector<double> pressure_vector_local = cell_pressures_global[proc];
        std::vector<std::vector<double>> quad_node_locations_local = quad_node_locations_global[proc];

        if (pressure_vector_local.empty()) { continue; }

        for (size_t m = 0; m < pressure_vector_local.size(); m++)
        {
            for (int n = 0; n < dim; n++)
            {
                data_output_matrix.set(row, n, quad_node_locations_local[m][n]);
            }
            // data_output_matrix.set(row, dim, pressure_vector_local[m]);
            ++row;
        }
    }

    std::string fn = all_parameters->solution_vtk_files_directory_name + "/point_locations.txt";
    std::ofstream snapshot_data_file(fn);
    unsigned int precision = 16;
    data_output_matrix.print_formatted(snapshot_data_file, precision);
    snapshot_data_file.close();

    if (all_parameters->reduced_order_param.save_snapshot_vtk)
    {        
        output_surface_solution_vtk(cell_pressures_dealii, flow_solver, mapping, filename);
        flow_solver->dg->output_face_results_vtk(0, 0);
        flow_solver->dg->output_results_vtk(0, 0);
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
Eigen::MatrixXd ROMSnapshots<dim, nstate>::get_halton_points(const int &n_points)
{
    this->n_snapshots = n_points;
    generate_snapshot_points_halton();
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



template<int dim, int nstate>
DataOutAirfoilSurface<dim, nstate>::DataOutAirfoilSurface(
    const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver)
    : flow_solver(flow_solver)
    , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank==0)
{
}

template <int dim, int nstate>
typename DataOutAirfoilSurface<dim, nstate>::FaceDescriptor
DataOutAirfoilSurface<dim, nstate>::first_face()
{
    pcout << "inside first_face" << std::endl;
    for (auto cell = flow_solver->dg->dof_handler.begin_active(); cell != flow_solver->dg->dof_handler.end(); ++cell)
    {
        if (!cell->is_locally_owned() || !cell->at_boundary()) { continue; }
        for (const auto &face_id : dealii::GeometryInfo<dim>::face_indices())
        {
            if (cell->face(face_id)->boundary_id() == 1001)
            {
                bool foo = cell->face(face_id)->boundary_id() != dealii::numbers::internal_face_boundary_id;
                pcout << "SHOULD BE TRUE: " << foo << std::endl;
                pcout << "Returning found cell from first_face" << face_id << std::endl; 
                return FaceDescriptor(cell, face_id); 
            }
            else { continue; }
        }
    }
    pcout << "returning empty FaceDescriptor from first_face" << std::endl;
    return FaceDescriptor();
}

template <int dim, int nstate>
typename DataOutAirfoilSurface<dim, nstate>::FaceDescriptor
DataOutAirfoilSurface<dim, nstate>::next_face(const FaceDescriptor &old_face)
{
    pcout << "inside next_face" << std::endl;
    FaceDescriptor face = old_face;
    Assert(face.first->is_locally_owned(), dealii::ExcInternalError());
    Assert(face.first->at_boundary(), dealii::ExcInternalError());
    for (auto face_id = face.second + 1; face_id < dealii::GeometryInfo<dimension>::faces_per_cell; ++face_id)
    {
        if (face.first->face(face_id)->boundary_id() == 1001)
        {
            face.second = face_id;
            bool foo = face.first->face(face_id)->boundary_id() != dealii::numbers::internal_face_boundary_id;
            pcout << "SHOULD BE TRUE: " << foo << std::endl;
            pcout << "returning found next face in current cell from next_face" << face_id << std::endl;
            return face;
        }
        else { continue; }

    }

    pcout << "did not find next face in current cell moving to other cells" << std::endl;
    auto active_cell = face.first;
    ++active_cell;
  
    for (; active_cell != flow_solver->dg->dof_handler.end(); ++active_cell)
    {
        if (!active_cell->is_locally_owned() || !active_cell->at_boundary()) { continue; }
        pcout << active_cell->at_boundary() << std::endl;
        for (const auto &face_id : dealii::GeometryInfo<dim>::face_indices())
        {
            if (active_cell->face(face_id)->boundary_id() == 1001)
            {
                face.first = active_cell;
                face.second = face_id;
                bool foo = face.first->face(face_id)->boundary_id() != dealii::numbers::internal_face_boundary_id;
                pcout << "SHOULD BE TRUE: " << foo << std::endl;
                pcout << "returning next valid face in new cell from next_face" << face_id << std::endl;
                return face; 
            }
            else { continue; }
        }
    }   

    face.first  = flow_solver->dg->dof_handler.end();
    face.second = 0;
    pcout << "fell off edge returning invalid pointer" << std::endl;
    return face;
    return FaceDescriptor();
}

template <int dim, int nstate>
void ROMSnapshots<dim, nstate>::output_volume_solution_vtk(
    const dealii::LinearAlgebra::distributed::Vector<double> &pressures,
    const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver,
    const dealii::Mapping<dim> &mapping,
    const std::string &filename) const
{
    dealii::DataOut<dim, dealii::DoFHandler<dim>> data_out;
    data_out.attach_dof_handler(flow_solver->dg->dof_handler);

    pcout << "create Postprocess" << std::endl;
    const std::unique_ptr< dealii::DataPostprocessor<dim> > post_processor = 
        Postprocess::PostprocessorFactory<dim>::create_Postprocessor(all_parameters);
    pcout << "add post process" << std::endl;
    data_out.add_data_vector(flow_solver->dg->solution, *post_processor);

    std::vector<std::string> position_names;
    for(int d=0;d<dim;++d) {
        if (d==0) position_names.push_back("x");
        if (d==1) position_names.push_back("y");
        if (d==2) position_names.push_back("z");
    }
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, dealii::DataComponentInterpretation::component_is_scalar);
    data_out.add_data_vector (flow_solver->dg->high_order_grid->dof_handler_grid, flow_solver->dg->high_order_grid->volume_nodes, position_names, data_component_interpretation);

    pcout << "PRESSURES SIZE: " << pressures.size() << std::endl;
    int foo = 0;
    int nonzero = 0;
    for (size_t i = 0; i < pressures.size(); i++)
    {
        pcout << pressures(i) << std::endl;
        if (pressures(i) != 0) {++nonzero;}
        ++foo;
    }
    pcout << "TEST COUNTER: " << foo << " NONZERO ELEMS: " << nonzero << std::endl;
    // dealii::Vector<double> pressures_dealii(pressures.begin(), pressures.end());
    // pcout << "add pressures" << std::endl;
    // const std::string name = "Volume Node Pressure";
    // data_out.add_data_vector(pressures_dealii, name,
    //     dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_cell_data);

    pcout << "build_patches" << std::endl;
    data_out.build_patches(mapping, all_parameters->flow_solver_param.grid_degree);

    pcout << "write vtus" << std::endl;
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    std::string fn = this->all_parameters->solution_vtk_files_directory_name + "/proc" +
        dealii::Utilities::int_to_string(mpi_rank, 4) + "_" + filename + ".vtu";
    std::ofstream output(fn);
    data_out.write_vtu(output);

    pcout << "write pvtus" << std::endl;
    if (mpi_rank == 0)
    {
        std::vector<std::string> filenames;
        unsigned int nproc = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
        for (unsigned int iproc = 0; iproc < nproc; iproc++)
        {
            filenames.push_back("proc" + dealii::Utilities::int_to_string(iproc, 4) + "_" + filename + ".vtu");
        }
        std::string master_fn = this->all_parameters->solution_vtk_files_directory_name + "/" + 
            filename + ".pvtu";
        std::ofstream master_output(master_fn);
        data_out.write_pvtu_record(master_output, filenames);
    }   
}

template <int dim, int nstate>
void ROMSnapshots<dim, nstate>::output_surface_solution_vtk(
    const dealii::LinearAlgebra::distributed::Vector<double> &pressures,
    const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver,
    const dealii::Mapping<dim> &mapping,
    const std::string &filename) const
{
    pcout << "PRESSURES SIZE: " << pressures.size() << std::endl;
    // std::shared_ptr<DGBase<dim, double>> dg = DGFactory<dim, double>::create_discontinuous_galerkin(
    //     all_parameters,
    //     all_parameters->flow_solver_param.poly_degree,
    //     all_parameters->flow_solver_param.max_poly_degree_for_adaptation,
    //     all_parameters->flow_solver_param.grid_degree,
    //     flow_solver->flow_solver_case->generate_grid());

    // dealii::DataOutFaces<dim, dealii::DoFHandler<dim>> data_out;
    // DataOutAirfoilSurface<dim, dealii::DoFHandler<dim>> data_out;
    pcout << "init" << std::endl;
    DataOutAirfoilSurface<dim, nstate> data_out(flow_solver);
    pcout << "attach dof handler" << std::endl;
    data_out.attach_dof_handler(flow_solver->dg->dof_handler);

    // std::vector<std::string> position_names;
    // for(int d=0;d<dim;++d) 
    // {
    //     if (d==0) position_names.push_back("x");
    //     if (d==1) position_names.push_back("y");
    //     if (d==2) position_names.push_back("z");
    // }
    // std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> 
    //     data_component_interpretation(dim, dealii::DataComponentInterpretation::component_is_scalar);
    // data_out.add_data_vector (flow_solver->dg->high_order_grid->dof_handler_grid, 
    //     flow_solver->dg->high_order_grid->volume_nodes, 
    //     position_names, data_component_interpretation);

    //     dealii::Vector<float> subdomain(flow_solver->dg->triangulation->n_active_cells());
    // for (unsigned int i = 0; i < subdomain.size(); ++i) 
    // {
    //     subdomain(i) = flow_solver->dg->triangulation->locally_owned_subdomain();
    // }
    // data_out.add_data_vector(subdomain, std::string("subdomain"),
    //     dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_cell_data);

    // pcout << "create Postprocess" << std::endl;
    const std::unique_ptr< dealii::DataPostprocessor<dim> > post_processor = 
        Postprocess::PostprocessorFactory<dim>::create_Postprocessor(all_parameters);
    pcout << "add post process" << std::endl;
    data_out.add_data_vector(flow_solver->dg->solution, *post_processor);

    // dealii::Vector<double> pressures_dealii(pressures.begin(), pressures.end());
    // pcout << "add pressures" << std::endl;
    // data_out.add_data_vector(pressures, std::string("Surface Pressure"),
    //     dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim-1,dim>::DataVectorType::type_cell_data);

    // const bool write_higher_order_cells = false;//(dim>1 && grid_degree > 1) ? true : false;
    // dealii::DataOutBase::VtkFlags vtkflags(
    //     all_parameters->ode_solver_param.initial_time_step,
    //     flow_solver->ode_solver->current_iteration,
    //     true,
    //     dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::best_compression,
    //     write_higher_order_cells);
    // data_out.set_flags(vtkflags);
    pcout << "build_patches" << std::endl;
    data_out.build_patches(mapping, all_parameters->flow_solver_param.grid_degree);

    pcout << "write vtus" << std::endl;
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    std::string fn = this->all_parameters->solution_vtk_files_directory_name + "/proc" +
        dealii::Utilities::int_to_string(mpi_rank, 4) + "_" + filename + ".vtu";
    std::ofstream output(fn);
    data_out.write_vtu(output);

    pcout << "write pvtus" << std::endl;
    if (mpi_rank == 0)
    {
        std::vector<std::string> filenames;
        unsigned int nproc = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
        for (unsigned int iproc = 0; iproc < nproc; iproc++)
        {
            filenames.push_back("proc" + dealii::Utilities::int_to_string(iproc, 4) + "_" + filename + ".vtu");
        }
        std::string master_fn = this->all_parameters->solution_vtk_files_directory_name + "/" + 
            filename + ".pvtu";
        std::ofstream master_output(master_fn);
        data_out.write_pvtu_record(master_output, filenames);
    }   
}

// template <int dim, int nstate>
// dealii::LinearAlgebra::distributed::Vector<double> ROMSnapshots<dim, nstate>::get_cell_volume_pressures(
//     const std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> &flow_solver,
//     const std::string &filename) const
// {

//     using DoFHandlerType = typename dealii::DoFHandler<dim>;
//     static const unsigned int dimension = DoFHandlerType::dimension;
//     static const unsigned int space_dimension = DoFHandlerType::space_dimension;
//     typename dealii::Triangulation<dimension, space_dimension>::active_cell_iterator
//         cell = flow_solver->dg->triangulation->begin_active();
//     int counter = 0;
//     int counter2 = 0;
//     for (; cell != flow_solver->dg->triangulation->end(); ++cell)
//     {
//         if (cell->is_locally_owned())
//         {
//             for (const unsigned int f : dealii::GeometryInfo<dimension>::face_indices())
//             {
//                 pcout << "FACE ID: " << cell->face(f)->boundary_id() << std::endl;
//                 if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1001)
//                 {
//                     ++counter;
//                 }
//                 if (cell->face(f)->boundary_id() != 1004 || cell->face(f)->boundary_id() != dealii::numbers::internal_face_boundary_id)
//                 {
//                     ++counter2;
//                 }
//             }
//         }
//     }
//     pcout << "COUNTER: " << counter << " COUNTER2: " << counter2 << std::endl;

//     const dealii::Mapping<dim> &mapping = (*(flow_solver->dg->high_order_grid->mapping_fe_field));
//     dealii::FEValues<dim,dim> fe_values(
//         mapping, 
//         flow_solver->dg->fe_collection[all_parameters->flow_solver_param.poly_degree], 
//         flow_solver->dg->volume_quadrature_collection[all_parameters->flow_solver_param.poly_degree],
//         dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
//     std::vector<dealii::types::global_dof_index> dofs_indices (fe_values.dofs_per_cell);

//     const unsigned int n_quad_pts = fe_values.n_quadrature_points;
//     std::array<double,nstate> soln_at_q;
//     // std::vector<double> cell_pressures;
//     dealii::LinearAlgebra::distributed::Vector<double> cell_pressures(flow_solver->dg->triangulation->n_global_active_cells());
//     int local_element_i = 0;
//     int foo = 0;

//     for (auto cell = flow_solver->dg->dof_handler.begin_active(); cell!=flow_solver->dg->dof_handler.end(); ++cell)
//     {
//         ++foo;
//         if (!cell->is_locally_owned()) { continue; }
//         fe_values.reinit(cell);
//         cell->get_dof_indices(dofs_indices);

//         for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad)
//         {   
//             std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
//             for (unsigned int idof=0; idof<fe_values.dofs_per_cell; ++idof)
//             {
//                 const unsigned int istate = fe_values.get_fe().system_to_component_index(idof).first;
//                 soln_at_q[istate] += flow_solver->dg->solution[dofs_indices[idof]] * 
//                     fe_values.shape_value_component(idof, iquad, istate);
//             }
//             // cell_pressures.local_element(local_element_i) = compute_pressure_at_q(soln_at_q);
//             unsigned int global_id = cell->active_cell_index();
//             cell_pressures(global_id + iquad) = compute_pressure_at_q(soln_at_q);
//             ++local_element_i;
//         }
//     }
//     cell_pressures.compress(dealii::VectorOperation::add);
//     cell_pressures.update_ghost_values();
//     pcout << "Foo: " << foo << std::endl;

//     if (all_parameters->reduced_order_param.save_snapshot_vtk)
//     {
//         flow_solver->dg->output_results_vtk(flow_solver->ode_solver->current_iteration);
//         output_volume_solution_vtk(cell_pressures, flow_solver, mapping, filename);
//     }
//     return cell_pressures;
// }

#if PHILIP_DIM!=1
    template class ROMSnapshots<PHILIP_DIM, PHILIP_DIM+2>;
    // template class DataOutAirfoilSurface<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}  // POD Namespace
}  // PHiLiP Namespace
