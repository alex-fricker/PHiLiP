#include "nn_lspg_rom_testing.hpp"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "ode_solver/ode_solver_factory.h"
#include "reduced_order/pod_basis_offline.h"

#include "/usr/include/python3.8/Python.h"

#include <iostream>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
NNLSPGROMTesting<dim, nstate>::NNLSPGROMTesting(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{

}

template <int dim, int nstate>
int NNLSPGROMTesting<dim, nstate>::run_test() const
{
    int num_eval_points = all_parameters->reduced_order_param.num_evaluation_points;
    std::string testing_savename = std::to_string(num_eval_points) +"_" +
        all_parameters->reduced_order_param.snapshot_type + "_testing";

    int runs = 1;
    // std::vector<int> n_snapshots{40};
    // std::vector<int> n_snapshots{15, 20, 25, 30};

    ProperOrthogonalDecomposition::ROMSnapshots<dim, nstate> testing_matrix(
        all_parameters, parameter_handler);
    testing_matrix.build_snapshot_matrix(num_eval_points, false, false);

    int result;
    Py_Initialize();
    for (int i = 0; i < runs; i++)
    {
        Parameters::AllParameters params_copy(*all_parameters);
        // params_copy.reduced_order_param.num_halton = n_snapshots[i];

        pcout << "\n\n\n\n#####################################\n"
              << "Running iteration " << i+1 << " of " << runs << "..." 
              << "\n#####################################\n\n\n\n"
              << std::endl;

        MPI_Barrier(mpi_communicator);
        result = run_iteration(testing_matrix, &params_copy);
        pcout << "\n\n\n\n#####################################\n"
              << "Done running iteration. \n#####################################" << std::endl;
    }
    Py_Finalize(); 
    return result;
}

template <int dim, int nstate>
int NNLSPGROMTesting<dim, nstate>::run_iteration(
    const ProperOrthogonalDecomposition::ROMSnapshots<dim, nstate> &testing_matrix,
    const Parameters::AllParameters *const test_parameters) const
{
    this->pcout << "Starting neural network reduced order model test" << std::endl;

    /*---------------------------------
    Preparing training and testing data
    ---------------------------------*/
    int num_eval_points = test_parameters->reduced_order_param.num_evaluation_points;

    std::string training_savename = std::to_string(test_parameters->reduced_order_param.num_halton) +
        "_" + test_parameters->reduced_order_param.snapshot_type + "_snapshots";

    std::string testing_savename = std::to_string(num_eval_points) +"_" +
        test_parameters->reduced_order_param.snapshot_type + "_testing";

    /// Snapshot matrix to train the neural network
    ProperOrthogonalDecomposition::ROMSnapshots<dim, nstate> training_snapshot_matrix(
        test_parameters, parameter_handler);

    double snapshot_generation_t1 = MPI_Wtime();
    training_snapshot_matrix.build_snapshot_matrix(test_parameters->reduced_order_param.num_halton, true, true);
    double snapshot_generation_t2 = MPI_Wtime();
    std::vector<std::string> training_pathnames = 
        training_snapshot_matrix.write_snapshot_data_to_file(training_savename);

    std::vector<std::string> testing_pathnames = 
        testing_matrix.write_snapshot_data_to_file(testing_savename);

    /*-------------------------------------------------------
    Building, training, and evaluating the neural network ROM
    -------------------------------------------------------*/

    /// Embedding the python code from pod_neural_network_rom.py
    this->pcout << "###################################\nInitializing neural network ROM" << std::endl;

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");

    // Import the neural network ROM python code
    std::string module_path = "pod_neural_network_rom";
    PyObject *nnrom_module_name = PyUnicode_DecodeFSDefault(module_path.c_str());
    if (nnrom_module_name == nullptr)
    {
        this->pcout << "Failed to generate module name." << std::endl;
        PyErr_Print();
        return 1;
    }

    PyObject *nnrom_module = PyImport_Import(nnrom_module_name);

    if (nnrom_module == nullptr)
    {
        this->pcout << "Failed to import the module." << std::endl;
        PyErr_Print();
        return 1;
    }
    Py_XDECREF(nnrom_module_name);

    PyObject *nnrom_dict = PyModule_GetDict(nnrom_module);
    if (nnrom_dict == nullptr)
    {
        this->pcout << "Failed to get the dictionary." << std::endl;
        PyErr_Print();
        return 1;
    }
    Py_XDECREF(nnrom_module);

    PyObject *nnrom_class_name = PyDict_GetItemString(nnrom_dict, "PODNeuralNetworkROM");
    if (nnrom_class_name == nullptr)
    {
        this->pcout << "Failed to get the python class." << std::endl;
        PyErr_Print();
        return 1;
    }
    Py_XDECREF(nnrom_dict);

    if (!PyCallable_Check(nnrom_class_name))
    {
        this->pcout << "PODNeuranNetworkROM class instance not callable." << std::endl;
        PyErr_Print();
        return 1;
    }
  
    /// Init a PODNeuralNetworkROM instance
    PyObject *nnrom_args = Py_BuildValue("(s)", test_parameters->solution_vtk_files_directory_name.c_str());
  
    PyObject *nnrom_instance = PyObject_CallObject(nnrom_class_name, nnrom_args);  
    if (nnrom_instance == nullptr)
    {
        this->pcout << "Failed to instantiate the python class." << std::endl;
        PyErr_Print();
        return 1;
    }
    Py_XDECREF(nnrom_class_name);
    Py_XDECREF(nnrom_args);
  
    /// Load data
    PyObject *result = PyObject_CallMethod(nnrom_instance, "load_data", "(sssi)",
        training_pathnames[0].c_str(),
        training_pathnames[1].c_str(),
        training_pathnames[2].c_str(),
        test_parameters->reduced_order_param.num_pod_modes);

    /// Setting up the neural network
    result = PyObject_CallMethod(nnrom_instance, "initialize_network", "(iidid)",
        test_parameters->reduced_order_param.architecture,
        test_parameters->reduced_order_param.epochs,
        test_parameters->reduced_order_param.learning_rate,
        test_parameters->reduced_order_param.training_batch_size,
        test_parameters->reduced_order_param.weight_decay);

    if (result == nullptr)
    {
        this->pcout << "Failed to initialize neural network class." << std::endl;
        PyErr_Print();
        return 1;
    }

    /// Build the neural network
    this->pcout << "Training neural network..." << std::endl;
    double nn_rom_train_t1 = MPI_Wtime();
    result = PyObject_CallMethod(nnrom_instance, "build_network", NULL);
    double nn_rom_train_t2 = MPI_Wtime();
    if (result == nullptr)
    {
        this->pcout << "Failed to build the neural network." << std::endl;
        PyErr_Print();
        return 1;
    }
    this->pcout << "Done training neural network." << std::endl;

    /// Evaluate the neural network
    this->pcout << "Evaluating the neural network rom..." << std::endl;
    double nn_rom_eval_t1 = MPI_Wtime();
    result = PyObject_CallMethod(nnrom_instance, "evaluate_network", "(s)", testing_pathnames[1].c_str());
    double nn_rom_eval_t2 = MPI_Wtime();
    if (result == nullptr)
    {
        this->pcout << "Failed to evaluate the neural network." << std::endl;
        PyErr_Print();
        return 1;
    }
    this->pcout << "Done evaluating the neural network rom.\n###################################\n" << std::endl;
    MPI_Barrier(mpi_communicator);
 
    Py_XDECREF(result);
    Py_XDECREF(nnrom_instance);

    /// Load NNROM solution data from the disk
    std::string filename = test_parameters->solution_vtk_files_directory_name + "/pod_nnrom_solution.txt";
    if(!std::filesystem::exists(filename))
    {
        this->pcout << "Neural network ROM solution file does not exist." << std::endl;
        return 1;
    }

    std::ifstream rom_file(test_parameters->solution_vtk_files_directory_name + "/pod_nnrom_solution.txt");
    if(!rom_file.is_open())
    {
        this->pcout << "Failed to open rom solution file." << std::endl;
        return 1;        
    }

    int solution_size = testing_matrix.pod->snapshotMatrix.rows();
    Eigen::MatrixXd nn_rom_solutions(solution_size, num_eval_points);

    std::string line;
    int i = 0;
    while (std::getline(rom_file, line))
    {
        int j = 0;
        std::istringstream iss(line);
        double value;
        while (iss >> value)
        {
            nn_rom_solutions(i, j) = value;
            ++j;
        }
        ++i;
    }

    /*------------------------------------------------------------------------
    Building and evaluating the LSPG ROM for the same test points as the NNROM
    ------------------------------------------------------------------------*/
    
    ProperOrthogonalDecomposition::ROMSnapshots<dim, nstate> lspg_rom_matrix(test_parameters, parameter_handler);
    Parameters::ODESolverParam::ODESolverEnum ode_solver_type = 
        Parameters::ODESolverParam::ODESolverEnum::pod_petrov_galerkin_solver;
    std::string lspg_rom_savename = std::to_string(test_parameters->reduced_order_param.num_halton) +
        "_" + test_parameters->reduced_order_param.snapshot_type + "_lspg_roms_solutions";

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver_petrov_galerkin = 
        FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(test_parameters, parameter_handler);
    std::shared_ptr<ProperOrthogonalDecomposition::OfflinePOD<dim>> pod_petrov_galerkin =
        std::make_shared<ProperOrthogonalDecomposition::OfflinePOD<dim>>(flow_solver_petrov_galerkin->dg);


    MPI_Barrier(mpi_communicator);
    this->pcout << "\n\n###################################\nEvaluating the lspg rom..." << std::endl;
    lspg_rom_matrix.snapshots_residual_L2_norm.resize(num_eval_points);
    double lspg_eval_time = 0;
    for (int i = 0; i < num_eval_points; ++i)
    {
        this->pcout << "\nEvaluating LSPG ROM point " << i+1 << " of " << num_eval_points << "..." << std::endl;
        Eigen::RowVectorXd test_params = testing_matrix.snapshot_points.col(i).transpose();
        const Parameters::AllParameters params = lspg_rom_matrix.reinit_parameters(test_params);

        flow_solver_petrov_galerkin = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(
            &params, parameter_handler);

        flow_solver_petrov_galerkin->ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver_manual(
            ode_solver_type, flow_solver_petrov_galerkin->dg, pod_petrov_galerkin);

        flow_solver_petrov_galerkin->ode_solver->allocate_ode_system();

        double lspg_rom_eval_t1 = MPI_Wtime();
        flow_solver_petrov_galerkin->ode_solver->steady_state();
        double lspg_rom_eval_t2 = MPI_Wtime();
        lspg_eval_time += lspg_rom_eval_t2 - lspg_rom_eval_t1;

        lspg_rom_matrix.snapshots_residual_L2_norm(i) = flow_solver_petrov_galerkin->ode_solver->residual_norm;

        dealii::LinearAlgebra::distributed::Vector<double> lspg_rom_solution;
        if (test_parameters->reduced_order_param.snapshot_type == "dg_solution")
        {
            lspg_rom_solution = flow_solver_petrov_galerkin->dg->solution;
        }
        else if (test_parameters->reduced_order_param.snapshot_type == "surface_pressure")
        {
            lspg_rom_solution = lspg_rom_matrix.get_boundary_face_pressures(flow_solver_petrov_galerkin);
        }
        else if (test_parameters->reduced_order_param.snapshot_type == "volume_pressure")
        {
            lspg_rom_solution = lspg_rom_matrix.get_cell_volume_pressures(flow_solver_petrov_galerkin);
        }
        else
        {
            pcout << "Invalid snapshot type selected." << std::endl;
            return 1;
        }
        lspg_rom_matrix.pod->addSnapshot(lspg_rom_solution);
    }

    lspg_rom_matrix.write_snapshot_data_to_file(lspg_rom_savename);
    this->pcout << "Done evaluating the lspg rom.\n###################################" << std::endl;

    if (test_parameters->reduced_order_param.save_snapshot)
    {
        output_rom_solution_to_csv(nn_rom_solutions, testing_matrix.snapshot_points, test_parameters,
            "nn_rom_solution");
        output_rom_solution_to_csv(lspg_rom_matrix.pod->snapshotMatrix, testing_matrix.snapshot_points,
            test_parameters, "lspg_rom_solution");
    }

    /*--------------------------
    Comparing NNROM and LSPG ROM
    --------------------------*/

    double snapshot_generation_time = snapshot_generation_t2 - snapshot_generation_t1;
    double nn_rom_training_time = nn_rom_train_t2 - nn_rom_train_t1;
    double nn_rom_eval_time = (nn_rom_eval_t2 - nn_rom_eval_t1) / training_snapshot_matrix.n_snapshots;
    lspg_eval_time /= training_snapshot_matrix.n_snapshots;

    pcout << "\nTotal time to generate training snapshots: " << snapshot_generation_time
          << "\nTotal time to train the neural network: " << nn_rom_training_time
          << "\nTime for one evaluation of the neural network rom: " << nn_rom_eval_time
          << "\nTime for one evaluation of the lspg rom: " << lspg_eval_time
          << std::endl;

    std::string times_filename = test_parameters->solution_vtk_files_directory_name + "/timings.txt";
    std::ofstream times_out(times_filename);
    if (!times_out.is_open())
    {
        pcout << "Failed to open timings file" << std::endl;
        return 1;
    }
    times_out << "Total time to generate training snapshots: " << snapshot_generation_time
              << "\nTotal time to train the neural network: " << nn_rom_training_time
              << "\nTime for one evaluation of the neural network rom: " << nn_rom_eval_time
              << "\nTime for one evaluation of the lspg rom: " << lspg_eval_time
              << std::endl;
    times_out.close();

    for (int i = 0; i < num_eval_points; ++i)
    {
        Eigen::RowVectorXd nn_rom_solution = nn_rom_solutions.col(i).transpose();
        Eigen::RowVectorXd fom_solution = testing_matrix.pod->snapshotMatrix.col(i).transpose();
        Eigen::RowVectorXd lspg_rom_solution = lspg_rom_matrix.pod->snapshotMatrix.col(i).transpose();

        double nn_rom_solution_error = (nn_rom_solution - fom_solution).norm() / fom_solution.norm();
        double lspg_rom_solution_error = (lspg_rom_solution - fom_solution).norm() / fom_solution.norm();

        pcout << "\nNeural network ROM test point: " << i 
              << "\n\tNN ROM Solution Error: " << nn_rom_solution_error 
              << "\n\tLSPG ROM Solution Error: " << lspg_rom_solution_error
              << std::endl;
    }

    /*------
    Clean up
    ------*/
    /// Create a new directory for the generated files
    MPI_Barrier(mpi_communicator);
    if (mpi_rank == 0)
    {
        std::string save_dir_suffix = "subsonic_surface_solutions";
        std::string src_dir = std::filesystem::current_path().string();
        std::string dst_dir = src_dir + "/" + std::to_string(test_parameters->reduced_order_param.num_halton) 
            + "snapshots_" + std::to_string(test_parameters->reduced_order_param.num_evaluation_points)
            + "pts_" + save_dir_suffix;

        std::string output_params_filename = test_parameters->solution_vtk_files_directory_name 
            + "/run_parameters.txt";
        std::ofstream output_params(output_params_filename);
        output_params << "mesh file: " << test_parameters->flow_solver_param.input_mesh_filename
                      << "\nmesh refinements: " << test_parameters->flow_solver_param.number_of_mesh_refinements
                      << "\npolynomial order: " << test_parameters->flow_solver_param.poly_degree
                      << "\nnumber of ROM DoFs: " << solution_size
                      << "\nnumber of snapshots: " << test_parameters->reduced_order_param.num_halton
                      << "\nparameter min values: " << test_parameters->reduced_order_param.parameter_min_values[0]
                      << " " << test_parameters->reduced_order_param.parameter_min_values[1]
                      << "\nparameter max values: " << test_parameters->reduced_order_param.parameter_max_values[0]
                      << " " << test_parameters->reduced_order_param.parameter_max_values[1]
                      << "\nreduced residual tolerance: " 
                      << test_parameters->reduced_order_param.reduced_residual_tolerance
                      << "\nsnapshot type: " << test_parameters->reduced_order_param.snapshot_type
                      << "\nnumber of pod modes: " << test_parameters->reduced_order_param.num_pod_modes
                      << "\narchitecture: " << test_parameters->reduced_order_param.architecture
                      << "\nepochs: " << test_parameters->reduced_order_param.epochs
                      << "\nlearning rate: " << test_parameters->reduced_order_param.learning_rate
                      << "\ntraining batch size: " << test_parameters->reduced_order_param.training_batch_size
                      << std::endl;

        int dir_counter = 0;
        bool dir_created = false;
        while (!dir_created)
        {
            if (std::filesystem::exists(dst_dir + std::to_string(dir_counter))) 
            {
                ++dir_counter;
            }
            else 
            {
                dst_dir += std::to_string(dir_counter);
                std::filesystem::create_directory(dst_dir);
                dir_created = true; 
            }
        }

        /// Move the generated files to the new directory
        for (const auto &entry : std::filesystem::directory_iterator("."))
        {
            if (entry.is_regular_file() && (entry.path().extension() == ".txt" || entry.path().extension() == ".csv"))
            {
                std::filesystem::rename(entry.path(),
                    dst_dir + "/" + entry.path().filename().string());
            }
        }       
    }
    return 0;
}

template <int dim, int nstate>
void NNLSPGROMTesting<dim, nstate>::output_rom_solution_to_csv(
    const Eigen::MatrixXd &rom_solutions,
    const Eigen::MatrixXd &eval_parameters, 
    const PHiLiP::Parameters::AllParameters *const all_parameters,
    const std::string &filename_suffix) const
{
    std::ifstream points_file(all_parameters->solution_vtk_files_directory_name + "/point_locations.txt");
    if(!points_file.is_open())
    {
        this->pcout << "Failed to open point locations file." << std::endl;        
    }
    else
    {
        Eigen::MatrixXd out_matrix(rom_solutions.rows(), dim+1);

        std::string line;
        int i = 0;
        while (std::getline(points_file, line))
        {
            int j = 0;
            std::istringstream iss(line);
            double value;
            while (iss >> value)
            {
                out_matrix(i, j) = value;
                ++j;
            }
            ++i;
        }

        const double pi = atan(1.0) * 4.0;
        const static Eigen::IOFormat csv_format(Eigen::FullPrecision, Eigen::DontAlignCols, ",", "\n");
        for (unsigned int i = 0; i < rom_solutions.cols(); i++)
        {
            double AoA = eval_parameters(1, i) * 180 / pi;
            std::string fn = all_parameters->solution_vtk_files_directory_name + "/mach_" 
            + std::to_string(eval_parameters(0, i)) + "_aoa_" + std::to_string(AoA) + "_rom_" 
            + all_parameters->reduced_order_param.snapshot_type + "_" + filename_suffix + ".csv";

            out_matrix.col(out_matrix.cols()-1) = rom_solutions.col(i);
            std::ofstream solution_out(fn);
            if (solution_out.is_open())
            {
                solution_out << out_matrix.format(csv_format);
                solution_out.close();
            }
        }
    }
    
}

#if PHILIP_DIM==1
        template class NNLSPGROMTesting<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class NNLSPGROMTesting<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}  ///< Tests namespace
}  ///< PHiLiP namespace  
