#include "neural_network_rom.hpp"
#include "parameters/all_parameters.h"
#include "reduced_order/rom_snapshots.hpp"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "functional/functional.h"
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

    std::vector<std::string> training_pathnames;
    std::string training_savename = std::to_string(all_parameters->reduced_order_param.num_halton) +
        "_" + all_parameters->reduced_order_param.snapshot_type + "_snapshots";

    ProperOrthogonalDecomposition::ROMSnapshots<dim, nstate> snapshot_matrix(
        all_parameters,
        parameter_handler);

    if (all_parameters->reduced_order_param.recompute_training_snapshot_matrix)
    {
        // Generate the snapshot matrix to train the neural network
        snapshot_matrix.build_snapshot_matrix(all_parameters->reduced_order_param.num_halton);
        training_pathnames = snapshot_matrix.write_snapshot_data_to_file(training_savename);
    } 
    else
    {
        training_pathnames = snapshot_matrix.get_pathnames(training_savename);
    }

    int num_eval_points = all_parameters->reduced_order_param.num_evaluation_points;
    // int num_params = all_parameters->reduced_order_param.parameter_names.size();

    std::string testing_savename = std::to_string(all_parameters->reduced_order_param.num_evaluation_points) +
        "_" + all_parameters->reduced_order_param.snapshot_type + "_testing";

    ProperOrthogonalDecomposition::ROMSnapshots<dim, nstate> testing_matrix(
        all_parameters, 
        parameter_handler);
    // testing_matrix.build_snapshot_matrix(num_eval_points);
    // std::vector<std::string> testing_pathnames = testing_matrix.write_snapshot_data_to_file(testing_savename);
    std::vector<std::string> testing_pathnames = testing_matrix.get_pathnames(testing_savename);

    // Embedding the python code from pod_neural_network_rom.py
    int fuck = 0;

    this->pcout << "\n\nInitializing neural network ROM" << std::endl;
    Py_Initialize();

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    // Import the neural network ROM python code
    std::string module_path = "pod_neural_network_rom";
    PyObject *nnrom_module_name = PyUnicode_DecodeFSDefault(module_path.c_str());
    if (nnrom_module_name == nullptr)
    {
        this->pcout << "Failed to generate module name." << std::endl;
        PyErr_Print();
        return 1;
    }

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    PyObject *nnrom_module = PyImport_Import(nnrom_module_name);

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    if (nnrom_module == nullptr)
    {
        this->pcout << "Failed to import the module." << std::endl;
        PyErr_Print();
        return 1;
    }
    Py_XDECREF(nnrom_module_name);

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    PyObject *nnrom_dict = PyModule_GetDict(nnrom_module);
    if (nnrom_dict == nullptr)
    {
        this->pcout << "Failed to get the dictionary." << std::endl;
        PyErr_Print();
        return 1;
    }
    Py_XDECREF(nnrom_module);

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    PyObject *nnrom_class_name = PyDict_GetItemString(nnrom_dict, "PODNeuralNetworkROM");
    if (nnrom_class_name == nullptr)
    {
        this->pcout << "Failed to get the python class." << std::endl;
        PyErr_Print();
        return 1;
    }
    Py_XDECREF(nnrom_dict);
    
    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    if (!PyCallable_Check(nnrom_class_name))
    {
        this->pcout << "PODNeuranNetworkROM class instance not callable." << std::endl;
        PyErr_Print();
        return 1;
    }
    
    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    // Init a PODNeuralNetworkROM instance
    PyObject *nnrom_args = Py_BuildValue("(s)", all_parameters->solution_vtk_files_directory_name.c_str());
    
    PyObject *nnrom_instance = PyObject_CallObject(nnrom_class_name, nnrom_args);  
    if (nnrom_instance == nullptr)
    {
        this->pcout << "Failed to instantiate the python class." << std::endl;
        PyErr_Print();
        return 1;
    }
    Py_XDECREF(nnrom_class_name);
    Py_XDECREF(nnrom_args);
    
    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    // Load data
    PyObject *result = PyObject_CallMethod(nnrom_instance, "load_data", "(sssi)",
        training_pathnames[0].c_str(),
        training_pathnames[1].c_str(),
        training_pathnames[2].c_str(),
        all_parameters->reduced_order_param.num_pod_modes);

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    // Setting up the neural network
    result = PyObject_CallMethod(nnrom_instance, "initialize_network", "(iidid)",
        all_parameters->reduced_order_param.architecture,
        all_parameters->reduced_order_param.epochs,
        all_parameters->reduced_order_param.learning_rate,
        all_parameters->reduced_order_param.training_batch_size,
        all_parameters->reduced_order_param.weight_decay);

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    if (result == nullptr)
    {
        this->pcout << "Failed to initialize neural network class." << std::endl;
        PyErr_Print();
        return 1;
    }

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    // Build the neural network
    result = PyObject_CallMethod(nnrom_instance, "build_network", NULL);
    if (result == nullptr)
    {
        this->pcout << "Failed to build the neural network." << std::endl;
        PyErr_Print();
        return 1;
    }

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    // Evaluate the neural network
    result = PyObject_CallMethod(nnrom_instance, "evaluate_network", "(s)", testing_pathnames[1].c_str());

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;

    Py_XDECREF(result);
    Py_XDECREF(nnrom_instance);
    Py_Finalize();

    pcout << "fuck: " << fuck << std::endl;
    ++fuck;   

    int solution_size = snapshot_matrix.pod->snapshotMatrix.rows();
    Eigen::MatrixXd rom_solutions(solution_size, num_eval_points);

    std::ifstream rom_file(all_parameters->solution_vtk_files_directory_name + "/pod_nnrom_solution.txt");
    if(!rom_file.is_open())
    {
        this->pcout << "Failed to open rom solution file." << std::endl;
        return 1;        
    }

    std::string line;
    int i = 0;
    while (std::getline(rom_file, line))
    {
        int j = 0;
        std::istringstream iss(line);
        double value;
        while (iss >> value)
        {
            rom_solutions(i, j) = value;
            ++j;
        }
        ++i;
    }

    double solution_error_tol = 1e-3;
    bool test_pass = true;
    for (int i = 0; i < num_eval_points; ++i)
    {
        Eigen::RowVectorXd rom_solution = testing_matrix.pod->snapshotMatrix.col(i).transpose();
        Eigen::RowVectorXd fom_solution = snapshot_matrix.pod->snapshotMatrix.col(i).transpose();
        double solution_error = (rom_solution - fom_solution).norm() / fom_solution.norm();

        if (solution_error > solution_error_tol) { test_pass = false;}

        pcout << "Neural network ROM test point: " << i << " error: " << solution_error << std::endl;

    }





        // bool functional_error_fail = false;
        // bool solution_error_fail = false;

        // while (int i = 0 < num_eval_points && !functional_error_fail && !solution_error_fail)
        // {
        //     Eigen::RowVectorXd test_parameters_rom = rom_testing_parameters.col(i).transpose();

        //     /// Evaluate the neural network
        //     PyObject *params_list = PyList_New(num_params);
        //     for (int j = 0; j < num_params; j++)
        //     {
        //         pcout << "Test paramerter: " << test_parameters_rom(j) << std::endl;
        //         int return_code = PyList_SetItem(params_list, j, PyLong_FromLong(test_parameters_rom(j)));
        //         if (return_code < 0)
        //         {
        //             this->pcout << "Failed to build parameters list." << std::endl;
        //             PyErr_Print();
        //             return 1;
        //         }
        //     }

        //     result = PyObject_CallMethod(nnrom_instance, "evaluate_network", "O", params_list);
        //     if (result == nullptr)
        //     {
        //         this->pcout << "Failed to evaluate neural network for parameters."
        //                     << test_parameters_rom
        //                     << std::endl;
        //         PyErr_Print();
        //         return 1;
        //     }
        //     Py_XDECREF(params_list);

        //     std::ifstream file("pod_nnrom_solution.txt");
        //     if (!file.is_open())
        //     {
        //         this->pcout << "Failed to open rom solution file." << std::endl;
        //         return 1;
        //     }
        //     else
        //     {
        //         PHiLiP::Parameters::AllParameters test_parameters_philip = snapshot_matrix.reinit_parameters(
        //             test_parameters_rom);
        //         std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> rom_flow_solver =
        //             FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(
        //                 &test_parameters_philip, 
        //                 parameter_handler);

        //         dealii::LinearAlgebra::ReadWriteVector<double> rom_solution_temp(rom_flow_solver->dg->solution.size());
        //         double value;
        //         int i = 0;
        //         while (file >> value)
        //         {
        //             rom_solution_temp[i] = value;
        //             i++;
        //         }
        //         file.close();

        //         rom_flow_solver->dg->solution.import(rom_solution_temp, dealii::VectorOperation::insert);
        //         dealii::LinearAlgebra::distributed::Vector<double> rom_solution(rom_flow_solver->dg->solution);
        //         auto rom_functional = FunctionalFactory<dim, nstate, double>::create_Functional(
        //             all_parameters->functional_param, rom_flow_solver->dg);
        //         double rom_functional_value = rom_functional->evaluate_functional(false, false);

        //         std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> fom_flow_solver = 
        //             snapshot_matrix.solve_snapshot_FOM(test_parameters_rom);
        //         dealii::LinearAlgebra::distributed::Vector<double> fom_solution(fom_flow_solver->dg->solution);
        //         auto fom_functional = FunctionalFactory<dim, nstate, double>::create_Functional(
        //             all_parameters->functional_param, fom_flow_solver->dg);
        //         double fom_fucntional_value = fom_functional->evaluate_functional(false, false);

        //         double rom_solution_error = (rom_solution -= fom_solution).l2_norm() / fom_solution.l2_norm();
        //         double rom_functional_error = std::abs(rom_functional_value - fom_fucntional_value) / 
        //             std::abs(fom_fucntional_value);

        //         this->pcout << "Neural network rom solution error: " << rom_solution_error << std::endl
        //                     << "Neural network rom functional error: " << rom_functional_error << std::endl;
                
        //         if (rom_solution_error > 1E-10 && rom_functional_error < 1E-11)
        //         {
        //             this->pcout << "ROM solution error too large, test failed." << std::endl;
        //             solution_error_fail = true;
        //         }
        //         else if (rom_solution_error < 1E-10 && rom_functional_error > 1E-11)
        //         {
        //             this->pcout << "ROM functional error too large, test failed." << std::endl;
        //             functional_error_fail = true;
        //         }
        //         else if (rom_solution_error > 1E-10 && rom_functional_error > 1E-11)
        //         {
        //             this->pcout << "ROM solution and functional errors too large, test failed!" << std::endl;
        //             solution_error_fail = true;
        //             functional_error_fail = true;
        //         }
        //     }
        //     i++;
        // }

    this->pcout << "Test passed!" << std::endl;
    return test_pass;
}

#if PHILIP_DIM==1
        template class NeuralNetworkROM<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class NeuralNetworkROM<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}  ///< Tests namespace
}  ///< PHiLiP namespace  
