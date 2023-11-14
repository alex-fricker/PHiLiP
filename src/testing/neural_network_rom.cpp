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

    std::string testing_savename = std::to_string(all_parameters->reduced_order_param.num_evaluation_points) +
        "_" + all_parameters->reduced_order_param.snapshot_type + "_testing";

    ProperOrthogonalDecomposition::ROMSnapshots<dim, nstate> testing_matrix(
        all_parameters, 
        parameter_handler);
    testing_matrix.build_snapshot_matrix(num_eval_points);
    std::vector<std::string> testing_pathnames = testing_matrix.write_snapshot_data_to_file(testing_savename);
    // std::vector<std::string> testing_pathnames = testing_matrix.get_pathnames(testing_savename);

    // Embedding the python code from pod_neural_network_rom.py
    this->pcout << "\n\nInitializing neural network ROM" << std::endl;
    Py_Initialize();
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
  
    // Load data
    PyObject *result = PyObject_CallMethod(nnrom_instance, "load_data", "(sssi)",
        training_pathnames[0].c_str(),
        training_pathnames[1].c_str(),
        training_pathnames[2].c_str(),
        all_parameters->reduced_order_param.num_pod_modes);

    // Setting up the neural network
    result = PyObject_CallMethod(nnrom_instance, "initialize_network", "(iidid)",
        all_parameters->reduced_order_param.architecture,
        all_parameters->reduced_order_param.epochs,
        all_parameters->reduced_order_param.learning_rate,
        all_parameters->reduced_order_param.training_batch_size,
        all_parameters->reduced_order_param.weight_decay);

    if (result == nullptr)
    {
        this->pcout << "Failed to initialize neural network class." << std::endl;
        PyErr_Print();
        return 1;
    }

    // Build the neural network
    this->pcout << "Training neural network..." << std::endl;
    result = PyObject_CallMethod(nnrom_instance, "build_network", NULL);
    if (result == nullptr)
    {
        this->pcout << "Failed to build the neural network." << std::endl;
        PyErr_Print();
        return 1;
    }
    this->pcout << "Done training neural network." << std::endl;

    // Evaluate the neural network
    result = PyObject_CallMethod(nnrom_instance, "evaluate_network", "(s)", testing_pathnames[1].c_str());
 
    Py_XDECREF(result);
    Py_XDECREF(nnrom_instance);
    Py_Finalize(); 

    std::string filename = all_parameters->solution_vtk_files_directory_name + "/pod_nnrom_solution.txt";
    if(!std::filesystem::exists(filename))
    {
        this->pcout << "Neural network ROM solution file does not exist." << std::endl;
        return 1;
    }

    std::ifstream rom_file(all_parameters->solution_vtk_files_directory_name + "/pod_nnrom_solution.txt");
    if(!rom_file.is_open())
    {
        this->pcout << "Failed to open rom solution file." << std::endl;
        return 1;        
    }
    
    int solution_size = testing_matrix.pod->snapshotMatrix.rows();
    Eigen::MatrixXd rom_solutions(solution_size, num_eval_points);

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

    double solution_error_tol = 1e3;
    bool test_fail = false;
    for (int i = 0; i < num_eval_points; ++i)
    {
        Eigen::RowVectorXd rom_solution = rom_solutions.col(i).transpose();
        Eigen::RowVectorXd fom_solution = testing_matrix.pod->snapshotMatrix.col(i).transpose();
        double solution_error = (rom_solution - fom_solution).norm() / fom_solution.norm();

        if (std::abs(solution_error) > solution_error_tol) { test_fail = true;}

        pcout << "Neural network ROM test point: " << i << " with L2 error: " << solution_error << std::endl;

    }

    return test_fail;
}

#if PHILIP_DIM==1
        template class NeuralNetworkROM<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class NeuralNetworkROM<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}  ///< Tests namespace
}  ///< PHiLiP namespace  
