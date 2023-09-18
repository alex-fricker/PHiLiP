#include "neural_network_rom.hpp"
#include <iostream>
#include <filesystem>
#include "parameters/all_parameters.h"
#include <eigen/Eigen/Dense>
#include "reduced_order/rom_snapshots.hpp"
#include "/usr/include/python3.8/Python.h"
#include <string>
#include <vector>

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

    // Generate the snapshot matrix to train the neural network
    ProperOrthogonalDecomposition::ROMSnapshots<dim, nstate> snapshot_matrix(
        all_parameters,
        parameter_handler);
    snapshot_matrix.build_snapshot_matrix(all_parameters->reduced_order_param.num_halton);
    std::string training_savename = "training_snapshot";
    std::string testing_savename = "testing_snapshot";
    std::vector<std::string> training_pathnames = snapshot_matrix.write_snapshot_data_to_file(training_savename);

    // Generate the snapshot matrices for checking the solution
    int number_evaluation_points = 1;
    snapshot_matrix.build_snapshot_matrix(number_evaluation_points);
    std::vector<std::string> testing_pathnames = snapshot_matrix.write_snapshot_data_to_file(testing_savename);

    // Embedding the python code from pod_neural_network_rom.py
    Py_Initialize();

    PyObject *nnrom_module_name = PyUnicode_DecodeFSDefault("neural_network_ROM");

    PyObject *nnrom_module = PyImport_Import(nnrom_module_name);
    if (nnrom_module == nullptr)
    {
        PyErr_Print();
        std::cerr << "Failed to import the module.\n";
        return 1;
    }
    Py_XDECREF(nnrom_module_name);
    
    PyObject *nnrom_dict = PyModule_GetDict(nnrom_module);
    if (nnrom_dict = nullptr)
    {
        PyErr_Print();
        std::cerr << "Failed to get the dictionary.\n";
        return 1;
    }
    Py_XDECREF(nnrom_module);

    PyObject *nnrom_class_name = PyDict_GetItemString(nnrom_dict, "PODNeuralNetworkROM");
    if (nnrom_class_name == nullptr)
    {
        PyErr_Print();
        std::cerr << "Failed to get the python class.\n";
        return 1;
    }
    Py_XDECREF(nnrom_dict);

    
    PyObject *nnrom_args = PyTuple_Pack(3,
                                        PyUnicode_DecodeFSDefault(training_pathnames[0].c_str()),
                                        PyUnicode_DecodeFSDefault(training_pathnames[1].c_str()),
                                        PyLong_FromLong(all_parameters->reduced_order_param.num_pod_modes));

    PyObject *nnrom_instance;
    if (PyCallable_Check(nnrom_class_name))
    {
        nnrom_instance = PyObject_CallObject(nnrom_class_name, nnrom_args);  // Init PODNeuralNetworkROM
        Py_XDECREF(nnrom_class_name);
        Py_XDECREF(nnrom_args);
    }
    else
    {
        PyErr_Print();
        std::cerr << "Failed to instantiate the python class.\n";
        return 1;
    }

    // Initializing the neural network
    PyObject *nnrom_init_kwargs = PyDict_New();
    std::vector<int> return_codes;
    return_codes.push_back(PyDict_SetItemString(nnrom_init_kwargs, "architecture", 
        PyLong_FromLong(all_parameters->reduced_order_param.architecture))) ;
    return_codes.push_back(PyDict_SetItemString(nnrom_init_kwargs, "epochs", 
        PyLong_FromLong(all_parameters->reduced_order_param.epochs)));
    return_codes.push_back(PyDict_SetItemString(nnrom_init_kwargs, "learning_rate", 
        PyFloat_FromDouble(all_parameters->reduced_order_param.learning_rate)));
    return_codes.push_back(PyDict_SetItemString(nnrom_init_kwargs, "training_batch_size", 
        PyLong_FromLong(all_parameters->reduced_order_param.training_batch_size)));
    return_codes.push_back(PyDict_SetItemString(nnrom_init_kwargs, "weight_decay", 
        PyFloat_FromDouble(all_parameters->reduced_order_param.weight_decay)));

    for (std::vector<int>::size_type i = 0; i < return_codes.size(); i++)
    {
        if (return_codes[i] < 0)
        {
            PyErr_Print();
            std::cerr << "Failed to generate initialization dictionary.\n";
            return 1;
        }
    }

    PyObject *result;
    result = PyObject_CallMethod(nnrom_instance, "initialize_network", "O", nnrom_init_kwargs);
    if (result == nullptr)
    {
        PyErr_Print();
        std::cerr << "Failed to initialize neural network class.\n";
        return 1;
    }
    Py_XDECREF(nnrom_init_kwargs);

    // Build the neural network
    result = PyObject_CallMethod(nnrom_instance, "build_network", "p", 
        all_parameters->reduced_order_param.print_plots);
    if (result == nullptr)
    {
        PyErr_Print();
        std::cerr <<"Failed to build the neural network.\n";
        return 1;
    }

    // Evaluate the neural network
    result = PyObject_CallMethod(nnrom_instance, "evaluate_network", "s", testing_pathnames[1]);
    if (result == nullptr)
    {
        PyErr_Print();
        std::cerr << "Faile to evaluate the neural network,\n";
        return 1;
    }
    Py_XDECREF(result);
    Py_XDECREF(nnrom_instance);
    Py_Finalize();

    return 0;
}

#if PHILIP_DIM==1
        template class NeuralNetworkROM<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class NeuralNetworkROM<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}  ///< Tests namespace
}  ///< PHiLiP namespace  
