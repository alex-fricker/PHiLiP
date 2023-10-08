#include "neural_network_rom.hpp"
#include "parameters/all_parameters.h"
#include "reduced_order/rom_snapshots.hpp"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include "functional/functional.h"
#include "/usr/include/python3.8/Python.h"
#include <iostream>
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
    std::vector<std::string> testing_pathnames;
    std::string training_savename = std::to_string(all_parameters->reduced_order_param.num_halton) +
        "_snapshots_training";

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

    // Embedding the python code from pod_neural_network_rom.py
    this->pcout << "\nInitializing neural network ROM" << std::endl;
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");

    std::string module_path = "pod_neural_network_rom";
    PyObject *nnrom_module_name = PyUnicode_DecodeFSDefault(module_path.c_str());

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

    PyObject *nnrom_args = PyTuple_Pack(3,
                                        PyUnicode_DecodeFSDefault(training_pathnames[0].c_str()),
                                        PyUnicode_DecodeFSDefault(training_pathnames[1].c_str()),
                                        PyUnicode_DecodeFSDefault(training_pathnames[2].c_str()),
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
        this->pcout << "Failed to instantiate the python class." << std::endl;
        PyErr_Print();
        return 1;
    }
    Py_XDECREF(nnrom_instance);

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
            this->pcout << "Failed to generate initialization dictionary." << std::endl;
            PyErr_Print();
            return 1;
        }
    }

    PyObject *result;
    result = PyObject_CallMethod(nnrom_instance, "initialize_network", "O", nnrom_init_kwargs);
    if (result == nullptr)
    {
        this->pcout << "Failed to initialize neural network class." << std::endl;
        PyErr_Print();
        return 1;
    }
    Py_XDECREF(nnrom_init_kwargs);
    

    // Build the neural network
    result = PyObject_CallMethod(nnrom_instance, "build_network", "p", 
        all_parameters->reduced_order_param.print_plots);
    if (result == nullptr)
    {
        this->pcout <<"Failed to build the neural network." << std::endl;
        PyErr_Print();
        return 1;
    }

    /// Evaluate the neural network
    int num_eval_points = all_parameters->reduced_order_param.num_evaluation_points;
    int num_params = all_parameters->reduced_order_param.parameter_names.size();
    Eigen::MatrixXd rom_testing_parameters = snapshot_matrix.get_halton_points(num_eval_points);
    bool functional_error_fail = false;
    bool solution_error_fail = false;
    int i = 0;

    while (i < num_eval_points && !functional_error_fail && !solution_error_fail)
    {
        Eigen::RowVectorXd test_parameters_rom = rom_testing_parameters.col(i).transpose();

        /// Evaluate the neural network
        PyObject *params_list = PyList_New(num_params);
        for (int j = 0; j < num_params; j++)
        {
            int return_code = PyList_SetItem(params_list, j, PyLong_FromLong(test_parameters_rom(j)));
            if (return_code < 0)
            {
                this->pcout << "Failed to build parameters list." << std::endl;
                PyErr_Print();
                return 1;
            }
        }

        result = PyObject_CallMethod(nnrom_instance, "evaluate_network", "O", params_list);
        if (result == nullptr)
        {
            this->pcout << "Failed to evaluate neural network for parameters."
                        << test_parameters_rom
                        << std::endl;
            PyErr_Print();
            return 1;
        }
        Py_XDECREF(params_list);

        std::ifstream file("pod_nnrom_solution.txt");
        if (!file.is_open())
        {
            this->pcout << "Failed to open rom solution file." << std::endl;
            return 1;
        }
        else
        {
            PHiLiP::Parameters::AllParameters test_parameters_philip = snapshot_matrix.reinit_parameters(
                test_parameters_rom);
            std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> rom_flow_solver =
                FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(
                    &test_parameters_philip, 
                    parameter_handler);

            dealii::LinearAlgebra::ReadWriteVector<double> rom_solution_temp(rom_flow_solver->dg->solution.size());
            double value;
            int i = 0;
            while (file >> value)
            {
                rom_solution_temp[i] = value;
                i++;
            }
            file.close();

            rom_flow_solver->dg->solution.import(rom_solution_temp, dealii::VectorOperation::insert);
            dealii::LinearAlgebra::distributed::Vector<double> rom_solution(rom_flow_solver->dg->solution);
            auto rom_functional = FunctionalFactory<dim, nstate, double>::create_Functional(
                all_parameters->functional_param, rom_flow_solver->dg);
            double rom_functional_value = rom_functional->evaluate_functional(false, false);

            std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> fom_flow_solver = 
                snapshot_matrix.solve_snapshot_FOM(test_parameters_rom);
            dealii::LinearAlgebra::distributed::Vector<double> fom_solution(fom_flow_solver->dg->solution);
            auto fom_functional = FunctionalFactory<dim, nstate, double>::create_Functional(
                all_parameters->functional_param, fom_flow_solver->dg);
            double fom_fucntional_value = fom_functional->evaluate_functional(false, false);

            double rom_solution_error = (rom_solution -= fom_solution).l2_norm() / fom_solution.l2_norm();
            double rom_functional_error = std::abs(rom_functional_value - fom_fucntional_value) / 
                std::abs(fom_fucntional_value);

            this->pcout << "Neural network rom solution error: " << rom_solution_error << std::endl
                        << "Neural network rom functional error: " << rom_functional_error << std::endl;
            
            if (rom_solution_error > 1E-10 && rom_functional_error < 1E-11)
            {
                this->pcout << "ROM solution error too large, test failed." << std::endl;
                solution_error_fail = true;
            }
            else if (rom_solution_error < 1E-10 && rom_functional_error > 1E-11)
            {
                this->pcout << "ROM functional error too large, test failed." << std::endl;
                functional_error_fail = true;
            }
            else if (rom_solution_error > 1E-10 && rom_functional_error > 1E-11)
            {
                this->pcout << "ROM solution and functional errors too large, test failed!" << std::endl;
                solution_error_fail = true;
                functional_error_fail = true;
            }
        }
        i++;
    }

    Py_XDECREF(result);
    Py_XDECREF(nnrom_instance);
    Py_Finalize();

    this->pcout << "Test passed!" << std::endl;
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
