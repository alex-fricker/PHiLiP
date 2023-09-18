#include "parameters/parameters_reduced_order.h"
namespace PHiLiP {
namespace Parameters {

// Reduced Order Model inputs
ReducedOrderModelParam::ReducedOrderModelParam () {}

void ReducedOrderModelParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        prm.declare_entry("adaptation_tolerance", "1",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Tolerance for POD adaptation");
        prm.declare_entry("path_to_search", ".",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Path to search for saved snapshots or POD basis.");
        prm.declare_entry("reduced_residual_tolerance", "1E-13",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Tolerance for nonlinear reduced residual");
        prm.declare_entry("num_halton", "0",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Number of Halton sequence points to add to initial snapshot set");
        prm.declare_entry("recomputation_coefficient", "5",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Number of Halton sequence points to add to initial snapshot set");
        prm.declare_entry("parameter_names", "mach, alpha",
                          dealii::Patterns::List(dealii::Patterns::Anything(), 0, 10, ","),
                          "Names of parameters for adaptive sampling");
        prm.declare_entry("parameter_min_values", "0.4, 0",
                          dealii::Patterns::List(dealii::Patterns::Double(), 0, 10, ","),
                          "Minimum values for parameters");
        prm.declare_entry("parameter_max_values", "0.7, 4",
                          dealii::Patterns::List(dealii::Patterns::Double(), 0, 10, ","),
                          "Maximum values for parameters");
    }
    prm.leave_subsection();

    prm.enter_subsection("Neural network");
    {
        prm.declare_entry("run_k_fold_cross_validation", "false",
                          dealii::Patterns::Bool(),
                          "Option to run k-fold cross validation.");
        prm.declare_entry("print_plots", "false",
                          dealii::Patterns::Bool(),
                          "Option to print the error loss plots.");
        prm.declare_entry("num_pod_modes", "0",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Number of POD modes to use, set to 0 to use all modes.");
        prm.declare_entry("architecture", "1",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "ID of neural network architecture to use.");
        prm.declare_entry("epochs", "500",
                          dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                          "Number of epochs to use in training the neural network.");
        prm.declare_entry("learning_rate", "5e-3",
                          dealii::Patterns::Double(dealii::Patterns::Double::max_double_value, 
                                                   dealii::Patterns::Double::max_double_value),
                          "Learning rate for the training of the neural network.");
        prm.declare_entry("training_batch_size", "15",
                          dealii::Patterns::Integer(1, dealii::Patterns::Integer::max_int_value),
                          "Batch size for training the neural network.");
        prm.declare_entry("testing_batch_size", "2",
                          dealii::Patterns::Integer(0, dealii::Patterns::Integer::max_int_value),
                          "Batch size for testing the neural network if doing k-fold cross validation.");
        prm.declare_entry("weight_decay", "1e-3",
                          dealii::Patterns::Double(dealii::Patterns::Double::min_double_value, 
                                                   dealii::Patterns::Double::max_double_value),
                          "Penalty applied to the L2 norm of the weights in the loss function.");
        prm.declare_entry("num_kf_splits", "5",
                          dealii::Patterns::Integer(2, dealii::Patterns::Integer::max_int_value),
                          "Number of splits to make if doing k-fold cross validation.");

    }
    prm.leave_subsection();
}

void ReducedOrderModelParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        adaptation_tolerance = prm.get_double("adaptation_tolerance");
        reduced_residual_tolerance = prm.get_double("reduced_residual_tolerance");
        num_halton = prm.get_integer("num_halton");
        recomputation_coefficient = prm.get_integer("recomputation_coefficient");
        path_to_search = prm.get("path_to_search");

        std::string parameter_names_string = prm.get("parameter_names");
        std::unique_ptr<dealii::Patterns::PatternBase> ListPatternNames(new dealii::Patterns::List(dealii::Patterns::Anything(), 0, 10, ",")); //Note, in a future version of dealii, this may change from a unique_ptr to simply the object. Will need to use std::move(ListPattern) in next line.
        parameter_names = dealii::Patterns::Tools::Convert<decltype(parameter_names)>::to_value(parameter_names_string, ListPatternNames);

        std::string parameter_min_string = prm.get("parameter_min_values");
        std::unique_ptr<dealii::Patterns::PatternBase> ListPatternMin(new dealii::Patterns::List(dealii::Patterns::Double(), 0, 10, ",")); //Note, in a future version of dealii, this may change from a unique_ptr to simply the object. Will need to use std::move(ListPattern) in next line.
        parameter_min_values = dealii::Patterns::Tools::Convert<decltype(parameter_min_values)>::to_value(parameter_min_string, ListPatternMin);

        std::string parameter_max_string = prm.get("parameter_max_values");
        std::unique_ptr<dealii::Patterns::PatternBase> ListPatternMax(new dealii::Patterns::List(dealii::Patterns::Double(), 0, 10, ",")); //Note, in a future version of dealii, this may change from a unique_ptr to simply the object. Will need to use std::move(ListPattern) in next line.
        parameter_max_values = dealii::Patterns::Tools::Convert<decltype(parameter_max_values)>::to_value(parameter_max_string, ListPatternMax);
    }
    prm.leave_subsection();

    prm.enter_subsection("Neural network");
    {
        run_k_fold_cross_validation = prm.get_bool("run_k_fold_cross_validation");
        print_plots = prm.get_bool("print_plots");
        num_pod_modes = prm.get_integer("num_pod_modes");
        architecture = prm.get_integer("architecture");
        epochs = prm.get_integer("epochs");
        learning_rate = prm.get_double("learning_rate");
        training_batch_size = prm.get_integer("training_batch_size");
        testing_batch_size = prm.get_integer("testing_batch_size");
        weight_decay = prm.get_double("weight_decay");
        num_kf_splits = prm.get_integer("num_kf_splits");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
