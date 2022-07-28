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
        prm.declare_entry("mach_lower_bound", "0.5",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Lower bound of mach number for the apative sampling algorithm for the naca0012 case.");
        prm.declare_entry("mach_upper_bound", "0.9",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Lower bound of mach number for the apative sampling algorithm for the naca0012 case.");
        prm.declare_entry("aoa_lower_bound", "0.0",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Lower bound of angle of attack for the apative sampling algorithm for the naca0012 case.");
        prm.declare_entry("aoa_upper_bound", "4.0",
                          dealii::Patterns::Double(0, dealii::Patterns::Double::max_double_value),
                          "Lower bound of angle of attack for the apative sampling algorithm for the naca0012 case.");
    }
    prm.leave_subsection();
}

void ReducedOrderModelParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("reduced order");
    {
        adaptation_tolerance = prm.get_double("adaptation_tolerance");
        reduced_residual_tolerance = prm.get_double("reduced_residual_tolerance");
        path_to_search = prm.get("path_to_search");
        mach_lower_bound = prm.get_double("mach_lower_bound");
        mach_upper_bound = prm.get_double("mach_upper_bound");
        aoa_lower_bound = prm.get_double("aoa_lower_bound");
        aoa_upper_bound = prm.get_double("aoa_upper_bound");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace