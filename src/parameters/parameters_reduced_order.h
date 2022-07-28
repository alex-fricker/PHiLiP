#ifndef __PARAMETERS_REDUCED_ORDER_H__
#define __PARAMETERS_REDUCED_ORDER_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {
/// Parameters related to reduced-order model
class ReducedOrderModelParam
{
public:

    /// Tolerance for POD adaptation
    double adaptation_tolerance;

    /// Path to search for snapshots or saved POD basis
    std::string path_to_search;

    /// Tolerance of the reduced-order nonlinear residual
    double reduced_residual_tolerance;

    /// Lower bound of mach number for the apative sampling algorithm for the naca0012 case.
    double mach_lower_bound;

    /// Upper bound of mach number for the apative sampling algorithm for the naca0012 case.
    double mach_upper_bound;

    /// Lower bound of angle of attack for the apative sampling algorithm for the naca0012 case.
    double aoa_lower_bound;

    /// Upper bound of angle of attack for the apative sampling algorithm for the naca0012 case.
    double aoa_upper_bound;

    ReducedOrderModelParam (); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);

};

} // Parameters namespace
} // PHiLiP namespace
#endif