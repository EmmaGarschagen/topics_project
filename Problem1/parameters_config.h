/* ---------------------------------------------------------------------
 * Problem 1: Finite Deformation Elasticity
 * Emma Garschagen
 * November 2022
 *
 * This code is based on a modified Cook's Membrane problem developed for
 * the deal.ii code gallery by Jean-Paul Pelteret, University of Erlangen-Nuremberg,
 * and Andrew McBride, University of Cape Town, 2015, 2017 and Step-44 from the deal.ii
 * tutorials contributed by Wolfgang Bangerth, Colorado State University.
 * ---------------------------------------------------------------------
 */

#ifndef EMMA_TOPICS_PROJECT_PARAMETERS_CONFIG_H
#define EMMA_TOPICS_PROJECT_PARAMETERS_CONFIG_H
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>

namespace Parameters{
    using namespace dealii;

struct FESystem {
    unsigned int poly_degree;
    unsigned int quad_order;
    unsigned int sim_geom;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
};



// @sect4{Materials}

// The shear modulus $ \mu $ and Poisson ration$ \nu $ for the
// neo-Hookean material:
struct Materials {
    double nu;
    double mu;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
};



// @sect4{Linear solver}

struct LinearSolver {
    std::string type_lin;
    double tol_lin;
    double max_iterations_lin;
    std::string preconditioner_type;
    double preconditioner_relaxation;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
};


// @sect4{Nonlinear solver}

struct NonlinearSolver {
    unsigned int max_iterations_NR;
    double tol_f;
    double tol_u;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
};

// @sect4{Time}

struct Time {
    double delta_t;
    double end_time;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
};


// @sect4{All parameters}

struct AllParameters :
        public FESystem,
        public Materials,
        public LinearSolver,
        public NonlinearSolver,
        public Time
{
    AllParameters(const std::string &input_file);

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
};


}
#endif //EMMA_TOPICS_PROJECT_PARAMETERS_CONFIG_H
