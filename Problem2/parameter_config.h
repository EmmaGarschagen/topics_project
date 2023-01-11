#ifndef PARAMETER_CONFIG_H
#define PARAMETER_CONFIG_H
/* ---------------------------------------------------------------------
 * Problem 2: Parameter configuration
 * Emma Garschagen
 * November 2022
 *
 * This code is based on a modified Cook's Membrane problem developed for
 * the deal.ii code gallery by Jean-Paul Pelteret, University of Erlangen-Nuremberg,
 * and Andrew McBride, University of Cape Town, 2015, 2017 and Step-77 from the deal.ii
 * tutorials contributed by Wolfgang Bangerth, Colorado State University.
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>

namespace Parameters
{
    using namespace dealii;

// @sect4{Finite Element system}

    struct FESystem {
        unsigned int poly_degree;
        unsigned int quad_order;
        unsigned int sim_geom;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
    };

// @sect4{Geometry}

    struct Geometry
    {
        unsigned int elements_per_edge;
        double scale;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
    };


// @sect4{Materials}

    struct Materials {
        double c0;
        double c1;
        double c2;

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
        std::string nonlinear_solver;
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
            public Geometry,
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

#endif // PARAMETER_CONFIG_H