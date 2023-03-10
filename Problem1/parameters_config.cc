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

#include "parameters_config.h"

namespace Parameters {
    using namespace dealii;
//    @sect4{Finite element system setup}
    void FESystem::declare_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Finite element system");
        {
            prm.declare_entry("Polynomial degree", "2",
                              Patterns::Integer(0),
                              "Displacement system polynomial order");

            prm.declare_entry("Quadrature order", "3",
                              Patterns::Integer(0),
                              "Gauss quadrature order");

            prm.declare_entry("Simulation geometry", "1",
                              Patterns::Integer(0),
                              "Simulation geometry");
        }
        prm.leave_subsection();
    }
    void FESystem::parse_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Finite element system");
        {
            poly_degree = prm.get_integer("Polynomial degree");
            quad_order = prm.get_integer("Quadrature order");
            sim_geom = prm.get_integer("Simulation geometry");
        }
        prm.leave_subsection();
    }

//    @sect4{Material parameter definition}
    void Materials::declare_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Material properties");
        {
            prm.declare_entry("Poisson's ratio", "0.3",
                              Patterns::Double(-1.0, 0.5),
                              "Poisson's ratio");

            prm.declare_entry("Shear modulus", "0.450e6",
                              Patterns::Double(),
                              "Shear modulus");
        }
        prm.leave_subsection();
    }
    void Materials::parse_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Material properties");
        {
            nu = prm.get_double("Poisson's ratio");
            mu = prm.get_double("Shear modulus");
        }
        prm.leave_subsection();
    }

//    @sect4{Linear solver parameters}
    void LinearSolver::declare_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Linear solver");
        {
            prm.declare_entry("Solver type", "CG",
                              Patterns::Selection("CG|Direct"),
                              "Type of solver used to solve the linear system");

            prm.declare_entry("Residual", "1e-6",
                              Patterns::Double(0.0),
                              "Linear solver residual (scaled by residual norm)");

            prm.declare_entry("Max iteration multiplier", "1",
                              Patterns::Double(0.0),
                              "Linear solver iterations (multiples of the system matrix size)");

            prm.declare_entry("Preconditioner type", "ssor",
                              Patterns::Selection("jacobi|ssor"),
                              "Type of preconditioner");

            prm.declare_entry("Preconditioner relaxation", "0.65",
                              Patterns::Double(0.0),
                              "Preconditioner relaxation value");
        }
        prm.leave_subsection();
    }
    void LinearSolver::parse_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Linear solver");
        {
            type_lin = prm.get("Solver type");
            tol_lin = prm.get_double("Residual");
            max_iterations_lin = prm.get_double("Max iteration multiplier");
            preconditioner_type = prm.get("Preconditioner type");
            preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
        }
        prm.leave_subsection();
    }

//    @sect4{Nonlinear solver parameters}
    void NonlinearSolver::declare_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Nonlinear solver");
        {
            prm.declare_entry("Max iterations Newton-Raphson", "10",
                              Patterns::Integer(0),
                              "Number of Newton-Raphson iterations allowed");

            prm.declare_entry("Tolerance force", "1.0e-9",
                              Patterns::Double(0.0),
                              "Force residual tolerance");

            prm.declare_entry("Tolerance displacement", "1.0e-6",
                              Patterns::Double(0.0),
                              "Displacement error tolerance");
        }
        prm.leave_subsection();
    }
    void NonlinearSolver::parse_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Nonlinear solver");
        {
            max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
            tol_f = prm.get_double("Tolerance force");
            tol_u = prm.get_double("Tolerance displacement");
        }
        prm.leave_subsection();
    }

//    @sect4{Time class parameters}
    void Time::declare_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Time");
        {
            prm.declare_entry("End time", "1",
                              Patterns::Double(),
                              "End time");

            prm.declare_entry("Time step size", "0.1",
                              Patterns::Double(),
                              "Time step size");
        }
        prm.leave_subsection();
    }
    void Time::parse_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Time");
        {
            end_time = prm.get_double("End time");
            delta_t = prm.get_double("Time step size");
        }
        prm.leave_subsection();
    }

    void AllParameters::declare_parameters(ParameterHandler &prm) {
        FESystem::declare_parameters(prm);
        Materials::declare_parameters(prm);
        LinearSolver::declare_parameters(prm);
        NonlinearSolver::declare_parameters(prm);
        Time::declare_parameters(prm);
    }
    void AllParameters::parse_parameters(ParameterHandler &prm) {
        FESystem::parse_parameters(prm);
        Materials::parse_parameters(prm);
        LinearSolver::parse_parameters(prm);
        NonlinearSolver::parse_parameters(prm);
        Time::parse_parameters(prm);
    }

    AllParameters::AllParameters(const std::string &input_file) {
        ParameterHandler prm;
        declare_parameters(prm);
        prm.parse_input(input_file);
        parse_parameters(prm);
    }

}
