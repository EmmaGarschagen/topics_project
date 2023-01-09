//
// Created by emma on 2023/01/07.
//

#ifndef EMMA_TOPICS_PROJECT_PARAMETER_CONFIG_H
#define EMMA_TOPICS_PROJECT_PARAMETER_CONFIG_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>

namespace Parameters
{
using namespace dealii;
    struct FESystem
    {
        unsigned int poly_degree;
        unsigned int quad_order;

        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm);
    };


    struct Geometry
    {
        unsigned int global_refinement;
        double       scale;
        double       p_p0;

        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm);
    };


    struct Materials
    {
        double nu;
        double mu;

        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm);
    };



    struct LinearSolver
    {
        std::string type_lin;
        double      tol_lin;
        double      max_iterations_lin;
        bool        use_static_condensation;
        std::string preconditioner_type;
        double      preconditioner_relaxation;

        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm);
    };



    struct NonlinearSolver
    {
        unsigned int max_iterations_NR;
        double       tol_f;
        double       tol_u;

        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm);
    };



    struct Time
    {
        double delta_t;
        double end_time;

        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm);
    };




    struct AllParameters : public FESystem,
                           public Geometry,
                           public Materials,
                           public LinearSolver,
                           public NonlinearSolver,
                           public Time

    {
        AllParameters(const std::string &input_file);

        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm);
    };


} // namespace Parameters


#endif //EMMA_TOPICS_PROJECT_PARAMETER_CONFIG_H
