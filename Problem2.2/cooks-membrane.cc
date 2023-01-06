/*---------------------------------------------------------------
 * Problem 2: Finite Deformation Elasticity
 * Emma Garschagen (2022)
 *
 * This problem has been solved with the use of the deal.II library and code
 * from the code gallery and various tutorials. These are referenced below:
 *
 * J-P. V. Pelteret and A. McBride, The deal.II code gallery: Quasi-Static Finite-Strain Compressible
 * Elasticity, 2016. DOI: 10.5281/zenodo.1228964
 *
 * W. Bangerth, The deal.II tutorial: Step-77 tutorial program. URL: dealii.org/current/doxygen/deal.II/step_77.html
 */


#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/config.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/transformations.h>

#include <deal.II/sundials/kinsol.h>

#include <fstream>
#include <iostream>
#include <memory>


namespace CooksMembrane
{
    using namespace dealii;
    using namespace Physics::Transformations;
    using namespace Physics::Elasticity;
    using namespace std;

    namespace Parameters
    {
        // Finite Element System
        struct FESystem
        {
            unsigned int poly_degree;
            unsigned int quad_order;

            static void
            declare_parameters(ParameterHandler &prm);

            void
            parse_parameters(ParameterHandler &prm);
        };

        void FESystem::declare_parameters(dealii::ParameterHandler &prm)
        {
            prm.enter_subsection("Finite element system");
            {
                prm.declare_entry("Polynomial degree", "2",
                                  Patterns::Integer(0),
                                  "Displacement system polynomial order");

                prm.declare_entry("Quadrature order", "3",
                                  Patterns::Integer(0),
                                  "Gauss quadrature order");
            }
            prm.leave_subsection();
        }

        void FESystem::parse_parameters(dealii::ParameterHandler &prm)
        {
            prm.enter_subsection("Finite element system");
            {
                poly_degree = prm.get_integer("Polynomial degree");
                quad_order = prm.get_integer("Quadrature order");
            }
            prm.leave_subsection();
        }

        // Geometry
        struct Geometry
        {
            unsigned int elements_per_edge;
            double scale;

            static void
            declare_parameters(ParameterHandler &prm);

            void
            parse_parameters(ParameterHandler &prm);
        };

        void Geometry::declare_parameters(dealii::ParameterHandler &prm)
        {
            prm.enter_subsection("Geometry");
            {
                prm.declare_entry("Elements per edge", "32",
                                  Patterns::Integer(0),
                                  "Number of elements per long edge of the beam");

                prm.declare_entry("Grid scale", "1e-3",
                                  Patterns::Double(0.0),
                                  "Global grid scaling factor");
            }
            prm.leave_subsection();
        }


        void Geometry::parse_parameters(dealii::ParameterHandler &prm)
        {
            prm.enter_subsection("Geometry");
            {
                elements_per_edge = prm.get_integer("Elements per edge");
                scale = prm.get_double("Grid scale");
            }
            prm.leave_subsection();
        }

        // Materials
        struct Materials
        {
            std::string material_model;
            double c0;
            double c1;
            double c2;
            double nu;
            double mu;

            static void
            declare_parameters(ParameterHandler &prm);

            void
            parse_parameters(ParameterHandler &prm);
        };

        void Materials::declare_parameters(ParameterHandler &prm)
        {
            prm.enter_subsection("Material properties");
            {
                prm.declare_entry("Material model", "mooney",
                                  Patterns::Selection("mooney|neo-hook"),
                                  "Material model");

                prm.declare_entry("Constant 0", "1e9",
                                  Patterns::Double(0.0),
                                  "Constant 0");

                prm.declare_entry("Constant 1", "92e3",
                                  Patterns::Double(0.0),
                                  "Constant 1");

                prm.declare_entry("Constant 2", "237e3",
                                  Patterns::Double(0.0),
                                  "Constant 2");

                prm.declare_entry("Poisson's ratio", "0.3",
                                  Patterns::Double(-1.0, 0.5),
                                  "Poisson's ratio");

                prm.declare_entry("Shear modulus", "0.450e6",
                                  Patterns::Double(),
                                  "Shear modulus");

            }
            prm.leave_subsection();
        }

        void Materials::parse_parameters(ParameterHandler &prm)
        {
            prm.enter_subsection("Material properties");
            {
                material_model = prm.get("Material model");
                c0 = prm.get_double("Constant 0");
                c1 = prm.get_double("Constant 1");
                c2 = prm.get_double("Constant 2");
                nu = prm.get_double("Poisson's ratio");
                mu = prm.get_double("Shear modulus");
            }
            prm.leave_subsection();
        }

        //Linear Solver
        struct LinearSolver
        {
            std::string type_lin;
            double      tol_lin;
            double      max_iterations_lin;
            std::string preconditioner_type;
            double      preconditioner_relaxation;

            static void
            declare_parameters(ParameterHandler &prm);

            void
            parse_parameters(ParameterHandler &prm);
        };

        void LinearSolver::declare_parameters(ParameterHandler &prm)
        {
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

        void LinearSolver::parse_parameters(ParameterHandler &prm)
        {
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

        // Nonlinear Solver
        struct NonlinearSolver
        {
            unsigned int max_iterations_NR;
            double       tol_f;
            double       tol_u;

            static void
            declare_parameters(ParameterHandler &prm);

            void
            parse_parameters(ParameterHandler &prm);
        };

        void NonlinearSolver::declare_parameters(ParameterHandler &prm)
        {
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

        // Container that hols all run-time selections
        struct AllParameters :
                public FESystem,
                public Geometry,
                public Materials,
                public LinearSolver,
                public NonlinearSolver

        {
            AllParameters(const std::string &input_file);

            static void
            declare_parameters(ParameterHandler &prm);

            void
            parse_parameters(ParameterHandler &prm);
        };

        AllParameters::AllParameters(const std::string &input_file)
        {
            ParameterHandler prm;
            declare_parameters(prm);
            prm.parse_input(input_file);
            parse_parameters(prm);
        }

        void AllParameters::declare_parameters(ParameterHandler &prm)
        {
            FESystem::declare_parameters(prm);
            Geometry::declare_parameters(prm);
            Materials::declare_parameters(prm);
            LinearSolver::declare_parameters(prm);
            NonlinearSolver::declare_parameters(prm);
        }

        void AllParameters::parse_parameters(ParameterHandler &prm)
        {
            FESystem::parse_parameters(prm);
            Geometry::parse_parameters(prm);
            Materials::parse_parameters(prm);
            LinearSolver::parse_parameters(prm);
            NonlinearSolver::parse_parameters(prm);
        }

    } //Parameters namespace

    template <int dim>
    class Material_Model{
    public:
        Material_Model(const Parameters::AllParameters &parameters)
        :
        parameters(parameters),
        neo_hook_c1(parameters.mu / 2.0),
        beta((parameters.nu) / (1 - 2 * parameters.nu)),
        c0(parameters.c0),
        c1(parameters.c1),
        c2(parameters.c2),
        material_model(parameters.material_model)
        {}

        ~Material_Model() {}

        SymmetricTensor<2, dim>
        get_tau(const SymmetricTensor<2, dim> &C,
                const double &det_F,
                const Tensor<2, dim> &F){
            return det_F * get_CauchyStress(C, det_F, F);
        }

        SymmetricTensor<2, dim>
        get_CauchyStress(const SymmetricTensor<2, dim> &C,
                         const double &det_F,
                         const Tensor<2, dim> &F) const {
            return symmetrize(pow(det_F, -1) * F * get_SecondPiolaStress(C, det_F) * transpose(F));
        }


        SymmetricTensor<2, dim>
        get_CauchyStress(const Tensor<2, dim> &F) {

            const double det_F = determinant(F);
            const SymmetricTensor<2, dim> C = symmetrize(transpose(F) * F);

            return this->get_CauchyStress(C, det_F, F);
        }

        SymmetricTensor<4, dim>
        get_Jc(const double &det_F,
               const SymmetricTensor<2, dim> &C_inv,
               const Tensor<2, dim> &F) const {
            return det_F *
                   Physics::Transformations::Piola::push_forward(get_LagrangianElasticityTensor(C_inv, det_F, F), F);
        }

        SymmetricTensor<2, dim>
        get_GreenLagrangeStrain(const Tensor<2, dim> &F) const {
            return 0.5 * (symmetrize(transpose(F) * F) - unit_symmetric_tensor<dim>());
        }

        SymmetricTensor<2, dim>
        get_SecondPiolaStress(const Tensor<2, dim> &F) {
            const double det_F = determinant(F);
            const Tensor<2, dim> C = transpose(F) * F;

            return this->get_SecondPiolaStress(C, det_F);
        }


    private:
        const Parameters::AllParameters &parameters;
        const double neo_hook_c1;
        const double beta;
        const double c0;
        const double c1;
        const double c2;
        std::string material_model;

        SymmetricTensor<4, dim, double>
        get_LagrangianElasticityTensor(const SymmetricTensor<2, dim>        &C_inv,
                                       const double                         &det_F,
                                       const Tensor<2, dim>                 &F) const
        {
            if(material_model == "mooney")
            {return 4*c1*Physics::Elasticity::StandardTensors< dim >::IxI + 2*c0*det_F*(2*det_F-1)*outer_product(C_inv, C_inv) -  2*(2*c0*det_F*(det_F-1) + 2*(c1 + 2*c2))*Physics::Elasticity::StandardTensors< dim >::dC_inv_dC(F) - 4*c2* Physics::Elasticity::StandardTensors< dim >::S;}
            else if(material_model == "neo-hook")
            {return 4 * neo_hook_c1 * pow(det_F, -2 * beta) *
                    (beta * outer_product(C_inv, C_inv) - Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F));}

        }


        SymmetricTensor<2, dim>
        get_SecondPiolaStress(const SymmetricTensor<2, dim>             &C,
                              const double                              &det_F) const
        {   if(material_model == "mooney")
            {return 2 * (c1 + c2 * trace(C)) * unit_symmetric_tensor<dim>() - 2 * c2 * C + (2*c0*det_F*(det_F-1) - 2*(c1 + 2*c2))*
                                                                                           invert(C);}
            else if(material_model == "neo-hook")
            {return 2 * neo_hook_c1 * (Physics::Elasticity::StandardTensors<dim>::I - pow(det_F, -2 * beta) * invert(C));}

        }

    };


    template<int dim>
    class GradientPostprocessor : public DataPostprocessorTensor<dim> {
    public:
        GradientPostprocessor()
                :
                DataPostprocessorTensor<dim>("grad_u",
                                             update_gradients) {}

        virtual
        void
        evaluate_vector_field
                (const DataPostprocessorInputs::Vector<dim> &input_data,
                 std::vector<Vector<double> > &computed_quantities) const override {

            AssertDimension(input_data.solution_gradients.size(),
                            computed_quantities.size());

            for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p) {
                AssertDimension(computed_quantities[p].size(),
                                (Tensor<2, dim>::n_independent_components));
                for (unsigned int d = 0; d < dim; ++d)
                    for (unsigned int e = 0; e < dim; ++e)
                        computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))]
                                = input_data.solution_gradients[p][d][e];
            }
        }
    };

    template<int dim>
    class DeformationGradientPostprocessor : public DataPostprocessorTensor<dim> {
    public:
        DeformationGradientPostprocessor()
                :
                DataPostprocessorTensor<dim>("F",
                                             update_gradients) {}

        virtual
        void
        evaluate_vector_field
                (const DataPostprocessorInputs::Vector<dim> &input_data,
                 std::vector<Vector<double> > &computed_quantities) const override {
            AssertDimension(input_data.solution_gradients.size(),
                            computed_quantities.size());

            for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p) {
                AssertDimension(computed_quantities[p].size(),
                                (Tensor<2, dim>::n_independent_components));
                for (unsigned int d = 0; d < dim; ++d)
                    for (unsigned int e = 0; e < dim; ++e)
                        computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))]
                                = input_data.solution_gradients[p][d][e] + StandardTensors<dim>::I[d][e];
            }
        }
    };

    template<int dim>
    class CauchyStressPostprocessor : public DataPostprocessorTensor<dim> {
    public:
        CauchyStressPostprocessor(const Parameters::AllParameters &parameters)
                : DataPostprocessorTensor<dim>("Cauchy_stress",
                                               update_gradients), parameters(parameters) {}

        virtual
        void
        evaluate_vector_field
                (const DataPostprocessorInputs::Vector<dim> &input_data,
                 std::vector<Vector<double> > &computed_quantities) const override {

            AssertDimension(input_data.solution_gradients.size(),
                            computed_quantities.size());
            SymmetricTensor<2, dim> stress;
            Tensor<2, dim> F;
            Material_Model<dim> material(parameters);
            for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p) {

                AssertDimension(computed_quantities[p].size(),
                                (Tensor<2, dim>::n_independent_components));

                for (unsigned int d = 0; d < dim; ++d)
                    for (unsigned int e = 0; e < dim; ++e)
                        F[d][e] = input_data.solution_gradients[p][d][e] + StandardTensors<dim>::I[d][e];

                stress = material.get_CauchyStress(F);

                for (unsigned int d = 0; d < dim; ++d)
                    for (unsigned int e = 0; e < dim; ++e)
                        computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))]
                                = stress[d][e];
            }
        }

    private:
        const Parameters::AllParameters parameters;
    };

    template<int dim>
    class LagrangeStrainPostprocessor : public DataPostprocessorTensor<dim> {
    public:
        LagrangeStrainPostprocessor(const Parameters::AllParameters &parameters)
                : DataPostprocessorTensor<dim>("strain",
                                               update_gradients), parameters(parameters) {}

        virtual
        void
        evaluate_vector_field
                (const DataPostprocessorInputs::Vector<dim> &input_data,
                 std::vector<Vector<double> > &computed_quantities) const override {

            AssertDimension(input_data.solution_gradients.size(),
                            computed_quantities.size());
            SymmetricTensor<2, dim> strain;
            Tensor<2, dim> F;
            Material_Model<dim> material(parameters);
            for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p) {

                AssertDimension(computed_quantities[p].size(),
                                (Tensor<2, dim>::n_independent_components));

                for (unsigned int d = 0; d < dim; ++d)
                    for (unsigned int e = 0; e < dim; ++e)
                        F[d][e] = input_data.solution_gradients[p][d][e] + StandardTensors<dim>::I[d][e];

                strain = material.get_GreenLagrangeStrain(F);

                for (unsigned int d = 0; d < dim; ++d)
                    for (unsigned int e = 0; e < dim; ++e)
                        computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))]
                                = strain[d][e];
            }
        }

    private:
        const Parameters::AllParameters parameters;
    };

    template <int dim>
    class Solid
    {
    public:
        Solid(const Parameters::AllParameters &parameters);

        virtual
        ~Solid();
        void run();

    private:

        void create_grid();
        void setup_system(const bool initial_step);
        void solve(const Vector<double> &rhs,
                   Vector<double>       &solution,
                   const double         tolerance);
        void refine_mesh();
        void output_results(const unsigned int refinement_cycle);
        void make_constraints();
        void compute_and_factorize_jacobian(const Vector<double> &evaluation_point);
        void compute_residual(const Vector<double> &evaluation_point,
                              Vector<double>       &residual);



        const Parameters::AllParameters &parameters;


        Triangulation<dim>                      triangulation;
        DoFHandler<dim>                         dof_handler;
        const unsigned int                      degree;
        FESystem<dim>                           fe;
        const FEValuesExtractors::Vector        u_fe;

        static const unsigned int               n_components = dim;
        static const unsigned int               first_u_component = 0;

        enum {
            u_dof = 0
        };

        const QGauss<dim>                       qf_cell;
        const QGauss<dim-1>                     qf_face;
        const unsigned int                      n_q_points;
        const unsigned int                      n_q_points_f;
        const unsigned int                      dofs_per_cell;

        AffineConstraints<double>               hanging_node_constraints;

        SparsityPattern                         sparsity_pattern;
        SparseMatrix<double>                    jacobian_matrix;
        std::unique_ptr<SparseDirectUMFPACK>    jacobian_matrix_factorization;

        Vector<double>                          current_solution;

        double vol_reference;
        double vol_current;
        TimerOutput    timer;


    };

    template<int dim>
    Solid<dim>::Solid(const Parameters::AllParameters &parameters)
            :
            parameters(parameters),
            dof_handler(triangulation),
            degree(parameters.poly_degree),
            fe(FE_Q<dim>(parameters.poly_degree), dim), //displacement
            u_fe(first_u_component),
            qf_cell(parameters.quad_order),
            qf_face(parameters.quad_order),
            n_q_points(qf_cell.size()),
            n_q_points_f(qf_face.size()),
            dofs_per_cell(fe.dofs_per_cell),
            vol_reference(0.0),
            vol_current(0.0),
            timer(std::cout,
                  TimerOutput::summary,
                  TimerOutput::wall_times)

    {}

    template <int dim>
    Solid<dim>::~Solid()
    {
        dof_handler.clear();
    }

    template <int dim>
    void Solid<dim>::run()
    {
        create_grid();
        setup_system(/*initial_step=*/ true);

        for (unsigned int refinement_cycle = 0; refinement_cycle < 3; ++refinement_cycle)
        {
            timer.reset();
            std::cout << "Mesh refinement step " << refinement_cycle << std::endl;

            if (refinement_cycle != 0)
                refine_mesh();

            const double target_tolerance = 1e-3 * std::pow(0.1, refinement_cycle);
            std::cout << " Target tolerance: " << target_tolerance << std::endl << std::endl;

            // Set up SUNDIALS functions
            {
                typename SUNDIALS::KINSOL<Vector<double>>::AdditionalData
                        additional_data;
                additional_data.function_tolerance = target_tolerance;
                additional_data.strategy = dealii::SUNDIALS::KINSOL<>::AdditionalData::SolutionStrategy::linesearch;


                SUNDIALS::KINSOL<Vector<double>> nonlinear_solver(additional_data);

                nonlinear_solver.reinit_vector = [&](Vector<double> &x) {
                    x.reinit(dof_handler.n_dofs());
                };

                nonlinear_solver.residual =
                        [&](const Vector<double> &evaluation_point,
                            Vector<double> &      residual) {
                            compute_residual(evaluation_point, residual);

                            return 0;
                        };

                nonlinear_solver.setup_jacobian =
                        [&](const Vector<double> &current_u,
                            const Vector<double> & /*current_f*/) {
                            compute_and_factorize_jacobian(current_u);

                            return 0;
                        };

                nonlinear_solver.solve_with_jacobian = [&](const Vector<double> &rhs,
                                                           Vector<double> &      dst,
                                                           const double tolerance) {
                    Vector<double> rhs_to_use = rhs;
//                    rhs_to_use *= 0.5;
                    this->solve(rhs_to_use, dst, tolerance);

                    return 0;
                };

                nonlinear_solver.solve(current_solution);

            }

            output_results(refinement_cycle);
            timer.print_summary();
            std::cout << std::endl;
        }
    }

    template <int dim>
    Point<dim> grid_y_transform (const Point<dim> &pt_in)
    {
        const double &x = pt_in[0];
        const double &y = pt_in[1];

        const double y_upper = 44.0 + (16.0/48.0)*x; // Line defining upper edge of beam
        const double y_lower =  0.0 + (44.0/48.0)*x; // Line defining lower edge of beam
        const double theta = y/44.0; // Fraction of height along left side of beam
        const double y_transform = (1-theta)*y_lower + theta*y_upper; // Final transformation

        Point<dim> pt_out = pt_in;
        pt_out[1] = y_transform;

        return pt_out;
    }


    template <int dim>
    void Solid<dim>::create_grid()
    {
        std::vector< unsigned int > repetitions(dim, parameters.elements_per_edge);
        if (dim == 3)
            repetitions[dim-1] = 1;

        const Point<dim> bottom_left = (dim == 3 ? Point<dim>(0.0, 0.0, -0.5) : Point<dim>(0.0, 0.0));
        const Point<dim> top_right = (dim == 3 ? Point<dim>(48.0, 44.0, 0.5) : Point<dim>(48.0, 44.0));

        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  repetitions,
                                                  bottom_left,
                                                  top_right);

        const double tol_boundary = 1e-6;
        typename Triangulation<dim>::active_cell_iterator cell =
                triangulation.begin_active(), endc = triangulation.end();
        for (; cell != endc; ++cell)
            for (unsigned int face = 0;
                 face < GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary() == true)
                {
                    if (std::abs(cell->face(face)->center()[0] - 0.0) < tol_boundary)
                        cell->face(face)->set_boundary_id(1); // -X faces
                    else if (std::abs(cell->face(face)->center()[0] - 48.0) < tol_boundary)
                        cell->face(face)->set_boundary_id(11); // +X faces
                    else if (dim == 3 && std::abs(std::abs(cell->face(face)->center()[2]) - 0.5) < tol_boundary)
                        cell->face(face)->set_boundary_id(2); // +Z and -Z faces
                }

        GridTools::transform(&grid_y_transform<dim>, triangulation);

        GridTools::scale(parameters.scale, triangulation);

        vol_reference = GridTools::volume(triangulation);
        vol_current = vol_reference;
        std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;
    }

    template <int dim>
    void Solid<dim>::setup_system(const bool initial_step)
    {
        timer.enter_subsection("Set up system");

        if (initial_step)
        {
            if(parameters.material_model == "neo-hook")
                std::cout << "Using a compressible Neo-Hookean single-field formulation." << std::endl;
            else
                std::cout << "Using a compressible Mooney-Rivlin single-field formulation." << std::endl;

            dof_handler.distribute_dofs(fe);
            current_solution.reinit(dof_handler.n_dofs());

            hanging_node_constraints.clear();
            DoFTools::make_hanging_node_constraints(dof_handler,
                                                    hanging_node_constraints);
            hanging_node_constraints.close();

            std::cout   << "Triangulation:"
                        << "\n\t Number of active cells: " << triangulation.n_active_cells()
                        << "\n\t Number of degrees of freedom: " << dof_handler.n_dofs()
                        << std::endl;
        }

        DynamicSparsityPattern dsp(dof_handler.n_dofs());

        Table<2, DoFTools::Coupling> coupling(n_components, n_components);
        for (unsigned int  ii = 0; ii < n_components; ++ii)
            for (unsigned int jj = 0; jj < n_components; ++jj)
                coupling[ii][jj] = DoFTools::always;


        DoFTools::make_sparsity_pattern(dof_handler,
                                        coupling,
                                        dsp);

        hanging_node_constraints.condense(dsp);
        sparsity_pattern.copy_from(dsp);
        jacobian_matrix.reinit(sparsity_pattern);
        jacobian_matrix_factorization.reset();

        timer.leave_subsection();
    }

    template <int dim>
    void Solid<dim>::solve(const Vector<double> &rhs,
                           Vector<double> &solution,
                           const double /*tolerance*/)
    {
        timer.enter_subsection("Linear system solve");
        std::cout<<"  Solving linear system"<<std::endl;
        jacobian_matrix_factorization->vmult(solution, rhs);
        hanging_node_constraints.distribute(solution);

        timer.leave_subsection();
    }

    template <int dim>
    void Solid<dim>::refine_mesh()
    {
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

        KellyErrorEstimator<dim>::estimate(dof_handler,
                                           QGauss<dim-1>(fe.degree + 1),
                                           std::map<types::boundary_id, const Function<dim> *>(),
                                           current_solution,
                                           estimated_error_per_cell);

        GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                        estimated_error_per_cell,
                                                        0.3,
                                                        0.03);

        triangulation.prepare_coarsening_and_refinement();

        SolutionTransfer<dim> solution_transfer(dof_handler);
        solution_transfer.prepare_for_coarsening_and_refinement(current_solution);

        triangulation.execute_coarsening_and_refinement();

        dof_handler.distribute_dofs(fe);

        Vector<double> tmp(dof_handler.n_dofs());
        solution_transfer.interpolate(current_solution, tmp);
        current_solution = std::move(tmp);

        hanging_node_constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                hanging_node_constraints);

        hanging_node_constraints.close();
        hanging_node_constraints.distribute(current_solution);
        make_constraints();
        setup_system(/*initial_step=*/false);
    }

    template <int dim>
    void Solid<dim>::output_results(const unsigned int refinement_cycle)
    {
        timer.enter_subsection("Graphical output");

        DataOut<dim> data_out;
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                data_component_interpretation(dim,
                                              DataComponentInterpretation::component_is_part_of_vector);
        std::vector<std::string> solution_name(dim, "Displacement");

        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(current_solution,
                                 solution_name,
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);

        GradientPostprocessor<dim> grad_post;
        data_out.add_data_vector(current_solution, grad_post);
        DeformationGradientPostprocessor<dim> F_post;
        data_out.add_data_vector(current_solution, F_post);
        CauchyStressPostprocessor<dim> sig_post(parameters);
        data_out.add_data_vector(current_solution, sig_post);
        LagrangeStrainPostprocessor<dim> E_post(parameters);
        data_out.add_data_vector(current_solution, E_post);

        const std::string filename = "Q"+to_string(parameters.poly_degree)+"cooksmembrane_solution-" + Utilities::int_to_string(refinement_cycle, 2) + ".vtu";
        std::ofstream output(filename);
        data_out.write_vtu(output);
    }

    template <int dim>
    void Solid<dim>::make_constraints()
    {
        std::map<types::global_dof_index, double> boundary_values;
        ;
        {
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components),
                                                     boundary_values,
                                                     fe.component_mask(u_fe));
        }

        if (dim == 3)
        {
            const int boundary_id = 2;
            const FEValuesExtractors::Scalar z_displacement(2);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components),
                                                     boundary_values,
                                                     fe.component_mask(z_displacement));
        }



        for (const auto &boundary_value : boundary_values)
            current_solution(boundary_value.first) = boundary_value.second;
        hanging_node_constraints.distribute(current_solution);
//        else
//        {
//            if (hanging_node_constraints.has_inhomogeneities())
//            {
//                AffineConstraints<double> homogeneous_constraints(hanging_node_constraints);
//                for (unsigned int dof = 0; dof != dof_handler.n_dofs(); ++dof)
//                    if (homogeneous_constraints.is_inhomogeneously_constrained(dof))
//                        homogeneous_constraints.set_inhomogeneity(dof, 0.0);
//                hanging_node_constraints.clear();
//                hanging_node_constraints.copy_from(homogeneous_constraints);
//            }
//        }

    }


    template <int dim>
    void Solid<dim>::compute_and_factorize_jacobian(const Vector<double> &evaluation_point)
    {
        //Assembling the Jacobian
        {
            timer.enter_subsection("Assembling the Jacobian");
            std::cout << "Computing the Jacobian matrix" << std::endl;
            Material_Model<dim> material(parameters);



            const QGauss<dim> quadrature_formula(fe.degree+1);
            jacobian_matrix = 0;

            FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    update_gradients | update_quadrature_points | update_JxW_values);

            const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;
            const unsigned int n_q_points = quadrature_formula.size();


            FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
            std::vector<Tensor<2, dim>> evaluation_point_gradients(n_q_points);
            std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


            for (const auto &cell : dof_handler.active_cell_iterators())
            {
                cell_matrix = 0;
                fe_values.reinit(cell);
                fe_values[u_fe].get_function_gradients(evaluation_point,
                                                       evaluation_point_gradients);

                std::vector<std::vector<Tensor<2, dim>>>            grad_Nx(qf_cell.size(),
                                                                            std::vector<Tensor<2, dim>> (dofs_per_cell));
                std::vector<std::vector<SymmetricTensor<2, dim>>>   symm_grad_Nx(qf_cell.size(),
                                                                                 std::vector<SymmetricTensor<2, dim>> (dofs_per_cell));

                const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    const Tensor<2, dim> &grad_u = evaluation_point_gradients[q_point];
                    const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(grad_u);
                    const double det_F = determinant(F);
                    const SymmetricTensor<2, dim> C = Physics::Elasticity::Kinematics::C(F);
                    const Tensor<2, dim> F_inv = invert(F);
                    const SymmetricTensor<2, dim> C_inv = invert(C);
                    Assert(det_F > double(0.0), ExcInternalError());
                    Assert(grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
                    Assert(symm_grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

                    const SymmetricTensor<2, dim> tau = material.get_tau(C, det_F, F);
                    const SymmetricTensor<4, dim> Jc = material.get_Jc(det_F, C_inv, F);
                    const Tensor<2, dim> tau_ns (tau);
                    const double JxW = fe_values.JxW(q_point);


                    for (unsigned int k = 0; k < dofs_per_cell; ++k) //Reset gradients
                    {
                        grad_Nx[q_point][k] = Tensor<2, dim>();
                        symm_grad_Nx[q_point][k] = SymmetricTensor<2, dim>();
                    }

                    for (unsigned int k = 0; k < dofs_per_cell; ++k) //Compute gradients
                    {
                        const unsigned k_group = fe.system_to_base_index(k).first.first;
                        if (k_group == u_dof)
                        {
                            grad_Nx[q_point][k] = fe_values[u_fe].gradient(k, q_point)*F_inv;
                            symm_grad_Nx[q_point][k] = symmetrize(grad_Nx[q_point][k]);
                        }
                        else
                            Assert(k_group <= u_dof, ExcInternalError());
                    }


                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const unsigned int component_i = fe.system_to_component_index(i).first;
                        const unsigned int i_group     = fe.system_to_base_index(i).first.first;

                        for (unsigned int j = 0; j <= i; ++j)
                        {
                            const unsigned int component_j = fe.system_to_component_index(j).first;
                            const unsigned int j_group     = fe.system_to_base_index(j).first.first;
                            if ((i_group == j_group) && (i_group == u_dof))
                            {
                                cell_matrix(i, j) += symm_grad_Nx[q_point][i] * Jc // The material contribution:
                                                     * symm_grad_Nx[q_point][j] * JxW;
                                if (component_i == component_j) // geometrical stress contribution
                                    cell_matrix(i, j) += grad_Nx[q_point][i][component_i] * tau_ns
                                                         * grad_Nx[q_point][j][component_j] * JxW;
                            }
                            else
                                Assert((i_group <= u_dof) && (j_group <= u_dof),
                                       ExcInternalError());
                        }
                    }
                }
                for (unsigned  int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                        cell_matrix(i, j) = cell_matrix(j, i);



                cell->get_dof_indices(local_dof_indices);
                hanging_node_constraints.distribute_local_to_global(cell_matrix,
                                                                    local_dof_indices,
                                                                    jacobian_matrix);
            }

            //Apply Dirichlet boundary condition

            std::map<types::global_dof_index, double> boundary_values;
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     0,
                                                     Functions::ZeroFunction<dim>(n_components),
                                                     boundary_values);

            Vector<double> dummy_solution(dof_handler.n_dofs());
            Vector<double> dummy_rhs(dof_handler.n_dofs());
//            {
//                {
//                    const int boundary_id = 1;
//                    VectorTools::interpolate_boundary_values(dof_handler,
//                                                             boundary_id,
//                                                             Functions::ZeroFunction<dim>(n_components),
//                                                             boundary_values,
//                                                             fe.component_mask(u_fe));
//                }
//
//                if (dim == 3) {
//                    const int boundary_id = 2;
//                    const FEValuesExtractors::Scalar z_displacement(2);
//                    VectorTools::interpolate_boundary_values(dof_handler,
//                                                             boundary_id,
//                                                             Functions::ZeroFunction<dim>(n_components),
//                                                             boundary_values,
//                                                             fe.component_mask(z_displacement));
//                }
//            }


            MatrixTools::apply_boundary_values(boundary_values,
                                               jacobian_matrix,
                                               dummy_solution,
                                               dummy_rhs);

            timer.leave_subsection();
        }

        //Factorizing the Jacobian
        {
            timer.enter_subsection("Factorizing the Jacobian");
            std::cout << "Factorizing the Jacobian" <<std::endl;

            jacobian_matrix_factorization = std::make_unique<SparseDirectUMFPACK>();
            jacobian_matrix_factorization->factorize(jacobian_matrix);

            timer.leave_subsection();
        }

    }

    template <int dim>
    void Solid<dim>::compute_residual(const Vector<double> &evaluation_point,
                                      Vector<double> &residual) // TODO: add Neumann BC
    {
        timer.enter_subsection("Assembling residual");
        std::cout<<"Assembling residual vector..."<<std::flush;

        Material_Model<dim> material(parameters);

        const QGauss<dim> quadrature_formula(fe.degree+1);

        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_gradients | update_quadrature_points | update_JxW_values);

        FEFaceValues<dim> fe_face_values(fe,
                                         qf_face,
                                         update_values | update_gradients | update_quadrature_points | update_JxW_values);

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        Vector<double> cell_residual(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<Tensor<2, dim>> evaluation_point_gradients(n_q_points);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            cell_residual = 0;
            fe_values.reinit(cell);
            fe_values[u_fe].get_function_gradients(evaluation_point,
                                                   evaluation_point_gradients);

            std::vector<std::vector<Tensor<2, dim>>>                grad_Nx(qf_cell.size(),
                                                                            std::vector<Tensor<2, dim>> (dofs_per_cell));
            std::vector<std::vector<SymmetricTensor<2, dim>>>       symm_grad_Nx(qf_cell.size(),
                                                                                 std::vector<SymmetricTensor<2, dim>> (dofs_per_cell));

            const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
                const Tensor<2, dim> &grad_u = evaluation_point_gradients[q_point];
                const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(grad_u);
                const double det_F = determinant(F);
                const SymmetricTensor<2, dim> C = Physics::Elasticity::Kinematics::C(F);
                const Tensor<2, dim> F_inv = invert(F);
                const SymmetricTensor<2, dim> C_inv = invert(C);
                Assert(det_F > double(0.0), ExcInternalError());
                Assert(grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
                Assert(symm_grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

                const SymmetricTensor<2, dim> tau = material.get_tau(C, det_F, F);
                const Tensor<2, dim> tau_ns (tau);
                const double JxW = fe_values.JxW(q_point);

                for (unsigned int k = 0; k < dofs_per_cell; ++k) //Reset gradients
                {
                    grad_Nx[q_point][k] = Tensor<2, dim>();
                    symm_grad_Nx[q_point][k] = SymmetricTensor<2, dim>();
                }

                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    const unsigned k_group = fe.system_to_base_index(k).first.first;

                    if (k_group == u_dof)
                    {
                        grad_Nx[q_point][k] = fe_values[u_fe].gradient(k, q_point)*F_inv;
                        symm_grad_Nx[q_point][k] = symmetrize(grad_Nx[q_point][k]);
                    }
                    else
                        Assert(k_group <= u_dof, ExcInternalError());
                }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int i_group     = fe.system_to_base_index(i).first.first;
                    if (i_group == u_dof)
                        cell_residual(i) += (symm_grad_Nx[q_point][i] * tau) * JxW;
                    else
                        Assert(i_group <= u_dof, ExcInternalError());
                }
            }

            // Neumann contribution per cell

            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
                 ++face)
                if (cell->face(face)->at_boundary() == true
                    && cell->face(face)->boundary_id() == 11)
                {
                    fe_face_values.reinit(cell, face);

                    for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                         ++f_q_point)
                    {
                        const double magnitude = (0.1/ (16.0 * parameters.scale * 1.0 * parameters.scale )); // (Total force) / (RHS surface area)
                        Tensor<1, dim> dir;
                        dir[1] = 1.0;
                        const Tensor<1, dim> traction = magnitude * dir;

                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                            const unsigned int i_group =
                                    fe.system_to_base_index(i).first.first;

                            if (i_group == u_dof)
                            {
                                const unsigned int component_i =
                                        fe.system_to_component_index(i).first;
                                const double Ni =
                                        fe_face_values.shape_value(i, f_q_point);
                                const double JxW = fe_face_values.JxW(f_q_point);

                                cell_residual(i) -= (Ni * traction[component_i])
                                                    * JxW;
                            }

                        }
                    }

                }

            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                residual(local_dof_indices[i]) += cell_residual(i);
        }

        hanging_node_constraints.condense(residual);

//        for (const types::global_dof_index i :
//            DoFTools::extract_boundary_dofs(dof_handler))
//            residual(i) = 0;

        //Fixed left side of beam
        const std::set< types::boundary_id > left_dirichlet_boundary_id = {1};
        for (const types::global_dof_index i :
                DoFTools::extract_boundary_dofs(dof_handler, fe.component_mask(u_fe), left_dirichlet_boundary_id))
            residual(i) = 0;

        //Zero z-displacement
        const std::set< types::boundary_id > z_dirichlet_boundary_id = {2};
        const FEValuesExtractors::Scalar z_displacement(2);
        for (const types::global_dof_index i :
                DoFTools::extract_boundary_dofs(dof_handler, fe.component_mask(z_displacement), z_dirichlet_boundary_id))
            residual(i) = 0;


        for (const types::global_dof_index i :
                DoFTools::extract_hanging_node_dofs(dof_handler))
            residual(i) = 0;

        std::cout << " norm = "<< residual.l2_norm() << std::endl;

        timer.leave_subsection();
    }






} //CooksMembrane namespace

int main()
{
    try
    {
        using namespace CooksMembrane;
        Parameters::AllParameters parameters("problem2_parameters.prm");
        const unsigned int dim = 3;
        Solid<dim> cooks_membrane_3d(parameters);
        cooks_membrane_3d.run();
    }

    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}















