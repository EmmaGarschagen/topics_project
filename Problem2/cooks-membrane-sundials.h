#ifndef COOKSMEMBRANE_H
#define COOKSMEMBRANE_H

/* ---------------------------------------------------------------------
 * Problem 2: Finite Deformation Elasticity
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
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/config.h>

#include <deal.II/sundials/kinsol.h>

#if DEAL_II_VERSION_MAJOR >= 9 && defined(DEAL_II_WITH_TRILINOS)

#include <deal.II/differentiation/ad.h>

#endif


#include <deal.II/physics/transformations.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <iostream>
#include <fstream>
#include <memory>

#include "parameter_config.h"
#include "time.h"

bool almost_equals(const double &a,
                   const double &b,
                   const double &tol = 1e-8);

namespace Cooks_Membrane {
    using namespace dealii;
    using namespace Physics::Transformations;
    using namespace Physics::Elasticity;
    using namespace std;

    inline double degrees_to_radians(const double &degrees) {
        return degrees * (M_PI / 180.0);
    }


// @sect3{Compressible Mooney_Rivlin material within a one-field formulation}
    template<int dim, typename NumberType>
    class Material_Compressible_Mooney_Rivlin_One_Field {
        public:
            /** Implementation of a single-field Mooney-Rivlin material model
             *
             * @param parameters From the parameter config file, the relevant material constants are extracted.
             */
        Material_Compressible_Mooney_Rivlin_One_Field(const Parameters::AllParameters &parameters)
                :
                c0(parameters.c0),
                c1(parameters.c1),
                c2(parameters.c2) {}


        ~Material_Compressible_Mooney_Rivlin_One_Field() {}


        SymmetricTensor<2, dim, NumberType>
        get_tau(const SymmetricTensor<2, dim, NumberType> &C,
                const NumberType &det_F,
                const Tensor<2, dim, NumberType> &F)
                /** Get the Kirchhoff stress, $\boldsymbol{\tau}$.
                 *
                 * @param C Right Cauchy-Green Ttnsor
                 * @param det_F Jacobian
                 * @param F Deformation gradient
                 * @return A symmetric tensor of rank 2
                 */
        {
            return det_F * get_CauchyStress(C, det_F, F);
        }

        SymmetricTensor<2, dim, NumberType>
        get_GreenLagrangeStrain(const Tensor<2, dim, NumberType> &F) const
        /** Get the Green-Lagrange strain for post-processing.
         *
         * @param F Deformation gradient
         * @return  A symmetric tensor of rank 2
         */
        {
            return 0.5 * (symmetrize(transpose(F) * F) - unit_symmetric_tensor<dim, NumberType>());
        }


        SymmetricTensor<2, dim, NumberType>
        get_CauchyStress(const SymmetricTensor<2, dim, NumberType> &C,
                         const NumberType &det_F,
                         const Tensor<2, dim, NumberType> &F)
                         /** Get the Cauchy stress, $\boldsymbol{\sigma}$.
                          *
                          * @param C Right Cauchy-Green tensor
                          * @param det_F Jacobian
                          * @param F Deformation gradient
                          * @return A symmetric tensor of rank 2
                          */
        {
            return symmetrize(pow(det_F, -1) * F * get_SecondPiolaStress(C, det_F) * transpose(F));
        }


        SymmetricTensor<2, dim, NumberType>
        get_CauchyStress(const Tensor<2, dim, NumberType> &F)
        /** Get the Cauchy stress for post-processing.
         *
         * @param F Deformation gradient
         * @return A symmetric tensor of rank 2
         */
        {
            const NumberType det_F = determinant(F);
            const SymmetricTensor<2, dim, NumberType> C = symmetrize(transpose(F) * F);

            return this->get_CauchyStress(C, det_F, F);
        }


        SymmetricTensor<2, dim, NumberType>
        get_SecondPiolaStress(const Tensor<2, dim, NumberType> &F)
        /** Get the second Piola stress for post-processing.
         *
         * @param F Deformation gradient
         * @return A symmetric tensor of rank 2
         */
        {
            const NumberType det_F = determinant(F);
            const SymmetricTensor<2, dim, NumberType> C = transpose(F) * F;

            return this->get_SecondPiolaStress(C, det_F);

        }


        SymmetricTensor<4, dim, NumberType>
        get_Jc(const SymmetricTensor<2, dim, NumberType> &C,
               const NumberType &det_F,
               const Tensor<2, dim, NumberType> &F) const
               /** The tangent, $J\mathfrak{c}$
                *
                * @param C Right Cauchy-Green tensor
                * @param det_F Jacobian
                * @param F Deformation gradient
                * @return A symmetric tensor of rank 2
                */
       {
            return det_F *
                   Physics::Transformations::Piola::push_forward(get_LagrangeElasticityTensor(C, det_F, F), F);
        }

        const double c0; /**< Material constant 0*/
        const double c1; /**< Material constant 1*/
        const double c2; /**< Material constant 2*/
        const double d = 2 * (c1 + 2 * c2); /**< Volumetric constant*/

        SymmetricTensor<2, dim, NumberType>
        get_SecondPiolaStress(const SymmetricTensor<2, dim, NumberType> &C,
                              const NumberType &det_F) const
                              /** Get second Piola stress.
                               *
                               * @param C Right Cauchy-Green tensor
                               * @param det_F Jacobian
                               * @return  A symmetric tensor of rank 2
                               */
       {
            return 2 * (c1 + c2 * trace(C)) * unit_symmetric_tensor<dim>() - 2 * c2 * C +
                   (2 * c0 * det_F * (det_F - 1) - d) * invert(C);
        }


        SymmetricTensor<4, dim, NumberType>
        get_LagrangeElasticityTensor(const SymmetricTensor<2, dim, NumberType> &C,
                                     const NumberType &det_F,
                                     const Tensor<2, dim, NumberType> &F) const
                                     /** Get Lagrangian elasticity tensor.
                                      *
                                      * @param C Right Cauchy-Green tensor
                                      * @param det_F Jacobian
                                      * @param F Deformation gradient
                                      * @return A symmetric tensor of rank 4
                                      */
         {
            const SymmetricTensor<2, dim, NumberType> C_inv = invert(C);

            return 4 * c2 * Physics::Elasticity::StandardTensors<dim>::IxI +
                   2 * c0 * det_F * (2 * det_F - 1) * outer_product(C_inv, C_inv) +
                   2 * (2 * c0 * det_F * (det_F - 1) - d) * Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F) -
                   4 * c2 * Physics::Elasticity::StandardTensors<dim>::S;
        }

    };

    template<int dim>
    class GradientPostprocessor : public DataPostprocessorTensor<dim> {
    public:
        /** Post-processor to find the solution gradient
         *
         */
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
        /** Post-processor to find the deformation gradient
        *
        */
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

    template<int dim, typename NumberType>
    class CauchyStressPostprocessor : public DataPostprocessorTensor<dim> {
    public:
        /** Post-processor to find the Cauchy stress
      *
      */
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
            SymmetricTensor<2, dim, NumberType> stress;
            Tensor<2, dim, NumberType> F;
            Material_Compressible_Mooney_Rivlin_One_Field<dim, NumberType> material(parameters);
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

    template<int dim, typename NumberType>
    class LagrangeStrainPostprocessor : public DataPostprocessorTensor<dim> {
    public:
        /** Post-processor to find the Green-Lagrange strain
      *
      */
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
            SymmetricTensor<2, dim, NumberType> strain;
            Tensor<2, dim, NumberType> F;
            Material_Compressible_Mooney_Rivlin_One_Field<dim, NumberType> material(parameters);
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





// @sect3{Quadrature point history}
    template<int dim, typename NumberType>
    class PointHistory {
    public:
        /** PointHistory
         * Each quadrature point holds a pointer to the material description;
         * the Kirchhoff stress $\boldsymbol{\tau}$ and tangent $J\mathfrak{c}$
         * are stored in these points.
         */
        PointHistory() {}

        virtual ~PointHistory() {}

        // Create material object
        void setup_lqp(const Parameters::AllParameters &parameters) {
            material.reset(new Material_Compressible_Mooney_Rivlin_One_Field<dim, NumberType>(parameters));
        }
        // Update second Piola stress
        SymmetricTensor<2, dim, NumberType>
        get_SecondPiolaStress(const Tensor<2, dim, NumberType> &C,
                              const NumberType &det_F) const {
            return material->get_SecondPiolaStress(C, det_F);
        }
        // Update Green-Lagrange strain
        SymmetricTensor<2, dim, NumberType>
        get_GreenLagrangeStrain(const Tensor<2, dim, NumberType> &F) const {
            return material->get_GreenLagrangeStrain(F);
        }

        // Update Kirchhoff stress
        SymmetricTensor<2, dim, NumberType>
        get_tau(const SymmetricTensor<2, dim, NumberType> &C,
                const NumberType &det_F,
                const Tensor<2, dim, NumberType> &F) const {
            return material->get_tau(C, det_F, F);
        }

        // Update tangent
        SymmetricTensor<4, dim, NumberType>
        get_Jc(const SymmetricTensor<2, dim, NumberType> &C,
               const NumberType &det_F,
               const Tensor<2, dim, NumberType> &F) const {
            return material->get_Jc(C, det_F, F);
        }

    private:
        std::shared_ptr<Material_Compressible_Mooney_Rivlin_One_Field<dim, NumberType> > material;
    };


// @sect3{Quasi-static compressible finite-strain solid}

    template<int dim, typename NumberType>
    struct Assembler_Base;
    template<int dim, typename NumberType>
    struct Assembler;


    template<int dim, typename NumberType>
    class Solid {
    public:
        Solid(const Parameters::AllParameters &parameters);

        virtual
        ~Solid();

        void
        run();


    private:

        /**
         * Create triangulation object
         */
        void
        make_grid();

        /**
         * Set up finite element system
         */
        void
        system_setup();

        /**
         * Assemble the system and right hand side matrices in WorkStream
         */
        void
        assemble_system(const BlockVector<double> &solution_delta, const bool rhs_only);

        /**
         * Data structures with necessary objects for assembly
         */
        friend struct Assembler_Base<dim, NumberType>;
        friend struct Assembler<dim, NumberType>;

        /**
         * Apply Dirichlet boundary conditions
         * @param it_nr Newton iteration
         */
        void
        make_constraints(const int &it_nr);

        /**
         * Create and update the quadrature points
         */
        void
        setup_qph();


        /**
         * Solve for displacement using a Newton-Raphson scheme as implemented in Problem 1
         * @param solution_delta Solution increment $\nabla\boldsymbol{u}$
         */
        void
        solve_nonlinear_timestep(BlockVector<double> &solution_delta);

        /**
         * Solve for displacement using a SUNDIALS::KINSOL
         * @param solution_delta Solution increment $\nabla\boldsymbol{u}$
         * @return
         */
        unsigned int
        solve_nonlinear_timestep_kinsol(BlockVector<double> &solution_delta);

        std::unique_ptr<SparseDirectUMFPACK> jacobian_matrix_factorization;

        /**
         * Compute and factorise the tangent matrix
         * @param newton_update_in Newton update
         * @param newton_iteration Newton iteration
         */
        void
        compute_and_factorize_jacobian(const BlockVector<double> &newton_update_in, const double newton_iteration);

        /**
         * Solve the linear system using Kinsol
         * @param rhs Righthand side vector
         * @param present_solution Current solution vector
         */
        void
        solve_linear_kinsol(const BlockVector<double> &rhs,
                            BlockVector<double> &present_solution);

        /**
         * Solve linear system using UMFPACK
         * @param newton_update Newton update
         */
        std::pair<unsigned int, double>
        solve_linear_system(BlockVector<double> &newton_update);

        // Solution retrieval as well as post-processing and writing data to file:
        /**
         * Retrieve total solution
         * @param solution_delta Solution increment $\nabla\boldsymbol{u}$
         */
        BlockVector<double>
        get_total_solution(const BlockVector<double> &solution_delta) const;

        /**
         * Post-process solution and write to file.
         */
        void
        output_results() const;

        const Parameters::AllParameters &parameters; /**< Parameters in config file loaded*/

        double vol_reference; /**< Reference volume */
        double vol_current; /**< Current volume */

        Triangulation<dim> triangulation; /**< Geometry of the problem*/

        Time time; /**< Keep track of current time */
        TimerOutput timer; /**< Keep track of compute time*/

              CellDataStorage<typename Triangulation<dim>::cell_iterator,
                PointHistory<dim, NumberType> > quadrature_point_history; /**< Storage object for quadrature point information */


        // The finite element system:
        const unsigned int degree; /**< Solution polynomial degree*/
        const FESystem<dim> fe;
        DoFHandler<dim> dof_handler_ref;
        const unsigned int dofs_per_cell;
        const FEValuesExtractors::Vector u_fe; /**< Extractor object to retrieve information from solution vector*/

        // Block system:
        static const unsigned int n_blocks = 1;
        static const unsigned int n_components = dim;
        static const unsigned int first_u_component = 0;

        enum {
            u_dof = 0
        };

        std::vector<types::global_dof_index> dofs_per_block;

        // Gauss quadature rules:
        const QGauss<dim> qf_cell;
        const QGauss<dim - 1> qf_face;
        const unsigned int n_q_points;
        const unsigned int n_q_points_f;

        AffineConstraints<double> constraints; /**< Keep track of constraints*/
        BlockSparsityPattern sparsity_pattern;
        BlockSparseMatrix<double> tangent_matrix; /**< System tangent matrix*/
        BlockVector<double> system_rhs; /**< System right-hand side vector*/
        BlockVector<double> solution_n; /**< Current solution*/

        // Store and update norms:
        struct Errors {
            Errors()
                    :
                    norm(1.0), u(1.0) {}

            void reset() {
                norm = 1.0;
                u = 1.0;
            }

            void normalise(const Errors &rhs) {
                if (rhs.norm != 0.0)
                    norm /= rhs.norm;
                if (rhs.u != 0.0)
                    u /= rhs.u;
            }

            double norm, u;
        };

        Errors error_residual, error_residual_0, error_residual_norm, error_update,
                error_update_0, error_update_norm;

        /**
         * Get the residual error
         * @param error_residual
         */
        void
        get_error_residual(Errors &error_residual);
        /**
         * Update the error
         * @param newton_update
         * @param error_update
         */
        void
        get_error_update(const BlockVector<double> &newton_update,
                         Errors &error_update);

        /**
         * Print to screen header
         */
        static
        void
        print_conv_header();
        /**
         * Print to screen footer
         */
        void
        print_conv_footer();
        /**
         * Calculate the vertical displacement of the right, top tip of the membrane
         * and print to screen.
         */
        void
        print_vertical_tip_displacement();


    };

// @sect3{Implementation of the <code>Solid</code> class}

// @sect4{Public interface}

    template<int dim, typename NumberType>
    Solid<dim, NumberType>::Solid(const Parameters::AllParameters &parameters)
            :
            parameters(parameters),
            vol_reference(0.0),
            vol_current(0.0),
            triangulation(Triangulation<dim>::maximum_smoothing),
            time(parameters.end_time, parameters.delta_t),
            timer(std::cout,
                  TimerOutput::summary,
                  TimerOutput::wall_times),
            degree(parameters.poly_degree),
            // The Finite Element System is composed of dim continuous displacement
            // DOFs.
            fe(FE_Q<dim>(parameters.poly_degree), dim), /**< Displacement*/
            dof_handler_ref(triangulation),
            dofs_per_cell(fe.dofs_per_cell),
            u_fe(first_u_component),
            dofs_per_block(n_blocks),
            qf_cell(parameters.quad_order),
            qf_face(parameters.quad_order),
            n_q_points(qf_cell.size()),
            n_q_points_f(qf_face.size()) {}


    template<int dim, typename NumberType>
    Solid<dim, NumberType>::~Solid()
    /** Class destructor
     * Clears data held by the <code>DOFHandler/code>
     */
    {
        dof_handler_ref.clear();
    }



    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::run()
    /**
     * Start the simulation with pre-processing, output an intial grid and then start
     * the simulation proper.
     */
    {
        make_grid();
        system_setup();
        output_results();
        time.increment();
        if (parameters.nonlinear_solver == "kinsol")
            std::cout<<"Using the SUNDIALS::KINSOL nonlinear solver"<<std::endl;
        else if (parameters.nonlinear_solver == "newton")
            std::cout<<"Using the Problem 1 nonlinear solver"<<std::endl;
        else
            throw "No nonlinear solver chosen.";


        // Reset solution update
        BlockVector<double> solution_delta(dofs_per_block);
        while (time.current() <= time.end()) {
            solution_delta = 0.0;

            // Solve current timestep
            if (parameters.nonlinear_solver == "kinsol")
                solve_nonlinear_timestep_kinsol(solution_delta);
            else if (parameters.nonlinear_solver == "newton")
                solve_nonlinear_timestep(solution_delta);
            else
                throw "No nonlinear solver chosen.";

            // Increment solution
            solution_n += solution_delta;
            // Output results
            output_results();
            time.increment();
        }
        print_vertical_tip_displacement();

    }


// @sect3{Private interface}


// @sect4{Solid::grid_y_transfrom}

    template<int dim>
    Point<dim> grid_y_transform(const Point<dim> &pt_in)
    /** Transform geometry
     *
     * @tparam dim FE system dimension
     * @param pt_in Point to be transformed
     * @return Transformed point
     */
    {
        const double &x = pt_in[0];
        const double &y = pt_in[1];

        const double y_upper = 44.0 + (16.0 / 48.0) * x; // Line defining upper edge of beam
        const double y_lower = 0.0 + (44.0 / 48.0) * x; // Line defining lower edge of beam
        const double theta = y / 44.0; // Fraction of height along left side of beam
        const double y_transform = (1 - theta) * y_lower + theta * y_upper; // Final transformation

        Point<dim> pt_out = pt_in;
        pt_out[1] = y_transform;

        return pt_out;
    }

// @sect4{Solid::make_grid}

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::make_grid()
    /** Create triangulation object
     *
     * @tparam dim FE system dimension
     * @tparam NumberType
     */
     {
        std::cout << "Polynomial degree: Q" << parameters.poly_degree << std::endl;
        std::vector<unsigned int> repetitions(dim, parameters.elements_per_edge);
        if (dim == 3)
            repetitions[dim - 1] = 1;

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
                if (cell->face(face)->at_boundary() == true) {
                    if (std::abs(cell->face(face)->center()[0] - 0.0) < tol_boundary)
                        cell->face(face)->set_boundary_id(1); /**< -X faces*/
                    else if (std::abs(cell->face(face)->center()[0] - 48.0) < tol_boundary)
                        cell->face(face)->set_boundary_id(11); /**< +X faces*/
                    else if (dim == 3 && std::abs(std::abs(cell->face(face)->center()[2]) - 0.5) < tol_boundary)
                        cell->face(face)->set_boundary_id(2); /**< +Z and -Z faces*/
                }

        GridTools::transform(&grid_y_transform<dim>, triangulation);

        GridTools::scale(parameters.scale, triangulation);

        vol_reference = GridTools::volume(triangulation);
        vol_current = vol_reference;
        std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;

    }


// @sect4{Solid::system_setup}

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::system_setup()
    /** Set up the FE system
     *
     * @tparam dim FE system dimension
     * @tparam NumberType
     */
    {
        timer.enter_subsection("Setup system");

        std::vector<unsigned int> block_component(n_components, u_dof); // Displacement

        // Initialise DOF handler
        dof_handler_ref.distribute_dofs(fe);
        // Renumber the grid
        DoFRenumbering::Cuthill_McKee(dof_handler_ref);
        DoFRenumbering::component_wise(dof_handler_ref, block_component);
        dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler_ref, block_component);

        std::cout << "Triangulation:"
                  << "\n\t Number of active cells: " << triangulation.n_active_cells()
                  << "\n\t Number of degrees of freedom: " << dof_handler_ref.n_dofs()
                  << std::endl;

        // Set up the sparsity pattern and tangent matrix
        tangent_matrix.clear();
        {
            const types::global_dof_index n_dofs_u = dofs_per_block[u_dof];

            BlockDynamicSparsityPattern csp(n_blocks, n_blocks);


            csp.block(u_dof, u_dof).reinit(n_dofs_u, n_dofs_u);
            csp.collect_sizes();


            Table<2, DoFTools::Coupling> coupling(n_components, n_components);
            for (unsigned int ii = 0; ii < n_components; ++ii)
                for (unsigned int jj = 0; jj < n_components; ++jj)
                    coupling[ii][jj] = DoFTools::always;
            DoFTools::make_sparsity_pattern(dof_handler_ref,
                                            coupling,
                                            csp,
                                            constraints,
                                            false);
            sparsity_pattern.copy_from(csp);
        }

        tangent_matrix.reinit(sparsity_pattern);

        // System RHs storage vector
        system_rhs.reinit(dofs_per_block);
        system_rhs.collect_sizes();

        // Current solution storage vector
        solution_n.reinit(dofs_per_block);
        solution_n.collect_sizes();

        // Set up quadrature point history
        setup_qph();

        timer.leave_subsection();
    }


// @sect4{Solid::setup_qph}

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::setup_qph()
    /** Set up quadrature point data
     *
     * @tparam dim FE system dimension
     * @tparam NumberType
     */
    {
        std::cout << "    Setting up quadrature point data..." << std::endl;

        quadrature_point_history.initialize(triangulation.begin_active(),
                                            triangulation.end(),
                                            n_q_points);

        // Set up initial quadrature point data
        for (typename Triangulation<dim>::active_cell_iterator cell =
                triangulation.begin_active(); cell != triangulation.end(); ++cell) {
            const std::vector<std::shared_ptr<PointHistory<dim, NumberType> > > lqph =
                    quadrature_point_history.get_data(cell); /**< Vector of smart pointers to quadrature point data*/
            Assert(lqph.size() == n_q_points, ExcInternalError());

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                lqph[q_point]->setup_lqp(parameters);
        }
    }


// @sect4{Solid::solve_nonlinear_timestep}

    template<int dim, typename NumberType>
    void
    Solid<dim, NumberType>::solve_nonlinear_timestep(BlockVector<double> &solution_delta)
    /** Solver nonlinear timestep using the Newton-Raphson scheme
     * As per Problem 1
     * @tparam dim FE system dimension
     * @tparam NumberType
     * @param solution_delta Solution increment $\nabla \boldsymbol{u}$
     */
    {
        std::cout << std::endl << "Timestep " << time.get_timestep() << " @ "
                  << time.current() << "s" << std::endl;

        BlockVector<double> newton_update(dofs_per_block);

        error_residual.reset();
        error_residual_0.reset();
        error_residual_norm.reset();
        error_update.reset();
        error_update_0.reset();
        error_update_norm.reset();

        print_conv_header();

        // Perform a number of Newton iterations to iteratively solve
        unsigned int newton_iteration = 0;
        for (; newton_iteration < parameters.max_iterations_NR;
               ++newton_iteration) {
            std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;

            // Impose Dirichlet boundary conditions
            make_constraints(newton_iteration);
            // Assemble the tangent matrix and residual
            assemble_system(solution_delta, /*rhs_only*/ false);

            get_error_residual(error_residual);

            if (newton_iteration == 0)
                error_residual_0 = error_residual;

            // Determine normalised residual error
            error_residual_norm = error_residual;
            error_residual_norm.normalise(error_residual_0);
            // Check for convergence
            if (newton_iteration > 0 && error_update_norm.u <= parameters.tol_u
                && error_residual_norm.u <= parameters.tol_f) {
                std::cout << " CONVERGED! " << std::endl;
                print_conv_footer();

                break;
            }

            const std::pair<unsigned int, double>
                    lin_solver_output = solve_linear_system(newton_update);

            get_error_update(newton_update, error_update);
            if (newton_iteration == 0)
                error_update_0 = error_update;


            // Determine the normalised Newton update error
            error_update_norm = error_update;
            error_update_norm.normalise(error_update_0);
            // Increment the solution
            solution_delta += newton_update;

            std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                      << std::scientific << lin_solver_output.first << "  "
                      << lin_solver_output.second << "  " << error_residual_norm.norm
                      << "  " << error_residual_norm.u << "  "
                      << "  " << error_update_norm.norm << "  " << error_update_norm.u
                      << "  " << std::endl;
        }

        AssertThrow(newton_iteration <= parameters.max_iterations_NR,
                    ExcMessage("No convergence in nonlinear solver!"));
    }


// @sect4{Solid::print_conv_header, Solid::print_conv_footer and Solid::print_vertical_tip_displacement}

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::print_conv_header()
    /** Print table header
     *
     */
    {
        static const unsigned int l_width = 87;

        for (unsigned int i = 0; i < l_width; ++i)
            std::cout << "_";
        std::cout << std::endl;

        std::cout << "    SOLVER STEP    "
                  << " |  LIN_IT   LIN_RES    RES_NORM    "
                  << " RES_U     NU_NORM     "
                  << " NU_U " << std::endl;

        for (unsigned int i = 0; i < l_width; ++i)
            std::cout << "_";
        std::cout << std::endl;
    }


    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::print_conv_footer()
    /** Print table footer
     *
     * @tparam dim FE system dimension
     * @tparam NumberType
     */
    {
        static const unsigned int l_width = 87;

        for (unsigned int i = 0; i < l_width; ++i)
            std::cout << "_";
        std::cout << std::endl;

        std::cout << "Relative errors:" << std::endl
                  << "Displacement:\t" << error_update.u / error_update_0.u << std::endl
                  << "Force: \t\t" << error_residual.u / error_residual_0.u << std::endl
                  << "v / V_0:\t" << vol_current << " / " << vol_reference
                  << std::endl;
    }

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::print_vertical_tip_displacement()
    /** Compute and print the vertical tip displacement
     *
     * @tparam dim FE system dimension
     * @tparam NumberType
     */
    {
        static const unsigned int l_width = 87;

        for (unsigned int i = 0; i < l_width; ++i)
            std::cout << "_";
        std::cout << std::endl;
        const Point<dim> soln_pt = (dim == 3 ?
                                    Point<dim>(48.0 * parameters.scale, 52.0 * parameters.scale, 0.5 * parameters.scale)
                                             :
                                    Point<dim>(48.0 * parameters.scale, 52.0 * parameters.scale));
        double vertical_tip_displacement = 0.0;
        double vertical_tip_displacement_check = 0.0;

        typename DoFHandler<dim>::active_cell_iterator cell =
                dof_handler_ref.begin_active(), endc = dof_handler_ref.end();
        for (; cell != endc; ++cell) {
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
                if (cell->vertex(v).distance(soln_pt) < 1e-6) {
                    vertical_tip_displacement = solution_n(cell->vertex_dof_index(v, u_dof + 1));

                    const MappingQ<dim> mapping(parameters.poly_degree);
                    const Point<dim> qp_unit = mapping.transform_real_to_unit_cell(cell, soln_pt);
                    const Quadrature<dim> soln_qrule(qp_unit);
                    AssertThrow(soln_qrule.size() == 1, ExcInternalError());
                    FEValues<dim> fe_values_soln(fe, soln_qrule, update_values);
                    fe_values_soln.reinit(cell);
                    std::vector<Tensor<1, dim>> soln_values(soln_qrule.size());
                    fe_values_soln[u_fe].get_function_values(solution_n,
                                                             soln_values);
                    vertical_tip_displacement_check = soln_values[0][u_dof + 1];
                    break;
                }
        }
        AssertThrow(vertical_tip_displacement > 0.0, ExcMessage("Found no cell with point inside!"))
        std::cout << "Vertical tip displacement: " << vertical_tip_displacement
                  << "\t Check: " << vertical_tip_displacement_check << std::endl;
    }



// @sect4{Solid::get_error_residual}

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::get_error_residual(Errors &error_residual) {
        BlockVector<double> error_res(dofs_per_block);

        for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
            if (!constraints.is_constrained(i))
                error_res(i) = system_rhs(i);

        error_residual.norm = error_res.l2_norm();
        error_residual.u = error_res.block(u_dof).l2_norm();
    }


// @sect4{Solid::get_error_update}

// Determine the true Newton update error for the problem
    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::get_error_update(const BlockVector<double> &newton_update,
                                                  Errors &error_update) {
        BlockVector<double> error_ud(dofs_per_block);
        for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
            if (!constraints.is_constrained(i))
                error_ud(i) = newton_update(i);

        error_update.norm = error_ud.l2_norm();
        error_update.u = error_ud.block(u_dof).l2_norm();
    }



// @sect4{Solid::get_total_solution}

    template<int dim, typename NumberType>
    BlockVector<double>
    Solid<dim, NumberType>::get_total_solution(const BlockVector<double> &solution_delta) const
    /** Get the total solution vector
     * @param solution_delta Solution increment $\nabla \boldsymbol{u}$
     */
    {
        BlockVector<double> solution_total(solution_n);
        solution_total += solution_delta;
        return solution_total;
    }


// @sect4{Solid::assemble_system}

    template<int dim, typename NumberType>
    struct Assembler_Base {
        virtual ~Assembler_Base() {}

        /**
         * Stores local contributions to the system matrix and RHS
         */
        struct Local_ASM{
            const Solid<dim, NumberType> *solid;
            FullMatrix<double> cell_matrix; /**< Local stiffness matrix*/
            Vector<double> cell_rhs; /**< Local RHS vector*/
            std::vector<types::global_dof_index> local_dof_indices;
            bool rhs_only;

            Local_ASM(const Solid<dim, NumberType> *solid, const double rhs_only_in)
                    :
                    solid(solid),
                    cell_matrix(solid->dofs_per_cell, solid->dofs_per_cell),
                    cell_rhs(solid->dofs_per_cell),
                    local_dof_indices(solid->dofs_per_cell),
                    rhs_only(rhs_only_in) {}

            void reset() {
                cell_matrix = 0.0;
                cell_rhs = 0.0;
            }
        };


        /**
         * Stores larger objects for assembly, such as the shape-function array (<code>Nx</code>), gradient vectors, etc.
         */
        struct ScratchData_ASM {
            const BlockVector<double> &solution_total;
            std::vector<Tensor<2, dim, NumberType> > solution_grads_u_total;

            FEValues<dim> fe_values_ref;
            FEFaceValues<dim> fe_face_values_ref;

            std::vector<std::vector<Tensor<2, dim, NumberType> > > grad_Nx; /**< Shape function gradient*/
            std::vector<std::vector<SymmetricTensor<2, dim, NumberType> > >
                    symm_grad_Nx; /**< Symmetric shape function gradient*/

            bool rhs_only;

            ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                            const QGauss<dim> &qf_cell,
                            const UpdateFlags uf_cell,
                            const QGauss<dim - 1> &qf_face,
                            const UpdateFlags uf_face,
                            const BlockVector<double> &solution_total,
                            const bool rhs_only_in)
                    :
                    solution_total(solution_total),
                    solution_grads_u_total(qf_cell.size()),
                    fe_values_ref(fe_cell, qf_cell, uf_cell),
                    fe_face_values_ref(fe_cell, qf_face, uf_face),
                    grad_Nx(qf_cell.size(),
                            std::vector<Tensor<2, dim, NumberType> >(fe_cell.dofs_per_cell)),
                    symm_grad_Nx(qf_cell.size(),
                                 std::vector<SymmetricTensor<2, dim, NumberType> >
                                         (fe_cell.dofs_per_cell)),
                    rhs_only(rhs_only_in) {}

            ScratchData_ASM(const ScratchData_ASM &rhs)
                    :
                    solution_total(rhs.solution_total),
                    solution_grads_u_total(rhs.solution_grads_u_total),
                    fe_values_ref(rhs.fe_values_ref.get_fe(),
                                  rhs.fe_values_ref.get_quadrature(),
                                  rhs.fe_values_ref.get_update_flags()),
                    fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                                       rhs.fe_face_values_ref.get_quadrature(),
                                       rhs.fe_face_values_ref.get_update_flags()),
                    grad_Nx(rhs.grad_Nx),
                    symm_grad_Nx(rhs.symm_grad_Nx) {}

            void reset() {
                const unsigned int n_q_points = fe_values_ref.get_quadrature().size();
                const unsigned int n_dofs_per_cell = fe_values_ref.dofs_per_cell;
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
                    Assert(grad_Nx[q_point].size() == n_dofs_per_cell,
                           ExcInternalError());
                    Assert(symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                           ExcInternalError());

                    solution_grads_u_total[q_point] = Tensor<2, dim, NumberType>();
                    for (unsigned int k = 0; k < n_dofs_per_cell; ++k) {
                        grad_Nx[q_point][k] = Tensor<2, dim, NumberType>();
                        symm_grad_Nx[q_point][k] = SymmetricTensor<2, dim, NumberType>();
                    }
                }
            }

        };

        void
        assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 ScratchData_ASM &scratch,
                                 Local_ASM &data)
                                 /**
                                  * Assemble the tangent matrix contribution for a single cell
                                  * @param cell
                                  * @param scratch Shape functions, shape function gradients, etc.
                                  * @param data Cell data
                                  */
                                 {
            // Assemble local stiffness matrix and residual
            assemble_system_tangent_residual_one_cell(cell, scratch, data);
            // Assemble Neumann contribution
                                     assemble_neumann_contribution_one_cell(cell, scratch, data);
        }

        void
        copy_local_to_global_ASM(const Local_ASM &data)
        /**
         * Add the local contributions to the system matrix
         */
        {
            const AffineConstraints<double> &constraints = data.solid->constraints;
            BlockSparseMatrix<double> &tangent_matrix = const_cast<Solid<dim, NumberType> *>(data.solid)->tangent_matrix;
            BlockVector<double> &system_rhs = const_cast<Solid<dim, NumberType> *>(data.solid)->system_rhs;

            // Assemble global tangent matrix and residual
            if (data.rhs_only == false) {
                constraints.distribute_local_to_global(
                        data.cell_matrix, data.cell_rhs,
                        data.local_dof_indices,
                        tangent_matrix, system_rhs);


            }
            // Assemble residual only
            else
            {
                constraints.distribute_local_to_global(
                        data.cell_rhs,
                        data.local_dof_indices,
                        system_rhs);
            }

        }

    protected:

        // This function needs to exist in the base class for
        // Workstream to work with a reference to the base class.
        virtual void
        assemble_system_tangent_residual_one_cell(const typename DoFHandler<dim>::active_cell_iterator &/*cell*/,
                                                  ScratchData_ASM &/*scratch*/,
                                                  Local_ASM &/*data*/) {
            AssertThrow(false, ExcPureFunctionCalled());
        }

        void
        assemble_neumann_contribution_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                               ScratchData_ASM &scratch, Local_ASM &data)
                                               /**
                                                * Assemble Neumann contribution for the local residual
                                                * @param cell
                                                * @param scratch Shape functions, shape function gradients, etc.
                                                * @param data Cell data
                                                */
                                               {

            const unsigned int &n_q_points_f = data.solid->n_q_points_f;
            const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
            const Parameters::AllParameters &parameters = data.solid->parameters;
            const Time &time = data.solid->time;
            const FESystem<dim> &fe = data.solid->fe;
            const unsigned int &u_dof = data.solid->u_dof;

            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
                 ++face)
                // Check if on the Neumann boundary
                if (cell->face(face)->at_boundary() == true
                    && cell->face(face)->boundary_id() == 11) {
                    scratch.fe_face_values_ref.reinit(cell, face);

                    for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                         ++f_q_point) {

                        const double time_ramp = (time.current() / time.end());
                        const double magnitude = (parameters.traction/ (16.0 * parameters.scale * 1.0 * parameters.scale)) * time_ramp; /**< Traction*/
                        Tensor<1, dim> dir;
                        // Specify traction in reference configuration
                        dir[1] = 1.0;
                        const Tensor<1, dim> traction = magnitude * dir;

                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            const unsigned int i_group =
                                    fe.system_to_base_index(i).first.first;

                            if (i_group == u_dof) {
                                const unsigned int component_i =
                                        fe.system_to_component_index(i).first;
                                const double Ni =
                                        scratch.fe_face_values_ref.shape_value(i,
                                                                               f_q_point);
                                const double JxW = scratch.fe_face_values_ref.JxW(
                                        f_q_point);

                                data.cell_rhs(i) += (Ni * traction[component_i])
                                                    * JxW;
                            }
                        }
                    }
                }
        }
    };

    template<int dim>
    struct Assembler<dim, double> : Assembler_Base<dim, double> {
        typedef double NumberType;
        using typename Assembler_Base<dim, NumberType>::ScratchData_ASM;
        using typename Assembler_Base<dim, NumberType>::Local_ASM;

        virtual ~Assembler() {}

        virtual void
        assemble_system_tangent_residual_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                  ScratchData_ASM &scratch,
                                                  Local_ASM &data)
                                                  /**
                                                   * Assemble the local tangent matrix and residual
                                                   * @param cell
                                                   * @param scratch
                                                   * @param data
                                                   */
                                                  {
            const unsigned int &n_q_points = data.solid->n_q_points;
            const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
            const FESystem<dim> &fe = data.solid->fe;
            const unsigned int &u_dof = data.solid->u_dof;
            const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;


            data.reset();
            scratch.reset();
            scratch.fe_values_ref.reinit(cell);
            cell->get_dof_indices(data.local_dof_indices);

            const std::vector<std::shared_ptr<const PointHistory<dim, NumberType> > > lqph =
                    const_cast<const Solid<dim, NumberType> *>(data.solid)->quadrature_point_history.get_data(cell);
            Assert(lqph.size() == n_q_points, ExcInternalError());


            // Find the solution gradients in the current cell and update solution
            scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total,
                                                               scratch.solution_grads_u_total);


            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            /**
             * Build the local stiffness matrix while making use of its symmetry by calculating only the
             * entries in the lower half. These are copied into the top half.
             */
            {
                const Tensor<2, dim, NumberType> &grad_u = scratch.solution_grads_u_total[q_point];
                const Tensor<2, dim, NumberType> F = Physics::Elasticity::Kinematics::F(grad_u);
                const NumberType det_F = determinant(F);
                const SymmetricTensor<2, dim, NumberType> C = Physics::Elasticity::Kinematics::C(F);
                const Tensor<2, dim, NumberType> F_inv = invert(F);
                Assert(det_F > NumberType(0.0), ExcInternalError());
                for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                    const unsigned int k_group = fe.system_to_base_index(k).first.first;

                    if (k_group == u_dof) {
                        scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv; /**< Grad(u)*/
                        scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);/**< Symm_Grad(u)*/
                    } else Assert(k_group <= u_dof, ExcInternalError());
                }

                const SymmetricTensor<2, dim, NumberType> tau = lqph[q_point]->get_tau(C, det_F, F); /**< Kirchhoff stress */
                const SymmetricTensor<4, dim, NumberType> Jc = lqph[q_point]->get_Jc(C, det_F, F); /**< Tangent */
                const Tensor<2, dim, NumberType> tau_ns(tau);


                const std::vector<SymmetricTensor<2, dim> > &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
                const std::vector<Tensor<2, dim> > &grad_Nx = scratch.grad_Nx[q_point];
                const double JxW = scratch.fe_values_ref.JxW(q_point);

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const unsigned int component_i = fe.system_to_component_index(i).first;
                    const unsigned int i_group = fe.system_to_base_index(i).first.first;

                    if (i_group == u_dof)
                        data.cell_rhs(i) -= (symm_grad_Nx[i] * tau) * JxW;
                    else Assert(i_group <= u_dof, ExcInternalError());

                    if (data.rhs_only == false) {
                        for (unsigned int j = 0; j <= i; ++j) {
                            const unsigned int component_j = fe.system_to_component_index(j).first;
                            const unsigned int j_group = fe.system_to_base_index(j).first.first;

                            if ((i_group == j_group) && (i_group == u_dof)) {
                                data.cell_matrix(i, j) += symm_grad_Nx[i] * Jc
                                                          * symm_grad_Nx[j] * JxW; /**<The material contribution*/
                                if (component_i == component_j)
                                    data.cell_matrix(i, j) += grad_Nx[i][component_i] * tau_ns
                                                              * grad_Nx[j][component_j] * JxW; /**< The geometrical stress contribution*/
                            } else Assert((i_group <= u_dof) && (j_group <= u_dof),
                                          ExcInternalError());
                        }
                    }
                }
            }


            // Copy the lower half of the local matrix into the upper half:
            if (data.rhs_only == false) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = i + 1; j < dofs_per_cell; ++j) {
                        data.cell_matrix(i, j) = data.cell_matrix(j, i);
                    }
            }
        }
    };



    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::assemble_system(const BlockVector<double> &solution_delta, const bool rhs_only)
    /**
     * Assemble system using WorkStream
     * @tparam dim FE system dimension
     * @tparam NumberType
     * @param solution_delta Solution increment $\nabla \boldsymbol{u}$
     * @param rhs_only
     */
    {
        timer.enter_subsection("Assemble linear system");
        if (rhs_only == false) {
            std::cout << " ASM_T " << std::flush;
            tangent_matrix = 0.0;
        } else {
            std::cout << " ASM_R " << std::flush;
        }

        system_rhs = 0.0;

        const UpdateFlags uf_cell(update_gradients |
                                  update_JxW_values);
        const UpdateFlags uf_face(update_values |
                                  update_JxW_values);

        const BlockVector<double> solution_total(get_total_solution(solution_delta));
        typename Assembler_Base<dim, NumberType>::Local_ASM per_task_data(this, rhs_only);
        typename Assembler_Base<dim, NumberType>::ScratchData_ASM scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face,
                                                                               solution_total, rhs_only);
        Assembler<dim, NumberType> assembler;

        WorkStream::run(dof_handler_ref.begin_active(),
                        dof_handler_ref.end(),
                        static_cast<Assembler_Base<dim, NumberType> &>(assembler),
                        &Assembler_Base<dim, NumberType>::assemble_system_one_cell,
                        &Assembler_Base<dim, NumberType>::copy_local_to_global_ASM,
                        scratch_data,
                        per_task_data);

        timer.leave_subsection();
    }


// @sect4{Solid::make_constraints}
    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::make_constraints(const int &it_nr)
    /**
     * Impose Dirichlet boundary conditions
     * @tparam dim FE system dimension
     * @tparam NumberType
     * @param it_nr Newton iteration number
     */
    {
        std::cout << " CST " << std::flush;

        if (it_nr > 1)
            return;
        const bool apply_dirichlet_bc = (it_nr == 0);

        if (apply_dirichlet_bc) {
            constraints.clear();

            {
                double time_end = time.end();
                double delta_t = time.get_delta_t();

                {
                    const int boundary_id = 1; /**< -X faces*/

                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                             boundary_id,
                                                             Functions::ZeroFunction<dim>(n_components),
                                                             constraints,
                                                             fe.component_mask(u_fe));
                }

                if (dim == 3) {
                    const int boundary_id = 2; /**< -Z and +Z faces*/
                    const FEValuesExtractors::Scalar z_displacement(2);
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                             boundary_id,
                                                             Functions::ZeroFunction<dim>(n_components),
                                                             constraints,
                                                             fe.component_mask(z_displacement));
                }

            }


        } else {
            if (constraints.has_inhomogeneities()) {
                AffineConstraints<double> homogeneous_constraints(constraints);
                for (unsigned int dof = 0; dof != dof_handler_ref.n_dofs(); ++dof)
                    if (homogeneous_constraints.is_inhomogeneously_constrained(dof))
                        homogeneous_constraints.set_inhomogeneity(dof, 0.0);
                constraints.clear();
                constraints.copy_from(homogeneous_constraints);
            }
        }

        constraints.close();
    }

// @sect4{Solid::solve_linear_system}

    template<int dim, typename NumberType>
    std::pair<unsigned int, double>
    Solid<dim, NumberType>::solve_linear_system(BlockVector<double> &newton_update) {
        BlockVector<double> A(dofs_per_block);
        BlockVector<double> B(dofs_per_block);

        unsigned int lin_it = 0;
        double lin_res = 0.0;


        {   // Solve for incremental displacement
            timer.enter_subsection("Linear solver");
            std::cout << " SLV " << std::flush;
            if (parameters.type_lin == "CG")
            /**
             * Solve using the Conjugate Gradient method
             */
            {
                const int solver_its = static_cast<unsigned int>(
                        tangent_matrix.block(u_dof, u_dof).m()
                        * parameters.max_iterations_lin);
                const double tol_sol = parameters.tol_lin
                                       * system_rhs.block(u_dof).l2_norm();

                SolverControl solver_control(solver_its, tol_sol);

                GrowingVectorMemory<Vector<double> > GVM;
                SolverCG<Vector<double> > solver_CG(solver_control, GVM);


                PreconditionSelector<SparseMatrix<double>, Vector<double> >
                        preconditioner(parameters.preconditioner_type,
                                       parameters.preconditioner_relaxation);
                preconditioner.use_matrix(tangent_matrix.block(u_dof, u_dof));

                solver_CG.solve(tangent_matrix.block(u_dof, u_dof),
                                newton_update.block(u_dof),
                                system_rhs.block(u_dof),
                                preconditioner);

                lin_it = solver_control.last_step();
                lin_res = solver_control.last_value();
            }

            else if (parameters.type_lin == "Direct")
            /**
             * Solve using the direct linear solver, UMFPACK
             */
            {
                SparseDirectUMFPACK A_direct;
                A_direct.initialize(tangent_matrix.block(u_dof, u_dof));
                A_direct.vmult(newton_update.block(u_dof), system_rhs.block(u_dof));

                lin_it = 1;
                lin_res = 0.0;
            } else Assert(false, ExcMessage("Linear solver type not implemented"));

            timer.leave_subsection();
        }


        // Distribute constraints to Newton update
        constraints.distribute(newton_update);

        return std::make_pair(lin_it, lin_res);
    }

// @sect4{Solid::output_results}

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::output_results() const
    /**
     * Post-process solution and write to file.
     * @tparam dim FE system dimension
     * @tparam NumberType
     */
    {
        DataOut<dim> data_out;
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

        std::vector<std::string> solution_name(dim, "Displacement");

        data_out.attach_dof_handler(dof_handler_ref);
        data_out.add_data_vector(solution_n,
                                 solution_name,
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);

        GradientPostprocessor<dim> grad_post; /**< Solution gradient post-processor object*/
        data_out.add_data_vector(solution_n, grad_post);
        DeformationGradientPostprocessor<dim> F_post; /**< Deformation gradient post-processor object*/
        data_out.add_data_vector(solution_n, F_post);
        CauchyStressPostprocessor<dim, NumberType> sig_post(parameters); /**< Cauchy stress post-processor object*/
        data_out.add_data_vector(solution_n, sig_post);
        LagrangeStrainPostprocessor<dim, NumberType> E_post(parameters); /**< Green-Lagrange strain post-processor object*/
        data_out.add_data_vector(solution_n, E_post);


        data_out.build_patches();
        std::ostringstream filename;
        string file;

        // Write to file
        file = "Q" + to_string(parameters.poly_degree) + "T" + to_string(parameters.traction)+ "cooks-membrane_soln-" + to_string(time.get_timestep()) +
               ".vtu";
        std::ofstream output(file.c_str());
        data_out.write_vtu(output);

    }
}


#endif // COOKSMEMBRANE_H