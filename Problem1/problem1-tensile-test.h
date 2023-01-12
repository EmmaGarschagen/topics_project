#ifndef EMMA_TOPICS_PROJECT_PROBLEM1_TENSILE_TEST_H
#define EMMA_TOPICS_PROJECT_PROBLEM1_TENSILE_TEST_H
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

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/config.h>

#include "time.h"
#include "parameters_config.h"

#if DEAL_II_VERSION_MAJOR >= 9 && defined(DEAL_II_WITH_TRILINOS)

#include <deal.II/differentiation/ad.h>

#endif


#include <deal.II/physics/transformations.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <iostream>
#include <fstream>
#include <memory>


bool almost_equals(const double &a,
                   const double &b,
                   const double &tol = 1e-8);

namespace Tensile_Test {
    using namespace dealii;
    using namespace Physics::Transformations;
    using namespace Physics::Elasticity;
    using namespace std;





// @sect3{Compressible neo-Hookean material within a one-field formulation}

    template<int dim, typename NumberType>
    class Material_Compressible_Neo_Hook_One_Field {
    public:
        Material_Compressible_Neo_Hook_One_Field(const double mu,
                                                 const double nu)
                :
                c_1(mu / 2.0),
                beta((nu) / (1 - 2 * nu)) {}

        ~Material_Compressible_Neo_Hook_One_Field() {}


        NumberType
        get_Psi(const NumberType &det_F,
                const SymmetricTensor<2, dim, NumberType> &C) const {
            return (c_1 / beta) * (std::pow(det_F, -2 * beta) - 1) + c_1 * (trace(C) - dim);
        }

        SymmetricTensor<2, dim, NumberType>
        get_tau(const NumberType &det_F,
                const Tensor<2, dim, NumberType> &F,
                const Tensor<2, dim, NumberType> &C_inv) {
            return det_F * get_CauchyStress(det_F, F, C_inv);
        }

        SymmetricTensor<2, dim, NumberType>
        get_GreenLagrangeStrain(const Tensor<2, dim, NumberType> &F) const {
            return 0.5 * (symmetrize(transpose(F) * F) - unit_symmetric_tensor<dim, NumberType>());
        }


        SymmetricTensor<2, dim, NumberType>
        get_CauchyStress(const NumberType &det_F,
                         const Tensor<2, dim, NumberType> &F,
                         const Tensor<2, dim, NumberType> &C_inv) {
            return symmetrize(pow(det_F, -1) * F * get_SecondPiolaStress(det_F, C_inv) * transpose(F));
        }

        SymmetricTensor<4, dim, NumberType>
        get_Jc(const NumberType &det_F,
               const SymmetricTensor<2, dim, NumberType> &C_inv,
               const Tensor<2, dim, NumberType> &F) const {
            return det_F *
                   Physics::Transformations::Piola::push_forward(get_LagrangeElasticityTensor(det_F, C_inv, F), F);
        }

        SymmetricTensor<2, dim, NumberType>
        get_CauchyStress(const Tensor<2, dim, NumberType> &F) {

            const NumberType det_F = determinant(F);
            const Tensor<2, dim, NumberType> C_inv = invert(transpose(F) * F);

            return this->get_CauchyStress(det_F, F, C_inv);
        } //Used by post-processors


        SymmetricTensor<2, dim, NumberType>
        get_SecondPiolaStress(const Tensor<2, dim, NumberType> &F) {
            const NumberType det_F = determinant(F);
            const Tensor<2, dim, NumberType> C_inv = inverse(transpose(F) * F);

            return this->get_SecondPiolaStress(det_F, C_inv);
        } //Used by post-processors

    private:

        const double c_1;
        const double beta;

        SymmetricTensor<2, dim, NumberType>
        get_SecondPiolaStress(const NumberType &det_F,
                              const Tensor<2, dim, NumberType> &C_inv) {
            return symmetrize(2 * c_1 * (Physics::Elasticity::StandardTensors<dim>::I - pow(det_F, -2 * beta) * C_inv));
        }

        SymmetricTensor<4, dim, NumberType>
        get_LagrangeElasticityTensor(const NumberType &det_F,
                                     const SymmetricTensor<2, dim, NumberType> &C_inv,
                                     const Tensor<2, dim, NumberType> &F) const {
            return 4 * c_1 * pow(det_F, -2 * beta) *
                   (beta * outer_product(C_inv, C_inv) - Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F));
        }

    };

// @sect3{Post-processors}

// @sect4{Gradient post-processor}
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

// @sect4{Deformation gradient post-processor}
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

// @sect4{Cauchy stress post-processor}
    template<int dim, typename NumberType>
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
            SymmetricTensor<2, dim, NumberType> stress;
            Tensor<2, dim, NumberType> F;
            Material_Compressible_Neo_Hook_One_Field<dim, NumberType> material(parameters.mu, parameters.nu);
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

// @sect4{Lagrange strain post-processor}
    template<int dim, typename NumberType>
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
            SymmetricTensor<2, dim, NumberType> strain;
            Tensor<2, dim, NumberType> F;
            Material_Compressible_Neo_Hook_One_Field<dim, NumberType> material(parameters.mu, parameters.nu);
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

//Here each quadrature point holds a pointer to a material description.
//Each point stores the Kirchhoff stress $\boldsymbol{\tau}$ and the tangent
//$J\mathfrak{c}$ for the quadrature points.

    template<int dim, typename NumberType>
    class PointHistory {
    public:
        PointHistory() {}

        virtual ~PointHistory() {}

        void setup_lqp(const Parameters::AllParameters &parameters) {
            material.reset(new Material_Compressible_Neo_Hook_One_Field<dim, NumberType>(parameters.mu,
                                                                                         parameters.nu));
        }

        NumberType
        get_Psi(const NumberType &det_F,
                const SymmetricTensor<2, dim, NumberType> &C) const {
            return material->get_Psi(det_F, C);
        }

        SymmetricTensor<2, dim, NumberType>
        get_SecondPiolaStress(const NumberType &det_F,
                              const Tensor<2, dim, NumberType> &C_inv) const {
            return material->get_SecondPiolaStress(det_F, C_inv);
        }

        SymmetricTensor<2, dim, NumberType>
        get_GreenLagrangeStrain(const Tensor<2, dim, NumberType> &F) const {
            return material->get_GreenLagrangeStrain(F);
        }

        SymmetricTensor<2, dim, NumberType>
        get_tau(const NumberType &det_F,
                const Tensor<2, dim, NumberType> &F,
                const Tensor<2, dim, NumberType> &C_inv) const {
            return material->get_tau(det_F, F, C_inv);
        }

        SymmetricTensor<4, dim, NumberType>
        get_Jc(const NumberType &det_F,
               const SymmetricTensor<2, dim, NumberType> &C_inv,
               const Tensor<2, dim, NumberType> &F) const {
            return material->get_Jc(det_F, C_inv, F);
        }

        SymmetricTensor<4, dim, NumberType>
        get_LagrangeElasticityTensor(const NumberType &det_F,
                                     const SymmetricTensor<2, dim, NumberType> &C_inv,
                                     const Tensor<2, dim, NumberType> &F) const {
            return material->get_LagrangeElasticityTensor(det_F, C_inv, F);
        }
    private:
        std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim, NumberType> > material;
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

        Triangulation<dim>
        make_dog_bone_geometry(const double &thickness,
                               const double &gauge_length,
                               const double &gauge_width,
                               const double &fillet_radius,
                               const double &clamp_width,
                               const double &clamp_length,
                               const int &n_refinements = 0,
                               const bool &output_all_triangulations = false,
                               const double &radius_multiplier = 1.25);

        void
        make_grid();

        void
        system_setup();

        void
        assemble_system(const BlockVector<double> &solution_delta);

        friend struct Assembler_Base<dim, NumberType>;
        friend struct Assembler<dim, NumberType>;


        void
        make_constraints(const int &it_nr);

        void
        setup_qph();

        void
        solve_nonlinear_timestep(BlockVector<double> &solution_delta);

        std::pair<unsigned int, double>
        solve_linear_system(BlockVector<double> &newton_update);

        BlockVector<double>
        get_total_solution(const BlockVector<double> &solution_delta) const;

        void
        output_results() const;

        const Parameters::AllParameters &parameters;

        double vol_reference;
        double vol_current;

        Triangulation<dim> triangulation;

        Time time;
        TimerOutput timer;

        CellDataStorage<typename Triangulation<dim>::cell_iterator,
                PointHistory<dim, NumberType> > quadrature_point_history;

        const unsigned int degree;
        const FESystem<dim> fe;
        DoFHandler<dim> dof_handler_ref;
        const unsigned int dofs_per_cell;
        const FEValuesExtractors::Vector u_fe;

        static const unsigned int n_blocks = 1;
        static const unsigned int n_components = dim;
        static const unsigned int first_u_component = 0;

        enum {
            u_dof = 0
        };

        std::vector<types::global_dof_index> dofs_per_block;

        const QGauss<dim> qf_cell;
        const QGauss<dim - 1> qf_face;
        const unsigned int n_q_points;
        const unsigned int n_q_points_f;

        AffineConstraints<double> constraints;
        BlockSparsityPattern sparsity_pattern;
        BlockSparseMatrix<double> tangent_matrix;
        BlockVector<double> green_strain;
        BlockVector<double> system_rhs;
        BlockVector<double> solution_n;

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

        void
        get_error_residual(Errors &error_residual);

        void
        get_error_update(const BlockVector<double> &newton_update,
                         Errors &error_update);

        static
        void
        print_conv_header();

        void
        print_conv_footer();

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
            fe(FE_Q<dim>(parameters.poly_degree), dim), // displacement
            dof_handler_ref(triangulation),
            dofs_per_cell(fe.dofs_per_cell),
            u_fe(first_u_component),
            dofs_per_block(n_blocks),
            qf_cell(parameters.quad_order),
            qf_face(parameters.quad_order),
            n_q_points(qf_cell.size()),
            n_q_points_f(qf_face.size())
    { }

    template<int dim, typename NumberType>
    Solid<dim, NumberType>::~Solid() {
        dof_handler_ref.clear();
    }


    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::run() {
        make_grid();
        system_setup();
        output_results();
        time.increment();

        // Reset solution for this timestep
        BlockVector<double> solution_delta(dofs_per_block);
        while (time.current() <= time.end()) {
            solution_delta = 0.0;

            // Solve the current time step and update total solution vector
            solve_nonlinear_timestep(solution_delta);
            solution_n += solution_delta;

            // Plot results
            output_results();
            time.increment();
        }


    }


// @sect3{Private interface}

// @sect4{Solid::make_dog_bone_geometry}

// This mesh was created by Benjamin Alheit and edited to fit the problem constraints. This function should only be called
// when the geometry is to be changed. Some faces in this geometry are inside-out and therefore lead to errors if the triangulation
// is not processed in gmsh first to reverse the offending faces.
    inline double degrees_to_radians(const double &degrees) {
        return degrees * (M_PI / 180.0);
    }

    template<int dim, typename NumberType>
    Triangulation<dim> Solid<dim, NumberType>::make_dog_bone_geometry(const double &thickness,
                                                                      const double &gauge_length,
                                                                      const double &gauge_width,
                                                                      const double &fillet_radius,
                                                                      const double &clamp_width,
                                                                      const double &clamp_length,
                                                                      const int &n_refinements,
                                                                      const bool &output_all_triangulations,
                                                                      const double &radius_multiplier) {
        Triangulation<dim> starting_triangulation
        , trimmed_triangulation
        , clamp_upper_portion
        , gauge_portion
        , final_triangulation;

        const double tol_boundary = 1e-6;

        static const unsigned int trimmed_x_boundary_id = 5;
        static const unsigned int trimmed_y_boundary_id = 6;
        static const Tensor<1, 3> x_axis({1, 0, 0});
        static const Tensor<1, 3> y_axis({0, 1, 0});
        static const double degrees_90 = degrees_to_radians(90);


        // Cells in this region will have TransfiniteInterpolationManifold with
        // manifold id tfi_manifold_id attached to them. Additionally, the boundary
        // faces of the hole will be associated with a PolarManifold (in 2D) or
        // CylindricalManifold (in 3D):
        double outer_radius = radius_multiplier * fillet_radius;

        double pad_bottom = gauge_width - (outer_radius - fillet_radius);
        double pad_top = pad_bottom;

        double pad_left = clamp_length - (outer_radius - fillet_radius);
        double pad_right = pad_left;

        GridGenerator::plate_with_a_hole(starting_triangulation,
                                         fillet_radius,
                                         outer_radius,
                                         pad_bottom,
                                         pad_top,
                                         pad_left,
                                         pad_right,
                /*const Point< dim > &  	center*/  Point<dim>(),
                /*const types::manifold_id  	polar_manifold_id*/  0,
                /*const types::manifold_id  	tfi_manifold_id*/  1,
                                         thickness,
                /*const unsigned int  	n_slices*/  2,
                /*const bool  	colorize*/ true
        );

        if (n_refinements > 0)
            starting_triangulation.refine_global((unsigned int) n_refinements);

        // starting_triangulation is a plate with a hole centred at (0,0)
        if (output_all_triangulations) {
            std::ofstream out("starting_triangulation.vtk");
            GridOut grid_out;
            grid_out.write_vtk(starting_triangulation, out);
        }

        // As the hole is centered at (0,0), removing cells with centres in the positive x- and y-directions
        // leaves the clamp and width areas adjacent to the fillet. This section shall be referred to as
        // the 'heel' of the quarter-specimen.
        set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;
        for (const auto &cell: starting_triangulation.active_cell_iterators())
            if (cell->center()[0] > 0 || cell->center()[1] > 0)
                cells_to_remove.insert(cell);

        GridGenerator::create_triangulation_with_removed_cells(starting_triangulation,
                                                               cells_to_remove,
                                                               trimmed_triangulation);

        if (output_all_triangulations) {
            std::ofstream out("trimmed_triangulation.vtk");
            GridOut grid_out;
            grid_out.write_vtk(trimmed_triangulation, out);
        }

        // The x and y boundaries of the heel are assigned boundary IDs to be used when the surface
        // boundary is extracted.
        for (const auto &face: trimmed_triangulation.active_face_iterators()) {
            if (std::abs(face->center()[0] - 0.0) < tol_boundary)
                face->set_boundary_id(trimmed_x_boundary_id);
            else if (std::abs(face->center()[1] - 0.0) < tol_boundary)
                face->set_boundary_id(trimmed_y_boundary_id);
        }

        Triangulation<dim - 1, dim> x_boundary_mesh, y_boundary_mesh;
        Triangulation<dim - 1, dim - 1> flat_x_boundary_mesh, flat_y_boundary_mesh;


        GridGenerator::extract_boundary_mesh(trimmed_triangulation,
                                             x_boundary_mesh,
                                             set<types::boundary_id>({trimmed_x_boundary_id}));
        GridTools::rotate(y_axis, degrees_90, x_boundary_mesh);
        GridGenerator::flatten_triangulation(x_boundary_mesh, flat_x_boundary_mesh);
        GridGenerator::extrude_triangulation(flat_x_boundary_mesh,
                                             (unsigned int) (4 * pow(2, n_refinements + 1) * gauge_length /
                                                             gauge_width),
                                             gauge_length,
                                             gauge_portion);
        GridTools::rotate(y_axis, degrees_90, gauge_portion);
        if (output_all_triangulations) {
            std::ofstream out("gauge_portion.vtk");
            GridOut grid_out;
            grid_out.write_vtk(gauge_portion, out);
        }

        double clamp_extrusion = clamp_width - gauge_width - fillet_radius;
        GridGenerator::extract_boundary_mesh(trimmed_triangulation,
                                             y_boundary_mesh,
                                             set<types::boundary_id>({trimmed_y_boundary_id}));
        GridTools::rotate(x_axis, -degrees_90, y_boundary_mesh);
        GridGenerator::flatten_triangulation(y_boundary_mesh, flat_y_boundary_mesh);
        GridGenerator::extrude_triangulation(flat_y_boundary_mesh,
                                             (unsigned int) (4 * pow(2, n_refinements + 3) * clamp_extrusion /
                                                             clamp_length),
                                             clamp_extrusion,
                                             clamp_upper_portion);
        GridTools::rotate(x_axis, -degrees_90, clamp_upper_portion);
        if (output_all_triangulations) {
            std::ofstream out("clamp_upper_portion.vtk");
            GridOut grid_out;
            grid_out.write_vtk(clamp_upper_portion, out);
        }

        GridGenerator::merge_triangulations(vector<const Triangulation<dim> *>({&trimmed_triangulation,
                                                                                &clamp_upper_portion,
                                                                                &gauge_portion}),
                                            final_triangulation);

        if (output_all_triangulations) {
            std::ofstream out("final_triangulation.vtk");
            GridOut grid_out;
            grid_out.write_vtk(final_triangulation, out);
        }


        for (auto &cell: final_triangulation.active_cell_iterators()) {
            for (unsigned int i_face = 0; i_face < GeometryInfo<dim>::faces_per_cell; ++i_face) {
                if (not cell->face_orientation(i_face))
                    cout << "Cell " << cell->index() << " face " << i_face << "might be inside-out." << endl;

            }
        }


        //The triangulation is exported in a gmsh format.
        if (output_all_triangulations) {
            GridOut grid_out;
            std::cout << "final_triangulation.msh saved" << std::endl;
            std::ofstream msh_out("final_triangulation.msh");
            grid_out.template write_msh(final_triangulation, msh_out);
        }

        return final_triangulation;
    }
// @sect4{Solid::make_grid}

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::make_grid() {
        std::cout << "Polynomial degree: Q"<<parameters.poly_degree<<std::endl;

        // Geometry of the dog bone in metres:
        const double thickness = 0.001;
        const double gauge_length = 0.0125;
        const double gauge_width = 0.003;
        const double fillet_radius = 0.001;
        const double clamp_width = 0.005;
        const double clamp_length = 0.012;
        const int n_refinements = 0;
        const bool output_all_triangulations = true;
        const double &radius_multiplier = 1.25;

        // Parameters relevant to full geometry
        double outer_radius = radius_multiplier * fillet_radius;
        double pad_bottom = gauge_width + fillet_radius;
        double pad_left = clamp_length + fillet_radius;
        std::ifstream input_file("reversed_triangulation_three.msh");
        GridIn<dim> grid_in;

        // Parameters for the gauge section
        std::vector<unsigned int> repetitions(dim, 32);
        const Point<dim> bottom_left = (dim == 3 ? Point<dim>(0.0, 0.0, -thickness) : Point<dim>(0.0, 0.0));
        const Point<dim> top_right = (dim == 3 ? Point<dim>(2 * gauge_length, 2 * gauge_width, thickness)
                                               : Point<dim>(2 * gauge_length, 2 * gauge_width));

        // Boundary IDs
        static const unsigned int x_symmetry_boundary_id = 11;
        static const unsigned int y_symmetry_boundary_id = 1;
        static const unsigned int z_symmetry_boundary_id = 22;
        static const unsigned int neumann_boundary_id = 2;
        static const unsigned int displacement_boundary_id = 4;



/*To change the geometry, uncomment the function below and generate another .msh to be edited in gmsh. This is
 * because some element faces are flipped during the creation of the triangulation; it is easiest to identify
 * the elements with negative volume in the gmsh GUI and reverse the faces in a new mesh that is exported from
 * gmsh and loaded into the programme as below*/

//        triangulation.copy_triangulation(Solid<dim,NumberType>::make_dog_bone_geometry(thickness,
//                                                                                       gauge_length,
//                                                                                       gauge_width,
//                                                                                       fillet_radius,
//                                                                                       clamp_width,
//                                                                                       clamp_length,
//                                                                                       n_refinements,
//                                                                                       output_all_triangulations));

        TriaIterator<TriaAccessor<dim - 1, dim, dim>>
                current_face;
        Point<dim> face_centre;

        // Mesh processed in gmsh (manually) is imported:
        switch(parameters.sim_geom){
            case 1: // Full geometry
                std::cout << "Geometry: Eighth symmetry of full geometry" << std::endl;
                grid_in.attach_triangulation(triangulation);
                grid_in.read_msh(input_file);
                triangulation.refine_global(1);
                for (auto &cell: triangulation.active_cell_iterators()) {
                    if (cell->at_boundary())
                        for (unsigned int i_face = 0; i_face < GeometryInfo<dim>::faces_per_cell; ++i_face) {
                            current_face = cell->face(i_face);
                            if (current_face->at_boundary()) {
                                face_centre = current_face->center();
                                if (almost_equals(face_centre[1], -pad_bottom)) {
                                    current_face->set_boundary_id(y_symmetry_boundary_id);
                                } // Y-face at symmetry line
                                else if (almost_equals(face_centre[0], gauge_length)) {
                                    current_face->set_boundary_id(x_symmetry_boundary_id);
                                } // X-face at symmetry line
                                else if (almost_equals(face_centre[0], -pad_left)) {
                                    current_face->set_boundary_id(displacement_boundary_id);
                                }// X-face at displaced boundary
                                else if (dim == 3 && almost_equals(face_centre[2], thickness * 0.5)) {
                                    current_face->set_boundary_id(neumann_boundary_id);
                                } // +Z-face
                                else if (dim == 3 && almost_equals(face_centre[2], -thickness * 0.5)) {
                                    current_face->set_boundary_id(z_symmetry_boundary_id);
                                } // -Z-face
                            }
                        }
                }

                break;

            case 2: // Gauge section only

                std::cout<< "Geometry: Gauge section only" << std::endl;

                if (dim == 3)
                    repetitions[dim - 1] = 1;

                GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                          repetitions,
                                                          bottom_left,
                                                          top_right);
                for (auto &cell: triangulation.active_cell_iterators()) {
                    if (cell->at_boundary())
                        for (unsigned int i_face = 0; i_face < GeometryInfo<dim>::faces_per_cell; ++i_face) {
                            current_face = cell->face(i_face);
                            if (current_face->at_boundary()) {
                                face_centre = current_face->center();
                                if (almost_equals(face_centre[1], 0)) {
                                    current_face->set_boundary_id(y_symmetry_boundary_id);
                                } // Y-face at symmetry line
                                else if (almost_equals(face_centre[0], 0)) {
                                    current_face->set_boundary_id(x_symmetry_boundary_id);
                                } // X-face at symmetry line
                                else if (almost_equals(face_centre[0], 2*gauge_length)) {
                                    current_face->set_boundary_id(displacement_boundary_id);
                                }// X-face at displaced boundary
                                else if (dim == 3 && almost_equals(face_centre[2], thickness)) {
                                    current_face->set_boundary_id(neumann_boundary_id);
                                } // +Z-face
                                else if (dim == 3 && almost_equals(face_centre[2], -thickness)) {
                                    current_face->set_boundary_id(z_symmetry_boundary_id);
                                } // -Z-face
                            }
                        }
                }

                break;

            default:
                AssertThrow(false,
                            ExcMessage("Nonexistent simulation geometry chosen."))


        }



        vol_reference = GridTools::volume(triangulation);
        vol_current = vol_reference;
        std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;
    }


// @sect4{Solid::system_setup}

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::system_setup() {
        timer.enter_subsection("Setup system");

        std::vector<unsigned int> block_component(n_components, u_dof); // Displacement

        // DoF handler initialised and grid renumbered
        dof_handler_ref.distribute_dofs(fe);
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

            // All components of the system are coupled.
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

        // Set up storage vectors
        system_rhs.reinit(dofs_per_block);
        system_rhs.collect_sizes();

        solution_n.reinit(dofs_per_block);
        solution_n.collect_sizes();

        // Set up point history:
        setup_qph();

        timer.leave_subsection();
    }


// @sect4{Solid::setup_qph}

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::setup_qph() {
        std::cout << "    Setting up quadrature point data..." << std::endl;

        quadrature_point_history.initialize(triangulation.begin_active(),
                                            triangulation.end(),
                                            n_q_points);

        // Set up initial quadrature point data. Retrieved QP data is returned
        // as a vector of smart pointers.
        for (typename Triangulation<dim>::active_cell_iterator cell =
                triangulation.begin_active(); cell != triangulation.end(); ++cell) {
            const std::vector<std::shared_ptr<PointHistory<dim, NumberType> > > lqph =
                    quadrature_point_history.get_data(cell);
            Assert(lqph.size() == n_q_points, ExcInternalError());

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                lqph[q_point]->setup_lqp(parameters);
        }
    }


// @sect4{Solid::solve_nonlinear_timestep}

// Newton-Raphson method solver
    template<int dim, typename NumberType>
    void
    Solid<dim, NumberType>::solve_nonlinear_timestep(BlockVector<double> &solution_delta) {
        std::cout << std::endl << "Timestep " << time.get_timestep() << " @ "
                  << time.current() << "s" << std::endl;

        // Vector to store Newton update
        BlockVector<double> newton_update(dofs_per_block);

        // Reset error objects
        error_residual.reset();
        error_residual_0.reset();
        error_residual_norm.reset();
        error_update.reset();
        error_update_0.reset();
        error_update_norm.reset();

        // Print solver header
        print_conv_header();

        unsigned int newton_iteration = 0;
        for (; newton_iteration < parameters.max_iterations_NR;
               ++newton_iteration) {
            std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;

            // Impose Dirichlet constraints
            make_constraints(newton_iteration);
            // Assemble tangent matrix and right-hand side vector
            assemble_system(solution_delta);

            // Compute residual error
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

            // We can now determine the normalised Newton update error, and
            // perform the actual update of the solution increment for the current
            // time step, update all quadrature point information pertaining to
            // this new displacement and stress state and continue iterating:
            error_update_norm = error_update;
            error_update_norm.normalise(error_update_0);

            solution_delta += newton_update;

            std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                      << std::scientific << lin_solver_output.first << "  "
                      << lin_solver_output.second << "  " << error_residual_norm.norm
                      << "  " << error_residual_norm.u << "  "
                      << "  " << error_update_norm.norm << "  " << error_update_norm.u
                      << "  " << std::endl;
        }

        // If more iterations than specified in the parameter file occurs, an exception is raised
        AssertThrow(newton_iteration <= parameters.max_iterations_NR,
                    ExcMessage("No convergence in nonlinear solver!"));
    }


// @sect4{Solid::print_conv_header, Solid::print_conv_footer and Solid::print_vertical_tip_displacement}

    // Print updated data for each iteration
    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::print_conv_header() {
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
    void Solid<dim, NumberType>::print_conv_footer() {
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



// @sect4{Solid::get_error_residual}

    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::get_error_residual(Errors &error_residual) {
        BlockVector<double> error_res(dofs_per_block);


        for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
            // Ignore constrained degrees of freedom
            if (!constraints.is_constrained(i))
                error_res(i) = system_rhs(i);

        // Compute residual error norm for unconstrained degrees of freedom
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

    // Total solution to be updated at the end of the timestep
    template<int dim, typename NumberType>
    BlockVector<double>
    Solid<dim, NumberType>::get_total_solution(const BlockVector<double> &solution_delta) const {
        BlockVector<double> solution_total(solution_n);
        solution_total += solution_delta;
        return solution_total;
    }


// @sect4{Solid::assemble_system}

    // @sect3{Assembler_Base}
    template<int dim, typename NumberType>
    struct Assembler_Base {
        virtual ~Assembler_Base() {}

        // Local_ASM object stores local contributions
        struct Local_ASM {
            const Solid<dim, NumberType> *solid;
            FullMatrix<double> cell_matrix;
            Vector<double> cell_rhs;
            std::vector<types::global_dof_index> local_dof_indices;

            Local_ASM(const Solid<dim, NumberType> *solid)
                    :
                    solid(solid),
                    cell_matrix(solid->dofs_per_cell, solid->dofs_per_cell),
                    cell_rhs(solid->dofs_per_cell),
                    local_dof_indices(solid->dofs_per_cell) {}

            void reset() {
                cell_matrix = 0.0;
                cell_rhs = 0.0;
            }
        };

        // ScratchData stores larger objects (shape function value arrays, gradient vectors)
        struct ScratchData_ASM {
            const BlockVector<double> &solution_total;
            std::vector<Tensor<2, dim, NumberType> > solution_grads_u_total;

            FEValues<dim> fe_values_ref;
            FEFaceValues<dim> fe_face_values_ref;

            std::vector<std::vector<Tensor<2, dim, NumberType> > > grad_Nx;
            std::vector<std::vector<SymmetricTensor<2, dim, NumberType> > >
            symm_grad_Nx;

            ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                            const QGauss<dim> &qf_cell,
                            const UpdateFlags uf_cell,
                            const QGauss<dim - 1> &qf_face,
                            const UpdateFlags uf_face,
                            const BlockVector<double> &solution_total)
                    :
                    solution_total(solution_total),
                    solution_grads_u_total(qf_cell.size()),
                    fe_values_ref(fe_cell, qf_cell, uf_cell),
                    fe_face_values_ref(fe_cell, qf_face, uf_face),
                    grad_Nx(qf_cell.size(),
                            std::vector<Tensor<2, dim, NumberType> >(fe_cell.dofs_per_cell)),
                    symm_grad_Nx(qf_cell.size(),
                                 std::vector<SymmetricTensor<2, dim, NumberType> >
                                                                     (fe_cell.dofs_per_cell)) {}

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

        // Assemble local tangent matrix
        void
        assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 ScratchData_ASM &scratch,
                                 Local_ASM &data) {

            assemble_system_tangent_residual_one_cell(cell, scratch, data);
            assemble_neumann_contribution_one_cell(cell, scratch, data);
        }

        // This function adds the local contribution to the system matrix.
        void
        copy_local_to_global_ASM(const Local_ASM &data) {
            const AffineConstraints<double> &constraints = data.solid->constraints;
            BlockSparseMatrix<double> &tangent_matrix = const_cast<Solid<dim, NumberType> *>(data.solid)->tangent_matrix;
            BlockVector<double> &system_rhs = const_cast<Solid<dim, NumberType> *>(data.solid)->system_rhs;

            constraints.distribute_local_to_global(
                    data.cell_matrix, data.cell_rhs,
                    data.local_dof_indices,
                    tangent_matrix, system_rhs);
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
                                               ScratchData_ASM &scratch,
                                               Local_ASM &data) {
            // Aliases for data referenced from the Solid class
            const unsigned int &n_q_points_f = data.solid->n_q_points_f;
            const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
            const Parameters::AllParameters &parameters = data.solid->parameters;
            const Time &time = data.solid->time;
            const FESystem<dim> &fe = data.solid->fe;
            const unsigned int &u_dof = data.solid->u_dof;
            static const unsigned int neumann_boundary_id = 2;

            // Assemble the Neumann contribution if the cell is on the Neumann boundary
            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
                 ++face)
                if (cell->face(face)->at_boundary() == true
                    && cell->face(face)->boundary_id() == neumann_boundary_id) {
                    scratch.fe_face_values_ref.reinit(cell, face);

                    for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                         ++f_q_point) {
                        // Homogeneous Neumann boundary condition
                        const double time_ramp = (time.current() / time.end());
                        const double magnitude = 0.0 * time_ramp;
                        Tensor<1, dim> dir;
                        dir[2] = 1.0;
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

    // @sect3{Assembler}
    template<int dim>
    struct Assembler<dim, double> : Assembler_Base<dim, double> {
        typedef double NumberType;
        using typename Assembler_Base<dim, NumberType>::ScratchData_ASM;
        using typename Assembler_Base<dim, NumberType>::Local_ASM;

        virtual ~Assembler() {}

        virtual void
        assemble_system_tangent_residual_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                  ScratchData_ASM &scratch,
                                                  Local_ASM &data) {
            // Aliases for data referenced from the Solid class
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

            // Find solution gradient at each quadrature point in the current cell and then
            // update each quadrature point using the displacement gradient
            scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total,
                                                               scratch.solution_grads_u_total);

            // Build the local stiffness matrix while making use of its symmetry
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
                // Solution gradient
                const Tensor<2, dim, NumberType> &grad_u = scratch.solution_grads_u_total[q_point];
                // Deformation gradient
                const Tensor<2, dim, NumberType> F = Physics::Elasticity::Kinematics::F(grad_u);
                // Jacobian
                const NumberType det_F = determinant(F);
                // Right Cauchy-Green tensor
                const SymmetricTensor<2, dim, NumberType> C = Physics::Elasticity::Kinematics::C(F);
                // Inverse of deformation gradient
                const Tensor<2, dim, NumberType> F_inv = invert(F);
                // Inverse of right Cauchy-Green tensor
                const SymmetricTensor<2, dim, NumberType> C_inv = invert(C);
                // Ensure that Jacobian is always positive
                Assert(det_F > NumberType(0.0), ExcInternalError());
                for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                    const unsigned int k_group = fe.system_to_base_index(k).first.first;

                    if (k_group == u_dof) {
                        // Grad u
                        scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                        scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
                    } else Assert(k_group <= u_dof, ExcInternalError());
                }

                // Kirchhoff stress
                const SymmetricTensor<2, dim, NumberType> tau = lqph[q_point]->get_tau(det_F, F, C_inv);
                // Tangent
                const SymmetricTensor<4, dim, NumberType> Jc = lqph[q_point]->get_Jc(det_F, C_inv, F);
                const Tensor<2, dim, NumberType> tau_ns(tau);

                // Define aliases
                const std::vector<SymmetricTensor<2, dim> > &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
                const std::vector<Tensor<2, dim> > &grad_Nx = scratch.grad_Nx[q_point];
                const double JxW = scratch.fe_values_ref.JxW(q_point);

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const unsigned int component_i = fe.system_to_component_index(i).first;
                    const unsigned int i_group = fe.system_to_base_index(i).first.first;

                    // Assemble local RHS vector
                    if (i_group == u_dof)
                        data.cell_rhs(i) -= (symm_grad_Nx[i] * tau) * JxW;
                    else Assert(i_group <= u_dof, ExcInternalError());


                    for (unsigned int j = 0; j <= i; ++j) {
                        const unsigned int component_j = fe.system_to_component_index(j).first;
                        const unsigned int j_group = fe.system_to_base_index(j).first.first;

                        if ((i_group == j_group) && (i_group == u_dof)) {
                            data.cell_matrix(i, j) += symm_grad_Nx[i] * Jc
                                                      * symm_grad_Nx[j] * JxW;// The material contribution:
                            if (component_i == component_j) // Geometric stress contribution
                                data.cell_matrix(i, j) += grad_Nx[i][component_i] * tau_ns
                                                          * grad_Nx[j][component_j] * JxW;
                        } else Assert((i_group <= u_dof) && (j_group <= u_dof),
                                      ExcInternalError());
                    }
                }
            }


            // Copy lower half of local matrix into upper half
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = i + 1; j < dofs_per_cell; ++j) {
                    data.cell_matrix(i, j) = data.cell_matrix(j, i);
                }
        }

    };


    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::assemble_system(const BlockVector<double> &solution_delta) {
        timer.enter_subsection("Assemble linear system");
        std::cout << " ASM " << std::flush;

        // Reset storage vectors
        tangent_matrix = 0.0;
        system_rhs = 0.0;

        // Update values
        const UpdateFlags uf_cell(update_gradients |
                                  update_JxW_values);
        const UpdateFlags uf_face(update_values |
                                  update_JxW_values);

        // Get total solution vector
        const BlockVector<double> solution_total(get_total_solution(solution_delta));
        // Compute assembly matrices (shape functions, gradients, etc)
        typename Assembler_Base<dim, NumberType>::Local_ASM per_task_data(this);
        typename Assembler_Base<dim, NumberType>::ScratchData_ASM scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face,
                                                                               solution_total);
        // Initialise assembler
        Assembler<dim, NumberType> assembler;

        // Pass copies of data structures to WorkStream object for processing
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
// Displacement constraints are made at zeroth iteration
    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::make_constraints(const int &it_nr) {
        std::cout << " CST " << std::flush;

        // Do not apply constraints after the first iteration
        if (it_nr > 1)
            return;
        const bool apply_dirichlet_bc = (it_nr == 0);

        // Apply Dirichlet constraints at the zeroth iteration
        if (apply_dirichlet_bc) {
            // Clear constraints
            constraints.clear();


            {
                double time_end = time.end();
                double delta_t = time.get_delta_t();
                int number_of_steps = time_end / delta_t;
                double symmetric_displacement;

                // Scale the displacement to match the simulation geometry symmetry
                switch(parameters.sim_geom){
                    case 1:
                        symmetric_displacement = -0.02;
                        break;
                    case 2:
                        symmetric_displacement = 0.04;
                        break;

                    default:
                        AssertThrow(false,
                                    ExcMessage("Dirichlet boundary condition cannot be applied with chosen geometry."))
                }

                // Boundary ID's
                static const unsigned int x_symmetry_boundary_id = 11;
                static const unsigned int y_symmetry_boundary_id = 1;
                static const unsigned int displacement_boundary_id = 4;

                const FEValuesExtractors::Scalar x_displacement(0);
                const FEValuesExtractors::Scalar y_displacement(1);

                // x-symmetry
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                         x_symmetry_boundary_id,
                                                         Functions::ZeroFunction<dim>(n_components),
                                                         constraints,
                                                         fe.component_mask(x_displacement));

                // y-symmetry
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                         y_symmetry_boundary_id,
                                                         Functions::ZeroFunction<dim>(n_components),
                                                         constraints,
                                                         fe.component_mask(y_displacement));

                // Displacement driven right hand side of the beam
                double displacement = symmetric_displacement / number_of_steps;
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                         displacement_boundary_id,
                                                         Functions::ConstantFunction<dim>(displacement, n_components),
                                                         constraints,
                                                         fe.component_mask(x_displacement));

            }

            // Zero Z-displacement through thickness direction
            // This corresponds to a plane stress condition being imposed on the beam
            if (dim == 3) {
                static const unsigned int z_symmetry_boundary_id = 22;
                const FEValuesExtractors::Scalar z_displacement(2);
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                         z_symmetry_boundary_id,
                                                         Functions::ZeroFunction<dim>(n_components),
                                                         constraints,
                                                         fe.component_mask(z_displacement));
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

        // Solve for incremental displacement
        {
            timer.enter_subsection("Linear solver");
            std::cout << " SLV " << std::flush;
            if (parameters.type_lin == "CG") {
                const int solver_its = static_cast<unsigned int>(
                        tangent_matrix.block(u_dof, u_dof).m()
                        * parameters.max_iterations_lin);
                const double tol_sol = parameters.tol_lin
                                       * system_rhs.block(u_dof).l2_norm();

                SolverControl solver_control(solver_its, tol_sol);

                GrowingVectorMemory<Vector<double> > GVM;
                SolverCG<Vector<double> > solver_CG(solver_control, GVM);

                // McBride and Pelteret recommend a SSOR preconditioner as it appears to
                // provide the fastest solver convergence characteristics for this
                // problem on a single-thread machine.
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
            } else if (parameters.type_lin == "Direct") {
                // A direct solver can be used for small problems
                SparseDirectUMFPACK A_direct;
                A_direct.initialize(tangent_matrix.block(u_dof, u_dof));
                A_direct.vmult(newton_update.block(u_dof), system_rhs.block(u_dof));

                lin_it = 1;
                lin_res = 0.0;
            } else Assert(false, ExcMessage("Linear solver type not implemented"));

            timer.leave_subsection();
        }

        // Distribute constraints back to the Newton update
        constraints.distribute(newton_update);

        return std::make_pair(lin_it, lin_res);
    }

// @sect4{Solid::output_results}
// Results are written to file
    template<int dim, typename NumberType>
    void Solid<dim, NumberType>::output_results() const {
        DataOut<dim> data_out;
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                data_component_interpretation(dim,DataComponentInterpretation::component_is_part_of_vector);

        std::vector<std::string> solution_name(dim, "Displacement");

        data_out.attach_dof_handler(dof_handler_ref);
        data_out.add_data_vector(solution_n,
                                 solution_name,
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);

        // Post-processors
        GradientPostprocessor<dim> grad_post;
        data_out.add_data_vector(solution_n, grad_post);
        DeformationGradientPostprocessor<dim> F_post;
        data_out.add_data_vector(solution_n, F_post);
        CauchyStressPostprocessor<dim, NumberType> sig_post(parameters);
        data_out.add_data_vector(solution_n, sig_post);
        LagrangeStrainPostprocessor<dim, NumberType> E_post(parameters);
        data_out.add_data_vector(solution_n, E_post);


        data_out.build_patches();
        std::ostringstream filename;
        string file;
        switch(parameters.sim_geom){
            case 1:
                file = "Q"+to_string(parameters.poly_degree)+"full_geom_tensiletest_solution-" + to_string(time.get_timestep()) + ".vtu";
                break;
            case 2:
                file = "Q"+to_string(parameters.poly_degree)+"gauge_tensiletest_solution-" + to_string(time.get_timestep()) + ".vtu";
                break;
        }
        std::ofstream output(file.c_str());
        data_out.write_vtu(output);

    }

}


#endif //EMMA_TOPICS_PROJECT_PROBLEM1_TENSILE_TEST_H
