/* ---------------------------------------------------------------------
 * Problem 2: SUNDIALS::KINSOL functions
 * Emma Garschagen
 * November 2022
 *
 * This code is based on Step-77 from the deal.ii
 * tutorials contributed by Wolfgang Bangerth, Colorado State University,
 * and significant refactoring by Ernesto Ismail.
 * ---------------------------------------------------------------------
 */
#include "cooks-membrane-sundials.h"

namespace Cooks_Membrane {
    using namespace dealii;
    using namespace std;


    template<int dim, typename NumberType>
    unsigned int
    Solid<dim, NumberType>::solve_nonlinear_timestep_kinsol(BlockVector<double> &solution_delta)
    /**
     * KINSOL nonlinear solver
     * Refactored by Ernesto Ismail
     * @tparam dim FE system dimension
     * @tparam NumberType
     * @param solution_delta Solution increment
     * @return The solution as an unsigned integer
     */
    {

        std::cout << std::endl << "Timestep " << time.get_timestep() << " @ "
                  << time.current() << "s" << std::endl;

        double newton_iteration = 0;

        // Assemble tangent matrix
        compute_and_factorize_jacobian(solution_n, newton_iteration);
        // Solve linear step
        solve_linear_kinsol(system_rhs,solution_n);
        // Increment Newton iteration counter
        newton_iteration++;

        double target_tolerance = 1.0e-8;

        typename SUNDIALS::KINSOL<BlockVector<double>>::AdditionalData
                additional_data;
        additional_data.function_tolerance = target_tolerance;
        // Select solver strategy (linesearch | newton)
        additional_data.strategy = dealii::SUNDIALS::KINSOL<BlockVector<double>>::AdditionalData::SolutionStrategy::linesearch;

        SUNDIALS::KINSOL<BlockVector<double>> nonlinear_solver(additional_data);

        nonlinear_solver.reinit_vector = [&](BlockVector<double> &x){

            const types::global_dof_index n_dofs_u = dofs_per_block[u_dof];
            x.reinit (dofs_per_block);
            x.collect_sizes ();
        };

        nonlinear_solver.residual =

                [&](const BlockVector<double> &evaluation_point_in,
                    BlockVector<double> & residual_vector_kinsol)
                /**
                 * Link system assembly functions with KINSOL's residual assembly function.
                 * Note that only the RHS vector is assembled.
                 * @see Solid<dim, NumberType>::assemble_system(const BlockVector<double> &solution_delta, const bool rhs_only)
                 * @param evaluation_point_in
                 * @param residual_vector_kinsol
                 * @return
                 */
                    {

                    BlockVector<double> solution_delta_internal = evaluation_point_in;
                    solution_delta_internal -= solution_n;

                    assemble_system(solution_delta_internal, /*rhs_only*/ true);

                    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
                        if (!constraints.is_constrained(i))
                            residual_vector_kinsol(i) = -system_rhs(i);

                    cout<<"res: "<<residual_vector_kinsol.l2_norm()<<endl;

                    return 0;
                };

        nonlinear_solver.setup_jacobian =

                [&](const BlockVector<double> &current_u,
                    const BlockVector<double> & /*current_f*/)
                /**
                 * Link system assembly functions with KINSOL's Jacobian assembly function.
                 * @see Solid<dim, NumberType>::compute_and_factorize_jacobian(const BlockVector<double> &newton_update_in, const double newton_iteration)
                 * @param current_u Current solution vector
                 * @return
                 */
                {

                    compute_and_factorize_jacobian(current_u, newton_iteration);
                    newton_iteration++;

                    return 0;
                };

        nonlinear_solver.solve_with_jacobian = [&](const BlockVector<double> &rhs,
                                                   BlockVector<double> &      dst,
                                                   const double tolerance)
       /**
        * Solve linear system
        * @param rhs System RHS vector
        * @param dst Present solution
        * @param tolerance
        * @return
        */
        {
            this->solve_linear_kinsol(rhs, dst);

            return 0;
        };

        return nonlinear_solver.solve(solution_n);
    }

    template<int dim, typename NumberType>
    void
    Solid<dim, NumberType>::compute_and_factorize_jacobian(const BlockVector<double> &newton_update_in, const double newton_iteration)
    /**
     * Assemble the system matrix and impose boundary conditions
     * @tparam dim FE system dimension
     * @tparam NumberType
     * @param newton_update_in Newton update
     * @param newton_iteration Newton iteration counter
     */
    {
        // Impose Dirichlet boundary conditions
        make_constraints(newton_iteration);

        // Calculate the solution increment from the Newton update and solution vector
        BlockVector<double> solution_delta_internal = newton_update_in;
        solution_delta_internal -= solution_n;

        assemble_system(solution_delta_internal, /*rhs_only*/ false);
        jacobian_matrix_factorization = std::make_unique<SparseDirectUMFPACK>();
        jacobian_matrix_factorization->factorize(tangent_matrix);
    }

    template<int dim, typename NumberType>
    void
    Solid<dim, NumberType>::solve_linear_kinsol(const BlockVector<double> &rhs,
                                                BlockVector<double> &present_solution)
    /**
     * Solve the linear timestep
     * @tparam dim FE system dimension
     * @tparam NumberType
     * @param rhs
     * @param present_solution
     */
    {
        jacobian_matrix_factorization->vmult(present_solution, rhs);
        constraints.distribute(present_solution);
    }

    template class Solid<3, double>;
}