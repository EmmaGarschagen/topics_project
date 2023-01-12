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

#include "problem1-tensile-test.h"

bool almost_equals(const double &a,
                   const double &b,
                   const double &tol) {
    return fabs(a - b) < tol;
}

namespace Tensile_Test {
    using namespace dealii;
    using namespace Physics::Transformations;
    using namespace Physics::Elasticity;
    using namespace std;
}

// @sect3{Main function}
// Lastly we provide the main driver function which appears
// no different to the other tutorials.
int main(int argc, char *argv[]) {
    using namespace dealii;
    using namespace Tensile_Test;

    const unsigned int dim = 3;

    try {
        deallog.depth_console(0);
        Parameters::AllParameters parameters("problem1_parameters.prm");


        std::cout << "Assembly method: Residual and linearisation are computed manually." << std::endl;

        // Allow multi-threading
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                            dealii::numbers::invalid_unsigned_int);
        typedef double NumberType;
        Solid<dim, NumberType> solid_3d(parameters);
        solid_3d.run();


    }
    catch (std::exception &exc) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl << exc.what()
                  << std::endl << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
    catch (...) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}