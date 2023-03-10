
/* ---------------------------------------------------------------------
 * Problem 2: Finite Deformation Elasticity with Adaptive Mesh Refinement.
 * Emma Garschagen
 * November 2022
 *
 * This code is based on a modified Cook's Membrane problem developed for
 * the deal.ii code gallery by Jean-Paul Pelteret, University of Erlangen-Nuremberg,
 * and Andrew McBride, University of Cape Town, 2015, 2017 and Step-77 from the deal.ii
 * tutorials contributed by Wolfgang Bangerth, Colorado State University.
 * Ernesto Ismail contributed significantly to the implementation of adaptive
 * mesh refinement.
 * ---------------------------------------------------------------------
 */


#include "cooks-membrane-refinement.h"

bool almost_equals(const double &a,
                   const double &b,
                   const double &tol) {
    return fabs(a - b) < tol;
}

namespace Cooks_Membrane {
    using namespace dealii;
    using namespace std;

}

// @sect3{Main function}

int main(int argc, char *argv[]) {
    using namespace dealii;
    using namespace Cooks_Membrane;

    const unsigned int dim = 3;



    try {
        deallog.depth_console(0);
        Parameters::AllParameters parameters("problem2_parameters_refinement.prm");


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
