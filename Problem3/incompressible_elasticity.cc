/* ---------------------------------------------------------------------
 * Problem 2: Finite Deformation Elasticity
 * Emma Garschagen
 * November 2022
 *
 * ---------------------------------------------------------------------
 */


#include "incompressible_elasticity.h"

bool almost_equals(const double &a,
                   const double &b,
                   const double &tol) {
    return fabs(a - b) < tol;
}

namespace Incompressible_Elasticity {
    using namespace dealii;
    using namespace std;

}

// @sect3{Main function}

int main() {
    using namespace dealii;
    using namespace Incompressible_Elasticity;

    try
    {
        const unsigned int dim = 3;
        Solid<dim>         solid("problem3_parameters.prm");
        solid.run();
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
