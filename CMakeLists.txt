cmake_minimum_required(VERSION 3.5)

PROJECT(emma-topics-project)

file(GLOB PART1_SRCS Problem1/*.cc Problem1/*.h)
file(GLOB PART2_SRCS Problem2/*.cc Problem2/*.h)
file(GLOB PART2R_SRCS Problem2_Refinement/*.cc Problem2_Refinement/*.h)




FIND_PACKAGE(deal.II 9.4.1
        HINTS /home/emma/dealii-candi/deal.II-v9.4.1 ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
        )
DEAL_II_INITIALIZE_CACHED_VARIABLES()


ADD_EXECUTABLE( part1 ${PART1_SRCS})
DEAL_II_SETUP_TARGET(part1)
ADD_EXECUTABLE( part2 ${PART2_SRCS} Problem2/cooks-membrane-sundials.h)
DEAL_II_SETUP_TARGET(part2)
ADD_EXECUTABLE( part2_refine ${PART2R_SRCS})
DEAL_II_SETUP_TARGET(part2_refine)



CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

IF(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
            "*** Could not locate deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
            )
ENDIF()

IF(NOT DEAL_II_WITH_SUNDIALS) # keep in one line
    MESSAGE(FATAL_ERROR "
Error! This programme requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_SUNDIALS = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options:
    DEAL_II_WITH_SUNDIALS = ${DEAL_II_WITH_SUNDIALS}
This conflicts with the requirements."
            )
ENDIF()
