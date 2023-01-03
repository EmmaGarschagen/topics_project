SET(TARGET "emma-topics-project")
SET(CMAKE_CXX_STANDARD 17)


FILE(GLOB_RECURSE TARGET_SRC "Problem2/*.cc" "Problem2/*.prm" )
#FILE(GLOB_RECURSE TARGET_SRC "Problem1/*.cc" "Problem1/*.prm" "Problem2/*.cc" "Problem2/*.prm" "Problem3/*.cc" "Problem3/*.prm")
SET(TARGET_SRC ${TARGET_SRC} ${TARGET_INC})

#include_directories(.)
#include_directories(Problem1)
#include_directories(Problem2)

#add_executable(emma-topics-project Problem1/tensile-test.cc)
#add_executable(emma-topics-project Problem2/cooks-membrane.cc)

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

FIND_PACKAGE(deal.II 9.4.0
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
        )

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

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()