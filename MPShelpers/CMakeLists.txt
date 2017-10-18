cmake_minimum_required(VERSION 3.2)
project(MPShelpers)

#set(CMAKE_CXX_STANDARD 11)

set(SOURCES
        src/helpers.cpp
        src/eigs.cpp
        src/Models.cpp
        src/parser.cpp)

set(HEADERS_BASE
        helpers.h
        eigs.h
        Models.h
        parser.h
        Defs.h
        MPSIncludesBasic.h
        MPSIncludesAdvanced.h)

set(HEADERS_BLOCK
        Block/BlockObj.hpp
        Block/EigsBlockFunctions.hpp
        Block/InvEBlockFunctions.hpp
        Block/ItoKey.hpp
        Block/KeyTypes.hpp
        Block/MPSBlockMat.hpp
        Block/MPSBlockUtilities.hpp
        Block/symobj.hpp
        Block/TMBlockFunctions.hpp)

set(HEADERS_INCONLY
        DimMaps.hpp
        helpers.hpp
        IterativeSolvers.hpp
        OperatorTypes.hpp
        tictoc.hpp)

### compiler specific settings
set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -parallel -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -parallel -march=native -g")

set(CMAKE_VERBOSE_MAKEFILE ON)
set(USE_NM_REP ON CACHE BOOL "Use NM Representation for Hubbard")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build Type")
set(ARMADILLO_INCLUDE $ENV{HOME}/libs/armadillo/include CACHE FILEPATH "Path to Armadillo Include")
include_directories(include)
include_directories(${ARMADILLO_INCLUDE})

add_definitions(-DSYMS)
if (${USE_NM_REP})
	add_definitions(-DFHUB_NM_REP)
endif()

# build both static and shared version
add_library(MPShelpers_static STATIC ${SOURCES})
add_library(MPShelpers SHARED ${SOURCES})

if (${USE_NM_REP})
#	message(STATUS "setting NM postfix")
	set(POSTFIX "${POSTFIX}_NM")
#	set(RELEASE_POSTFIX "${RELEASE_POSTFIX}_NM")
#	set(RELDEB_POSTFIX "${RELDEB_POSTFIX}_NM")
endif()
set_target_properties(MPShelpers_static PROPERTIES DEBUG_POSTFIX "${POSTFIX}_debug")
set_target_properties(MPShelpers_static PROPERTIES RELEASE_POSTFIX "${POSTFIX}_release")
set_target_properties(MPShelpers_static PROPERTIES RELWITHDEBINFO_POSTFIX "${POSTFIX}_reldeb")
set_target_properties(MPShelpers PROPERTIES DEBUG_POSTFIX "${POSTFIX}_debug")
set_target_properties(MPShelpers PROPERTIES RELEASE_POSTFIX "${POSTFIX}_release")
set_target_properties(MPShelpers PROPERTIES RELWITHDEBINFO_POSTFIX "${POSTFIX}_reldeb")

install (TARGETS MPShelpers_static DESTINATION ${CMAKE_SOURCE_DIR}/lib)
install (TARGETS MPShelpers DESTINATION ${CMAKE_SOURCE_DIR}/lib)
#install (TARGETS MPShelpers CONFIGURATIONS Debug DESTINATION ${CMAKE_SOURCE_DIR}/lib RENAME libMPShelpers_debug)
