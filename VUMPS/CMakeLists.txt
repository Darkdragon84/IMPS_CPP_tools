cmake_minimum_required(VERSION 3.2)
project(VUMPS)

# set(CMAKE_CXX_COMPILER /opt/intel/bin/icpc)# CACHE STRING "C++ Compiler")

set(SOURCES VUMPS_main.cpp)
#set(INTELROOT $ENV{mc_intel_16})
#set(INTELLIBS ${INTELROOT}/compiler/lib/intel64)
set(INTELROOT /opt/intel)
set(INTELLIBS ${INTELROOT}/lib/intel64)
set(MKLLIBS $ENV{MKLROOT}/lib/intel64 )
#set(MKLROOT /opt/intel/mkl)

#set(MKL_STATIC_LIBS -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group)

set(USE_NM_REP ON CACHE BOOL "Use NM Representation for Hubbard")
set(STATIC_MKL_LIBS ON CACHE BOOL "Link statically")

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build Type")
set(ARMADILLO_INCLUDE $ENV{HOME}/libs/armadillo/include CACHE FILEPATH "Path to Armadillo Include")

set(MPS_HELPERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../MPShelpers CACHE FILEPATH "Path to MPS helpers")
#set(MPS_HELPERS_DIR $ENV{HOME}/C++/MPShelpers CACHE FILEPATH "Path to MPS helpers")

set(ARPACK_DIR $ENV{HOME}/libs/arpack CACHE FILEPATH "Path to Arpack Library")
set(ARPACK_DEBUG libarpack_debug.a CACHE STRING "Arpack Debug Library")
set(ARPACK_RELEASE libarpack_intel_release.a CACHE STRING "Arpack Release Library")


## LAPACK/BLAS Debug Libraries ##################################################################
set(LIB_BLAS blas)
set(LIB_LAPACK lapack)
#find_library(LIB_BLAS NAMES blas libblas HINTS /lib /usr/lib)
#find_library(LIB_LAPACK NAMES lapack liblapack HINTS /lib /usr/lib)
set(LAPACKBLAS_DEBUG ${LIB_BLAS} ${LIB_LAPACK} CACHE STRING "BLAS/Lapack Debug Libraries")

## LAPACK/BLAS Release Libraries (for MKL) ##################################################################

if (${STATIC_MKL_LIBS})
	set(LAPACKBLAS_RELEASE 
	-static-intel
	-Wl,--start-group 
	${MKLLIBS}/libmkl_intel_lp64.a 
	${MKLLIBS}/libmkl_intel_thread.a 
	${MKLLIBS}/libmkl_core.a 
	-Wl,--end-group 
	pthread m dl iomp5 svml imf intlc
	CACHE STRING "BLAS/Lapack Release Libraries")
else ()
	set(LAPACKBLAS_RELEASE 
	-L${MKLLIBS}
	mkl_intel_lp64 mkl_core mkl_intel_thread pthread m dl iomp5 svml imf intlc
	CACHE STRING "BLAS/Lapack Release Libraries")
endif()

set(FORTRAN_LIB_DEBUG gfortran CACHE STRING "Fortran Release Library")
set(FORTRAN_LIB_RELEASE ifcore CACHE STRING "Fortran Release Library")

### compiler specific settings
set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -parallel -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -parallel -march=native -g")

#set(CMAKE_EXE_LINK_STATIC_C_FLAGS)       # remove -Wl,-Bstatic
#set(CMAKE_EXE_LINK_STATIC_CXX_FLAGS)
#set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)       # remove -Wl,-Bdynamic
#set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)
#set(CMAKE_SHARED_LIBRARY_C_FLAGS)         # remove -fPIC
#set(CMAKE_SHARED_LIBRARY_CXX_FLAGS)
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)    # remove -rdynamic
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)


set(INSTALL_DEST ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})

include_directories(${ARMADILLO_INCLUDE})

# MPS helpers (link the separately built libraries statically, instead of compiling here)
if (${USE_NM_REP})
	add_definitions(-DFHUB_NM_REP)
	set(MPS_HELP_LIB_DEBUG ${MPS_HELPERS_DIR}/lib/libMPShelpers_static_NM_debug.a)
	set(MPS_HELP_LIB_RELASE ${MPS_HELPERS_DIR}/lib/libMPShelpers_static_NM_release.a)
	set(MPS_HELP_LIB_RELDEB ${MPS_HELPERS_DIR}/lib/libMPShelpers_static_NM_reldeb.a)
else ()
	set(MPS_HELP_LIB_DEBUG ${MPS_HELPERS_DIR}/lib/libMPShelpers_static_debug.a)
	set(MPS_HELP_LIB_RELASE ${MPS_HELPERS_DIR}/lib/libMPShelpers_static_release.a)
	set(MPS_HELP_LIB_RELDEB ${MPS_HELPERS_DIR}/lib/libMPShelpers_static_reldeb.a)
endif()

#support for symmetric MPS implementation
add_definitions(-DSYMS)

add_executable(VUMPS ${SOURCES})

if (${USE_NM_REP})
	message(STATUS "using NM representation of FHUB")
	# is there a single POSTFIX property?
	set_target_properties(VUMPS PROPERTIES DEBUG_POSTFIX "_NM")
	set_target_properties(VUMPS PROPERTIES RELEASE_POSTFIX "_NM")
	set_target_properties(VUMPS PROPERTIES RELWITHDEBINFO_POSTFIX "_NM")
endif()

# link libraries
link_directories(${MKLLIBS})

# link MPS helper library
target_link_libraries(VUMPS debug ${MPS_HELP_LIB_DEBUG} optimized ${MPS_HELP_LIB_RELASE})
include_directories(${MPS_HELPERS_DIR}/include)

# link arpack
target_link_libraries(VUMPS debug ${ARPACK_DIR}/${ARPACK_DEBUG} optimized ${ARPACK_DIR}/${ARPACK_RELEASE})

# we have to loop, as we have to specify for each single library, if it is to be linked for debug or optimized
foreach (_lib ${LAPACKBLAS_DEBUG})
target_link_libraries(VUMPS debug ${_lib})
endforeach()

foreach (_lib ${LAPACKBLAS_RELEASE})
target_link_libraries(VUMPS optimized ${_lib})
endforeach()

# link fortran libraries
target_link_libraries(VUMPS debug ${FORTRAN_LIB_DEBUG} optimized ${FORTRAN_LIB_RELEASE})

# install destinations
install(TARGETS VUMPS COMPONENT RUNTIME DESTINATION ${INSTALL_DEST})
