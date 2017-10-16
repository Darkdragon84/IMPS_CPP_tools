#ifndef MPS_INCLUDES_ADV_
#define MPS_INCLUDES_ADV_

/// legacy includes
#include "MPSIncludesBasic.h"

/// general MPS specific includes
#include "IterativeSolvers.hpp" // GMRES
#include "eigs.h"

/// includes for quantum numbers
#ifdef _USE_SYMMETRIES_
#include "Block/BlockObj.hpp" // elementary building blocks for MPS (Schmidt value vectors, block (diagonal) matrices and arrays of all these)
#include "Block/MPSBlockMat.hpp" // MPS tensor class
#include "Block/TMBlockFunctions.hpp" // Transfer Matrix Operations
#include "Block/MPSBlockUtilities.hpp" // various MPS helper functions
#include "Block/InvEBlockFunctions.hpp" // Routines for calculating infinite geometric sums of transfer matrices
#include "Block/EigsBlockFunctions.hpp" // Routines for calculating dominant left and right eigenvectors of transfer matrices
#endif // _USE_SYMMETRIES_

#endif // MPS_INCLUDES_ADV_
