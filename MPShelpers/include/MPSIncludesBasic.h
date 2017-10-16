#ifndef MPS_INCLUDES_BASIC_
#define MPS_INCLUDES_BASIC_

/// legacy includes
#include <assert.h>
#include <time.h>

/// general includes
#include <iostream>
#include <fstream>
#include <sstream>

#include <utility>
#include <functional>
#include <algorithm>

#include <complex>
#include <cmath>

#include <vector>
#include <map>
#include <tuple>
#include <deque>
#include <initializer_list>

#include <string>
#include <memory>

#include <armadillo>

/// general MPS specific includes
#include "Defs.h"
#ifdef _USE_SYMMETRIES_
#include "DimMaps.hpp" // definition of certain often needed special containers containing symmetry sector dimension information
#endif // _USE_SYMMETRIES_
#include "arma_typedefs.h" // lots of typedefs used throughout and
#include "helpers.h" // header for general helper functions that need to be compiled separately
#include "helpers.hpp" // header for general include only helper functions
#include "OperatorTypes.hpp" // Sparse Operator class (sparse matrix + some additional functionality)

/// includes for quantum numbers
#ifdef _USE_SYMMETRIES_
#include "Block/symobj.hpp" // elementary object encoding the symmetry group operation. Neede to perform group operations with quantum numbers defined in KeyTypes.hpp
#include "Block/KeyTypes.hpp" // Quantum number object, used to label blocks in symmetric MPS tensors
#include "Block/ItoKey.hpp" // Object translating physical indices into quantum numbers, used for filling symmetric MPS tensors with correct labels of quantum number sectors
#endif // _USE_SYMMETRIES_

#endif // MPS_INCLUDES_BASIC_
