#ifndef TYPEDEFS_ARM_H_
#define TYPEDEFS_ARM_H_
//
//#include <assert.h>
//#include <complex>
//#include <map>
//#include <tuple>
//#include <string>
//#include <memory>
#include <armadillo>
#include "Defs.h"
//#include <time.h>


using namespace arma; // I do realize this is not optimal, but removing this would take a lot of fixing throughout all sorts of other .hpp's and .cpp's

/// typedefs
using Real = double;
using uint = unsigned int;
using Complex = std::complex<double>;

using RMatType = Mat<Real>;
using CMatType = Mat<Complex>;
using RVecType = Col<Real>;
using CVecType = Col<Complex>;
using IVecType = Col<int>;
using UIVecType = Col<uint>;
using BoolVecType = Col<bool>;

#ifdef COMPLEX
using Scalar = Complex;
using VecType = CVecType;
using MatType = CMatType;
#else
using Scalar = Real;
using VecType = RVecType;
using MatType = RMatType;
#endif

/// additional typedefs
enum dirtype {r,l,s,c};

#endif // TYPEDEFS_ARM_H_
