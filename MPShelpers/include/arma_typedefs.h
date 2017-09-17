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

//using arma::Mat;
//using arma::Col;

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

//using pRMatType = std::shared_ptr<RMatType>;
//using pCMatType = std::shared_ptr<CMatType>;
//using pRVecType = std::shared_ptr<RVecType>;
//using pCVecType = std::shared_ptr<CVecType>;
//using pIVecType = std::shared_ptr<IVecType>;

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
