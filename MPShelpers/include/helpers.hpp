#ifndef HELPERSHPP_H_
#define HELPERSHPP_H_

#include <cmath>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <string>
#include <sstream>
#include <utility>
#include <time.h>

#include "arma_typedefs.h"

using std::cout;
using std::endl;


/**< DENSE MATRIX MULTIPLICATIONS AND DIVISIONS BY SCHMIDT VALUES *********************************************************/
/**< THESE ARE ALL MODIFYING OPERATIONS, FOR NON-MODIFYING ONE USE THE NON-MODIFYING OPERATORS << AND >> */
/**< multiplication */
template<typename T>
inline
void
MultMatLamRight(Mat<T>& in, const RVecType& lam)
{
    assert(in.n_cols==lam.n_elem);
//    for (uint i=0;i<lam.n_elem;++i) in.unsafe_col(i)*=lam(i);
    for (uint i=0;i<lam.n_elem;++i) in.col(i)*=lam(i);
}

template<typename T>
inline
void
MultMatLamLeft(const RVecType& lam, Mat<T>& in)
{
    assert(in.n_rows==lam.n_elem);
    for (uint i=0;i<lam.n_elem;++i) in.row(i)*=lam(i);
}

/**< division */
template<typename T>
inline
void
DivMatLamRight(Mat<T>& in, const RVecType& lam, double thresh=1e-14) /// for now, check for threshold, but divide any ways...
{
    assert(in.n_cols==lam.n_elem);
    for (uint i=0;i<lam.n_elem;++i)
    {
        if(lam(i)<thresh)cerr<<"DivMatLamRight: division by "<<lam(i)<<endl;
//        in.unsafe_col(i)/=lam(i);
        in.col(i)/=lam(i);
    }
}

template<typename T>
inline
void
DivMatLamLeft(const RVecType& lam, Mat<T>& in, double thresh=1e-14)
{
    assert(in.n_rows==lam.n_elem);
    for (uint i=0;i<lam.n_elem;++i)
    {
        if(lam(i)<thresh)cerr<<"DivMatLamLeft: division by "<<lam(i)<<endl;
        in.row(i)/=lam(i);
    }
}

/**< modifying operations */
/**< multiplication */
template<typename T>
inline
Mat<T>&
operator<(Mat<T>& mat, const RVecType& lam) {MultMatLamRight(mat,lam); return mat;}

template<typename T>
inline
Mat<T>&
operator>(const RVecType& lam, Mat<T>& mat) {MultMatLamLeft(lam,mat); return mat;}

/**< division */
template<typename T>
inline
Mat<T>&
operator>(Mat<T>& mat, const RVecType& lam) {DivMatLamRight(mat,lam); return mat;}

template<typename T>
inline
Mat<T>&
operator<(const RVecType& lam, Mat<T>& mat) {DivMatLamLeft(lam,mat); return mat;}

/**< non-modifying operations (pass by value -> create copies first)*/
/**< multiplication */
template<typename T>
inline
Mat<T>
operator<<(Mat<T> mat, const RVecType& lam) {MultMatLamRight(mat,lam); return mat;}

template<typename T>
inline
Mat<T>
operator>>(const RVecType& lam, Mat<T> mat) {MultMatLamLeft(lam,mat); return mat;}

/**< division */
template<typename T>
inline
Mat<T>
operator>>(Mat<T> mat, const RVecType& lam) {DivMatLamRight(mat,lam); return mat;}

template<typename T>
inline
Mat<T>
operator<<(const RVecType& lam, Mat<T> mat) {DivMatLamLeft(lam,mat); return mat;}

/**< dot product for matrices */
template<typename T>
inline
T
dot(const Mat<T>& lhs, const Mat<T>& rhs);

template<>
inline
Real
dot(const Mat<Real>& lhs, const Mat<Real>& rhs)
{
    assert(lhs.n_elem==rhs.n_elem);
    return dot(Col<Real>(const_cast<Real*>(lhs.memptr()),lhs.n_elem,false),Col<Real>(const_cast<Real*>(rhs.memptr()),rhs.n_elem,false));
}

template<>
inline
Complex
dot(const Mat<Complex>& lhs, const Mat<Complex>& rhs)
{
    assert(lhs.n_elem==rhs.n_elem);
    Mat<Complex> lhc = conj(lhs);
    return dot(Col<Complex>(const_cast<Complex*>(lhc.memptr()),lhs.n_elem,false),Col<Complex>(const_cast<Complex*>(rhs.memptr()),rhs.n_elem,false));
}

/// this corrects for the square root scaling of the 2-norm of random vectors with the number of elements.
/// This is especially useful for vectors which should go to zero and where machine precision starts playing a role.
/// In principle, the 2-norm of an N-element vector can then never go below ~ eps*sqrt(N), where eps is the machine precision and N is the number of elements.
template<typename VT>
inline
Real
norm_scaled(const VT& x) {return norm(x)/std::sqrt(x.size());}

template<typename VT>
inline
Real
norm_inf(const VT& x) {return abs(x).max();}

template<typename T>
inline
Real
dot(const Mat<T>& mat);

template<>
inline
Real
dot(const Mat<Real>& mat)
{return dot(mat,mat);}

template<>
inline
Real
dot(const Mat<Complex>& mat)
{return real(dot(mat,mat));}

/**< for printing c-style arrays */
template<typename T>
void
PrintArray(T arr[], uint N, std::string name="")
{
    if (name!="")cout<<name<<":"<<endl;
    for (uint i=0;i<N;++i)cout<<arr[i]<<endl;
    cout<<endl;
}


template<typename T>
void
qr_pos(Mat<T>& Q, Mat<T>& R, const Mat<T>& X)
{
    qr_econ(Q,R,X);
    Col<T> rdgl = R.diag();
    for (uint i=0;i<rdgl.n_elem;++i)
    {
        if (rdgl(i)<0)
        {
            Q.col(i) *= -1;
            R.row(i) *= -1;
        }
    }
}

template<typename T>
inline
Mat<T>
qr_pos(Mat<T>& R, const Mat<T>& X)
{
    Mat<T> Q;
    qr_pos(Q,R,X);
    return Q;
}

/// the library constructor SpMat<eT>(const Base<eT,T1>& ) points to SpMat<eT>::operator=(const Base<eT, T1>&) (SpMat_meat.hpp:805)
/// it only checks for nonzero elements via x[i] != eT(0), which is insane!! any ridiculously small number like e.g. 1e-300 would still be considered finite.
/// This is not useful for our purpose, so define sth similar with a threshold. It would make sense to use it as a relative threshold, e.g. thresh = max(in)*absthresh, where absthresh e.g. ~1e-15
template<typename T>
SpMat<T>
Dense2Sparse(const Mat<T>& in, T thresh=1e-15)
{
    SpMat<T> out(in.n_rows,in.n_cols);
    for (uint j=0;j<in.n_cols;++j)
    {
        for (uint i=0;i<in.n_rows;++i) if (abs(in(i,j))>thresh) out(i,j) = in(i,j);
    }
    return out;
}

#ifdef _USE_SYMMETRIES_

#include <map>
#include <deque>
#include "DimMaps.hpp"

/// TODO (valentin#1#): find out why this doesn't work:
///// general method to apply single MT -> MT functions to every single element of the MTArray
///// this is for BlockDiagMat/BlockMat/MPSBlockMat Arrays
//template<typename MT>
//std::deque<MT>
//ApplyFun(const std::deque<MT>& in, const std::function<MT (const MT&)>& F)
//{
//    std::deque<MT> out;
//    for (const MT& it : in) out.emplace_back(F(it));
//    return out;
//}

/// determine minima and maxima of a map
template<typename KT, typename VT>
inline
VT
min(const std::map<KT,VT>& X)
{
    auto it = X.cbegin();
    VT val = it->second;
    ++it;
    for ( ; it!=X.cend();++it) val = min(val,it->second);
    return val;
}


template<typename KT, typename VT>
inline
VT
max(const std::map<KT,VT>& X)
{
    auto it = X.cbegin();
    VT val = it->second;
    ++it;
    for ( ; it!=X.cend();++it) val = std::max(val,it->second);
    return val;
}

template<typename KT>
inline
bool
all(const std::map<KT,bool>& X)
{
    bool out = true;
    /// as soon as we encounter a false entry, we can return false
    for (const auto& it : X) if (!it.second) return false;
    return out;
}


template<typename KT>
inline
bool
any(const std::map<KT,bool>& X)
{
    bool out = false;
    /// as soon as we encounter a true entry, we can return true
    for (const auto& it : X) if (it.second) return true;
    return out;
}

/// check if arrays of containers are empty or not
/// DEFINE SEPARATELY FOR MPSBLOCKMATS, AS THEY INITIALLY ARE ALWAYS OF LENGTH d, BUT THE CONTAINED BLOCKMATS COULD BE EMPTY
template<typename CT>
inline
bool
all(const std::deque<CT>& X)
{
    bool out = true;
    /// as soon as we encounter a false entry, we can return false
    for (const auto& it : X) if (it.empty()) return false;
    return out;
}

template<typename CT>
inline
bool
any(const std::deque<CT>& X)
{
    bool out = false;
    /// as soon as we encounter a true entry, we can return true
    for (const auto& it : X) if (!it.empty()) return true;
    return out;
}

template<typename CT>
inline
bool
all(const std::vector<CT>& X)
{
    bool out = true;
    /// as soon as we encounter a false entry, we can return false
    for (const auto& it : X) if (it.empty()) return false;
    return out;
}

template<typename CT>
inline
bool
any(const std::vector<CT>& X)
{
    bool out = false;
    /// as soon as we encounter a true entry, we can return true
    for (const auto& it : X) if (!it.empty()) return true;
    return out;
}

template<typename KT, typename VT>
inline
VT
sum(const std::map<KT,VT>& X)
{
    auto it = X.cbegin();
    VT val = it->second;
    while (++it!=X.cend()) val += it->second;
    return val;
}

template<typename KT>
std::ostream& operator<<(std::ostream& os, const std::tuple<KT,KT,uint,uint>& X)
{
    os<<"("<<std::get<0>(X)<<","<<std::get<1>(X)<<"): "<<std::get<2>(X)<<" x "<<std::get<3>(X);
    return os;
}

template<typename KT>
std::ostream& operator<<(std::ostream& os, const std::tuple<KT,uint,uint>& X)
{
    os<<std::get<0>(X)<<": "<<std::get<1>(X)<<" x "<<std::get<2>(X);
    return os;
}

template<typename KT>
std::ostream& operator<<(std::ostream& os, const std::pair<KT,uint>& X)
{
    os<<X.first<<": "<<X.second<<" x "<<X.second;
    return os;
}

template<typename KT>
std::ostream& operator<<(std::ostream& os, const dimkeypair_vec<KT>& V)
{
    for (const auto& it : V) os<<it<<endl;
    return os;
}

template<typename KT>
std::ostream& operator<<(std::ostream& os, const dimpair_vec<KT>& V)
{
    for (const auto& it : V) os<<it<<endl;
    return os;
}

template<typename KT>
std::ostream& operator<<(std::ostream& os, const dim_vec<KT>& V)
{
    for (const auto& it : V) os<<it<<endl;
    return os;
}

template<typename VT>
VT sign(VT x) { return x/std::abs(x);};

//template<>
//Real sign(Real x){ return (x > 0) ? Real(1) : ( (x < 0) ? Real(-1) : Real(0) );}
//
//template<>
//Complex sign(Complex x){ return x/std::abs(x);}

template<typename VT>
std::string to_varstring(const VT val, const uint prec = 6)
{
    std::stringstream out;
    out << std::setprecision(prec) << val;
    return out.str();
}


#endif // _USE_SYMMETRIES_

#endif
