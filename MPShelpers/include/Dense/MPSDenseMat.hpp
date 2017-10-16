#ifndef MPS_DENSE_MAT_H
#define MPS_DENSE_MAT_H

#include <vector>
#include <cmath>
#include <assert.h>
#include <fstream>

#include "arma_typedefs.h"
//#include "tictoc.hpp"


using namespace std;

/// MPS typedefs
typedef RVecType Lambda;

template<typename T>
inline
T
entropy(const Col<T>& lam)
{
    Col<T> lam2 = pow(lam,2);
    return -dot(lam2,log(lam2));
}

/// MPS MATRIX CLASS (WITHOUT QUANTUM NUMBERS) ************************************************************************************/
/// DECLARATION -------------------------------------------------------------------------------------------------------------------*/
template<typename T>
class MPSDenseMat : public std::vector<Mat<T> >
{
/// TODO (valentin#1#): move c'tor and assignment op
public:
    MPSDenseMat(uint dim, uint NSites=1):std::vector<Mat<T> >(std::pow(dim,NSites)),dim_(dim),NSites_(NSites) {}; /// std c'tor

    template<typename fill_type>
    MPSDenseMat(uint dim, uint ml, uint mr, uint NSites ,const fill::fill_class<fill_type>& filler);

    MPSDenseMat(uint dim, uint ml, uint mr, uint NSites = 1) : MPSDenseMat(dim,ml,mr,NSites,fill::zeros) {};
//    MPSDenseMat(const MPSDenseMat& other):std::vector<Mat<T> >(other),dim_(other.GetDim()),NSites_(other.GetNSites()) {cout<<"copied"<<endl;}; /// copy c'tor
//    MPSDenseMat(MPSDenseMat&& other):std::vector<Mat<T> >(std::move(other)),dim_(other.GetDim()),NSites_(other.GetNSites()) {cout<<"moved"<<endl;}; /// move c'tor
    MPSDenseMat(const std::vector<Mat<T> >& matvec,uint dim);
    MPSDenseMat(const Mat<T>& M, const uint dim, const uint NSites, dirtype dir=l);
    /// Create MPS from linear C-style array: beware! The MPS matrices are sequentially filled by linear parts of mem, so e.g. MPS.GetDenseMat(l) and Mat<T>(mem,ml,mr) will NOT be the same!!
    MPSDenseMat(T* mem, const uint ml, const uint mr, const uint dim, const uint NSites, dirtype dir=l);

//    MPSDenseMat& operator=(const MPSDenseMat&) = default;
//    MPSDenseMat& operator=(MPSDenseMat&& lhs);

//    void Set(const std::vector<Mat<T> >& matvec);
//    void Set(const Mat<T>& M, mattype mt=col);

    MPSDenseMat& MultLamLeft(const Lambda& lam);
    MPSDenseMat& MultLamInvLeft(const Lambda& lam, double thresh=1e-14);
    MPSDenseMat& MultLamRight(const Lambda& lam);
    MPSDenseMat& MultLamInvRight(const Lambda& lam, double thresh=1e-14);

    MPSDenseMat& MultMatLeft(const Mat<T>& mat);
    MPSDenseMat& MultMatRight(const Mat<T>& mat);

    inline MPSDenseMat& operator+=(const MPSDenseMat<T>& other);
    inline MPSDenseMat& operator-=(const MPSDenseMat<T>& other);

    Mat<T> GetDenseMat(dirtype dir) const;
    Col<T> GetVector() const;
//    inline MPSDenseMat& t();
    inline MPSDenseMat t() const;
    inline MPSDenseMat st() const;

    ///  ROUTINES FOR ARPACK -----------------------------------------*/

//    inline void ArMultLeftUni(T* in, T* out) const; /// Armadillo: just wow!
//    inline void ArMultRightUni(T* in, T* out) const; /// Armadillo: just wow!
    ///-----------------------------------------------------------------


    /// DiskIO
    bool save(std::string name) const;
    bool save(std::ofstream& file) const;
    bool load(std::string name);
    bool load(std::ifstream& file);

    /// Getters
    inline uint GetNSites() const {return NSites_;};
    inline uint GetDim() const {return dim_;};
    inline uint GetMl() const;
    inline uint GetMr() const;

    inline void Disp(uint ind) const;

    inline void print(string name="") const;
protected:
    uint dim_,NSites_;
};

/// typedefs
typedef MPSDenseMat<Real> RMPSDenseMat;
typedef MPSDenseMat<Complex> CMPSDenseMat;
typedef MPSDenseMat<Scalar> MPSMat;

/// IMPLEMENTATION --------------------------------------------------------------------------------------------------------- */
template<typename T>
MPSDenseMat<T>::MPSDenseMat(const std::vector<Mat<T> >& matvec,uint dim):
    std::vector<Mat<T> >(matvec),
    dim_(dim),
    NSites_(round(log(matvec.size())/log(dim)))
{
    assert(this->size() % this->dim_==0);
#ifndef NDEBUG // put explicitly in preproc clause to avoid loop in release mode
    uint ml=this->at(0).n_rows;
    uint mr=this->at(0).n_cols;
//    for (auto it=this->begin(); it!=this->end(); ++it)
    for (const auto& vit : *this)
    {
        assert(vit.n_rows==ml);
        assert(vit.n_cols==mr);
    }
#endif
};

//template<typename T>
//MPSDenseMat<T>::MPSDenseMat(uint dim, uint ml, uint mr, uint NSites):dim_(dim),NSites_(NSites)
////MPSDenseMat<T>::MPSDenseMat(uint dim, uint ml, uint mr, uint NSites):dim_(dim),NSites_(NSites)
//{
//    uint numel=std::pow(dim_,NSites_);
//    this->reserve(numel);
//    for (uint i=0; i<numel; ++i) this->emplace_back(Mat<T>(ml,mr,filler));
////    for (uint i=0; i<numel; ++i) this->emplace_back(zeros<Mat<T> >(ml,mr));
//}

template<typename T>
template<typename fill_type>
MPSDenseMat<T>::MPSDenseMat(uint dim, uint ml, uint mr, uint NSites, const fill::fill_class<fill_type>& filler):dim_(dim),NSites_(NSites)
//MPSDenseMat<T>::MPSDenseMat(uint dim, uint ml, uint mr, uint NSites):dim_(dim),NSites_(NSites)
{
    uint numel=std::pow(dim_,NSites_);
    this->reserve(numel);
    for (uint i=0; i<numel; ++i) this->emplace_back(Mat<T>(ml,mr,filler));
//    for (uint i=0; i<numel; ++i) this->emplace_back(zeros<Mat<T> >(ml,mr));
}

template<typename T>
MPSDenseMat<T>::MPSDenseMat(const Mat<T>& M, const uint dim, uint NSites, dirtype dir):dim_(dim),NSites_(NSites)
{
    uint numel=std::pow(dim_,NSites_);
    this->reserve(numel);
    if (dir == l)
    {
        assert(M.n_rows%numel==0);
        uint mr=M.n_cols;
        uint ml=M.n_rows/numel;
        for (uint i=0; i<numel; ++i)this->push_back(M.submat(i*ml,0,(i+1)*ml-1,mr-1));
    }
    else if(dir == r)
    {
        assert(M.n_cols%numel==0);
        uint mr=M.n_cols/numel;
        uint ml=M.n_rows;
        for (uint i=0; i<numel; ++i)this->push_back(M.submat(0,i*mr,ml-1,(i+1)*mr-1));
    }
    else cerr<<"wrong structure"<<endl;
}

template<typename T>
MPSDenseMat<T>::MPSDenseMat(T* mem, const uint ml, const uint mr, const uint dim, const uint NSites, dirtype dir):std::vector<Mat<T> >(dim),dim_(dim),NSites_(NSites)
{
    uint pos=0;
    for (auto& it : *this) {it = Mat<T>(&mem[pos],ml,mr);pos+=ml*mr;}
}

//template<typename T>
//MPSDenseMat<T>&
//MPSDenseMat<T>::operator=(MPSDenseMat<T>&& other)
//{
//    if (this != &other)
//    {
//        MPSDenseMat(std::move(other));
//    }
//    return *this;
//}

template<typename T>
inline
uint
MPSDenseMat<T>::GetMl() const
{
    uint ml = this->at(0).n_rows;
#ifndef NDEBUG // put explicitly in preproc clause to avoid loop in release mode
    for (const auto& vit : *this) assert(vit.n_rows==ml);
#endif
    return ml;
}

template<typename T>
inline
uint
MPSDenseMat<T>::GetMr() const
{
    uint mr = this->at(0).n_cols;
#ifndef NDEBUG // put explicitly in preproc clause to avoid loop in release mode
    for (const auto& vit : *this) assert(vit.n_cols==mr);
#endif
    return mr;
};

template<typename T>
inline
void
MPSDenseMat<T>::Disp(uint ind) const
{
    assert(ind<this->size());
    cout<<ind<<":"<<endl;
    cout<<this->at(ind);
}

template<typename T>
inline
void
MPSDenseMat<T>::print(string name) const
{
    if(name!="")cout<<name<<endl;
    for (uint i=0; i<this->size(); ++i)Disp(i);
}

template<typename T>
ostream&
operator<<(ostream& os, const MPSDenseMat<T>& MPS)
{
    for (uint s=0;s<MPS.size();++s)
    {
        os<<s<<":"<<endl;
        os<<MPS[s]<<endl;
    }
    os<<endl;
    return os;
}

template<typename T>
MPSDenseMat<T>
MPSDenseMat<T>::t() const
{
    MPSDenseMat<T> out(this->dim_,this->NSites_);
    typename MPSDenseMat<T>::iterator outit;
    typename MPSDenseMat<T>::const_iterator init;

    for (outit=out.begin(),init=this->begin();init!=this->end();++outit,++init) (*outit) = init->t();
    return out;
}

template<typename T>
MPSDenseMat<T>
MPSDenseMat<T>::st() const
{
    MPSDenseMat<T> out(this->dim_,this->NSites_);
    typename MPSDenseMat<T>::iterator outit;
    typename MPSDenseMat<T>::const_iterator init;

    for (outit=out.begin(),init=this->begin();init!=this->end();++outit,++init) (*outit) = init->st();
    return out;
}

/// return matrix form
template<typename T>
Mat<T>
MPSDenseMat<T>::GetDenseMat(dirtype dir) const
{
    Mat<T> out;
    uint ml=this->GetMl(),mr=this->GetMr();
    if (dir==l)
    {
        out.set_size(this->size()*ml,mr);
        uint i=0;
        for (const auto& vit : *this)
        {
            out.rows(i*ml,(i+1)*ml-1) = vit;
            ++i;
        }
    }
    else if (dir==r)
    {
        out.set_size(ml,this->size()*mr);
        uint i=0;
        for (const auto& vit : *this)
        {
            out.cols(i*mr,(i+1)*mr-1) = vit;
            ++i;
        }
    }
    else {cerr<<"GetDenseMat: wrong matrix format specified"<<endl;abort();}
    return out;
}
template<typename T>
Col<T>
MPSDenseMat<T>::GetVector() const
{
    uint d = this->GetDim();
    uint ml = this->GetMl();
    uint mr = this->GetMr();

    Col<T> out(d*ml*mr);
    T* mem = out.memptr();
    uint pos=0;
    for (const auto& it : *this)
    {
        std::copy(it.memptr(),it.memptr()+ml*mr,&mem[pos]);
        pos+=ml*mr;
    }
    return out;
}


/// DiskIO*********************************************************************************************/
template<typename T>
bool
MPSDenseMat<T>::save(std::string name) const
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay = save(file);
    file.close();

    return save_okay;
}

template<typename T>
bool
MPSDenseMat<T>::save(std::ofstream& file) const
{
    bool save_okay = file.is_open();
    uint dim=this->dim_,NS=this->NSites_;

    file.write(reinterpret_cast<char*>(&dim), sizeof(uint));
    file.write(reinterpret_cast<char*>(&NS), sizeof(uint));

    for (const auto& vit : *this) save_okay = save_okay && diskio::save_arma_binary(vit,file);

    return save_okay;
}

bool load(std::string name);
bool load(std::ifstream& file);

/// Multiplication and Division by Schmidt Values *********************************************************************************************/
template<typename T>
MPSDenseMat<T>&
MPSDenseMat<T>::MultLamLeft(const Lambda& lam)
{
    uint m=lam.n_elem;
    for (auto it=this->begin(); it!=this->end(); ++it)
    {
        assert(it->n_rows==m);
        for (uint i=0; i<m; ++i) it->row(i)*=lam(i);
    }
    return *this;
}

template<typename T>
MPSDenseMat<T>&
MPSDenseMat<T>::MultLamInvLeft(const Lambda& lam, double thresh)
{
    uint m=lam.n_elem;
    for (auto it=this->begin(); it!=this->end(); ++it)
    {
        assert(it->n_rows==m);
        for (uint i=0; i<m; ++i)
        {
            if (lam(i)<thresh) cerr<<"MultLamInvLeft: division by "<<lam(i)<<endl;
            it->row(i)/=lam(i);
        }
    }
    return *this;
}

template<typename T>
MPSDenseMat<T>&
MPSDenseMat<T>::MultLamRight(const Lambda& lam)
{
    uint m=lam.n_elem;
    for (auto it=this->begin(); it!=this->end(); ++it)
    {
        assert(it->n_cols==m);
        for (uint i=0; i<m; ++i) it->unsafe_col(i)*=lam(i);
    }
    return *this;
}

template<typename T>
MPSDenseMat<T>&
MPSDenseMat<T>::MultLamInvRight(const Lambda& lam, double thresh)
{
    uint m=lam.n_elem;
    for (auto it=this->begin(); it!=this->end(); ++it)
    {
        assert(it->n_cols==m);
        for (uint i=0; i<m; ++i)
        {
            if (lam(i)<thresh) cerr<<"MultLamInvRight: division by "<<lam(i)<<endl;
            it->unsafe_col(i)/=lam(i);
        }
    }
    return *this;
}

template<typename T>
inline
MPSDenseMat<T>&
MPSDenseMat<T>::MultMatLeft(const Mat<T>& mat)
{
    for (auto& it : *this) it = mat*it;
    return *this;
}

template<typename T>
inline
MPSDenseMat<T>&
MPSDenseMat<T>::MultMatRight(const Mat<T>& mat)
{
    for (auto& it : *this) it = it*mat;
    return *this;
}

template<typename T>
inline
MPSDenseMat<T>&
MPSDenseMat<T>::operator+=(const MPSDenseMat<T>& other)
{
    assert(this->GetDim() == other.GetDim() && "MPSDenseMat<T> operator+: lhs and rhs need to be of same physical dimension");
    assert(this->GetNSites() == other.GetNSites() && "MPSDenseMat<T> operator+: lhs and rhs need to be of same size");

    typename MPSDenseMat<T>::const_iterator rhit;
    typename MPSDenseMat<T>::iterator lhit;

    for (lhit=this->begin(),rhit=other.begin();lhit!=this->end();++lhit,++rhit) *lhit += *rhit;
    return *this;
}

template<typename T>
inline
MPSDenseMat<T>&
MPSDenseMat<T>::operator-=(const MPSDenseMat<T>& other)
{
    assert(this->GetDim() == other.GetDim() && "MPSDenseMat<T> operator+: lhs and rhs need to be of same physical dimension");
    assert(this->GetNSites() == other.GetNSites() && "MPSDenseMat<T> operator+: lhs and rhs need to be of same size");

    typename MPSDenseMat<T>::const_iterator rhit;
    typename MPSDenseMat<T>::iterator lhit;

    for (lhit=this->begin(),rhit=other.begin();lhit!=this->end();++lhit,++rhit) *lhit -= *rhit;
    return *this;
}

/// OUTSIDE HELPER FUNCTIONS *******************************************************************************************************/

/// LAMBDA OPERATIONS ----------------------------------------------------------------------------------------------------
/// NON MODIFYING --------------------------------------------------------------------------------------------------------------
template<typename T>
inline
MPSDenseMat<T>
operator<<(MPSDenseMat<T> in, const Lambda& lam)
{
    /**< M*lam = M<<lam */
    return in.MultLamRight(lam);
}

template<typename T>
inline
MPSDenseMat<T>
operator>>(MPSDenseMat<T> in, const Lambda& lam)
{
    /**< M/lam = M>>lam */
    return in.MultLamInvRight(lam);
}

template<typename T>
inline
MPSDenseMat<T>
operator>>(const Lambda& lam, MPSDenseMat<T> in)
{
    /**< lam*M = lam>>M */
    return in.MultLamLeft(lam);
}

template<typename T>
inline
MPSDenseMat<T>
operator<<(const Lambda& lam, MPSDenseMat<T> in)
{
    /**< lam\M = lam<<M */
    return in.MultLamInvLeft(lam);
}


template<typename T>
inline
MPSDenseMat<T>
operator*(const Mat<T>& mat, MPSDenseMat<T> in)
{
    /**< mat*M */
    return in.MultMatLeft(mat);
}


template<typename T>
inline
MPSDenseMat<T>
operator*(MPSDenseMat<T> in, const Mat<T>& mat)
{
    /**< M*mat */
    return in.MultMatRight(mat);
}

/// multiplying two MPS to create a joint MPS matrix
template<typename T>
MPSDenseMat<T>
operator*(const MPSDenseMat<T>& lhs, const MPSDenseMat<T>& rhs)
{
    assert(lhs.GetDim()==rhs.GetDim() && "MPSDenseMat<T> operator*: lhs and rhs have different LocalDim");
    MPSDenseMat<T> out(lhs.GetDim(),lhs.GetNSites()+rhs.GetNSites());
    for (uint i = 0;i<lhs.size();++i)
    {
        for (uint j=0;j<rhs.size();++j) out[i*rhs.size()+j] = lhs[i]*rhs[j];
    }
    return out;
}

/// Adding or subtracting two MPS matrices of equal size

template<typename T>
inline
MPSDenseMat<T>
operator+(MPSDenseMat<T> lhs, const MPSDenseMat<T>& rhs)
{
    return lhs+=rhs;
}

template<typename T>
inline
MPSDenseMat<T>
operator-(MPSDenseMat<T> lhs, const MPSDenseMat<T>& rhs)
{
    return lhs-=rhs;
}
//template<typename T>
//MPSDenseMat<T>
//operator+(const MPSDenseMat<T>& lhs, const MPSDenseMat<T>& rhs)
//{
//    assert(lhs.GetDim() == rhs.GetDim() && "MPSDenseMat<T> operator+: lhs and rhs need to be of same physical dimension");
//    assert(lhs.GetNSites() == rhs.GetNSites() && "MPSDenseMat<T> operator+: lhs and rhs need to be of same size");
//    MPSDenseMat<T> out(lhs.GetDim(),lhs.GetNSites());
//
//    typename MPSDenseMat<T>::const_iterator lhit,rhit;
//    typename MPSDenseMat<T>::iterator outit=out.begin();
//    lhit = lhs.begin();
//    rhit = rhs.begin();
//
//
//    for (lhit=lhs.begin(),rhit=rhs.begin();lhit!=lhs.end();++lhit,++rhit)
//    {
//        *(outit++) = *lhit + *rhit;
//    }
//    return out;
//}

/// MODIFYING --------------------------------------------------------------------------------------------------------------
template<typename T>
inline MPSDenseMat<T>& operator<(MPSDenseMat<T>& in, const Lambda& lam)
{
    /**< M*lam = M<lam */
    in.MultLamRight(lam);
    return in;
}

template<typename T>
inline MPSDenseMat<T>& operator>(MPSDenseMat<T>& in, const Lambda& lam)
{
    /**< M/lam = M>lam */
    in.MultLamInvRight(lam);
    return in;
}

template<typename T>
inline MPSDenseMat<T>& operator>( const Lambda& lam, MPSDenseMat<T>& in)
{
    /**< lam*M = lam>M */
    in.MultLamLeft(lam);
    return in;
}

template<typename T>
inline MPSDenseMat<T>& operator<(const Lambda& lam, MPSDenseMat<T>& in)
{
    /**< lam\M = lam<<M */
    in.MultLamInvLeft(lam);
    return in;
}
#endif // MPS_MAT_H


//
///// NON MODIFYING --------------------------------------------------------------------------------------------------------------
//template<typename T>
//inline
//MPSDenseMat<T>
//operator<<(const MPSDenseMat<T>& in, const Lambda& lam)
//{
//    /**< M*lam = M<<lam */
//    MPSDenseMat<T> out(in);
//    out.MultLamRight(lam);
//    return out;
//}
//
//template<typename T>
//inline
//MPSDenseMat<T>
//operator>>(const MPSDenseMat<T>& in, const Lambda& lam)
//{
//    /**< M/lam = M>>lam */
//    MPSDenseMat<T> out(in);
//    out.MultLamInvRight(lam);
//    return out;
//}
//
//template<typename T>
//inline
//MPSDenseMat<T>
//operator>>(const Lambda& lam, const MPSDenseMat<T>& in)
//{
//    /**< lam*M = lam>>M */
//    MPSDenseMat<T> out(in);
//    out.MultLamLeft(lam);
//    return out;
//}
//
//template<typename T>
//inline
//MPSDenseMat<T>
//operator<<(const Lambda& lam, const MPSDenseMat<T>& in)
//{
//    /**< lam\M = lam<<M */
//    MPSDenseMat<T> out(in);
//    out.MultLamInvLeft(lam);
//    return out;
//}
//
//
//template<typename T>
//inline
//MPSDenseMat<T>
//operator*(const Mat<T>& mat, const MPSDenseMat<T>& in)
//{
//    MPSDenseMat<T> out(in);
//    return out.MultMatLeft(mat);
//}
//
//
//template<typename T>
//inline
//MPSDenseMat<T>
//operator*(const MPSDenseMat<T>& in, const Mat<T>& mat)
//{
//    MPSDenseMat<T> out(in);
//    return out.MultMatRight(mat);
//}
//
//template<typename T>
//MPSDenseMat<T>
//operator*(const MPSDenseMat<T>& lhs, const MPSDenseMat<T>& rhs)
//{
//    assert(lhs.GetDim()==rhs.GetDim() && "MPSDenseMat<T> operator*(): lhs and rhs have different LocalDim");
//    MPSDenseMat<T> out(lhs.GetDim(),lhs.GetNSites()+rhs.GetNSites());
//    for (uint i = 0;i<lhs.size();++i)
//    {
//        for (uint j=0;j<rhs.size();++j) out[i*rhs.size()+j] = lhs[i]*rhs[j];
//    }
//    return out;
//}
