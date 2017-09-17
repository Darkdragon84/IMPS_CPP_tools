#ifndef MPS_BLOCK_MAT_H
#define MPS_BLOCK_MAT_H

//#include <vector>
//#include <deque>
//#include <map>
//#include <cmath>
//#include <cassert>
//#include <fstream>
//
//#include "arma_typedefs.h"
//#include "BlockObj.hpp"
//#include "ItoKey.hpp"

using std::map;
using std::cout;
using std::endl;
using std::get;
using std::min;
using std::max;

/// fwd declarations
template<typename KT,typename VT>
class MPSBlockMatArray;

/** MPS MATRIX CLASS (WITH QUANTUM NUMBERS) *****************************************************************************************************
defined as a vector of maps. The vector indices correspond to the MPS physical indices and for each of them there is a matrix in blockform containing
values of VT type. These blockmatrices are stored as a map, where KT is the Key after which the separate blocks are ordered within the map
(see class BlockMat in BlockObj.hpp). In more detail, the map Key is actually the left (set of) quantum number(s), which is connected to the right (set of)
quantum number(s) via the physical index according to the underlying abelian group operation, i.e. [right = left + index] or [left = right - index].
a pair of KT, where the left and right element are connected via the physical index according to the underlying abelian group operation, i.e.
[right = left + index] or [left = right - index]. The blocks are therefore stored according to their left quantum numbers.

KT can be any type viable as a key for a std::map, i.e. it needs to have a copy (and if possible, move) constructor and there must be an operation < that compares
two elements of KT type. Additionally, for the MPS container, the + and - operators need to be defined. They should correspond to the abelian group action.
/// DECLARATION --------------------------------------------------------------------------------------------------------------------------------*/

/// TODO (valentin#1#): We should really include I2K as member variable, as we need this many times! That means we have to add an additional template parameter N... also think about if we can just use a reference of I2K or if we should store a copy (probably copy is safer)

template<typename KT,typename VT>
class MPSBlockMat : public std::vector<BlockMat<KT,VT> >
{
public:
    typedef KT key_type;
    typedef VT scalar_type;
    typedef MPSBlockMatArray<KT,VT> array_type;

    typedef typename BlockMat<KT,VT>::iterator miter;
    typedef typename BlockMat<KT,VT>::const_iterator mciter;

    typedef struct
    {
        /// define pair of phys. index and iterator to corresponding element (matrix within a BlockMat)
        /// we will need it to fill the matrix for which we will determine the null space
        typedef typename std::pair<uint,mciter> ph_ind_iter; /// pair to store phys. index and iterator to matrix within a BlockMat

        uint mr_tot, ml_tot; /// bond dimension of current QN (for which we are obtaining this secdata)
        std::vector<ph_ind_iter> v_phys_ind; /// stores pairs of (phys. index and iterator to matrix within a BlockMat)

        /// maps phys. index to tuple of matrix dimension, corresponding span (in dense matrix) and corresponding symmetry sector
        std::map<uint,std::tuple<uint,span,const KT*> > sizes_spans; /// use pointer to KeyType to avoid copying
    } singlesecdata;

    typedef struct
    {
        /// define tuple of left phys. index, right phys. index iterator to corresponding element (matrix within a BlockMat)
        /// we will need it to fill the matrix to decompose via SVD with the corresponding entries
        typedef typename std::tuple<uint,uint,mciter> ph_ind_iter; /// triple to store left phys. index, right phys. index, iterator to matrix within a BlockMat

        uint ml_tot, mr_tot; /// left and right bond dimensions of current QN (for which we are obtaining this secdata)
        std::vector<ph_ind_iter> v_phys_ind; /// stores tuples of (left phys. index, right phys. index, iterator to matrix within a BlockMat)

        /// maps phys. index to tuple of matrix dimension, corresponding span (in dense matrix) and corresponding symmetry sector
        /// we will fill two separate with all contributing left and right phys. indices and their corresponding sizes/spans/QNs
        std::map<uint,std::tuple<uint,span,const KT*> > sizes_spans_left,sizes_spans_right; /// use pointer to KeyType to avoid copying
    } doublesecdata;

    MPSBlockMat() = delete;
    MPSBlockMat(const MPSBlockMat&) = default;
    MPSBlockMat(MPSBlockMat&&) = default;
    MPSBlockMat(uint d, uint N=1):std::vector<BlockMat<KT,VT> >(pow(d,N)),d_(d),NSites_(N) {};
    MPSBlockMat(const Col<VT>& vec, const std::vector<dimkeypair_vec<KT> >& dimvec, uint d, uint N=1); /// reconstruct MPS from vectorization (e.g. from Vectorize())

    template<uint N, typename fill_type>
    MPSBlockMat(const ItoKey<N,KT>& I2K, const dim_map<KT>& ml, const dim_map<KT>& mr, const fill::fill_class<fill_type>& filler = fill::randn);

    template<typename GO>
    MPSBlockMat(std::ifstream& file, const GO& GroupObj) {this->load(file,GroupObj);}

    /// mixed type copy constructor (only implemented for complex from real)
    template<typename VTO>
    MPSBlockMat(const MPSBlockMat<KT,VTO>& other) {*this = MPSFromMPSCpCt(other);};

    MPSBlockMat& operator=(const MPSBlockMat&) = default;
    MPSBlockMat& operator=(MPSBlockMat&&) = default;

    inline uint GetLocalDim() const {return d_;};
    inline uint GetNSites() const {return NSites_;};

    /// helpers
    void purge() {for (auto& it : *this) it.clear();};
    void ShowDims(const std::string& name="") const;
    void print(const std::string& name="") const;
    inline dim_map<KT> GetUniformSizes() const;
    inline dim_map<KT> GetMl() const;
    inline dim_map<KT> GetMr() const;
    inline uint GetTotalMl() const;
    inline uint GetTotalMr() const;
    inline uint GetNElem() const;
    std::vector<dimkeypair_vec<KT> > GetSizesVector() const;

    typedef typename std::pair<Mat<VT>,singlesecdata> mat_sec_pair;
    typedef typename std::map<KT,mat_sec_pair> dense_map;
    std::map<KT, singlesecdata> GetSingleSecData(dirtype dir) const;
//    std::map<KT, std::pair<Mat<VT>,singlesecdata> > GetDenseMat(dirtype dir) const;
    dense_map GetDenseMat(dirtype dir) const;

    template<uint NL>
    std::map<KT, doublesecdata> GetDoubleSecData(const ItoKey<NL,KT>& I2K) const;
    template<uint NL>
    std::map<KT, std::pair<Mat<VT>,doublesecdata> > GetDenseSplitMat(const ItoKey<NL,KT>& I2K) const;


    Col<VT> Vectorize() const;

    inline MPSBlockMat t() const;
    inline MPSBlockMat st() const;

    inline MPSBlockMat ShiftBothQN(const KT& KL, const KT& KR) const;

    template<uint N>
    MPSBlockMat FlipQN(const ItoKey<N,KT>& I2K, const std::vector<bool>& which = std::vector<bool>()) const;

    template<uint N>
    MPSBlockMat PermuteQN(const ItoKey<N,KT>& I2K, const std::vector<uint>& perm) const;

    /// operators
    inline MPSBlockMat operator-() const {MPSBlockMat out(this->GetLocalDim(),this->GetNSites()); for (uint s=0;s<this->size();++s) out[s] = -(this->at(s)); return out;};
    MPSBlockMat& operator+=(const MPSBlockMat& other);
    MPSBlockMat& operator-=(const MPSBlockMat& other);
    inline MPSBlockMat& operator*=(VT x) {for (auto& it : *this) it*=x; return *this;};
    inline MPSBlockMat& operator/=(VT x) {for (auto& it : *this) it/=x; return *this;};

    /// modifying: pass by reference to modify
    friend inline MPSBlockMat<KT,VT>& operator<(MPSBlockMat<KT,VT>& MPS, const BlockLam<KT>& lam) {MPS.MultMPSLamRight(lam); return MPS;}
    friend inline MPSBlockMat<KT,VT>& operator>(const BlockLam<KT>& lam, MPSBlockMat<KT,VT>& MPS) {MPS.MultMPSLamLeft(lam); return MPS;}
    friend inline MPSBlockMat<KT,VT>& operator>(MPSBlockMat<KT,VT>& MPS, const BlockLam<KT>& lam) {MPS.DivMPSLamRight(lam); return MPS;}
    friend inline MPSBlockMat<KT,VT>& operator<(const BlockLam<KT>& lam, MPSBlockMat<KT,VT>& MPS) {MPS.DivMPSLamLeft(lam); return MPS;}

    /// non-modifying: pass by value to create copy, which is then modified (this is preferred over passing by reference and then copying!)
    friend inline MPSBlockMat<KT,VT> operator<<(MPSBlockMat<KT,VT> MPS, const BlockLam<KT>& lam) {MPS.MultMPSLamRight(lam); return MPS;}
    friend inline MPSBlockMat<KT,VT> operator>>(const BlockLam<KT>& lam, MPSBlockMat<KT,VT> MPS) {MPS.MultMPSLamLeft(lam); return MPS;}
    friend inline MPSBlockMat<KT,VT> operator>>(MPSBlockMat<KT,VT> MPS, const BlockLam<KT>& lam) {MPS.DivMPSLamRight(lam); return MPS;}
    friend inline MPSBlockMat<KT,VT> operator<<(const BlockLam<KT>& lam, MPSBlockMat<KT,VT> MPS) {MPS.DivMPSLamLeft(lam); return MPS;}

    /// DiskIO
    bool save(std::string name) const;
    bool save(std::ofstream& file) const;
    bool load(std::string name);
    template<typename GO>
    bool load(std::ifstream& file, const GO& GroupObj);
//    template<uint N>
//    bool load(std::ifstream& file, const ItoKey<N,KT>& I2K);

protected:
    uint d_,NSites_;
//    ItoKey I2K_;

    /// Multpilication and Division with Schmidt Values
    inline MPSBlockMat& MultMPSLamLeft(const BlockLam<KT>& lam) {for (auto& it : *this) it.MultBlockMatLamLeft(lam); return *this;};
    inline MPSBlockMat& MultMPSLamRight(const BlockLam<KT>& lam) {for (auto& it : *this) it.MultBlockMatLamRight(lam); return *this;};
    inline MPSBlockMat& DivMPSLamLeft(const BlockLam<KT>& lam) {for (auto& it : *this) it.DivBlockMatLamLeft(lam); return *this;};
    inline MPSBlockMat& DivMPSLamRight(const BlockLam<KT>& lam) {for (auto& it : *this) it.DivBlockMatLamRight(lam); return *this;};

};

/**< HELPER FUNCTION TEMPLATES FOR PARTIAL SPECIALIZATION OF MIXED CONSTRUCTOR ***********************************************/
/// This is necessary since for some stupid reason it is not allowed to partially specialize class template member function templates (i.e. the mixed constructor)
template<typename KT, typename VTI, typename VTO>
inline
MPSBlockMat<KT,VTO>
MPSFromMPSCpCt(const MPSBlockMat<KT,VTI>& in);

template<typename KT>
inline
MPSBlockMat<KT,Complex>
MPSFromMPSCpCt(const MPSBlockMat<KT,Real>& in)
{
    MPSBlockMat<KT,Complex> out(in.GetLocalDim(),in.GetNSites());
    for (uint s=0;s<in.size();++s) out[s] = BlockMat<KT,Complex>(in[s]);
    return out;
}


/// TODO (valentin#1#): figure out a decent way to initialize the rng
template<typename KT, typename VT>
MPSBlockMat<KT,VT>::MPSBlockMat(const Col<VT>& vec, const std::vector<dimkeypair_vec<KT> >& dimvec, uint d, uint N)
:MPSBlockMat(d,N)
//:std::vector<BlockMat<KT,VT> >(pow(d,N)),d_(d),NSites_(N),
{
    assert(dimvec.size() == pow(d,N));
    #ifndef NDEBUG
    uint m_tot = 0;
    for (const auto& vit : dimvec) for (const auto& it : vit) m_tot += get<2>(it)*get<3>(it);
    assert(vec.size() == m_tot && "MPSBlockMat<KT,VT>::MPSBlockMat(const Col<VT>&, const vector<dimkeypair_vec<KT> >&): vec and dimvec need to account for the same number of elements");
    #endif // NDEBUG

    this->resize(dimvec.size());
    uint pos = 0;
    for (uint s = 0; s<this->size(); ++s)
    {
        for (const auto& it : dimvec[s])
        {
            this->at(s).emplace_hint(this->at(s).end(),get<0>(it),std::make_pair(get<1>(it),MatType(&vec.memptr()[pos],get<2>(it),get<3>(it))));
            pos += get<2>(it)*get<3>(it);
        }
    }

}

template<typename KT, typename VT>
template<uint N, typename fill_type>
MPSBlockMat<KT,VT>::MPSBlockMat(const ItoKey<N,KT>& I2K, const dim_map<KT>& ml, const dim_map<KT>& mr, const fill::fill_class<fill_type>& filler):MPSBlockMat(I2K.GetLocalDim(),N)
{
    for (uint s=0;s<this->size();++s)
    {
        for (const auto& mlit : ml) /// loop through left quantum numbers and corresponding sizes
            {
                auto mrit = mr.find(mlit.first + I2K[s]); /// if a matching quantum number is found on the right, insert a random matrix with dimensions specified in ml and mr
                if (mrit!=mr.end())
                {
                    this->at(s).emplace_hint(this->at(s).end(),mlit.first,std::make_pair(mrit->first,Mat<VT>(mlit.second, mrit->second, filler)));
                }
            }
    }
}

/**< std ostream output for e.g. screen output with cout */
template<typename KT, typename VT>
ostream&
operator<<(ostream& os, const MPSBlockMat<KT,VT>& MPS)
{
    for (uint i=0;i<MPS.size();++i)
    {
        os<<i<<":"<<endl;
        os<<MPS[i]<<endl;
    }
    return os;
}

/**< Show present quantum number sectors and corresponding matrix for each physical index */
template<typename KT, typename VT>
void
MPSBlockMat<KT,VT>::print(const std::string& name) const
{
    if(name!="")cout<<name<<endl;
    for (uint i=0;i<this->size();++i)
    {
        cout<<i<<":"<<endl;
        this->at(i).print();
    }
}

/**< Show present quantum number sectors and corresponding matrix sizes only for each physical index */
template<typename KT, typename VT>
void
MPSBlockMat<KT,VT>::ShowDims(const std::string& name) const
{
    if(name!="")cout<<name<<endl;
    for (uint i=0;i<this->size();++i)
    {
        cout<<i<<":"<<endl;
        this->at(i).ShowDims();
    }
}


template<typename KT,typename VT>
inline
dim_map<KT>
MPSBlockMat<KT,VT>::GetMl() const
{
    dim_map<KT> diml;
    for (const auto& Ait : *this)
    {
        for (const auto& matit : Ait)
        {
            /// emplace returns pair<iterator,bool>, if bool=true, iterator points to the newly inserted element, if false it points to the existing element in its place

            #ifndef NDEBUG
            auto res = diml.emplace(Qin(matit),QMat(matit).n_rows);
            if ( !res.second ) assert(res.first->second == QMat(matit).n_rows && "GetUniformSizes(): encountered same symmetry sector with different ml") ;
            #else
            diml.emplace(Qin(matit),QMat(matit).n_rows);
            #endif // NDEBUG
//            auto res2 = dims.emplace(Qout(matit),QMat(matit).n_cols);
//            if ( !res2.second ) assert(res2.first->second == QMat(matit).n_cols && "GetUniformSizes(): encountered same symmetry sector with different mr") ;
        }
    }
    return diml;
}

template<typename KT,typename VT>
inline
dim_map<KT>
MPSBlockMat<KT,VT>::GetMr() const
{
    dim_map<KT> dimr;
    for (const auto& Ait : *this)
    {
        for (const auto& matit : Ait)
        {
            /// emplace return pair<iterator,bool>, if bool=true, iterator points to the newly inserted element, if false it points to the existing element in its place
//            auto res1 = diml.emplace(Qin(matit),QMat(matit).n_rows);
//            if ( !res1.second ) assert(res1.first->second == QMat(matit).n_rows && "GetUniformSizes(): encountered same symmetry sector with different ml") ;

            #ifndef NDEBUG
            auto res = dimr.emplace(Qout(matit),QMat(matit).n_cols);
            if ( !res.second ) assert(res.first->second == QMat(matit).n_cols && "GetMr(): encountered same symmetry sector with different mr") ;
            #else
            dimr.emplace(Qout(matit),QMat(matit).n_cols);
            #endif // NDEBUG
        }
    }
    return dimr;
}

/** \brief Returns all left and right keys and corresponding sizes for all block elements in the current MPS. Specifically, returns a vector of length d, containing vectors of length Ns of tuples (Qin,Qout,ml,mr). Here d is the physical dimension and Ns is the number of symmetry sectors for the current physical index.
 *
 * \return
 *
 */
template<typename KT,typename VT>
inline
std::vector<dimkeypair_vec<KT> >
MPSBlockMat<KT,VT>::GetSizesVector() const
{
    std::vector<dimkeypair_vec<KT> > dims(this->size());
    auto Ait = this->cbegin();
    auto dimit = dims.begin();
    for ( ; Ait!=this->end(); ++Ait, ++dimit)
    {
        dimit->reserve(Ait->size());
        for (const auto& matit : *Ait) dimit->emplace_back(std::make_tuple(Qin(matit),Qout(matit),QMat(matit).n_rows,QMat(matit).n_cols));
    }
    return dims;
}

//template<typename KT,typename VT>
//MPSBlockMat<KT,VT>::GetUniformSizes() const
//{
//    dim_map<KT> dims;
//    for (const auto& Ait : *this)
//    {
//        for (const auto& matit : Ait)
//        {
///// TODO (valentin#1#2015-05-12): consider using lower_bound to get rid of emplace()
//            auto res1 = dims.emplace(Qin(matit),QMat(matit).n_rows);
//            if ( !res1.second ) assert(res1.first->second == QMat(matit).n_rows && "GetUniformSizes(): encountered same symmetry sector with different ml") ;
//            auto res2 = dims.emplace(Qout(matit),QMat(matit).n_cols);
//            if ( !res2.second ) assert(res2.first->second == QMat(matit).n_cols && "GetUniformSizes(): encountered same symmetry sector with different mr") ;
//        }
//    }
//    return dims;
//}

template<typename KT,typename VT>
dim_map<KT>
MPSBlockMat<KT,VT>::GetUniformSizes() const
{
    dim_map<KT> dims;
    for (const auto& Ait : *this)
    {
        for (const auto& matit : Ait)
        {
//            cout<<"processing ("<<Qin(matit)<<","<<Qout(matit)<<"): "<<QMat(matit).n_rows<<" x "<<QMat(matit).n_cols<<endl;
            auto res1 = dims.emplace(Qin(matit),QMat(matit).n_rows); /// emplace returns pair<iter,bool>
            if ( !res1.second )
            {
                assert(res1.first->second == QMat(matit).n_rows && "GetUniformSizes(): encountered same symmetry sector with different ml") ;
            }
            auto res2 = dims.emplace(Qout(matit),QMat(matit).n_cols);
            if ( !res2.second )
            {
//                cout<<res2.first->first<<endl;
                assert(res2.first->second == QMat(matit).n_cols && "GetUniformSizes(): encountered same symmetry sector with different mr") ;
            }
        }
    }
    return dims;
}

template<typename KT,typename VT>
inline
uint
MPSBlockMat<KT,VT>::GetTotalMl() const
{
    uint dim=0;
    dim_map<KT> sizes;
    for (const auto& vit : *this) for(const auto& mit : vit) sizes.emplace(Qin(mit),QMat(mit).n_rows);
    for (const auto& sit : sizes) dim+= sit.second;
    return dim;
}

template<typename KT,typename VT>
inline
uint
MPSBlockMat<KT,VT>::GetTotalMr() const
{
    uint dim=0;
    dim_map<KT> sizes;
    for (const auto& vit : *this) for(const auto& mit : vit) sizes.emplace(Qout(mit),QMat(mit).n_cols);
    for (const auto& sit : sizes) dim+= sit.second;
    return dim;
}

template<typename KT,typename VT>
inline
uint
MPSBlockMat<KT,VT>::GetNElem() const
{
    uint n=0;
    for (const auto& vit : *this) for (const auto& matit : vit) n += QMat(matit).n_elem;
    return n;
}

//template<typename KT,typename VT>
//Col<VT>
//MPSBlockMat<KT,VT>::Vectorize() const
//{
//    Col<VT> out(this->GetNElem());
//    uint pos = 0,dpos = 0;
//    for (const auto& vit : *this)
//    {
//        for (const auto& matit : vit)
//        {
//            dpos = QMat(matit).n_rows * QMat(matit).n_cols;
//            out.subvec(pos,pos + dpos - 1) = VecType(QMat(matit).memptr(),dpos);
//            pos += dpos;
//        }
//    }
//    return out;
//}

/// supposedly faster vectorization
template<typename KT,typename VT>
Col<VT>
MPSBlockMat<KT,VT>::Vectorize() const
{
    Col<VT> out(this->GetNElem());
    VT* outmem = out.memptr();

    uint pos = 0,dpos = 0;
    for (const auto& vit : *this)
    {
        for (const auto& matit : vit)
        {
            dpos = QMat(matit).n_rows * QMat(matit).n_cols;
//            out.subvec(pos,pos + dpos - 1) = VecType(QMat(matit).memptr(),dpos);
            memcpy(&outmem[pos],QMat(matit).memptr(),dpos*sizeof(*outmem));
            pos += dpos;
        }
    }
    return out;
}

/// TODO (valentin#1#): implement version that takes already calculated map of singlesecdata
template<typename KT, typename VT>
std::map<KT, std::pair<Mat<VT>,typename MPSBlockMat<KT,VT>::singlesecdata> >
MPSBlockMat<KT,VT>::GetDenseMat(dirtype dir) const
{
    auto QNmap = this->GetSingleSecData(dir);
    std::map<KT, std::pair<Mat<VT>,typename MPSBlockMat<KT,VT>::singlesecdata> > DenseMat;

    uint s;

    if (dir==l)
    {
        for (const auto& qnit : QNmap)
        {
            /// try to avoid unnecessary copying by directly emplacing empty matrix and filling it later. We can also move the singlesecdata, as we don't need it in QNmap anymore
            /// DenseMat.first = current Qout
            /// DenseMat.second.first = actual dense matrix to be composed from this
            /// DenseMat.second.second = singlesecdata object to access sizes, spans and keys
            auto iter = DenseMat.emplace_hint(DenseMat.end(),qnit.first,std::make_pair(Mat<VT>(qnit.second.ml_tot,qnit.second.mr_tot,fill::zeros),std::move(qnit.second)));

            for (const auto& vit : iter->second.second.v_phys_ind)
            {
                s = vit.first;
                auto spanit = iter->second.second.sizes_spans.find(s);
                if (spanit != iter->second.second.sizes_spans.end()) iter->second.first(get<1>(spanit->second),span::all) = QMat(*vit.second);
                else
                {
                    cerr<<"MPSBlockMat<KT,VT>::GetDenseMat(dir): span for phys. index "<<s<<" not found!"<<endl;
                    abort();
                }
            }
        }
    }
    else if (dir==r)
    {
        for (const auto& qnit : QNmap)
        {
            /// try to avoid unnecessary copying by directly emplacing empty matrix and filling it later. We can also move the singlesecdata, as we don't need it in QNmap anymore
            /// DenseMat.first = current Qout
            /// DenseMat.second.first = actual dense matrix to be composed from this
            /// DenseMat.second.second = singlesecdata object to access sizes, spans and keys
            auto iter = DenseMat.emplace_hint(DenseMat.end(),qnit.first,std::make_pair(Mat<VT>(qnit.second.ml_tot,qnit.second.mr_tot,fill::zeros),std::move(qnit.second)));

            for (const auto& vit : iter->second.second.v_phys_ind)
            {
                s = vit.first;
                auto spanit = iter->second.second.sizes_spans.find(s);
                if (spanit != iter->second.second.sizes_spans.end()) iter->second.first(span::all,get<1>(spanit->second)) = QMat(*vit.second);
                else
                {
                    cerr<<"MPSBlockMat<KT,VT>::GetDenseMat(dir): span for phys. index "<<s<<" not found!"<<endl;
                    abort();
                }
            }
        }
    }
    else {cerr<<"MPSBlockMat<KT,VT>::GetDenseMat(): wrong direction specified!"<<endl; abort();}

    return DenseMat;
}


/// TODO (valentin#1#): implement version that takes already calculated map of singlesecdata
template<typename KT, typename VT>
template<uint NL>
std::map<KT, std::pair<Mat<VT>,typename MPSBlockMat<KT,VT>::doublesecdata> >
MPSBlockMat<KT,VT>::GetDenseSplitMat(const ItoKey<NL,KT>& I2K) const
{
    std::map<KT, std::pair<Mat<VT>,typename MPSBlockMat<KT,VT>::doublesecdata> > DenseMat;
    auto QNmap = this->GetDoubleSecData(I2K);

    uint left, right;
    /// key of this map will be the quantum number in the middel (i.e. where we want to perform the split or whatever)
    for (const auto& qnit : QNmap)
    {
        /// try to avoid unnecessary copying by directly emplacing empty matrix and filling it later. We can also move the doublesecdata, as we don't need it in QNmap anymore
        /// DenseMat.first = current center QN
        /// DenseMat.second.first = actual dense matrix to be composed from this
        /// DenseMat.second.second = doublesecdata object to access sizes, spans and keys
        auto iter = DenseMat.emplace_hint(DenseMat.end(),qnit.first,std::make_pair(Mat<VT>(qnit.second.ml_tot, qnit.second.mr_tot,fill::zeros),std::move(qnit.second)));

        const doublesecdata& sec(iter->second.second);
        for (const auto& vit : sec.v_phys_ind)
        {
            /// vit = tuple[left phys. index, right phys. index, corresponding block matrix]
            left=get<0>(vit);
            right=get<1>(vit);

            /// here we need to access the spans associatively, hence the maps
            auto lspan = sec.sizes_spans_left.find(left);
            auto rspan = sec.sizes_spans_right.find(right);
            if (lspan != sec.sizes_spans_left.end() && rspan != sec.sizes_spans_right.end()) iter->second.first(get<1>(lspan->second),get<1>(rspan->second)) = QMat(*get<2>(vit));
            else
            {
                cerr<<"MPSBlockMat<KT,VT>::GetDenseMat(dir): spans for phys. indices ("<<left<<","<<right<<") not found!"<<endl;
                abort();
            }
        }
    }
    return DenseMat;
}


template<typename KT, typename VT>
template<uint NL>
std::map<KT, typename MPSBlockMat<KT,VT>::doublesecdata>
MPSBlockMat<KT,VT>::GetDoubleSecData(const ItoKey<NL,KT>& I2K) const
{
    assert(this->NSites_ > NL && "MPS needs to span more than NL sites");
    assert(this->d_ == I2K.GetLocalDim() && "MPS and I2K need to have the same physical dimension");

    std::map<KT,typename MPSBlockMat<KT,VT>::doublesecdata> QNmap;
    uint left,right;
    uint dimright = pow(this->d_,this->NSites_-NL);

    typename MPSBlockMat<KT,VT>::mciter iter;

    /// determine all possible QN (also possibly newly generated ones) that can occur at the desired bond
    for (uint s=0; s<this->size(); ++s)
    {
        /// s = left*dimright + right
        left=s/dimright;
        right=s%dimright;

        for (iter=this->at(s).begin(); iter!=this->at(s).end(); ++iter) /// use C++99 iteration for having iterators which we can store in a map
        {
            /// if current center QN (iter->first + left) is not yet present in QNmap, generate corresponding doublesecdata object with std.c'tor
            doublesecdata& sec(QNmap[iter->first+I2K[left]]);

            /// for current center QN, collect all contributing elements and their sizes and (left or right) QNs
            sec.v_phys_ind.emplace_back(left,right,iter);
            sec.sizes_spans_left.emplace(left,make_tuple(QMat(*iter).n_rows,span(),&Qin(*iter)));
            sec.sizes_spans_right.emplace(right,make_tuple(QMat(*iter).n_cols,span(),&Qout(*iter)));
        }
    }
    uint ml_tot,mr_tot,currsize;

    /// no const as we are modifying the entries by adding the correct spans
    for (auto& qnit : QNmap)
    {
        doublesecdata& sec(qnit.second);
        ml_tot=0;
        mr_tot=0;

        /// determine spans for left and right indices
        /// we can only do that now, since before we didn't know yet, which phys. indices actually contribute
        /// This way we can avoid decomposing matrices that are actually mostly filled with zeros (for the phys. indices that don't contribute)
        for (auto& mlit : sec.sizes_spans_left)
        {
            /// mlit.first = phys. ind. s
            /// mlit.second = tuple[left size for s, span in dense mat for that s (to be determined later), pointer to ingoing QN]
            currsize = get<0>(mlit.second);
            get<1>(mlit.second) = span(ml_tot,ml_tot + currsize - 1);
            ml_tot += currsize;
        }
        for (auto& mrit : sec.sizes_spans_right)
        {
            /// mlit.first = phys. ind. s
            /// mlit.second = tuple[left size for s, span in dense mat for that s (to be determined later), pointer to outgoing QN]
            currsize = get<0>(mrit.second);
            get<1>(mrit.second) = span(mr_tot,mr_tot + currsize - 1);
            mr_tot += currsize;
        }
        sec.ml_tot = ml_tot;
        sec.mr_tot = mr_tot;
    }
    return QNmap;
}

template<typename KT, typename VT>
std::map<KT,typename MPSBlockMat<KT,VT>::singlesecdata>
MPSBlockMat<KT,VT>::GetSingleSecData(dirtype dir) const
{
    std::map<KT,typename MPSBlockMat<KT,VT>::singlesecdata> QNmap;
    typename MPSBlockMat<KT,VT>::mciter iter;
    uint m_tot, currsize;

    /// dense left representation A_(s,a)(b)
    if (dir == l)
    {
        for (uint s=0; s<this->d_; ++s)
        {
            for (iter = this->at(s).begin(); iter!=this->at(s).end(); ++iter) /// use C++99 iteration for having map-iterators which we can store in another map
            {
                /// if current right QN (iter->first + s) is not yet present in QNmap, generate corresponding sectordata object with std.c'tor
                typename MPSBlockMat<KT,VT>::singlesecdata& sec(QNmap[Qout(*iter)]); /// if current right QN not yet present in QNmaps, create new empty secdata

                /// for current right QN, collect all contributing elements, their sizes and right QNs (spans in dense matrix rep will be determined later, for now put empty span)
                sec.mr_tot = QMat(*iter).n_cols;
                /// store pair of contrib. phys. index and iterator to the corresponding block
                sec.v_phys_ind.emplace_back(s,iter); /// C++11: use emplace_back instead of push_back
                /// store for each contrib. phys. index the left size and (a pointer to) the actual ingoing QN.
                sec.sizes_spans.emplace(s,make_tuple(QMat(*iter).n_rows,span(),&Qin(*iter)));
            }
        }

        /// no const as we are modifying the entries by adding the correct spans
        for (auto& qnit : QNmap)
        {
            typename MPSBlockMat<KT,VT>::singlesecdata& sec(qnit.second); /// just as a placeholder name

            m_tot = 0;
            for (auto& mlit : sec.sizes_spans)
            {
                /// mlit.first = phys. ind. s
                /// mlit.second = pair(left size for s and that QN, pointer to that ingoing QN)
                currsize = get<0>(mlit.second);
                get<1>(mlit.second) = span(m_tot,m_tot+currsize-1);
                m_tot += currsize;
            }
            sec.ml_tot = m_tot;
        }
    }

    /// dense right representation A_(a)(s,b)
    else if (dir == r)
    {
        for (uint s=0; s<this->d_; ++s)
        {
            for (iter = this->at(s).begin(); iter!=this->at(s).end(); ++iter) /// use C++99 iteration for having map-iterators which we can store in another map
            {
                typename MPSBlockMat<KT,VT>::singlesecdata& sec(QNmap[Qin(*iter)]); /// if current right QN not yet present in QNmaps, create new empty secdata

                /// for current left QN, collect all contributing elements, their sizes and right QNs (spans in dense matrix rep will be determined later, for now put empty span)
                sec.ml_tot = QMat(*iter).n_rows;
                /// store pair of contrib. phys. index and iterator to the corresponding block
                sec.v_phys_ind.emplace_back(s,iter); /// C++11: use emplace_back instead of push_back
                /// store for each contrib. phys. index the right size and (a pointer to) the actual outgoing QN.
                sec.sizes_spans.emplace(s,make_tuple(QMat(*iter).n_cols,span(),&Qout(*iter)));
            }
        }

        /// no const as we are modifying the entries by adding the correct spans
        for (auto& qnit : QNmap)
        {
            typename MPSBlockMat<KT,VT>::singlesecdata& sec(qnit.second); /// just as a placeholder name

            m_tot = 0;
            for (auto& mlit : sec.sizes_spans)
            {
                /// mlit.first = phys. ind. s
                /// mlit.second = pair(left size for s and that QN, pointer to that ingoing QN)
                currsize = get<0>(mlit.second);
                get<1>(mlit.second) = span(m_tot,m_tot+currsize-1);
                m_tot += currsize;
            }
            sec.mr_tot = m_tot;
        }
    }
    else {cerr<<"MPSBlockMat<KT,VT>::GetSingleSecData(): wrong direction specified!"<<endl;abort();}

    return QNmap;
}



template<typename KT,typename VT>
inline
MPSBlockMat<KT,VT>
MPSBlockMat<KT,VT>::t() const
{
    MPSBlockMat<KT,VT> out(this->d_,this->NSites_);
    for (uint s=0;s<this->size();++s) out[s] = this->at(s).t();
    return out;
}

template<typename KT,typename VT>
inline
MPSBlockMat<KT,VT>
MPSBlockMat<KT,VT>::st() const
{
    MPSBlockMat<KT,VT> out(this->d_,this->NSites_);
    for (uint s=0;s<this->size();++s) out[s] = this->at(s).st();
    return out;
}


//template<typename KT,typename VT>
//inline
//MPSBlockMat<KT,VT>
//MPSBlockMat<KT,VT>::ShiftQN(const KT& K) const
//{
//
//}

template<typename KT,typename VT>
inline
MPSBlockMat<KT,VT>
MPSBlockMat<KT,VT>::ShiftBothQN(const KT& KL, const KT& KR) const
{
    MPSBlockMat out(this->d_,this->NSites_);
    for (uint s=0;s<this->size();++s) out[s] = this->at(s).ShiftBothQN(KL,KR);
    return out;
};

template<typename KT,typename VT>
template<uint N>
MPSBlockMat<KT,VT>
MPSBlockMat<KT,VT>::FlipQN(const ItoKey<N,KT>& I2K, const std::vector<bool>& which) const
{
    assert(this->NSites_ == N && "MPS and I2K need to span the same amount of sites");

    MPSBlockMat<KT,VT> out(this->d_,this->NSites_);

    /// put target I2K into map<Key,index>, s.t. we can lookup the index for the target MPS after flipping the QN
    std::map<KT,uint> order;
    for (uint i=0;i<I2K.size();++i) order.emplace_hint(order.end(),std::make_pair(I2K[i],i));

    for (uint i=0;i<this->size();++i)
    {
        KT IKflip = FlipK(this->at(i).dK(),which); /// here we are sort of recalculating the I2K of the current MPS

        auto it = order.find(IKflip);
        if (it != order.end()) out[it->second] = this->at(i).FlipQN(which);
        else throw std::domain_error("MPSBlockMat::FlipQN: Key "+std::to_string(IKflip)+" not found in target I2K");
    }

    return out;
}


template<typename KT,typename VT>
template<uint N>
MPSBlockMat<KT,VT>
MPSBlockMat<KT,VT>::PermuteQN(const ItoKey<N,KT>& I2K, const std::vector<uint>& perm) const
{
    assert(this->NSites_ == N && "MPS and I2K need to span the same amount of sites");

    MPSBlockMat<KT,VT> out(this->d_,this->NSites_);

    /// put target I2K into map<Key,index>, s.t. we can lookup the index for the target MPS after permuting the QN
    std::map<KT,uint> order;
    for (uint i=0;i<I2K.size();++i) order.emplace_hint(order.end(),std::make_pair(I2K[i],i));

    for (uint i=0;i<this->size();++i)
    {
        KT IKperm = PermuteK(this->at(i).dK(),perm); /// here we are sort of recalculating the I2K of the current MPS

        auto it = order.find(IKperm);
        if (it != order.end()) out[it->second] = this->at(i).PermuteQN(perm);
        else throw std::domain_error("MPSBlockMat::PermuteQN: Key "+std::to_string(IKperm)+" not found in target I2K");
    }

    return out;
}

template<typename KT,typename VT>
inline
MPSBlockMat<KT,VT>&
MPSBlockMat<KT,VT>::operator+=(const MPSBlockMat<KT,VT>& other)
{
    assert(this->d_==other.GetLocalDim() && "MPSBlockMat::operator+= : this and other must have the same physical dimension");
    assert(this->NSites_==other.GetNSites() && "MPSBlockMat::operator+= : this and other must span the same number of sites");
    for (uint s=0;s<other.size();++s) this->at(s) += other[s];
    return *this;
}

template<typename KT,typename VT>
inline
MPSBlockMat<KT,VT>&
MPSBlockMat<KT,VT>::operator-=(const MPSBlockMat<KT,VT>& other)
{
    assert(this->d_==other.GetLocalDim() && "MPSBlockMat::operator+= : this and other must have the same physical dimension");
    assert(this->NSites_==other.GetNSites() && "MPSBlockMat::operator+= : this and other must span the same number of sites");
    for (uint s=0;s<other.size();++s) this->at(s) -= other[s];
    return *this;
}

///**< multiplication by BlockDiagMat from the left */
//template<typename KT,typename VT>
//inline
//MPSBlockMat<KT,VT>&
//MPSBlockMat<KT,VT>::MultMatLeft(const BlockDiagMat<KT,VT>& mat)
//{
////    MPSBlockMat<KT,VT> MPSout(MPSin.GetLocalDim(),MPSin.GetNSites());
///// TODO (valentin#1#2015-04-19): switch to iterators
//    for (uint i=0;i<MPSin.size();++i) MPSout[i] = mat*MPSin[i];
//    for (auto& it : this) it
////    typename MPSBlockMat<KT,VT>::const_iterator init;
////    typename MPSBlockMat<KT,VT>::iterator outit = MPSout.begin();
////    for (init=MPSin.begin(); init!=MPSin.end(); init++,outit++) *outit = mat*(*init);
//
//    return *this;
//}
/**< MPSBLOCKMAT BASIC UTILITIES ***********************************************************************************************/

/**< add or subtract two MPS */

/// template for same type additions
template<typename KT,typename VT>
inline
MPSBlockMat<KT,VT>
operator+(MPSBlockMat<KT,VT> lhs, const MPSBlockMat<KT,VT>& rhs) {return lhs+=rhs;}

/// general template for mixed type additions
/// we cannot pass lhs by copy here, as we don't know if lhs will be promoted or not
template<typename KT,typename VTL, typename VTR>
inline
MPSBlockMat<KT,typename promote_type<VTL,VTR>::result>
operator+(const MPSBlockMat<KT,VTL>& lhs, const MPSBlockMat<KT,VTR>& rhs);
//{
//    MPSBlockMat<KT,typename promote_type<VTL,VTR>::result> out(lhs);
//    return out+=rhs;
//}

/// specialization for mixed complex/real additions
/// we cannot pass lhs by copy here, as we don't know if lhs will be promoted or not
template<typename KT>
inline
MPSBlockMat<KT,Complex>
operator+(const MPSBlockMat<KT,Complex>& lhs, const MPSBlockMat<KT,Real>& rhs)
{
    MPSBlockMat<KT,Complex> out(rhs);
    return out+=lhs;
}

/// specialization for mixed complex/real additions
/// we cannot pass lhs by copy here, as we don't know if lhs will be promoted or not
template<typename KT>
inline
MPSBlockMat<KT,Complex>
operator+(const MPSBlockMat<KT,Real>& lhs, const MPSBlockMat<KT,Complex>& rhs)
{
    MPSBlockMat<KT,Complex> out(lhs);
    return out+=rhs;
}

/// template for same type subtractions
template<typename KT,typename VT>
inline
MPSBlockMat<KT,VT>
operator-(MPSBlockMat<KT,VT> lhs, const MPSBlockMat<KT,VT>& rhs) {return lhs-=rhs;}

/// general template for mixed type subtractions
/// we cannot pass lhs by copy here, as we don't know if lhs will be promoted or not
template<typename KT,typename VTL, typename VTR>
inline
MPSBlockMat<KT,typename promote_type<VTL,VTR>::result>
operator-(const MPSBlockMat<KT,VTL>& lhs, const MPSBlockMat<KT,VTR>& rhs);
//{
//    MPSBlockMat<KT,typename promote_type<VTL,VTR>::result> out(lhs);
//    return out-=rhs;
//}

/// specialization for mixed complex/real subtractions
/// we cannot pass lhs by copy here, as we don't know if lhs will be promoted or not
template<typename KT>
inline
MPSBlockMat<KT,Complex>
operator-(const MPSBlockMat<KT,Complex>& lhs, const MPSBlockMat<KT,Real>& rhs)
{
    MPSBlockMat<KT,Complex> out(lhs);
    return out-=MPSBlockMat<KT,Complex>(rhs);
}

/// specialization for mixed complex/real subtractions
/// we cannot pass lhs by copy here, as we don't know if lhs will be promoted or not
template<typename KT>
inline
MPSBlockMat<KT,Complex>
operator-(const MPSBlockMat<KT,Real>& lhs, const MPSBlockMat<KT,Complex>& rhs)
{
    MPSBlockMat<KT,Complex> out(lhs);
    return out-=rhs;
}

/**< concatenate two MPS matrices ************************************************************************************************/
template<typename KT,typename VTL,typename VTR>
inline
MPSBlockMat<KT,typename promote_type<VTL,VTR>::result>
operator*(const MPSBlockMat<KT,VTL>& lhs,const MPSBlockMat<KT,VTR>& rhs)
{
    assert(lhs.GetLocalDim()==rhs.GetLocalDim() && "MPSBlockMat<KT,VT> operator*(): lhs and rhs have different LocalDim");
    MPSBlockMat<KT,typename promote_type<VTL,VTR>::result> out(lhs.GetLocalDim(),lhs.GetNSites() + rhs.GetNSites());
/// TODO (valentin#1#2015-04-20): switch to iterators
    for (uint i=0;i<lhs.size();++i)
    {
        for (uint j=0;j<rhs.size();++j) out[i*rhs.size() + j]=lhs[i]*rhs[j];
    }
    return out;
}

/**< multiplication by Scalar from the left ********************************************************************************/

/// same type multiplication
template<typename KT, typename VT>
inline
MPSBlockMat<KT,VT>
operator*(VT x, MPSBlockMat<KT,VT> MPS) {return MPS*=x;};

/// general template for mixed type multiplication
/// we cannot pass MPS by copy here, as we don't know if MPS will be promoted or not
template<typename KT,typename VT, typename VTMPS>
inline
MPSBlockMat<KT,typename promote_type<VT,VTMPS>::result>
operator*(VT x, const MPSBlockMat<KT,VTMPS>& MPS);


/// specialization for mixed complex/real multiplication
/// we cannot pass MPS by copy here, as we don't know if MPS will be promoted or not
template<typename KT>
inline
MPSBlockMat<KT,Complex>
operator*(Real x, const MPSBlockMat<KT,Complex>& MPS)
{
    MPSBlockMat<KT,Complex> out(MPS);
    return out*=Complex(x,0.);
}

/// specialization for mixed complex/real multiplication
/// we cannot pass MPS by copy here, as we don't know if MPS will be promoted or not
template<typename KT>
inline
MPSBlockMat<KT,Complex>
operator*(Complex x, const MPSBlockMat<KT,Real>& MPS)
{
    MPSBlockMat<KT,Complex> out(MPS);
    return out*=x;
}

/**< multiplication by Scalar from the right ********************************************************************************/

/// same type multiplication
template<typename KT, typename VT>
inline
MPSBlockMat<KT,VT>
operator*(MPSBlockMat<KT,VT> MPS, VT x) {return MPS*=x;};

//template<typename KT, typename VTMPS, typename VT>

/**< multiplication by BlockMat from the left ********************************************************************************/
template<typename KT,typename VTL,typename VTR>
inline
MPSBlockMat<KT,typename promote_type<VTL,VTR>::result>
operator*(const BlockMat<KT,VTL>& mat, const MPSBlockMat<KT,VTR>& MPSin)
{
    MPSBlockMat<KT,typename promote_type<VTL,VTR>::result> MPSout(MPSin.GetLocalDim(),MPSin.GetNSites());
/// TODO (valentin#1#2015-04-19): switch to iterators
    for (uint i=0;i<MPSin.size();++i) MPSout[i] = mat*MPSin[i];
    return MPSout;
}

/**< multiplication by BlockMat from the right ********************************************************************************/
template<typename KT,typename VTL,typename VTR>
inline
MPSBlockMat<KT,typename promote_type<VTL,VTR>::result>
operator*(const MPSBlockMat<KT,VTL>& MPSin, const BlockMat<KT,VTR>& mat)
{
    MPSBlockMat<KT,typename promote_type<VTL,VTR>::result> MPSout(MPSin.GetLocalDim(),MPSin.GetNSites());
/// TODO (valentin#1#2015-04-19): switch to iterators
    for (uint i=0;i<MPSin.size();++i) MPSout[i] = MPSin[i]*mat;
    return MPSout;
}

/**< multiplication by BlockDiagMat from the left ********************************************************************************/
template<typename KT,typename VTL,typename VTR>
inline
MPSBlockMat<KT,typename promote_type<VTL,VTR>::result>
operator*(const BlockDiagMat<KT,VTL>& mat, const MPSBlockMat<KT,VTR>& MPSin)
{
    MPSBlockMat<KT,typename promote_type<VTL,VTR>::result> MPSout(MPSin.GetLocalDim(),MPSin.GetNSites());
/// TODO (valentin#1#2015-04-19): switch to iterators
    for (uint i=0;i<MPSin.size();++i) MPSout[i] = mat*MPSin[i];
    return MPSout;
}

/**< multiplication by BlockDiagMat from the right ********************************************************************************/
template<typename KT,typename VTL,typename VTR>
inline
MPSBlockMat<KT,typename promote_type<VTL,VTR>::result>
operator*(const MPSBlockMat<KT,VTL>& MPSin, const BlockDiagMat<KT,VTR>& mat)
{
    MPSBlockMat<KT,typename promote_type<VTL,VTR>::result> MPSout(MPSin.GetLocalDim(),MPSin.GetNSites());
/// TODO (valentin#1#2015-04-19): switch to iterators
    for (uint i=0;i<MPSin.size();++i) MPSout[i] = MPSin[i]*mat;
    return MPSout;
}

/**< Frobenius norm of MPS Matrices */
template<typename KT,typename VT>
inline
Real
norm(const MPSBlockMat<KT,VT>& MPS)
{
    Real nrm2 = 0.;
    for (const auto& it : MPS) nrm2 += pow(norm(it),2);
    return sqrt(nrm2);
}


template<typename KT,typename VT>
inline
Real
norm_inf(const MPSBlockMat<KT,VT>& MPS)
{
    Real nrmi = 0.;
    for (const auto& it : MPS) nrmi = max(nrmi,norm_inf(it));
    return nrmi;
}

///**< DISK IO ******************************************************************************************/
template<typename KT,typename VT>
bool
MPSBlockMat<KT,VT>::save(std::string name) const
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay = save(file);
    if (!save_okay)
    {
        cerr << "MPSBlockMat<KT,VT>::save(): could not save "<<name<<endl;
    }
    file.close();

    return save_okay;
}

template<typename KT,typename VT>
bool
MPSBlockMat<KT,VT>::save(std::ofstream& file) const
{
    bool save_okay = file.good();
    if (!save_okay)
    {
        cerr << "MPSBlockMat<KT,VT>::save(): bad file handle"<<endl;
        return false;
    }
    uint d(this->d_);
    uint N(this->NSites_);

//    file << "BLOCKMPS" << '\n';
    file << GetHeader(this) << '\n';

    file.write(reinterpret_cast<const char*>(&d), std::streamsize(sizeof(uint)));
    file.write(reinterpret_cast<const char*>(&N), std::streamsize(sizeof(uint)));

    for (const auto& it : *this) it.save(file);
    return save_okay;
}

template<typename KT,typename VT>
bool
MPSBlockMat<KT,VT>::load(std::string name)
{
    std::ifstream file(name.c_str(), std::ifstream::binary);
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"could not open "<<name<<endl;
        return false;
    }

    load_okay = this->load(file);
    if(!load_okay)
    {
        cerr<<"MPSBlockMat<KT,VT>::load(): could not load "<<name<<endl;
    }
    file.close();

    return load_okay;
}

template<typename KT, typename VT>
template<typename GO>
bool
MPSBlockMat<KT,VT>::load(std::ifstream& file, const GO& GroupObj)
{
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"MPSBlockMat<KT,VT>::load(): bad file handle"<<endl;
        return false;
    }
    this->clear();

    std::string header;
    file >> header;
    file.get(); /// somehow we need this to get over the newline stored to terminate header. WTF it took me 1 hr to find that out!!2
//    if (header != "BLOCKMPS")
    if (header != GetHeader(this))
    {
        cerr << "MPSBlockMat<KT,VT>::load(): wrong header "<<header<<", should be "<<GetHeader(this)<<endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&d_), std::streamsize(sizeof(uint)));
    file.read(reinterpret_cast<char*>(&NSites_), std::streamsize(sizeof(uint)));
    this->resize(pow(d_,NSites_));

    for (uint s=0; s<this->size(); ++s)
    {
        if(!this->at(s).load(file,GroupObj))
        {
            cerr << "MPSBlockMat<KT,VT>::load(): failed to load phys index "<<s<<endl;
            load_okay = false;
            break;
        }
    }
    return load_okay;
}

/**< STANDALONE FUNCTIONS TO SAVE AND LOAD **************************************************************************************/


//template<typename KT,typename VT>
//MPSBlockMat<KT,VT>
//load(std::ifstream& file)
//{
//    bool load_okay = file.good();
//    if (!load_okay)
//    {
//        cerr<<"MPSBlockMat<KT,VT>::load(): bad file handle"<<endl;
//        abort();
//    }
//
//    std::string header;
//    file >> header;
//    file.get(); /// somehow we need this to get over the newline stored to terminate header. WTF it took me 1 hr to find that out!!
//    if (header != "BLOCKMPS")
//    {
//        cerr << "MPSBlockMat<KT,VT>::load(): wrong header "<<header<<endl;
//        return false;
//    }
//    uint d=0,N=0;
//    file.read(reinterpret_cast<char*>(&d), std::streamsize(sizeof(uint)));
//    file.read(reinterpret_cast<char*>(&N), std::streamsize(sizeof(uint)));
//    MPSBlockMat<KT,VT> out(d,N);
//
//    for (uint s=0; s<out.size(); ++s)
//    {
//        if(!out[s].load(file))
//        {
//            cerr << "MPSBlockMat<KT,VT>::load(): phys index "<<s<<endl;
//            abort();
//        }
//    }
//    return out;
//}

//template<typename KT,typename VT>
//MPSBlockMat<KT,VT>
//load(std::string name)
//{
//    std::ifstream file(name.c_str(), std::fstream::binary);
//    bool load_okay = file.good();
//    if (!load_okay)
//    {
//        cerr<<"could not open "<<name<<endl;
//        abort();
//    }
//    MPSBlockMat<KT,VT> out = load<KT,VT>(file);
//    file.close();
//    return out;
//}
/**< ARRAY CONTAINERS FOR MULTI SITE PURPOSES ***********************************************************************************/
//template<typename KT, typename VT>
//using MPSArray = std::deque<MPSBlockMat<KT,VT> >;
//
//template<typename KT, typename VT>
//using BlockDiagarray = std::deque<BlockDiagMat<KT,VT> >;

template<typename KT, typename VT>
class MPSBlockMatArray : public std::deque<MPSBlockMat<KT,VT> >
{
public:
    typedef KT key_type;
    typedef VT scalar_type;
    typedef std::function<MPSBlockMat<KT,VT> (const MPSBlockMat<KT,VT>&)> fun_type;

    MPSBlockMatArray() = default;
    MPSBlockMatArray(const MPSBlockMatArray&) = default;
//    MPSBlockMatArray(const MPSBlockMatArray&&) = default;

    template<typename VTO>
    MPSBlockMatArray(const MPSBlockMatArray<KT,VTO>& other) {for (const auto& it : other) this->emplace_back(MPSBlockMat<KT,VT>(it));};

    inline void print(const std::string str="") const;
    inline void ShowDims(const std::string str="") const;

    inline dim_map<KT> GetMl() const {return this->front().GetMl();};
    inline dim_map<KT> GetMr() const {return this->back().GetMr();};
//    dim_map<KT>
    inline MPSBlockMatArray operator-() const
    {
        MPSBlockMatArray out;
        for (const auto& it : *this) out.emplace_back(-it);
        return out;
    };

    template<uint N>
    MPSBlockMatArray FlipQN(const ItoKey<N,KT>& I2K, const std::vector<bool>& which = std::vector<bool>()) const;

    /// general method to apply single MPSBlockMat -> MPSBlockMat functions to every single element of the MPSBlockMatArray
    MPSBlockMatArray ApplyFun(const std::function<MPSBlockMat<KT,VT> (const MPSBlockMat<KT,VT>&)>& F) const
    {
        MPSBlockMatArray out;
        for (const auto& it : *this) out.emplace_back(F(it));
        return out;
    };

    /// DiskIO
    bool save(std::string name) const;
    bool save(std::ofstream& file) const;
    bool load(std::string name);
    template<typename GO>
    bool load(std::ifstream& file, const GO& GroupObj);

};


template<typename KT, typename VT>
template<uint N>
MPSBlockMatArray<KT,VT>
MPSBlockMatArray<KT,VT>::FlipQN(const ItoKey<N,KT>& I2K, const std::vector<bool>& which) const
{
    MPSBlockMatArray<KT,VT> out;
    for (const auto& it : *this) out.emplace_back(it.FlipQN(I2K,which));
    return out;
}

template<typename KT, typename VT>
inline
void
MPSBlockMatArray<KT,VT>::print(const std::string str) const
{
    cout<<endl;
    if (str!="") cout<<str<<endl;
    uint ct=1;
    for (const auto& it : *this) it.print("site "+std::to_string(ct++));
}


template<typename KT, typename VT>
inline
void
MPSBlockMatArray<KT,VT>::ShowDims(const std::string str) const
{
    cout<<endl;
    if (str!="") cout<<str<<endl;
    uint ct=1;
    for (const auto& it : *this) it.ShowDims("site "+std::to_string(ct++));
}

template<typename KT, typename VT>
bool
MPSBlockMatArray<KT,VT>::save(std::string name) const
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay = this->save(file);
    if (!save_okay)
    {
        cerr << "MPSBlockMatArray<KT,VT>::save(): could not save "<<name<<endl;
    }
    file.close();

    return save_okay;
}

template<typename KT, typename VT>
bool
MPSBlockMatArray<KT,VT>::save(std::ofstream& file) const
{
    bool save_okay = file.good();
    if (!save_okay)
    {
        cerr << "MPSBlockMatArray<KT,VT>::save(): bad file handle"<<endl;
        return false;
    }
    uint Nsites = this->size();

//    file << "BLOCKMPSARRAY" << '\n';
    file << GetHeader(this) << '\n';
    file.write(reinterpret_cast<const char*>(&Nsites), std::streamsize(sizeof(uint)));

    for (const auto& it : *this) it.save(file);
    return save_okay;
}




//template<typename KT,typename VT>
//bool
//MPSArray<KT,VT>::load(std::string name)
//{
//    std::ifstream file(name.c_str(), std::fstream::binary);
//    bool load_okay = file.good();
//    if (!load_okay)
//    {
//        cerr<<"could not open "<<name<<endl;
//        return false;
//    }
//
//    load_okay = this->load(file);
//    if(!load_okay)
//    {
//        cerr<<"MPSArray<KT,VT>::load(): could not load "<<name<<endl;
//    }
//    file.close();
//
//    return load_okay;
//}


template<typename KT,typename VT>
template<typename GO>
bool
MPSBlockMatArray<KT,VT>::load(std::ifstream& file, const GO& GroupObj)
{
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"MPSBlockMatArray<KT,VT>::load(): bad file handle"<<endl;
        return false;
    }
    this->clear();

    std::string header;
    file >> header;
    file.get(); /// somehow we need this to get over the newline stored to terminate header. WTF it took me 1 hr to find that out!!
//    if (header != "BLOCKMPSARRAY")
    if (header != GetHeader(this))
    {
        cerr << "MPSBlockMatArray<KT,VT>::load(): wrong header "<<header<<", should be "<<GetHeader(this)<<endl;
        return false;
    }
    uint Nsites;
    file.read(reinterpret_cast<char*>(&Nsites), std::streamsize(sizeof(uint)));
    this->clear();

    for (uint n=0; n<Nsites; ++n)
    {
        /// We need to call ::load (using standalone version), otherwise the member function of MPSArray gets called (how stupid is that?)
        this->emplace_back(MPSBlockMat<KT,VT>(file,GroupObj));
    };
    return load_okay;
}

template<typename KT,typename VT>
ostream&
operator<<(ostream& os, const MPSBlockMatArray<KT,VT>& MPS)
{
    for (uint n=0;n<MPS.size();++n)
    {
        os<<"site "<<n+1<<endl<<endl;
        os<<MPS[n];
    }
    return os;
}

template<typename KT>
string
GetHeader(const MPSBlockMat<KT,Real>*) {return "RBLOCKMPS";}

template<typename KT>
string
GetHeader(const MPSBlockMat<KT,Complex>*) {return "CBLOCKMPS";}

template<typename KT>
string
GetHeader(const MPSBlockMatArray<KT,Real>*) {return "RBLOCKMPSARRAY";}

template<typename KT>
string
GetHeader(const MPSBlockMatArray<KT,Complex>*) {return "CBLOCKMPSARRAY";}

#endif // MPS_BLOCK_MAT_H
