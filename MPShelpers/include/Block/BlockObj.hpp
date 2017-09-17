#ifndef BLOCK_OBJ_H
#define BLOCK_OBJ_H

//#include <map>
//#include <fstream>
//#include <deque>
//#include <string>
//#include "helpers.hpp"
//#include "IKey.hpp"
//#include "KeyTypes.hpp"

using std::string;
using std::ostream;
using std::endl;
using std::cout;
using std::get;


/**< BLOCK OBJECTS AS BUILDING BLOCKS FOR MPS  */

/// fwd declarations
template<typename KT>
class BlockLam;

template<typename KT,typename VT>
class BlockMat;

template<typename KT,typename VT>
class BlockDiagMat;

template<typename KT>
class BlockLamArray;

template<typename KT,typename VT>
class BlockMatArray;

template<typename KT,typename VT>
class BlockDiagMatArray;

/**< Block Schmidt Value Type *****************************************************************************************
defined as a map of Real valued vectors (RVecType), where the quantum number(s) of the blocks are the keys for the map
*/
template<typename KT>
class BlockLam : public std::map<KT,RVecType>
{
public:
    typedef KT key_type;
    typedef BlockLamArray<KT> array_type;

    void ShowDims(const std::string& name="") const;
    void ShowDimsMins(const std::string& name="") const;
    void print(const std::string& name="") const;
    void print_sorted(const std::string& name="") const;

    inline dim_map<KT> GetSizes() const;
    inline uint GetNElem() const;

    BlockLam& operator+=(const BlockLam<KT>& other);
    BlockLam& operator-=(const BlockLam<KT>& other);
    inline BlockLam& operator*=(Real val);
    inline BlockLam& operator/=(Real val);

    bool save(std::string name) const;
    bool save(std::ofstream& file) const;
    bool load(std::string name);
    template<typename GO>
    bool load(std::ifstream& file, const GO& GroupObj);
};

template<typename KT>
inline
dim_map<KT>
BlockLam<KT>::GetSizes() const
{
    dim_map<KT> dims;
    for (const auto& it : *this) dims.emplace_hint(dims.end(),it.first,it.second.n_elem);
    return dims;
}

template<typename KT>
inline
uint
BlockLam<KT>::GetNElem() const
{
    uint dim=0;
    for (const auto it : *this) dim+=it.second.n_elem;
    return dim;
}

template<typename KT>
void
BlockLam<KT>::ShowDimsMins(const std::string& name) const
{
    if (name!="")cout<<name<<endl;
    if (!this->empty())
    {
        for (const auto& it : *this) cout<<it.first<<": "<<it.second.size()<<", min. = "<<min(it.second)<<endl;
    }
    else cout<<"---"<<endl;
}

template<typename KT>
void
BlockLam<KT>::ShowDims(const std::string& name) const
{
    if (name!="")cout<<name<<endl;
    if (!this->empty())
    {
        for (const auto& it : *this) cout<<it.first<<": "<<it.second.size()<<endl;
    }
    else cout<<"---"<<endl;
}

template<typename KT>
void
BlockLam<KT>::print_sorted(const std::string& name) const
{
    if (name!="")cout<<name<<endl;
    std::vector<std::pair<const KT*, Real> > lamvec;
    for (const auto& lamit : *this) for (const auto& valit : lamit.second) lamvec.emplace_back(std::make_pair(&lamit.first,valit));
    std::sort(lamvec.begin(),lamvec.end(),[](const std::pair<const KT*, Real>& lhs, const std::pair<const KT*, Real>& rhs) {return lhs.second > rhs.second;});
    for (const auto& lamit : lamvec) cout<<*lamit.first<<": "<<lamit.second<<endl;
}

template<typename KT>
void
BlockLam<KT>::print(const std::string& name) const
{
    if (name!="")cout<<name<<endl;
    cout<<*this<<endl;
}

template<typename KT>
bool
BlockLam<KT>::save(std::string name) const
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay = save(file);
    if (!save_okay)
    {
        cerr << "BlockLambda<KT,VT>::save(): could not save "<<name<<endl;
    }
    file.close();

    return save_okay;
}

template<typename KT>
bool
BlockLam<KT>::save(std::ofstream& file) const
{
    bool save_okay = file.good();
    if (!save_okay)
    {
        cerr << "BlockLambda<KT,VT>::save(): bad file handle"<<endl;
        return false;
    }
    uint Nsec=this->size();

//    file << "BLOCKLAMBDA" << '\n';
    file << GetHeader(this) << '\n';
    file.write(reinterpret_cast<const char*>(&Nsec), std::streamsize(sizeof(uint)));

    for (const auto& it : *this)
    {
//        file << it.first;
//        save_okay = it.first.save(file);

        if (!it.first.save(file))
        {
            cerr << "BlockLambda<KT,VT>::save(): saving of Q = "<<it.first<<" failed" << endl;
            save_okay = false;
            break;
        }

        if (!it.second.save(file,arma_binary))
        {
            cerr << "BlockLambda<KT,VT>::save(): saving of lam failed" << endl;
            save_okay = false;
            break;
        }
    }
    return save_okay;
}

template<typename KT>
bool
BlockLam<KT>::load(std::string name)
{
    std::ifstream file(name.c_str(), std::fstream::binary);
    if (!file.good())
    {
        cerr<<"BlockLambda<KT,VT>::load(): could not open "<<name<<endl;
        return false;
    }
    bool load_okay = this->load(file);
    if(!load_okay)
    {
        cerr<<"BlockLambda<KT,VT>::load(): could not load "<<name<<endl;
    }
    file.close();

    return load_okay;
}


template<typename KT>
template<typename GO>
bool
BlockLam<KT>::load(std::ifstream& file, const GO& GroupObj)
{
    this->clear();
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"BlockLambda<KT,VT>::load(): bad file handle"<<endl;
        return false;
    }

    RVecType tmp;
    std::string header;
    uint Nsec=0;

    file >> header;
    file.get(); /// somehow we need this to get over the newline stored to terminate header. WTF it took me 1 hr to find that out!!

//    if (header != "BLOCKLAMBDA")
    if (header != GetHeader(this))
    {
        cerr << "wrong header "<<header<<", should be "<<GetHeader(this)<<endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&Nsec), std::streamsize(sizeof(uint)));
    for (uint i=0;i<Nsec; ++i)
    {
        KT Kin(file,GroupObj);

        if (!tmp.load(file,arma_binary))
        {
            cerr<<"BlockLambda<KT,VT>::load(): unable to load sector "<<Kin<<endl;
            load_okay = false;
            break;
        }
        this->emplace_hint(this->end(),Kin,tmp);
    }
    return load_okay;
}

template<typename KT>
BlockLam<KT>&
BlockLam<KT>::operator+=(const BlockLam<KT>& other)
{

    for (const auto& oit : other) /// loop through symmetry sectors in other (it is important to loop through OTHER, s.t. no contributions are lost!!)
    {
        auto thisit = this->lower_bound(oit.first); /// look for current QN of other in this
        if (thisit != this->end() && thisit->first == oit.first) thisit->second += oit.second; /// if it is found, add other to this
        else this->emplace_hint(thisit,oit.first,oit.second); /// if it is not found, insert
    }
    return *this;
}

template<typename KT>
BlockLam<KT>&
BlockLam<KT>::operator-=(const BlockLam<KT>& other)
{

    for (const auto& oit : other) /// loop through symmetry sectors in other (it is important to loop through OTHER, s.t. no contributions are lost!!)
    {
        auto thisit = this->lower_bound(oit.first); /// look for current QN of other in this
        if (thisit != this->end() && thisit->first == oit.first) thisit->second -= oit.second; /// if it is found, add other to this
        else this->emplace_hint(thisit,oit.first,-oit.second); /// if it is not found, insert
    }
    return *this;
}

template<typename KT>
inline
BlockLam<KT>&
BlockLam<KT>::operator*=(Real val)
{
    for (auto& it : *this) it.second*=val;
    return *this;
}

template<typename KT>
inline
BlockLam<KT>&
BlockLam<KT>::operator/=(Real val)
{
    for (auto& it : *this) it.second/=val;
    return *this;
}

template<typename KT>
inline
BlockLam<KT>
operator+(BlockLam<KT> lhs, const BlockLam<KT>& rhs) {return lhs+=rhs;}

template<typename KT>
inline
BlockLam<KT>
operator-(BlockLam<KT> lhs, const BlockLam<KT>& rhs) {return lhs-=rhs;}


template<typename KT>
inline
BlockLam<KT>
operator*(Real val, const BlockLam<KT>& lamin)
{
    BlockLam<KT> lamout(lamin);
    return lamout*=val;
}

template<typename KT>
inline
BlockLam<KT>
operator*(const BlockLam<KT>& lamin, Real val)
{
    BlockLam<KT> lamout(lamin);
    return lamout*=val;
}


template<typename KT>
inline
BlockLam<KT>
operator/(const BlockLam<KT>& lamin, Real val)
{
    BlockLam<KT> lamout(lamin);
    return lamout/=val;
}


template<typename KT>
inline
Real
norm(const BlockLam<KT>& lam)
{
    #ifndef NDEBUG
    if (lam.empty()) cerr<<"norm: in is empty"<<endl;
    #endif // NDEBUG

    Real nrm2=0;
    for (const auto& it : lam) nrm2+=dot(it.second,it.second);
    return sqrt(nrm2);
}

template<typename KT>
inline
Real
norm_inf(const BlockLam<KT>& lam)
{
    #ifndef NDEBUG
    if (lam.empty()) cerr<<"norm: in is empty"<<endl;
    #endif // NDEBUG

    Real nrmi=0;
    for (const auto& it : lam) nrmi = max(nrmi,abs(it.second).max());
    return nrmi;
}

template<typename KT>
inline
BlockLam<KT>
sqrt(const BlockLam<KT>& in)
{
    BlockLam<KT> out;
    for (const auto& it : in) out.emplace_hint(out.end(),it.first,sqrt(it.second));
    return out;
}

template<typename KT,typename VT>
inline
BlockLam<KT>
pow(const BlockLam<KT>& in, VT expo)
{
    BlockLam<KT> out;
    for (const auto& it : in) out.emplace_hint(out.end(),it.first,pow(it.second,expo));
    return out;
}


template<typename KT>
ostream& operator<<(ostream& os, const BlockLam<KT>& lam)
{
    if (!lam.empty())
    {
        for (const auto& it : lam)
        {
            os<<it.first<<endl;
            os<<it.second;
        }
    }
    else os<<"---"<<endl;
    return os;
}

/// TODO (valentin#1#): How to define diagmat returning a real valued vector for complex matrices? Consider implementing a general BlockVector, where BlockLambda is a real-valued specialization

template<typename KT,typename VT>
inline
BlockLam<KT>
diagmat(const BlockDiagMat<KT,VT>& mat);

template<typename KT>
inline
BlockLam<KT>
diagmat(const BlockDiagMat<KT,Real>& mat)
{
    BlockLam<KT> lam;
    for (const auto& mit : mat) lam.emplace_hint(lam.end(),mit.first,mit.second.diag());
    return lam;
}

template<typename KT>
std::map<KT,Real>
min(const BlockLam<KT>& lam)
{
    std::map<KT,Real> minlam;
    for (const auto& it : lam) minlam.emplace_hint(minlam.end(),it.first,min(it.second));
    return minlam;
}

template<typename KT>
std::map<KT,Real>
max(const BlockLam<KT>& lam)
{
    std::map<KT,Real> maxlam;
    for (const auto& it : lam) maxlam.emplace_hint(maxlam.end(),it.first,max(it.second));
    return maxlam;
}

template<typename KT>
std::map<KT,bool>
operator>(const BlockLam<KT>& lam, Real val)
{
    std::map<KT,bool> res;
    for (const auto& it : lam) res.emplace_hint(res.end(),it.first,min(it.second)>val);
    return res;
}


template<typename KT>
std::map<KT,bool>
operator<(const BlockLam<KT>& lam, Real val)
{
    std::map<KT,bool> res;
    for (const auto& it : lam) res.emplace_hint(res.end(),it.first,max(it.second)<val);
    return res;
}


template<typename KT>
std::map<KT,bool>
operator<(Real val, const BlockLam<KT>& lam)
{
    std::map<KT,bool> res;
    for (const auto& it : lam) res.emplace_hint(res.end(),it.first,min(it.second)>val);
    return res;
}


template<typename KT>
std::map<KT,bool>
operator>(Real val, const BlockLam<KT>& lam)
{
    std::map<KT,bool> res;
    for (const auto& it : lam) res.emplace_hint(res.end(),it.first,max(it.second)<val);
    return res;
}


/** Block Matrix Type *************************************************************************************************************************************************************************
 *  Defined as a map of matrices of numerical type VT. Each block has an ingoing and outgoing (set of)
 *  quantum number(s) of type KT. The blocks are ordered according to the ingoing quantum number (serves
 *  as map key), the mapped type is then a pair consisting of the outgoing quantum number(s) and the matrix
 */
template<typename KT,typename VT>
using QMatPair = std::pair<KT,Mat<VT> >;

/**< ACCESS HELPERS FOR BlockMat -------------------- */
/**< if O is a Block in a BlockMat object, call like Qin(O), Qout(O), QMat(O) to get the respective quantum numbers and the entry */
template<typename KT,typename VT>
inline
const KT&
Qin(const std::pair<const KT,QMatPair<KT,VT> >& E) { return E.first;}

template<typename KT,typename VT>
inline
const KT&
Qout(const std::pair<const KT,QMatPair<KT,VT> >& E){ return E.second.first;}

template<typename KT,typename VT>
inline
Mat<VT>&
QMat(std::pair<const KT,std::pair<KT,Mat<VT> > >& E){ return E.second.second;}

template<typename KT,typename VT>
inline
const Mat<VT>&
QMat(const std::pair<const KT,std::pair<KT,Mat<VT> > >& E) { return E.second.second;}

template<typename KT,typename VT>
class BlockMat : public std::map<KT,QMatPair<KT,VT> >
{
public:
    typedef KT key_type;
    typedef VT scalar_type;

    typedef typename std::map<KT,QMatPair<KT,VT> > maptype;
    typedef typename maptype::mapped_type entry;
    typedef typename maptype::value_type value;
    typedef BlockMatArray<KT,VT> array_type;

    BlockMat() = default;
    BlockMat(const BlockMat& other) = default;
    BlockMat(BlockMat&& other) = default;
    BlockMat(const BlockDiagMat<KT,VT>& other);
    BlockMat(BlockDiagMat<KT,VT>&& other); /// steal all matrices from other, so effectively replace its structure from map<KT,MAT> to map<KT,pair<KT,MAT> >

    template<typename fill_type>
    BlockMat(const dim_map<KT>& ml, const dim_map<KT>& mr, const fill::fill_class<fill_type>& filler);

    template<typename fill_type>
    BlockMat(const dimkeypair_vec<KT>& mlr, const fill::fill_class<fill_type>& filler);

    /// constructors from vectors/arrays and QN sector dimension info
    BlockMat(const VT* vec, const dimkeypair_map<KT>& dimmap, bool copy_aux_mem=true, bool strict=false);
    BlockMat(      VT* vec, const dimkeypair_map<KT>& dimmap, bool copy_aux_mem=true, bool strict=false);

    BlockMat(const VT* vec, const dimkeypair_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false);
    BlockMat(      VT* vec, const dimkeypair_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false);

    template<typename DIMT>
    BlockMat(      Col<VT>& vec, const DIMT& dimvec, bool copy_aux_mem=true, bool strict=false);

    template<typename DIMT>
    BlockMat(const Col<VT>& vec, const DIMT& dimvec, bool copy_aux_mem=true, bool strict=false);

    /// mixed copy constructors
    template<typename VTO>
    inline explicit BlockMat(const BlockMat<KT,VTO>& other){BlockFromBlockCpCt(*this,other);};

    template<typename VTO>
    inline explicit BlockMat(const BlockDiagMat<KT,VTO>& other);

    /// Unfortunately there is no move constructor for complex matrices generated from two real matrices (or just one + zero matrix), so we also can't have a mixed move constructor

    template<typename GO>
    BlockMat(std::ifstream& file, const GO& GroupObj);

    /// ACCESS, HELPERS, INFO
    inline uint GetNElem() const;
    inline dimkeypair_vec<KT> GetSizesVector() const;
    Col<VT> Vectorize() const;
    inline bool IsDiag() const;
    inline void Fill(VT val) {for (auto& it : *this) QMat(it).fill(val);};

    inline KT dK() const { assert(!this->empty());return (this->begin()->second.first - this->begin()->first);};
    BlockMat ShiftLeftQN(const KT& K) const;
    BlockMat ShiftRightQN(const KT& K) const;
    BlockMat ShiftBothQN(const KT& KL, const KT& KR) const;
    BlockMat FlipQN(const std::vector<bool>& which = std::vector<bool>()) const;
    BlockMat PermuteQN(const std::vector<uint>& perm) const;

    BlockMat& operator=(const BlockMat& ) = default;
    BlockMat& operator=(      BlockMat&&) = default; /// this is good for now

    /// BASIC MODIFYING METHODS
    BlockMat& MultBlockMatLamLeft(const BlockLam<KT>& lam);
    BlockMat& MultBlockMatLamRight(const BlockLam<KT>& lam);
    BlockMat& DivBlockMatLamLeft(const BlockLam<KT>& lam);
    BlockMat& DivBlockMatLamRight(const BlockLam<KT>& lam);

//    BlockMat& MultDiagMatLeft(const BlockDiagMat<KT,VT>& mat);
//    BlockMat& MultDiagMatRight(const BlockDiagMat<KT,VT>& mat);
    /// BASIC NON-MODIFYING METHODS
    inline BlockMat t() const;
    inline BlockMat st() const;
    /// OPERATORS

    inline BlockMat operator-() const {BlockMat out; for (auto& it : *this) out.emplace_hint(out.end(),Qin(it),QMatPair<KT,VT>(Qout(it),-QMat(it))); return out;};
    BlockMat& operator+=(const BlockMat& other);
    BlockMat& operator-=(const BlockMat& other);
    inline BlockMat& operator*=(VT x) {for (auto& it : *this) QMat(it)*=x;return *this;};
    inline BlockMat& operator/=(VT x) {for (auto& it : *this) QMat(it)/=x;return *this;};

    void ShowDims(const std::string& name="") const;
    void print(const std::string& name="") const;

    /// DISK-IO
    bool save(std::string name) const;
    bool save(std::ofstream& file) const;
    bool load(std::string name);
    template<typename GO>
    bool load(std::ifstream& file, const GO& GroupObj);

};

/**< HELPER FUNCTION TEMPLATES FOR PARTIAL SPECIALIZATION OF MIXED CONSTRUCTOR ***********************************************/
/// This is necessary since for some stupid reason it is not allowed to partially specialize class template member function templates (i.e. the mixed constructor)
template<typename KT, typename VTI, typename VTO>
inline
void
BlockFromBlockCpCt(BlockMat<KT,VTO>& out, const BlockMat<KT,VTI>& in);

template<typename KT>
inline
void
BlockFromBlockCpCt(BlockMat<KT,Real>& out, const BlockMat<KT,Complex>& in)
{
    for (const auto& it : in)
    {
        /// it.first = ingoinf QN
        /// it.second = pair(QN,Mat)
        /// it.second.first = outgoing QN
        /// it.second.second = actual matrix for this QN sector
        #ifndef NDEBUG
        if (norm_inf(imag(it.second.second)) > 1e-12) cerr<<"imaginary part > 1e-12 in sector ("<<it.first<<","<<it.second.first<<endl;
        #endif // NDEBUG
        out.emplace_hint(out.end(),
                         it.first,
                         std::make_pair(it.second.first,
                                        real(it.second.second))
                         );
    }
}

template<typename KT>
inline
void
BlockFromBlockCpCt(BlockMat<KT,Complex>& out, const BlockMat<KT,Real>& in)
{
    for (const auto& it : in)
    {
        out.emplace_hint(out.end(),
                         it.first,
                         std::make_pair(it.second.first,
                                        Mat<Complex>(it.second.second,
                                                     Mat<double>(it.second.second.n_rows,it.second.second.n_cols,fill::zeros)
                                                     )
                                        )
                         );
    }
}

template<typename KT, typename VT>
template<typename fill_type>
BlockMat<KT,VT>::BlockMat(const dim_map<KT>& ml, const dim_map<KT>& mr, const fill::fill_class<fill_type>& filler)
{
    assert(ml.size() == mr.size() && "ml and mr need to be of same size");
    for (typename dim_map<KT>::const_iterator mlit = ml.begin(), mrit = mr.begin(); mlit!=ml.end(); ++mlit,++mrit)
    {
        if (mlit->second > 0 && mrit->second > 0) this->emplace_hint(this->end(),mlit->first,std::make_pair(mrit->first,Mat<VT>(mlit->second,mrit->second,filler)));
    }
}

/**< construct BlockMat with certain dimensions and fill with specified filler */
template<typename KT, typename VT>
template<typename fill_type>
BlockMat<KT,VT>::BlockMat(const dimkeypair_vec<KT>& mlr, const fill::fill_class<fill_type>& filler)
{
    for (const auto& mlrit : mlr)
    {
        if (get<2>(mlrit) > 0 && get<3>(mlrit) > 0)
            this->emplace_hint(this->end(),get<0>(mlrit),std::make_pair(get<1>(mlrit),Mat<VT>(get<2>(mlrit),get<3>(mlrit),filler)));
    }
}

/**< COPY FROM  BLOCKDIAGMATS **********************************************************************************************/
template<typename KT,typename VT>
BlockMat<KT,VT>::BlockMat(const BlockDiagMat<KT,VT>& other)
{
    for (const auto& oit : other) this->emplace_hint(this->end(),oit.first,std::make_pair(oit.first,oit.second));
}

/// this will leave other in a state, where the map has the same size as before, but
/// all the matrices in other have been stolen (moved) to this and replaced with 0x0 matrices.
/// the only overhead is creating and filling the map of this and copying the keys
template<typename KT,typename VT>
BlockMat<KT,VT>::BlockMat(BlockDiagMat<KT,VT>&& other)
{
//    cout<<"BlockMat(BlockDiagMat<KT,VT>&&) move"<<endl;
    for (auto& oit : other ) this->emplace_hint(this->end(),oit.first,std::make_pair(oit.first,std::move(oit.second)));
}

template<typename KT,typename VT>
inline
uint
BlockMat<KT,VT>::GetNElem() const
{
    uint n=0;
    for (const auto& it : *this) n+= it.second.second.n_elem;
    return n;
}

template<typename KT,typename VT>
inline
dimkeypair_vec<KT>
BlockMat<KT,VT>::GetSizesVector() const
{
    dimkeypair_vec<KT> dims;
    for (const auto& it : *this) dims.emplace_back(std::make_tuple(it.first,it.second.first,it.second.second.n_rows,it.second.second.n_cols));
    return dims;
}


/** constructors from vectors/array and QN sector dimension information *******************************************************************/

/// This is dangerous! The constructed BlockMat should actually not be altered, define it as const!!
template<typename KT,typename VT>
BlockMat<KT,VT>::BlockMat(const VT* vec, const dimkeypair_vec<KT>& dimvec, bool copy_aux_mem, bool strict)
{
    /// MAKE SURE THAT THE DIMENSION OF VEC IS AT LEAST dimvec.GetNElem()
    /// this is faster if the keys in dimvec are in order
    uint pos = 0;
    for (const auto& it : dimvec)
    {
        uint m=get<2>(it);
        uint n=get<3>(it);
        this->emplace_hint(this->end(),
                           get<0>(it),
                           std::make_pair(get<1>(it),
                                          Mat<VT>(const_cast<VT*>(&vec[pos]),m,n,copy_aux_mem,strict)
                                          )
                           );
        pos += m*n;
    }
}

template<typename KT,typename VT>
BlockMat<KT,VT>::BlockMat(      VT* vec, const dimkeypair_vec<KT>& dimvec, bool copy_aux_mem, bool strict)
{
    /// MAKE SURE THAT THE DIMENSION OF VEC IS AT LEAST dimvec.GetNElem()
    uint pos = 0;
    for (const auto& it : dimvec)
    {
        uint m=get<2>(it);
        uint n=get<3>(it);
        this->emplace_hint(this->end(),get<0>(it),std::make_pair(get<1>(it),Mat<VT>(&vec[pos],m,n,copy_aux_mem,strict)));
        pos += m*n;
    }
}

template<typename KT,typename VT>
template<typename DIMT>
BlockMat<KT,VT>::BlockMat(const Col<VT>& vec, const DIMT& dims, bool copy_aux_mem, bool strict) :
    BlockMat(vec.memptr(),dims, copy_aux_mem, strict)
{
    assert(vec.n_elem == dims.GetNElem() && "BlockMat(const Col& vec, const DIMT& dims): vec and dims need to account for the same number of elements");
}

template<typename KT,typename VT>
template<typename DIMT>
BlockMat<KT,VT>::BlockMat(      Col<VT>& vec, const DIMT& dims, bool copy_aux_mem, bool strict) :
    BlockMat(vec.memptr(),dims, copy_aux_mem, strict)
{
    assert(vec.n_elem == dims.GetNElem() && "BlockMat(Col& vec, const DIMT& dims): vec and dims need to account for the same number of elements");
}

/// supposedly faster vectorization
template<typename KT,typename VT>
inline
Col<VT>
BlockMat<KT,VT>::Vectorize() const
{
    Col<VT> out(this->GetNElem());
    VT* outmem = out.memptr();
    uint pos=0,dpos=0;

    for (const auto& it : *this)
    {
        dpos = QMat(it).n_rows*QMat(it).n_cols;
        memcpy(&outmem[pos],QMat(it).memptr(),dpos*sizeof(*outmem));
        pos += dpos;
    }
    return out;
}


/**< helper function to show all present quantum number sectors and their matrix sizes*/
template<typename KT,typename VT>
void
BlockMat<KT,VT>::ShowDims(const std::string& name) const
{
    if (name!="")cout<<name<<endl;
    if (!this->empty())
    {
        for (const auto& it : *this)cout<<"["<<Qin(it)<<","<<Qout(it)<<"]: "<<QMat(it).n_rows<<" x "<<QMat(it).n_cols<<endl;
    }
    else cout<<"---"<<endl;
}

/**< helper function to show all present quantum number sectors and their matrices */
template<typename KT,typename VT>
void
BlockMat<KT,VT>::print(const std::string& name) const
{
    if (name!="")cout<<name<<endl;
    cout<<*this<<endl;
}

template<typename KT,typename VT>
inline
bool
BlockMat<KT,VT>::IsDiag() const
{
    bool isdiag = true;
    for (const auto& it : *this) isdiag = isdiag && Qin(it) == Qout(it);
    return isdiag;
}

template<typename KT,typename VT>
BlockMat<KT,VT>
BlockMat<KT,VT>::ShiftLeftQN(const KT& K) const
{
    BlockMat<KT,VT> out;
    for (const auto& it : *this) out.emplace_hint(out.end(),std::make_pair(Qin(it)+K,it.second));
    return out;
}

template<typename KT,typename VT>
BlockMat<KT,VT>
BlockMat<KT,VT>::ShiftRightQN(const KT& K) const
{
    BlockMat<KT,VT> out;
    for (const auto& it : *this) out.emplace_hint(out.end(),std::make_pair(Qin(it),std::make_pair(Qout(it)+K,QMat(it))));
    return out;
}

template<typename KT,typename VT>
BlockMat<KT,VT>
BlockMat<KT,VT>::ShiftBothQN(const KT& KL, const KT& KR) const
{
    BlockMat<KT,VT> out;
    for (const auto& it : *this) out.emplace_hint(out.end(),std::make_pair(Qin(it)+KL,std::make_pair(Qout(it)+KR,QMat(it))));
    return out;
}


template<typename KT,typename VT>
BlockMat<KT,VT>
BlockMat<KT,VT>::FlipQN(const std::vector<bool>& which) const
{
    BlockMat<KT,VT> out;

    for (const auto& it : *this)
        out.emplace(std::make_pair(FlipK(Qin(it),which),std::make_pair(FlipK(Qout(it),which),QMat(it))));

    return out;
}


template<typename KT,typename VT>
BlockMat<KT,VT>
BlockMat<KT,VT>::PermuteQN(const std::vector<uint>& perm) const
{
    BlockMat<KT,VT> out;

    /// TODO (valentin#1#): properly implement a minus operator for KeyTypes

    for (const auto& it : *this)
        out.emplace(std::make_pair(PermuteK(Qin(it),perm),std::make_pair(PermuteK(Qout(it),perm),QMat(it))));

    return out;
}

/**< output stream operator overload for screen output */
template<typename KT,typename VT>
ostream&
operator<<(ostream& os, const BlockMat<KT,VT>& M)
{
    if (!M.empty())
    {
        for (const auto& it : M)
        {
            os<<"["<<Qin(it)<<","<<Qout(it)<<"]:"<<endl;
            os<<QMat(it)<<endl;
        }
    }
    else os<<"---"<<endl;
    return os;
}

template<typename KT,typename VT>
bool
BlockMat<KT,VT>::save(std::string name) const
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay =  this->save(file);
    if (!save_okay)
    {
        cerr << "BlockMat<KT,VT>::save(): could not save "<<name<<endl;
    }
    file.close();

    return save_okay;
}


template<typename KT,typename VT>
bool
BlockMat<KT,VT>::save(std::ofstream& file) const
{
    bool save_okay = file.good();
    if (!save_okay)
    {
        cerr<<"BlockMat<KT,VT>::save(): bad file handle"<<endl;
        return false;
    }
    uint Nsec=this->size();

/// TODO (valentin#1#): implement distinct headers for real and complex (to be able to check when loading)
//    file << "BLOCKMAT" << '\n';
    file << GetHeader(this) << '\n';
    file.write(reinterpret_cast<const char*>(&Nsec), std::streamsize(sizeof(uint)));

    for (const auto& it : *this)
    {
//        file << Qin(it);
//        file << Qout(it);
        if(!Qin(it).save(file))
        {
            cerr<<"BlockMat<KT,VT>::save(): saving of Qin = "<<Qin(it)<<" failed"<<endl;
            save_okay = false;
            break;
        }

        if(!Qout(it).save(file))
        {
            cerr<<"BlockMat<KT,VT>::save(): saving of Qin = "<<Qout(it)<<" failed"<<endl;
            save_okay = false;
            break;
        }
//        if (!diskio::save_arma_binary(QMat(it),file))
        if (!QMat(it).save(file,arma_binary))
        {
            cerr<<"BlockMat<KT,VT>::save(): saving of mat failed"<<endl;
            save_okay = false;
            break;
        }
    }
    return save_okay;
}

template<typename KT,typename VT>
bool
BlockMat<KT,VT>::load(std::string name)
{
    std::ifstream file(name.c_str(), std::fstream::binary);
    if (!file.is_open())
    {
        cerr<<"BlockMat<KT,VT>::load(): could not open "<<name<<endl;
        return false;
    }
    bool load_okay = this->load(file);
    if(!load_okay)
    {
        cerr<<"BlockDiagMat<KT,VT>::load(): could not load "<<name<<endl;
    }
    file.close();

    return load_okay;
}

/// for now we only implement the specialization for IKey, i.e. a vector of ints
template<typename KT, typename VT>
template<typename GO>
bool
BlockMat<KT,VT>::load(std::ifstream& file, const GO& GroupObj)
{
    this->clear();
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"BlockMat<KT,VT>::load(): bad file handle"<<endl;
        return false;
    }

    Mat<VT> tmp;
    std::string header;

    std::vector<int> in,out;
    uint Nsec=0;

    file >> header;
    file.get(); /// somehow we need this to get over the newline stored to terminate header. WTF it took me 1 hr to find that out!!
//    if (header != "BLOCKMAT")
    if (header != GetHeader(this))
    {
        cerr << "wrong header "<<header<<", should be "<<GetHeader(this)<<endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&Nsec), std::streamsize(sizeof(uint)));
    for (uint i=0; i<Nsec; ++i)
    {
        KT Kin(file,GroupObj);
        KT Kout(file,GroupObj);

        if (!tmp.load(file,arma_binary))
        {
            cerr<<"BlockMat<KT,VT>::load(): failed to load sector "<<Kin<<","<<Kout<<endl;
            load_okay = false;
            break;
        }

        this->emplace_hint(this->end(),Kin,std::make_pair(Kout,tmp));
    }
    return load_okay;
}



template<typename KT,typename VT>
inline
BlockMat<KT,VT>
BlockMat<KT,VT>::t() const
{
    BlockMat<KT,VT> out;
    for (const auto& it : *this)
    {
        out.emplace_hint(out.end(),Qout(it),std::make_pair(Qin(it),QMat(it).t()));
    }
    return out;
}


template<typename KT,typename VT>
inline
BlockMat<KT,VT>
BlockMat<KT,VT>::st() const
{
    BlockMat<KT,VT> out;
    for (const auto& it : *this)
    {
        out.emplace_hint(out.end(),Qout(it),std::make_pair(Qin(it),QMat(it).st()));
    }
    return out;
}

template<typename KT,typename VT>
BlockMat<KT,VT>&
BlockMat<KT,VT>::operator+=(const BlockMat<KT,VT>& other)
{
    for (const auto& oit : other) /// loop through symmetry sectors in other (it is important to loop through OTHER, s.t. no contributions are lost!!)
    {
        auto thisit = this->lower_bound(oit.first); /// look for current QN of other in this
        if (thisit != this->end() && thisit->first == oit.first)
//        if (thisit->first == oit.first)
        {
//            QMat(thisit->second) += QMat(oit.second);/// if it is found, add other to this
            assert(Qout(*thisit) == Qout(oit) && "BlockMat::operator+= : this and other have different Qout");
//            thisit->second.second += oit.second.second;/// if it is found, add other to this
            QMat(*thisit) += QMat(oit);/// if it is found, add other to this
        }
        else this->emplace_hint(thisit,oit.first,oit.second); /// if it is not found, insert

    }
    return *this;
}


template<typename KT,typename VT>
BlockMat<KT,VT>&
BlockMat<KT,VT>::operator-=(const BlockMat<KT,VT>& other)
{
    for (const auto& oit : other) /// loop through symmetry sectors in other (it is important to loop through OTHER, s.t. no contributions are lost!!)
    {
        auto thisit = this->lower_bound(oit.first); /// look for current QN of other in this
        if (thisit != this->end() && thisit->first == oit.first)
        {
//            QMat(thisit->second) += QMat(oit.second);/// if it is found, add other to this
            assert(Qout(*thisit) == Qout(oit) && "BlockMat::operator+= : this and other have different Qout");
//            thisit->second.second += oit.second.second;/// if it is found, add other to this
            QMat(*thisit) -= QMat(oit);/// if it is found, add other to this
        }
        else this->emplace_hint(thisit,oit.first,std::make_pair(Qout(oit),-QMat(oit))); /// if it is not found, insert

    }
    return *this;
}



/**< BASIC MULTIPLICATION AND DIVISION BY DIAGONAL MATRICES (E.G. SCHMIDT VALUES) ******************************************************/

/// modifying
template<typename KT,typename VT>
BlockMat<KT,VT>&
BlockMat<KT,VT>::MultBlockMatLamLeft(const BlockLam<KT>& lam)
{
    /// std paradigm for looping through map while also deleting elements
    /// for true iterators we don't need to put auto&, in fact that's not even allowed syntax
    for (auto thisit = this->begin(); thisit!=this->end(); )
    {
        const auto lamit = lam.find(Qin(*thisit));
        if (lamit!=lam.end())
        {
            (lamit->second) > QMat(*thisit);
            ++thisit;
        }
        else thisit = this->erase(thisit);
    }
    return *this;
}

template<typename KT,typename VT>
BlockMat<KT,VT>&
BlockMat<KT,VT>::DivBlockMatLamLeft(const BlockLam<KT>& lam)
{
    /// std paradigm for looping through map while also deleting elements
    /// for true iterators we don't need to put auto&, in fact that'not even allowed syntax
    for (auto thisit = this->begin(); thisit!=this->end(); )
    {
        const auto lamit = lam.find(Qin(*thisit));
        if (lamit!=lam.end())
        {
            (lamit->second) < QMat(*thisit);
            ++thisit;
        }
        else thisit = this->erase(thisit);
    }
    return *this;
}

template<typename KT,typename VT>
BlockMat<KT,VT>&
BlockMat<KT,VT>::MultBlockMatLamRight(const BlockLam<KT>& lam)
{
    /// std paradigm for looping through map while also deleting elements
    /// for true iterators we don't need to put auto&, in fact that'not even allowed syntax
    for (auto thisit = this->begin(); thisit!=this->end(); )
    {
        const auto lamit = lam.find(Qout(*thisit));
        if (lamit!=lam.end())
        {
            QMat(*thisit) < (lamit->second);
            ++thisit;
        }
        else thisit = this->erase(thisit);
    }
    return *this;
}


template<typename KT,typename VT>
BlockMat<KT,VT>&
BlockMat<KT,VT>::DivBlockMatLamRight(const BlockLam<KT>& lam)
{
    /// std paradigm for looping through map while also deleting elements
    /// for true iterators we don't need to put auto&, in fact that'not even allowed syntax
    for (auto thisit = this->begin(); thisit!=this->end(); )
    {
        const auto lamit = lam.find(Qout(*thisit));
        if (lamit!=lam.end())
        {
            QMat(*thisit) > (lamit->second);
            ++thisit;
        }
        else thisit = this->erase(thisit);
    }
    return *this;
}

/**< OPERATORS FOR MULTIPLYING/DIVIDING WITH DIAGONAL MATRICES (E.G. SCHMIDT VALUES) */
/// modifying -> return references to modified object
template<typename KT,typename VT>
inline BlockMat<KT,VT>& operator>(const BlockLam<KT>& lam, BlockMat<KT,VT>& mat) {return mat.MultBlockMatLamLeft(lam);}

template<typename KT,typename VT>
inline BlockMat<KT,VT>& operator<(BlockMat<KT,VT>& mat, const BlockLam<KT>& lam) {return mat.MultBlockMatLamRight(lam);}

template<typename KT,typename VT>
inline BlockMat<KT,VT>& operator<(const BlockLam<KT>& lam, BlockMat<KT,VT>& mat) {return mat.DivBlockMatLamLeft(lam);}

template<typename KT,typename VT>
inline BlockMat<KT,VT>& operator>(BlockMat<KT,VT>& mat, const BlockLam<KT>& lam) {return mat.DivBlockMatLamRight(lam);}

/// non-modifying -> pass Blockmats by VALUE to create copies
/// TODO (valentin#1#): consider implementing a separate non-modifying routine, creating the output object completely anew (for now the input one is copied and some parts of it might get deleted, so we are possibly copying unneeded data)
template<typename KT,typename VT>
inline BlockMat<KT,VT> operator>>(const BlockLam<KT>& lam, BlockMat<KT,VT> mat) {return mat.MultBlockMatLamLeft(lam);}

template<typename KT,typename VT>
inline BlockMat<KT,VT> operator<<(BlockMat<KT,VT> mat, const BlockLam<KT>& lam) {return mat.MultBlockMatLamRight(lam);}

template<typename KT,typename VT>
inline BlockMat<KT,VT> operator<<(const BlockLam<KT>& lam, BlockMat<KT,VT> mat) {return mat.DivBlockMatLamLeft(lam);}

template<typename KT,typename VT>
inline BlockMat<KT,VT> operator>>(BlockMat<KT,VT> mat, const BlockLam<KT>& lam) {return mat.DivBlockMatLamRight(lam);}

/**< other non-member functions */

template<typename KT,typename VT>
inline
Real
norm(const BlockMat<KT,VT>& in)
{
    #ifndef NDEBUG
    if (in.empty()) cerr<<"norm: in is empty"<<endl;
    #endif // NDEBUG

    Real nrm2=0;
//    for (const auto& it : in) nrm2+=abs(dot(it.second,it.second));
    for (const auto& it : in) nrm2 += dot(QMat(it));
//    for (const auto& it : in) nrm2+=pow(norm(QMat(it),"fro"),2);
    return sqrt(nrm2);
}


template<typename KT,typename VT>
inline
Real
norm_inf(const BlockMat<KT,VT>& in)
{
    #ifndef NDEBUG
    if (in.empty()) cerr<<"norm_inf: in is empty"<<endl;
    #endif // NDEBUG

    Real nrmi=0;
//    for (const auto& it : in) nrm2+=abs(dot(it.second,it.second));
    for (const auto& it : in) nrmi = std::max(nrmi,abs(QMat(it)).max());
//    for (const auto& it : in) nrm2+=pow(norm(QMat(it),"fro"),2);
    return nrmi;
}

/**< MULTIPLICATION AND DIVISION WITH SCALARS */

template<typename KT,typename VTx, typename VTM>
inline
BlockMat<KT,typename promote_type<VTx,VTM>::result>
operator*(const BlockMat<KT,VTM>& mat, VTx x)
{
    BlockMat<KT,typename promote_type<VTx,VTM>::result> out(mat);
    return out*=x;
}

template<typename KT,typename VTx, typename VTM>
inline
BlockMat<KT,typename promote_type<VTx,VTM>::result>
operator*(VTx x, const BlockMat<KT,VTM>& mat)
{
    BlockMat<KT,typename promote_type<VTx,VTM>::result> out(mat);
    return out*=x;
}

template<typename KT,typename VTx, typename VTM>
inline
BlockMat<KT,typename promote_type<VTx,VTM>::result>
operator/(const BlockMat<KT,VTM>& mat, VTx x)
{
    BlockMat<KT,typename promote_type<VTx,VTM>::result> out(mat);
    return out/=x;
}


/**< ADDITION AND SUBTRACTION ROUTINES FOR BLOCK MATRICES */

/// template for general mixed addition
template<typename KT,typename VTL, typename VTR>
inline
BlockMat<KT,typename promote_type<VTL,VTR>::result>
operator+(const BlockMat<KT,VTL>& lhs, const BlockMat<KT,VTR>& rhs)
{
    BlockMat<KT,typename promote_type<VTL,VTR>::result> out(lhs);
    return lhs+=rhs;
}

/// specialization for mixed same type addition
template<typename KT,typename VT>
inline
BlockMat<KT,VT>
operator+(BlockMat<KT,VT> lhs, const BlockMat<KT,VT>& rhs)
{
    return lhs+=rhs;
}

/// specialization for mixed complex real addition
template<typename KT>
inline
BlockMat<KT,Complex>
operator+(const BlockMat<KT,Complex>& lhs, const BlockMat<KT,Real>& rhs)
{
    BlockMat<KT,Complex> out(rhs);
    return out+=lhs;
}

/// specialization for mixed complex real addition
template<typename KT>
inline
BlockMat<KT,Complex>
operator+(const BlockMat<KT,Real>& lhs, const BlockMat<KT,Complex>& rhs)
{
    BlockMat<KT,Complex> out(lhs);
    return out+=rhs;
}

/// template for general mixed subtraction
template<typename KT,typename VTL, typename VTR>
inline
BlockMat<KT,typename promote_type<VTL,VTR>::result>
operator-(const BlockMat<KT,VTL>& lhs, const BlockMat<KT,VTR>& rhs)
{
    BlockMat<KT,typename promote_type<VTL,VTR>::result> out(lhs);
    return lhs-=rhs;
}

/// template for same type subtraction
template<typename KT,typename VT>
inline
BlockMat<KT,VT>
operator-(BlockMat<KT,VT> lhs, const BlockMat<KT,VT>& rhs)
{
    return lhs-=rhs;
}

/// template for mixed complex/real subtraction
template<typename KT>
inline
BlockMat<KT,Complex>
operator-(const BlockMat<KT,Complex>& lhs, const BlockMat<KT,Real>& rhs)
{
    BlockMat<KT,Complex> out(lhs);
    return out-=BlockMat<KT,Complex>(rhs);
}

/// template for mixed complex/real subtraction
template<typename KT>
inline
BlockMat<KT,Complex>
operator-(const BlockMat<KT,Real>& lhs, const BlockMat<KT,Complex>& rhs)
{
    BlockMat<KT,Complex> out(lhs);
    return out-=rhs;
}

/// trace for BlockMats
/// this automatically zero if the BlockMat is not block diagonal
template<typename KT,typename VT>
inline
VT
trace(const BlockMat<KT,VT>& mat)
{
    VT out=0;
    for (const auto& it : mat) if (it.first == it.second.first) out+=trace(it.second.second);
    return out;
}



/** BlockDiagonal Matrix Type ************************************************************************
 *  the ingoing and outgoing quantum number is the same, therefore the second needs not to be stored
 */

template<typename KT,typename VT>
class BlockDiagMat : public std::map<KT,Mat<VT> >
{
public:
    typedef KT key_type;
    typedef VT scalar_type;
    typedef typename std::map<KT,Mat<VT> > maptype;
    typedef BlockDiagMatArray<KT,VT> array_type;

    BlockDiagMat() = default;
    BlockDiagMat(const BlockDiagMat& in) = default;
    BlockDiagMat(BlockDiagMat&& in) = default;
    BlockDiagMat& operator=(const BlockDiagMat& other) & = default; /// the & before = default means, this operator can only be called on lvalues (for the rvalue variant, add &&)
    BlockDiagMat& operator=(BlockDiagMat&& other) & = default;
    BlockDiagMat(const BlockLam<KT>& lam);
    BlockDiagMat(BlockLam<KT>&& lam);
    BlockDiagMat(const BlockMat<KT,VT>& other);
    BlockDiagMat(BlockMat<KT,VT>&& other);

    template<typename fill_type>
    BlockDiagMat(const dim_map<KT>& dims, const fill::fill_class<fill_type>& filler);

    template<typename fill_type>
    BlockDiagMat(const dim_vec<KT>& dims, const fill::fill_class<fill_type>& filler);

    template<typename fill_type>
    BlockDiagMat(const dimpair_vec<KT>& dims, const fill::fill_class<fill_type>& filler);

    /// mixed copy constructors
    template<typename VTO>
    inline explicit BlockDiagMat(const BlockDiagMat<KT,VTO>& other){DiagFromDiagCpCt(*this,other);};

    /// GETTERS
    inline dimpair_map<KT> GetSizes() const;
    inline dimpair_vec<KT> GetSizesVector() const;
    inline dim_map<KT> GetUniformSizes() const;
    inline dim_vec<KT> GetUniformSizesVector() const;
    inline dim_map<KT> GetMl() const;
    inline dim_map<KT> GetMr() const;
    inline uint GetTotalMl() const;
    inline uint GetTotalMr() const;
    inline uint GetNElem() const;

    /// if the block matrices are square
    BlockDiagMat(const VT* vec, const dim_map<KT>& dimmap, bool copy_aux_mem=true, bool strict=false);
    BlockDiagMat(      VT* vec, const dim_map<KT>& dimmap, bool copy_aux_mem=true, bool strict=false);
//    BlockDiagMat(Col<VT>&& vec, const dim_map<KT>& dimmap, bool copy_aux_mem=true, bool strict=false);

    BlockDiagMat(const VT* vec, const dim_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false);
    BlockDiagMat(      VT* vec, const dim_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false);
//    BlockDiagMat(Col<VT>&& vec, const dim_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false);

    /// if the block matrices are NOT square
    BlockDiagMat(const VT* vec, const dimpair_map<KT>& dimmap, bool copy_aux_mem=true, bool strict=false);
    BlockDiagMat(      VT* vec, const dimpair_map<KT>& dimmap, bool copy_aux_mem=true, bool strict=false);
//    BlockDiagMat(Col<VT>&& vec, const dimpair_map<KT>& dimmap, bool copy_aux_mem=true, bool strict=false);

    BlockDiagMat(const VT* vec, const dimpair_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false);
    BlockDiagMat(      VT* vec, const dimpair_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false);

    template<typename DIMT>
    BlockDiagMat(      Col<VT>& vec, const DIMT& dimvec, bool copy_aux_mem=true, bool strict=false);

    template<typename DIMT>
    BlockDiagMat(const Col<VT>& vec, const DIMT& dimvec, bool copy_aux_mem=true, bool strict=false);

//    BlockDiagMat(const Col<VT>& vec, const dim_vec<KT>& dimvec);
//    BlockDiagMat(const Col<VT>& vec, const dim_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false);
//    BlockDiagMat(      Col<VT>& vec, const dim_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false);
//    BlockDiagMat(      Col<VT>& vec, const dim_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false); /// if the block matrices are square
//    BlockDiagMat(      Col<VT>& vec, const dimpair_vec<KT>& dimvec, bool copy_aux_mem=true, bool strict=false); /// if the block matrices are NOT square


    Col<VT> Vectorize() const;

    inline KT dK() const { assert(!this->empty());return KT(this->begin()->first.GetGroupObj());}; /// return zero key, BlockDiagMats are, well, block diagonal (added for consistency)
    BlockDiagMat ShiftQN(const KT& K) const;
    BlockDiagMat FlipQN(const std::vector<bool>& which = std::vector<bool>()) const;
    BlockDiagMat PermuteQN(const std::vector<uint>& perm) const;

    /// BASIC HELPERS
    void ShowDims(const std::string& name="") const;
    void print(const std::string& name="") const;

    /// BASIC NON-MODIFYING METHODS
    inline BlockDiagMat t() const; /// hermitian transpose
    inline BlockDiagMat st() const; /// regular transpose

    /// OPERATORS
    inline BlockDiagMat operator-() const {BlockDiagMat out; for (auto& it : *this) out.emplace_hint(out.end(),it.first,-it.second);return out;};
    inline BlockDiagMat& operator*=(VT scalar) {for (auto& it : *this) it.second*=scalar;return *this;};
    inline BlockDiagMat& operator/=(VT scalar) {for (auto& it : *this) it.second/=scalar;return *this;};

    BlockDiagMat& operator+=(const BlockDiagMat& other);
    BlockDiagMat& operator-=(const BlockDiagMat& other);

    inline BlockDiagMat& operator+=(const BlockLam<KT>& lam) {return (*this)+=BlockDiagMat(lam);};
    inline BlockDiagMat& operator-=(const BlockLam<KT>& lam) {return (*this)-=BlockDiagMat(lam);};

//    friend BlockDiagMat operator*(VT scalar, BlockDiagMat mat) {return mat*=scalar;}
//    friend BlockDiagMat operator*(BlockDiagMat mat, VT scalar) {return mat*=scalar;}

    /// DISK-IO
    bool save(std::string name) const;
    bool save(std::ofstream& file) const;
    bool load(std::string name);
    template<typename GO>
    bool load(std::ifstream& file, const GO& GroupObj);
};

template<typename KT, typename VT>
template<typename fill_type>
BlockDiagMat<KT,VT>::BlockDiagMat(const dim_map<KT>& dims, const fill::fill_class<fill_type>& filler)
{
    for (const auto& dimit : dims)
    {
        if (dimit.second > 0) this->emplace_hint(this->end(),dimit.first,Mat<VT>(dimit.second,dimit.second,filler));
    }
}

template<typename KT, typename VT>
template<typename fill_type>
BlockDiagMat<KT,VT>::BlockDiagMat(const dim_vec<KT>& dims, const fill::fill_class<fill_type>& filler)
{
    for (const auto& dimit : dims)
    {
        if (dimit.second > 0) this->emplace_hint(this->end(),dimit.first,Mat<VT>(dimit.second,dimit.second,filler));
    }
}

template<typename KT, typename VT>
template<typename fill_type>
BlockDiagMat<KT,VT>::BlockDiagMat(const dimpair_vec<KT>& dims, const fill::fill_class<fill_type>& filler)
{
    uint ml,mr;
    for (const auto& dimit : dims)
    {
        ml = get<1>(dimit);
        mr = get<2>(dimit);
        if (ml > 0 && mr > 0) this->emplace_hint(this->end(),get<0>(dimit),Mat<VT>(ml,mr,filler));
    }
}

/**< HELPER FUNCTION TEMPLATES FOR PARTIAL SPECIALIZATION OF MIXED CONSTRUCTOR ***********************************************/
/// This is necessary since for some stupid reason it is not allowed to partially specialize class template member function templates (i.e. the mixed constructor)
template<typename KT, typename VTI, typename VTO>
inline
void
DiagFromDiagCpCt(BlockDiagMat<KT,VTO>& out, const BlockDiagMat<KT,VTI>& in);


template<typename KT>
inline
void
DiagFromDiagCpCt(BlockDiagMat<KT,Complex>& out, const BlockDiagMat<KT,Real>& in)
{
    for (auto& it : in) out.emplace_hint(out.end(),it.first,Mat<Complex>(it.second,Mat<Real>(it.second.n_rows,it.second.n_cols,fill::zeros)));
}

/**< conversion copy constructors */

template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(const BlockLam<KT>& lam)
{
    for (const auto& it : lam) this->emplace_hint(this->end(),it.first,diagmat(it.second));
}

template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(const BlockMat<KT,VT>& other)
{
    for (const auto& oit : other)
    {
        if (Qin(oit) == Qout(oit)) this->emplace_hint(this->end(),Qin(oit),QMat(oit));
        #ifndef NDEBUG
        else cerr<<"warning in BlockDiagMat<KT,VT>::BlockDiagMat(const BlockMat<KT,VT>&) conversion: off-diagonal element ("<<Qin(oit)<<","<<Qout(oit)<<") found."<<endl;
        #endif // NDEBUG
    }
}

/**< move conversion constructors */
/// these will leave other in a state, where the map has the same size as before, but
/// all the matrices/vectors in other have been stolen (moved) to this and replaced with 0x0 matrices/0 vectors.
/// the only overhead is creating and filling the map of this and copying the keys

/// TODO (valentin#1#2016-09-14): Do we have to copy the keys or can we move them too?
template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(BlockLam<KT>&& lam)
{
//    cout<<"BlockDiagMat(BlockLambda<KT>&&) move"<<endl;
    for (auto& it : lam) this->emplace_hint(this->end(),it.first,diagmat(std::move(it.second)));
}

template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(BlockMat<KT,VT>&& other)
{
//    cout<<"BlockDiagMat(BlockMat<KT,VT>&&) move"<<endl;
    for (const auto& oit : other)
    {
        if (Qin(oit) == Qout(oit)) this->emplace_hint(this->end(),Qin(oit),std::move(QMat(oit)));
        #ifndef NDEBUG
        else cerr<<"warning in BlockDiagMat(const BlockMat<KT,VT>&) conversion: off-diagonal element ("<<Qin(oit)<<","<<Qout(oit)<<") found."<<endl;
        #endif // NDEBUG
    }
}

/** constructors from vectors/array and QN sector dimension information *******************************************************************/
template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(const VT* vec, const dim_map<KT>& dimmap, bool copy_aux_mem, bool strict)
{
    uint pos = 0;
    for (const auto& it : dimmap)
    {
        uint m=it.second;
        this->emplace_hint(this->end(),it.first,Mat<VT>(const_cast<VT*>(&vec[pos]),m,m));
        pos += m*m;
    }
}

template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(      VT* vec, const dim_map<KT>& dimmap, bool copy_aux_mem, bool strict)
{
    uint pos = 0;
    for (const auto& it : dimmap)
    {
        uint m=it.second;
        this->emplace_hint(this->end(),it.first,Mat<VT>(&vec[pos],m,m));
        pos += m*m;
    }
}

//template<typename KT,typename VT>
//BlockDiagMat<KT,VT>::BlockDiagMat(Col<VT>&& vec, const dim_map<KT>& dimmap, bool copy_aux_mem, bool strict) :
//    BlockDiagMat(vec.memptr(), dimmap, copy_aux_mem, strict)
//{
//    assert(vec.size() == dimmap.GetNElem() && "vec and dimmap must cover the same amount of elements");
//}


template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(const VT* vec, const dim_vec<KT>& dimvec, bool copy_aux_mem, bool strict)
{
    uint pos = 0;
    for (const auto& it : dimvec)
    {
        uint m=it.second;
        this->emplace_hint(this->end(),it.first,Mat<VT>(const_cast<VT*>(&vec[pos]),m,m));
        pos += m*m;
    }
}

template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(      VT* vec, const dim_vec<KT>& dimvec, bool copy_aux_mem, bool strict)
{
    uint pos = 0;
    for (const auto& it : dimvec)
    {
        uint m=it.second;
        this->emplace_hint(this->end(),it.first,Mat<VT>(&vec[pos],m,m));
        pos += m*m;
    }
}

//template<typename KT,typename VT>
//BlockDiagMat<KT,VT>::BlockDiagMat(Col<VT>&& vec, const dim_vec<KT>& dimvec, bool copy_aux_mem, bool strict) :
//    BlockDiagMat(vec.memptr(), dimvec, copy_aux_mem, strict)
//{
//    assert(vec.size() == dimvec.GetNElem() && "vec and dimvec must cover the same amount of elements");
//}


template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(const VT* vec, const dimpair_map<KT>& dimmap, bool copy_aux_mem, bool strict)
{
    uint pos = 0;
    for (const auto& it : dimmap)
    {
        uint m=it.second.first;
        uint n=it.second.second;
        this->emplace_hint(this->end(),it.first,Mat<VT>(const_cast<VT*>(&vec[pos]),m,n,copy_aux_mem,strict));
        pos += m*n;
    }
}

template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(      VT* vec, const dimpair_map<KT>& dimmap, bool copy_aux_mem, bool strict)
{
    uint pos = 0;
    for (const auto& it : dimmap)
    {
        uint m=it.second.first;
        uint n=it.second.second;
        this->emplace_hint(this->end(),it.first,Mat<VT>(&vec[pos],m,n,copy_aux_mem,strict));
        pos += m*n;
    }
}


//template<typename KT,typename VT>
//BlockDiagMat<KT,VT>::BlockDiagMat(Col<VT>&& vec, const dimpair_map<KT>& dimmap, bool copy_aux_mem, bool strict) :
//    BlockDiagMat(vec.memptr(),dimmap,copy_aux_mem,strict)
//{
//    assert(vec.size() == dimmap.GetNElem() && "vec and dimmap must cover the same amount of elements");
//}


template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(const VT* vec, const dimpair_vec<KT>& dimvec, bool copy_aux_mem, bool strict)
{
    uint pos = 0;
    for (const auto& it : dimvec)
    {
        uint m=get<1>(it);
        uint n=get<2>(it);
        this->emplace_hint(this->end(),get<0>(it),Mat<VT>(const_cast<VT*>(&vec[pos]),m,n,copy_aux_mem,strict));
        pos += m*n;
    }
}

template<typename KT,typename VT>
BlockDiagMat<KT,VT>::BlockDiagMat(      VT* vec, const dimpair_vec<KT>& dimvec, bool copy_aux_mem, bool strict)
{
    uint pos = 0;
    for (const auto& it : dimvec)
    {
        uint m=get<1>(it);
        uint n=get<2>(it);
        this->emplace_hint(this->end(),get<0>(it),Mat<VT>(&vec[pos],m,n,copy_aux_mem,strict));
        pos += m*n;
    }
}

template<typename KT,typename VT>
template<typename DIMT>
BlockDiagMat<KT,VT>::BlockDiagMat(const Col<VT>& vec, const DIMT& dims, bool copy_aux_mem, bool strict) :
    BlockDiagMat(vec.memptr(),dims,copy_aux_mem,strict)
{
    assert(vec.n_elem == dims.GetNElem() && "BlockDiagMat(const Col& vec, const DIMT& dims, bool copy_aux_mem, bool strict): vec and dims must cover the same amount of elements");
}

template<typename KT,typename VT>
template<typename DIMT>
BlockDiagMat<KT,VT>::BlockDiagMat(      Col<VT>& vec, const DIMT& dims, bool copy_aux_mem, bool strict) :
    BlockDiagMat(vec.memptr(),dims,copy_aux_mem,strict)
{
    assert(vec.n_elem == dims.GetNElem() && "BlockDiagMat(Col& vec, const DIMT& dims, bool copy_aux_mem, bool strict): vec and dims must cover the same amount of elements");
}

/// supposedly faster vectorization
/// unfortunately there is no way around copying once the BlockDiagMat has been created by itself,
/// as the respective blocks will in general not be contiguous in memory
template<typename KT,typename VT>
inline
Col<VT>
BlockDiagMat<KT,VT>::Vectorize() const
{
    Col<VT> out(this->GetNElem());
    VT* outmem = out.memptr();
    uint pos = 0, dpos=0;

    for (const auto& it : *this)
    {
        dpos = it.second.n_elem;
//        memcpy(&outmem[pos],it.second.memptr(),dpos*sizeof(*outmem));
        std::copy(it.second.memptr(),it.second.memptr()+dpos,&outmem[pos]);
        pos += dpos;
    }
    return out;
}

template<typename KT,typename VT>
BlockDiagMat<KT,VT>
BlockDiagMat<KT,VT>::ShiftQN(const KT& K) const
{
    BlockDiagMat<KT,VT> out;
    for (const auto& it : *this) out.emplace_hint(out.end(),std::make_pair(it.first+K,it.second));
    return out;
}


template<typename KT,typename VT>
BlockDiagMat<KT,VT>
BlockDiagMat<KT,VT>::FlipQN(const std::vector<bool>& which) const
{
    BlockDiagMat<KT,VT> out;

    for (const auto& it : *this)
        out.emplace(std::make_pair(FlipK(it.first,which),it.second));

    return out;
}

template<typename KT,typename VT>
BlockDiagMat<KT,VT>
BlockDiagMat<KT,VT>::PermuteQN(const std::vector<uint>& perm) const
{
    BlockDiagMat<KT,VT> out;

    for (const auto& it : *this)
        out.emplace(std::make_pair(PermuteK(it.first,perm),it.second));

    return out;
}


/**< helper function to show all present quantum number sectors */
template<typename KT,typename VT>
void
BlockDiagMat<KT,VT>::ShowDims(const std::string& name) const
{
    if (name!="")cout<<name<<endl;
    if (!this->empty())
    {
        for (const auto& it : *this) cout<<it.first<<": "<<it.second.n_rows<<"x"<<it.second.n_cols<<endl;
    }
    else cout<<"---"<<endl;
}


template<typename KT,typename VT>
inline
dimpair_map<KT>
BlockDiagMat<KT,VT>::GetSizes() const
{
    dimpair_map<KT> dims;
    for (const auto& it : *this) dims.emplace_hint(dims.end(),it.first,make_pair(it.second.n_rows,it.second.n_cols));
    return dims;
}

template<typename KT,typename VT>
inline
dimpair_vec<KT>
BlockDiagMat<KT,VT>::GetSizesVector() const
{
    dimpair_vec<KT> dims;
    dims.reserve(this->size());
    for (const auto& it : *this) dims.emplace_back(std::make_tuple(it.first,it.second.n_rows,it.second.n_cols));
    return dims;
}

template<typename KT,typename VT>
inline
dim_map<KT>
BlockDiagMat<KT,VT>::GetUniformSizes() const
{
    dim_map<KT> dims;
    for (const auto& it : *this)
    {
        if (it.second.n_rows == it.second.n_cols) dims.emplace_hint(dims.end(),it.first,it.second.n_rows);
        else {cerr<<"GetUniformSizes(): sector "<<it.first<<" is not square"<<endl;abort();}
    }
    return dims;
}

template<typename KT, typename VT>
inline
dim_vec<KT>
BlockDiagMat<KT,VT>::GetUniformSizesVector() const
{
    dim_vec<KT> dims;
    dims.reserve(this->size());
    for (const auto& it : *this)
    {
        if (it.second.n_rows == it.second.n_cols) dims.emplace_back(std::make_pair(it.first,it.second.n_rows));
        else {cerr<<"GetUniformSizesVector(): sector "<<it.first<<" is not square"<<endl;abort();}
    }
    return dims;
}


template<typename KT, typename VT>
inline
uint
BlockDiagMat<KT,VT>::GetNElem() const
{
    uint n=0;
    for (const auto& it : *this) n+= it.second.n_elem;
    return n;
}


template<typename KT,typename VT>
inline
dim_map<KT>
BlockDiagMat<KT,VT>::GetMl() const
{
    dim_map<KT> dims;
    for (const auto& it : *this) dims.emplace_hint(dims.end(),it.first,it.second.n_rows);
    return dims;
}

template<typename KT,typename VT>
inline
dim_map<KT>
BlockDiagMat<KT,VT>::GetMr() const
{
    dim_map<KT> dims;
    for (const auto& it : *this) dims.emplace_hint(dims.end(),it.first,it.second.n_cols);
    return dims;
}


template<typename KT,typename VT>
inline
uint
BlockDiagMat<KT,VT>::GetTotalMl() const
{
    uint m=0;
    for (const auto& it : *this) m+=it.second.n_rows;
    return m;
}


template<typename KT,typename VT>
inline
uint
BlockDiagMat<KT,VT>::GetTotalMr() const
{
    uint m=0;
    for (const auto& it : *this) m+=it.second.n_cols;
    return m;
}

template<typename KT,typename VT>
void
BlockDiagMat<KT,VT>::print(const std::string& name) const
{
    if (name!="")cout<<name<<endl;
    cout<<*this<<endl;
}


template<typename KT,typename VT>
BlockDiagMat<KT,VT>&
BlockDiagMat<KT,VT>::operator+=(const BlockDiagMat<KT,VT>& other)
{
    if (this->empty()) *this = other;
    else
    {
        for (const auto& oit : other) /// loop through symmetry sectors in other (it is important to loop through OTHER, s.t. no (possibly new) contributions are lost!!)
        {
            auto thisit = this->lower_bound(oit.first); /// look for current QN of other in this
//        if (thisit == this->end() || thisit->first != oit.first) this->emplace_hint(thisit,oit.first,oit.second);/// if it is NOT found, insert
//        else thisit->second += oit.second; /// otherwise add other to this
            if (thisit != this->end() && thisit->first == oit.first) thisit->second += oit.second; /// if it is found, add other to this
            else this->emplace_hint(thisit,oit.first,oit.second); /// if it is not found, insert
        }
    }
    return *this;
}

template<typename KT,typename VT>
BlockDiagMat<KT,VT>&
BlockDiagMat<KT,VT>::operator-=(const BlockDiagMat<KT,VT>& other)
{
    if (this->empty()) *this = -(other);
    else
    {
        for (const auto& oit : other) /// loop through symmetry sectors in other (it is important to loop through OTHER, s.t. no (possibly new) contributions are lost!!)
        {
            auto thisit = this->lower_bound(oit.first); /// look for current QN of other in this
            if (thisit != this->end() && thisit->first == oit.first) thisit->second -= oit.second; /// if it is found, add other to this
            else this->emplace_hint(thisit,oit.first,-oit.second); /// if it is not found, insert

        }
    }
    return *this;
}

template<typename KT,typename VT>
inline
BlockDiagMat<KT,VT>
BlockDiagMat<KT,VT>::t() const
{
    BlockDiagMat<KT,VT> out;
    for (const auto& it : *this) out.emplace_hint(out.end(),it.first,it.second.t());
    return out;
//    for (auto& it : *this) inplace_trans(it);
//    return out;
}


template<typename KT,typename VT>
inline
BlockDiagMat<KT,VT>
BlockDiagMat<KT,VT>::st() const
{
    BlockDiagMat<KT,VT> out;
    for (const auto& it : *this) out.emplace_hint(out.end(),it.first,it.second.st());
    return out;
}


template<typename KT, typename VT>
bool
BlockDiagMat<KT,VT>::save(std::string name) const
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay = save(file);
    if (!save_okay)
    {
        cerr << "BlockDiagMat<KT,VT>::save(): could not save "<<name<<endl;
    }
    file.close();

    return save_okay;
}

template<typename KT, typename VT>
bool
BlockDiagMat<KT,VT>::save(std::ofstream& file) const
{
    bool save_okay = file.good();
    if (!save_okay)
    {
        cerr << "BlockDiagMat<KT,VT>::save(): bad file handle"<<endl;
        return false;
    }
    uint Nsec=this->size();

//    file << "BLOCKDIAGMAT" << '\n';
    file << GetHeader(this) << '\n';
    file.write(reinterpret_cast<const char*>(&Nsec), std::streamsize(sizeof(uint)));

    for (const auto& it : *this)
    {
//        save_okay = it.first.save(file);

        if (!it.first.save(file))
        {
            cerr << "BlockDiagMat<KT,VT>::save(): saving of Q = "<<it.first<<" failed" << endl;
            save_okay = false;
            break;
        }
        if (!it.second.save(file,arma_binary))
        {
            cerr << "BlockDiagMat<KT,VT>::save(): saving of mat failed" << endl;
            save_okay = false;
            break;
        }
    }
    return save_okay;
}


template<typename KT,typename VT>
bool
BlockDiagMat<KT,VT>::load(std::string name)
{
    std::ifstream file(name.c_str(), std::fstream::binary);
    if (!file.good())
    {
        cerr<<"BlockDiagMat<KT,VT>::load(): could not open "<<name<<endl;
        return false;
    }
    bool load_okay = this->load(file);
    if(!load_okay)
    {
        cerr<<"BlockDiagMat<KT,VT>::load(): could not load "<<name<<endl;
    }
    file.close();

    return load_okay;
}

template<typename KT, typename VT>
template<typename GO>
bool
BlockDiagMat<KT,VT>::load(std::ifstream& file, const GO& GroupObj)
{
    this->clear();
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"BlockDiagMat<KT,VT>::load(): bad file handle"<<endl;
        return false;
    }

    Mat<VT> tmp;
    std::string header;
    uint Nsec=0;

    file >> header;
    file.get(); /// somehow we need this to get over the newline stored to terminate header. WTF it took me 1 hr to find that out!!

//    cout<<GetHeader(this)<<" vs. "<<header<<endl;
//    if (header != "BLOCKDIAGMAT")
    if (header != GetHeader(this))
    {
        cerr << "wrong header "<<header<<", should be "<<GetHeader(this)<<endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&Nsec), std::streamsize(sizeof(uint)));
    for (uint i=0;i<Nsec; ++i)
    {
        KT Kin(file,GroupObj);
        if (!tmp.load(file,arma_binary))
        {
            cerr<<"BlockDiagMat<KT,VT>::load(): failed to load sector "<<Kin<<endl;
            load_okay = false;
            break;
        }
        this->emplace_hint(this->end(),Kin,tmp);
    }
    return load_okay;
}
/**< NON MEMBER FUNCTIONS */

/**< output stream operator overload for screen output */
template<typename KT,typename VT>
ostream&
operator<<(ostream& os, const BlockDiagMat<KT,VT>& M)
{
    if (!M.empty())
    {
        for (const auto& it : M)
        {
            os<<it.first<<":"<<endl;
            os<<it.second<<endl;
        }
    }
    else os<<"---"<<endl;
    return os;
}

/**< TRIVIAL ARITHMETICS FOR BLOCKDIAGMATS ***********************************************************/

template<typename KT,typename VT>
inline
BlockDiagMat<KT,VT>
trans(const BlockDiagMat<KT,VT>& in)
{
    BlockDiagMat<KT,VT> out(in);
    for (auto& it : out) it.second.t();
    return out;
}


template<typename KT,typename VT>
inline
Real
norm(const BlockDiagMat<KT,VT>& in)
{
    #ifndef NDEBUG
    if (in.empty()) cerr<<"norm: in is empty"<<endl;
    #endif // NDEBUG

    Real nrm2=0;
    for (const auto& it : in) nrm2+=abs(dot(it.second,it.second));
//    for (const auto& it : in) nrm2+=pow(norm(it.second,"fro"),2);
    return sqrt(nrm2);
}

template<typename KT,typename VT>
inline
Real
norm_inf(const BlockDiagMat<KT,VT>& in)
{
    #ifndef NDEBUG
    if (in.empty()) cerr<<"norm: in is empty"<<endl;
    #endif // NDEBUG

    Real nrmi=0;
    for (const auto& it : in) nrmi = max(nrmi,abs(it.second).max());
//    for (const auto& it : in) nrm2+=pow(norm(it.second,"fro"),2);
    return nrmi;
}

/// multiplication with scalars
template<typename KT,typename VTM, typename VTx>
inline
BlockDiagMat<KT,typename promote_type<VTM,VTx>::result>
operator*(VTx x, const BlockDiagMat<KT,VTM>& in)
{
    BlockDiagMat<KT,typename promote_type<VTM,VTx>::result> out(in);
    return out*=x;
}

template<typename KT,typename VTM, typename VTx>
inline
BlockDiagMat<KT,typename promote_type<VTM,VTx>::result>
operator*(const BlockDiagMat<KT,VTM>& in, VTx x)
{
    BlockDiagMat<KT,typename promote_type<VTM,VTx>::result> out(in);
    return out*=x;
}

/// template for same type additions
template<typename KT,typename VT>
inline
BlockDiagMat<KT,VT>
operator+(BlockDiagMat<KT,VT> lhs, const BlockDiagMat<KT,VT>& rhs)
{
    return lhs+=rhs;
}

/// general template for mixed type additions
template<typename KT,typename VTL, typename VTR>
inline
BlockDiagMat<KT,typename promote_type<VTL,VTR>::result>
operator+(const BlockDiagMat<KT,VTL>& lhs, const BlockDiagMat<KT,VTR>& rhs)
{
    BlockDiagMat<KT,typename promote_type<VTL,VTR>::result> out(lhs);
    return out+=rhs;
}

/// specialization for mixed complex/real additions
template<typename KT>
inline
BlockDiagMat<KT,Complex>
operator+(const BlockDiagMat<KT,Complex>& lhs, const BlockDiagMat<KT,Real>& rhs)
{
    BlockDiagMat<KT,Complex> out(rhs);
    return out+=lhs;
}

/// specialization for mixed complex/real additions
template<typename KT>
inline
BlockDiagMat<KT,Complex>
operator+(const BlockDiagMat<KT,Real>& lhs, const BlockDiagMat<KT,Complex>& rhs)
{
    BlockDiagMat<KT,Complex> out(lhs);
    return out+=rhs;
}

/// general template for mixed type additions
template<typename KT,typename VTL, typename VTR>
inline
BlockDiagMat<KT,typename promote_type<VTL,VTR>::result>
operator-(const BlockDiagMat<KT,VTL>& lhs, const BlockDiagMat<KT,VTR>& rhs)
{
    BlockDiagMat<KT,typename promote_type<VTL,VTR>::result> out(lhs);
    return out-=rhs;
}

/// specialization for same type additions
template<typename KT,typename VT>
inline
BlockDiagMat<KT,VT>
operator-(BlockDiagMat<KT,VT> lhs, const BlockDiagMat<KT,VT>& rhs)
{
    return lhs-=rhs;
}

/// specialization for mixed complex/real additions
template<typename KT>
inline
BlockDiagMat<KT,Complex>
operator-(const BlockDiagMat<KT,Complex>& lhs, const BlockDiagMat<KT,Real>& rhs)
{
    BlockDiagMat<KT,Complex> out(lhs);
    return out -= BlockDiagMat<KT,Complex>(rhs);
}

/// specialization for mixed complex/real additions
template<typename KT>
inline
BlockDiagMat<KT,Complex>
operator-(const BlockDiagMat<KT,Real>& lhs, const BlockDiagMat<KT,Complex>& rhs)
{
    BlockDiagMat<KT,Complex> out(lhs);
    return out-=rhs;
}
template<typename KT,typename VT>
inline
VT
trace(const BlockDiagMat<KT,VT>& mat)
{
    VT out=0;
    for (const auto& it : mat) out+=trace(it.second);
    return out;
}

/**< BASIC MULTIPLICATION AND DIVISION BY DIAGONAL MATRICES (E.G. SCHMIDT VALUES) ******************************************************/

template<typename KT,typename VT>
BlockDiagMat<KT,VT>&
MultBlockDiagMatLamLeft(const BlockLam<KT>& lam, BlockDiagMat<KT,VT>& in)
{
    if (!lam.empty())
    {
        for (auto& matit : in)
        {
            const auto lamit = lam.find(matit.first);
            if (lamit != lam.end()) MultMatLamLeft(lamit->second,matit.second);
        }
    }
    return in;
}


template<typename KT,typename VT>
BlockDiagMat<KT,VT>&
MultBlockDiagMatLamRight(BlockDiagMat<KT,VT>& in, const BlockLam<KT>& lam)
{
    if (!lam.empty())
    {
        for (auto& matit : in)
        {
            const auto lamit = lam.find(matit.first);
            if (lamit!=lam.end()) matit.second < (lamit->second);
        }
    }
    return in;
}

template<typename KT,typename VT>
BlockDiagMat<KT,VT>&
DivBlockDiagMatLamLeft(const BlockLam<KT>& lam,BlockDiagMat<KT,VT>& in)
{
    if (!lam.empty())
    {
        for (auto& matit : in)
        {
            const auto lamit = lam.find(matit.first);
            if (lamit!=lam.end()) (lamit->second) < matit.second;
        }
    }
    return in;
}


template<typename KT,typename VT>
BlockDiagMat<KT,VT>&
DivBlockDiagMatLamRight(BlockDiagMat<KT,VT>& in, const BlockLam<KT>& lam)
{
    if (!lam.empty())
    {
        for (auto& matit : in)
        {
            const auto lamit = lam.find(matit.first);
            if (lamit!=lam.end()) matit.second > (lamit->second);
        }
    }
    return in;
}

/// modifying
template<typename KT,typename VT>
inline BlockDiagMat<KT,VT>& operator>(const BlockLam<KT>& lam, BlockDiagMat<KT,VT>& in) {return MultBlockDiagMatLamLeft(lam,in);}

template<typename KT,typename VT>
inline BlockDiagMat<KT,VT>& operator<(BlockDiagMat<KT,VT>& in, const BlockLam<KT>& lam) {return MultBlockDiagMatLamRight(in,lam);}

template<typename KT,typename VT>
inline BlockDiagMat<KT,VT>& operator<(const BlockLam<KT>& lam, BlockDiagMat<KT,VT>& in) {return DivBlockDiagMatLamLeft(lam,in);}

template<typename KT,typename VT>
inline BlockDiagMat<KT,VT>& operator>(BlockDiagMat<KT,VT>& in, const BlockLam<KT>& lam) {return DivBlockDiagMatLamRight(in,lam);}


/// non-modifying
template<typename KT,typename VT>
inline BlockDiagMat<KT,VT> operator>>(const BlockLam<KT>& lam, BlockDiagMat<KT,VT> in) {return MultBlockDiagMatLamLeft(lam,in);}

template<typename KT,typename VT>
inline BlockDiagMat<KT,VT> operator<<(BlockDiagMat<KT,VT> in, const BlockLam<KT>& lam) {return MultBlockDiagMatLamRight(in,lam);}

template<typename KT,typename VT>
inline BlockDiagMat<KT,VT> operator<<(const BlockLam<KT>& lam, BlockDiagMat<KT,VT> in) {return DivBlockDiagMatLamLeft(lam,in);}

template<typename KT,typename VT>
inline BlockDiagMat<KT,VT> operator>>(BlockDiagMat<KT,VT> in, const BlockLam<KT>& lam) {return DivBlockDiagMatLamRight(in,lam);}

///// non-modifying
//template<typename KT,typename VT>
//inline BlockDiagMat<KT,VT> operator>>(const BlockLambda<KT>& lam, const BlockDiagMat<KT,VT>& in) {BlockDiagMat<KT,VT> out(in); return MultBlockDiagMatLamLeft(lam,out);}
//
//template<typename KT,typename VT>
//inline BlockDiagMat<KT,VT> operator<<(const BlockDiagMat<KT,VT>& in, const BlockLambda<KT>& lam) {BlockDiagMat<KT,VT> out(in); return MultBlockDiagMatLamRight(out,lam);}
//
//template<typename KT,typename VT>
//inline BlockDiagMat<KT,VT> operator<<(const BlockLambda<KT>& lam, const BlockDiagMat<KT,VT>& in) {BlockDiagMat<KT,VT> out(in); return DivBlockDiagMatLamLeft(lam,out);}
//
//template<typename KT,typename VT>
//inline BlockDiagMat<KT,VT> operator>>(const BlockDiagMat<KT,VT>& in, const BlockLambda<KT>& lam) {BlockDiagMat<KT,VT> out(in); return DivBlockDiagMatLamRight(out,lam);}

/**< addition and subtraction with (diagonal matrices of) Schmidt Values */

template<typename KT, typename VT>
inline
BlockDiagMat<KT,VT>
operator+(BlockDiagMat<KT,VT> mat, const BlockLam<KT>& lam) {return mat+=lam;}


template<typename KT, typename VT>
inline
BlockDiagMat<KT,VT>
operator-(BlockDiagMat<KT,VT> mat, const BlockLam<KT>& lam) {return mat-=lam;}

/**< SIMPLE GENERATION OF SPECIAL BLOCKDIAGMATS ************************************************************************************************/
template<typename KT,typename VT>
inline
BlockDiagMat<KT,VT>
eye(const dim_map<KT>& dims)
{
    BlockDiagMat<KT,VT> mat;
    for (const auto& dimit : dims) mat.emplace_hint(mat.end(),dimit.first,eye(dims.second,dims.second));
    return mat;
}

template<typename KT>
inline
BlockDiagMat<KT,Real>
diagmat(const BlockLam<KT>& lam) {return BlockDiagMat<KT,Real>(lam);}

/**< MORE ADVANCED FUNCTIONS OF BLOCKDIAGMATS ************************************************************************************************/
template<typename KT, typename VT>
bool
eig_sym(BlockLam<KT>& EVal, BlockDiagMat<KT,VT>& EVec, const BlockDiagMat<KT,VT>& X)
{
    RVecType D;
    Mat<VT> U;
    EVal.clear();
    EVec.clear();
    bool conv=true;
    for (const auto& it : X)
    {
        conv = conv && eig_sym(D,U,it.second);
        EVal.emplace_hint(EVal.end(),it.first,D);
        EVec.emplace_hint(EVec.end(),it.first,U);
    }
    return conv;
}

template<typename KT, typename VT>
BlockDiagMat<KT,VT>
chol(const BlockDiagMat<KT,VT>& X, const char* layout = "upper")
{
    BlockDiagMat<KT,VT> R;
    for (const auto& it : X) R.emplace_hint(R.end(),it.first,chol(it.second,layout));
    return R;
}

/**< Create Identity matrices */
template<typename VT, typename KT> /// order of template params matters, WTF!!
BlockDiagMat<KT,VT>
eye(const dim_map<KT>& dims)
{
    BlockDiagMat<KT,VT> ID;
    for (const auto& dimit : dims) ID.emplace_hint(ID.end(),dimit.first,eye(dimit.second,dimit.second));
    return ID;
}

template<typename VT, typename KT>
BlockDiagMat<KT,VT>
eye(const dimpair_map<KT>& dims)
{
    BlockDiagMat<KT,VT> ID;
    for (const auto& dimit : dims) ID.emplace_hint(ID.end(),dimit.first,eye(dimit.second.first,dimit.second.second));
    return ID;
}

template<typename VT, typename KT> /// order of template params matters, WTF!!
BlockDiagMat<KT,VT>
eye(const dimpair_vec<KT>& dims)
{
    BlockDiagMat<KT,VT> ID;
    for (const auto& dimit : dims) ID.emplace_hint(ID.end(),get<0>(dimit),eye(get<1>(dimit),get<2>(dimit)));
    return ID;
}

template<typename KT, typename VT>
BlockDiagMat<KT,VT>
qr(BlockDiagMat<KT,VT>& R, const BlockDiagMat<KT,VT>& X, dirtype dir)
{
    R.clear();
    BlockDiagMat<KT,VT> Q;
    Mat<VT> Qtmp,Rtmp;

    if (dir == l)
    {
        for (const auto& it : X)
        {
//            qr_pos(Qtmp,Rtmp,it.second);
//            Q.emplace_hint(Q.end(),it.first,Qtmp);
//            R.emplace_hint(R.end(),it.first,Rtmp);
            auto qit = Q.emplace_hint(Q.end(),it.first,Mat<VT>());
            auto rit = R.emplace_hint(R.end(),it.first,Mat<VT>());
            qr_pos(qit->second,rit->second,it.second);

        }
    }
    else if (dir == r)
    {
        for (const auto& it : X)
        {
//            qr_pos(Qtmp,Rtmp,it.second.t().eval());
//            Q.emplace_hint(Q.end(),it.first,Qtmp.t());
//            R.emplace_hint(R.end(),it.first,Rtmp.t());
            auto qit = Q.emplace_hint(Q.end(),it.first,Mat<VT>());
            auto rit = R.emplace_hint(R.end(),it.first,Mat<VT>());
            qr_pos(qit->second,rit->second,it.second.t().eval());
            inplace_trans(qit->second);
            inplace_trans(rit->second);
        }
    }
    else throw std::logic_error("BlockDiagMat<KT,VT> qr(): wrong direction specified!");
    return Q;
}

template<typename KT, typename VT>
inline
void
qr(BlockDiagMat<KT,VT>& Q, BlockDiagMat<KT,VT>& R, const BlockDiagMat<KT,VT>& X, dirtype dir) {Q = qr(R,X,dir);}


/// only works for Hermitian matrices!!
template<typename KT, typename VT>
inline
BlockLam<KT>
eig_sym(const BlockDiagMat<KT,VT>& X)
{
    BlockLam<KT> D;
    for (const auto& it : X) D.emplace_hint(D.end(),it.first,eig_sym(it.second));
    return D;
}

template<typename KT, typename VT>
BlockDiagMat<KT,VT>
eig_sym(BlockLam<KT>& D, const BlockDiagMat<KT,VT>& X)
{
    BlockDiagMat<KT,VT> U;
    D.clear();

    Mat<VT> Utmp;
    RVecType Dtmp;
    for (const auto& it : X)
    {
        eig_sym(Dtmp,Utmp,it.second);
        D.emplace_hint(D.end(),it.first,Dtmp);
        U.emplace_hint(U.end(),it.first,Utmp);
    }
    return U;
}

template<typename KT, typename VT>
inline
void
eig_sym(BlockDiagMat<KT,VT>& U, BlockLam<KT>& D, const BlockDiagMat<KT,VT>& X) {U = eig_sym(D,X);}
//{
//    U.clear();
//    D.clear();
//
//    Mat<VT> Utmp;
//    RVecType Dtmp;
//    for (const auto& it : X)
//    {
//        eig_sym(Dtmp,Utmp,it.second);
//        D.emplace_hint(D.end(),it.first,Dtmp);
//        U.emplace_hint(U.end(),it.first,Utmp);
//    }
//}

template<typename KT, typename VT>
inline
BlockLam<KT>
svd(const BlockDiagMat<KT,VT>& X)
{
    BlockLam<KT> S;
    for (const auto& it : X) S.emplace_hint(S.end(),it.first,svd(it.second));
    return S;
}

template<typename KT, typename VT>
void
svd(BlockDiagMat<KT,VT>& U, BlockLam<KT>& S, BlockDiagMat<KT,VT>& V, const BlockDiagMat<KT,VT>& X, const char* mode = "both")
{
    U.clear();
    V.clear();
    S.clear();

    Mat<VT> Utmp,Vtmp;
    RVecType Stmp;
    for (const auto& it : X)
    {
        svd_econ(Utmp,Stmp,Vtmp,it.second,mode);
        U.emplace_hint(U.end(),it.first,Utmp);
        V.emplace_hint(V.end(),it.first,Vtmp);
        S.emplace_hint(S.end(),it.first,Stmp);
    }
}


//template<typename KT, typename VT>
//void
//svd_econ(BlockDiagMat<KT,VT>& U, BlockLambda<KT>& S, BlockDiagMat<KT,VT>& V, const BlockDiagMat<KT,VT>& X, const dimpair_map<KT>& dims)
//{
//
//}



/**< MULTIPLICATION ROUTINES FOR BLOCK MATRICES ****************************************************/

/**< multiplication operator for two BlockMats */
/// for now we don't need a modifying version of this
template<typename KT,typename VT1, typename VT2>
BlockMat<KT,typename promote_type<VT1,VT2>::result>
operator*(const BlockMat<KT,VT1>& lhs, const BlockMat<KT,VT2>& rhs)
{
    BlockMat<KT,typename promote_type<VT1,VT2>::result> out;
    for (const auto& lit : lhs)
    {
        const auto rit = rhs.find(Qout(lit));
        if (rit!=rhs.end()) out.emplace_hint(out.end(),Qin(lit),std::make_pair(Qout(*rit) , QMat(lit) * QMat(*rit) ));
    }
    return out;
}

/**< multiplication operator for BlockMat (left) and BlockDiagMat (right) */
/// for now we don't need a modifying version of this
template<typename KT,typename VT1,typename VT2> /// it could in principle happen that BlockMat is complex and BlockDiagMat is real
BlockMat<KT,typename promote_type<VT1,VT2>::result>
operator*(const BlockMat<KT,VT1>& lhs, const BlockDiagMat<KT,VT2>& rhs)
{
    BlockMat<KT,typename promote_type<VT1,VT2>::result> out;
    for (const auto& lit : lhs)
    {
        const auto rit = rhs.find(Qout(lit));
        if (rit!=rhs.end())
        {
            out.emplace_hint(out.end(),
                             Qin(lit),
                             QMatPair<KT,typename promote_type<VT1,VT2>::result>(Qout(lit) , QMat(lit) * rit->second)
                             );
        }
    }
    return out;
}

/**< multiplication operator for BlockDiagMat (left) and BlockMat (right) */
/// for now we don't need a modifying version of this
template<typename KT,typename VT1,typename VT2>
BlockMat<KT,typename promote_type<VT1,VT2>::result>
operator*(const BlockDiagMat<KT,VT1>& lhs, const BlockMat<KT,VT2>& rhs)
{
    BlockMat<KT,typename promote_type<VT1,VT2>::result> out;
    for (const auto& rit : rhs)
    {
        const auto lit = lhs.find(Qin(rit));
        if (lit!=lhs.end())
        {
            out.emplace_hint(out.end(),
                             Qin(rit),
                             QMatPair<KT,typename promote_type<VT1,VT2>::result>(Qout(rit) , lit->second * QMat(rit))
                             );
        }
    }
    return out;
}

/**< multiplication operator for two BlockDiagMats */
/// for now we don't need a modifying version of this
template<typename KT,typename VT1, typename VT2>
BlockDiagMat<KT,typename promote_type<VT1,VT2>::result>
operator*(const BlockDiagMat<KT,VT1>& lhs, const BlockDiagMat<KT,VT2>& rhs)
{
    BlockDiagMat<KT,typename promote_type<VT1,VT2>::result> out;
    for (const auto& lit : lhs)
    {
        const auto rit = rhs.find(lit.first);
        if (rit!=rhs.end()) out.emplace_hint(out.end(),lit.first, lit.second*rit->second );
    }
    return out;
}


/**< ARRAY CONTAINERS FOR MULTI SITE PURPOSES ***********************************************************************************/
template<typename KT>
class BlockLamArray : public std::deque<BlockLam<KT> >
{
    public:
    typedef KT key_type;

    BlockLamArray() = default;
    BlockLamArray(uint N):std::deque<BlockLam<KT> >(N) {};

    inline void print(const std::string str="") const;
    inline void ShowDims(const std::string str="") const;

    /// DiskIO
    bool save(std::string name) const;
    bool save(std::ofstream& file) const;
    bool load(std::string name);
    template<typename GO>
    bool load(std::ifstream& file, const GO& GroupObj);
};


template<typename KT>
inline
void
BlockLamArray<KT>::print(const std::string str) const
{
    cout<<endl;
    if (str!="") cout<<str<<endl;
    uint ct=1;
    for (const auto& it : *this) it.print("site "+std::to_string(ct++));
}


template<typename KT>
inline
void
BlockLamArray<KT>::ShowDims(const std::string str) const
{
    cout<<endl;
    if (str!="") cout<<str<<endl;
    uint ct=1;
    for (const auto& it : *this) it.ShowDims("site "+std::to_string(ct++));
}



template<typename KT>
bool
BlockLamArray<KT>::save(std::string name) const
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay = this->save(file);
    if (!save_okay)
    {
        cerr << "BlockLamArray<KT>::save(): could not save "<<name<<endl;
    }
    file.close();

    return save_okay;
}

template<typename KT>
bool
BlockLamArray<KT>::save(std::ofstream& file) const
{
    bool save_okay = file.good();
    if (!save_okay)
    {
        cerr << "BlockLamArray<KT>::save(): bad file handle"<<endl;
        return false;
    }
    uint Nsites = this->size();

//    file << "BLOCKLAMARRAY" << '\n';
    file << GetHeader(this) << '\n';
    file.write(reinterpret_cast<const char*>(&Nsites), std::streamsize(sizeof(uint)));

    for (const auto& it : *this) if(!it.save(file)) return false;
    return true;
}

template<typename KT>
bool
BlockLamArray<KT>::load(std::string name)
{
    std::ifstream file(name.c_str(), std::fstream::binary);
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"could not open "<<name<<endl;
        return false;
    }

    load_okay = this->load(file);
    if(!load_okay)
    {
        cerr<<"BlockLamArray<KT>::load(): could not load "<<name<<endl;
    }
    file.close();

    return load_okay;
}

template<typename KT>
template<typename GO>
bool
BlockLamArray<KT>::load(std::ifstream& file, const GO& GroupObj)
{
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"BlockLamArray<KT>::load(): bad file handle"<<endl;
        return false;
    }
    this->clear();

    std::string header;
    file >> header;
    file.get(); /// somehow we need this to get over the newline stored to terminate header. WTF it took me 1 hr to find that out!!
//    if (header != "BLOCKLAMARRAY")
    if (header != GetHeader(this))
    {
        cerr << "wrong header "<<header<<", should be "<<GetHeader(this)<<endl;
        return false;
    }
    uint Nsites;
    file.read(reinterpret_cast<char*>(&Nsites), std::streamsize(sizeof(uint)));
    this->resize(Nsites);

    for (uint n=0; n<Nsites; ++n)
    {
        if(!this->at(n).load(file,GroupObj))
        {
            cerr << "BlockLamArray<KT>::load(): failed to load site index "<<n<<endl;
            load_okay = false;
            break;
        }
    }
    return load_okay;
}

template<typename KT>
ostream&
operator<<(ostream& os, const BlockLamArray<KT>& lam)
{
    for (uint n=0;n<lam.size();++n)
    {
        os<<"site "<<n+1<<endl<<endl;
        os<<lam[n];
    }
    return os;
}

/**< BlockDiagArray *************************************************************************************++*/

template<typename KT, typename VT>
class BlockDiagMatArray : public std::deque<BlockDiagMat<KT,VT> >
{
    public:
    typedef KT key_type;
    typedef VT scalar_type;
    typedef std::function<BlockDiagMat<KT,VT> (const BlockDiagMat<KT,VT>&)> fun_type;

    BlockDiagMatArray() = default;
    BlockDiagMatArray(uint N):std::deque<BlockDiagMat<KT,VT> >(N) {};


    BlockDiagMatArray ShiftQN(const KT& K) const;
    BlockDiagMatArray FlipQN(const std::vector<bool>& which = std::vector<bool>()) const;

    inline void print(const std::string str="") const;
    inline void ShowDims(const std::string str="") const;

    /// general method to apply single BlockDiagMat -> BlockDiagMat functions to every single element of the MBlockDiagMatArray
    BlockDiagMatArray ApplyFun(const std::function<BlockDiagMat<KT,VT> (const BlockDiagMat<KT,VT>&)>& F) const
    {
        BlockDiagMatArray out;
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



template<typename KT,typename VT>
BlockDiagMatArray<KT,VT>
BlockDiagMatArray<KT,VT>::ShiftQN(const KT& K) const
{
    BlockDiagMatArray<KT,VT> out;
    for (const auto& it : *this) out.emplace_back(it.ShiftQN(K));
    return out;
}

template<typename KT,typename VT>
BlockDiagMatArray<KT,VT>
BlockDiagMatArray<KT,VT>::FlipQN(const std::vector<bool>& which) const
{
    BlockDiagMatArray<KT,VT> out;
    for (const auto& it : *this) out.emplace_back(it.FlipQN(which));
    return out;
}

template<typename KT,typename VT>
inline
void
BlockDiagMatArray<KT,VT>::print(const std::string str) const
{
    cout<<endl;
    if (str!="") cout<<str<<endl;
    uint ct=1;
    for (const auto& it : *this) it.print("site "+std::to_string(ct++));
}


template<typename KT, typename VT>
inline
void
BlockDiagMatArray<KT,VT>::ShowDims(const std::string str) const
{
    cout<<endl;
    if (str!="") cout<<str<<endl;
    uint ct=1;
    for (const auto& it : *this) it.ShowDims("site "+std::to_string(ct++));
}


template<typename KT, typename VT>
bool
BlockDiagMatArray<KT,VT>::save(std::string name) const
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay = this->save(file);
    if (!save_okay) cerr << "BlockDiagArray<KT,VT>::save(): could not save "<<name<<endl;
    file.close();

    return save_okay;
}

template<typename KT, typename VT>
bool
BlockDiagMatArray<KT,VT>::save(std::ofstream& file) const
{
    bool save_okay = file.good();
    if (!save_okay)
    {
        cerr << "BlockDiagArray<KT,VT>::save(): bad file handle"<<endl;
        return false;
    }
    uint Nsites = this->size();

//    file << "BLOCKDIAGMATARRAY" << '\n';
    file << GetHeader(this) << '\n';
    file.write(reinterpret_cast<const char*>(&Nsites), std::streamsize(sizeof(uint)));

    for (const auto& it : *this) if(!it.save(file)) return false;
    return true;
}


template<typename KT,typename VT>
bool
BlockDiagMatArray<KT,VT>::load(std::string name)
{
    std::ifstream file(name.c_str(), std::fstream::binary);
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"could not open "<<name<<endl;
        return false;
    }

    load_okay = this->load(file);
    if(!load_okay)
    {
        cerr<<"BlockDiagArray<KT,VT>::load(): could not load "<<name<<endl;
    }
    file.close();

    return load_okay;
}


template<typename KT,typename VT>
template<typename GO>
bool
BlockDiagMatArray<KT,VT>::load(std::ifstream& file, const GO& GroupObj)
{
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"BlockDiagMatArray<KT,VT>::load(): bad file handle"<<endl;
        return false;
    }
    this->clear();

    std::string header;
    file >> header;
    file.get(); /// somehow we need this to get over the newline stored to terminate header. WTF it took me 1 hr to find that out!!
//    if (header != "BLOCKDIAGMATARRAY")
    if (header != GetHeader(this))
    {
        cerr << "wrong header "<<header<<", should be "<<GetHeader(this)<<endl;
        return false;
    }
    uint Nsites;
    file.read(reinterpret_cast<char*>(&Nsites), std::streamsize(sizeof(uint)));
    this->resize(Nsites);

    for (uint n=0; n<Nsites; ++n)
    {
        if(!this->at(n).load(file,GroupObj))
        {
            cerr << "BlockDiagMatArray<KT,VT>::load(): failed to load site index "<<n<<endl;
            load_okay = false;
            break;
        }
    }
    return load_okay;
}

template<typename KT,typename VT>
ostream&
operator<<(ostream& os, const BlockDiagMatArray<KT,VT>& mat)
{
    for (uint n=0;n<mat.size();++n)
    {
        os<<"site "<<n+1<<endl<<endl;
        os<<mat[n];
    }
    return os;
}

/**< BlockMatArray *************************************************************************************++*/

template<typename KT, typename VT>
class BlockMatArray : public std::deque<BlockMat<KT,VT> >
{
public:
    typedef KT key_type;
    typedef VT scalar_type;
    typedef std::function<BlockMat<KT,VT> (const BlockMat<KT,VT>&)> fun_type;

    BlockMatArray() = default;
    BlockMatArray(uint N):std::deque<BlockMat<KT,VT> >(N) {};

    /// construct from vector and dims
    template<typename DIMT>
    BlockMatArray(const VT* vec, const std::vector<DIMT>& dims, bool copy_aux_mem=true, bool strict=false);
    template<typename DIMT>
    BlockMatArray(      VT* vec, const std::vector<DIMT>& dims, bool copy_aux_mem=true, bool strict=false);

    template<typename DIMT>
    BlockMatArray(const Col<VT>& vec, const std::vector<DIMT>& dims, bool copy_aux_mem=true, bool strict=false);
    template<typename DIMT>
    BlockMatArray(      Col<VT>& vec, const std::vector<DIMT>& dims, bool copy_aux_mem=true, bool strict=false);

    template<typename fill_type>
    BlockMatArray(const std::vector<dimkeypair_vec<KT> >& dims, const fill::fill_class<fill_type>& filler)
    {
        for (const auto& it : dims) this->emplace_back(BlockMat<KT,VT>(it,filler));
    }

    inline void Fill(VT val) {for (auto& it : *this) it.Fill(val);};

    inline void print(const std::string str="") const;
    inline void ShowDims(const std::string str="") const;

    /// GETTERS, HELPERS
    uint GetNElem() const {uint nelem = 0; for (const auto& it : *this) nelem += it.GetNElem(); return nelem;};
    Col<VT> Vectorize() const;


    /// general method to apply single BlockDiagMat -> BlockDiagMat functions to every single element of the MBlockDiagMatArray
    BlockMatArray ApplyFun(const std::function<BlockMat<KT,VT> (const BlockMat<KT,VT>&)>& F) const
    {
        BlockMatArray out;
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

template<typename KT,typename VT>
template<typename DIMT>
BlockMatArray<KT,VT>::BlockMatArray(const VT* vec, const std::vector<DIMT>& dims, bool copy_aux_mem, bool strict)
{
    /// MAKE SURE THAT THE DIMENSION OF VEC IS AT LEAST sum_n dims[n].GetNElem()
    uint pos = 0;
    for (const auto& dimit : dims)
    {
        this->emplace_back(BlockMat<KT,VT>(&vec[pos],dimit,copy_aux_mem,strict));
        pos += dimit.GetNElem();
    }
}


template<typename KT,typename VT>
template<typename DIMT>
BlockMatArray<KT,VT>::BlockMatArray(      VT* vec, const std::vector<DIMT>& dims, bool copy_aux_mem, bool strict)
{
    /// MAKE SURE THAT THE DIMENSION OF VEC IS AT LEAST sum_n dims[n].GetNElem()
    uint pos = 0;
    for (const auto& dimit : dims)
    {
        this->emplace_back(BlockMat<KT,VT>(&vec[pos],dimit,copy_aux_mem,strict));
        pos += dimit.GetNElem();
    }
}

template<typename KT,typename VT>
template<typename DIMT>
BlockMatArray<KT,VT>::BlockMatArray(const Col<VT>& vec, const std::vector<DIMT>& dims, bool copy_aux_mem, bool strict):
    BlockMatArray(vec.memptr(),dims,copy_aux_mem,strict)
{
    #ifndef NDEBUG
    uint mtot = 0;
    for (const auto& it : dims) mtot += it.GetNElem();
    assert(vec.size() == mtot && "BlockMatArray(const Col& vec, const vector<DIMT>& dims, bool copy_aux_mem, bool strict): vec and dims need to account for the same number of elements");
    #endif // NDEBUG
}

template<typename KT,typename VT>
template<typename DIMT>
BlockMatArray<KT,VT>::BlockMatArray(      Col<VT>& vec, const std::vector<DIMT>& dims, bool copy_aux_mem, bool strict):
    BlockMatArray(vec.memptr(),dims,copy_aux_mem,strict)
{
    #ifndef NDEBUG
    uint mtot = 0;
    for (const auto& it : dims) mtot += it.GetNElem();
    assert(vec.size() == mtot && "BlockMatArray(Col& vec, const vector<DIMT>& dims, bool copy_aux_mem, bool strict): vec and dims need to account for the same number of elements");
    #endif // NDEBUG
}


template<typename KT,typename VT>
Col<VT>
BlockMatArray<KT,VT>::Vectorize() const
{
    Col<VT> out(this->GetNElem());
    VT* outmem = out.memptr();
    uint pos=0,dpos=0;

    for (const auto& Vit : *this)
    {
        for (const auto& it : Vit)
        {
            dpos = QMat(it).n_rows*QMat(it).n_cols;
            memcpy(&outmem[pos],QMat(it).memptr(),dpos*sizeof(*outmem));
            pos += dpos;
        }
    }
    return out;
}


template<typename KT, typename VT>
bool
BlockMatArray<KT,VT>::save(std::string name) const
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay = this->save(file);
    if (!save_okay) cerr << "BlockMatArray<KT,VT>::save(): could not save "<<name<<endl;
    file.close();

    return save_okay;
}

template<typename KT, typename VT>
bool
BlockMatArray<KT,VT>::save(std::ofstream& file) const
{
    bool save_okay = file.good();
    if (!save_okay)
    {
        cerr << "BlockMatArray<KT,VT>::save(): bad file handle"<<endl;
        return false;
    }
    uint Nsites = this->size();

    file << GetHeader(this) << '\n';
    file.write(reinterpret_cast<const char*>(&Nsites), std::streamsize(sizeof(uint)));

    for (const auto& it : *this) if(!it.save(file)) return false;
    return true;
}


template<typename KT,typename VT>
inline
void
BlockMatArray<KT,VT>::print(const std::string str) const
{
    cout<<endl;
    if (str!="") cout<<str<<endl;
    uint ct=1;
    for (const auto& it : *this) it.print("site "+std::to_string(ct++));
}


template<typename KT, typename VT>
inline
void
BlockMatArray<KT,VT>::ShowDims(const std::string str) const
{
    cout<<endl;
    if (str!="") cout<<str<<endl;
    uint ct=1;
    for (const auto& it : *this) it.ShowDims("site "+std::to_string(ct++));
}

/**< HELPER FUNCTIONS FOR HEADERS FOR SAVING AND LOADING */
template<typename KT>
string
GetHeader(const BlockLam<KT>*) {return "BLOCKLAM";}

template<typename KT>
string
GetHeader(const BlockLamArray<KT>*) {return "BLOCKLAMARRAY";}

template<typename KT>
string
GetHeader(const BlockDiagMat<KT,Real>*) {return "RBLOCKDIAGMAT";}

template<typename KT>
string
GetHeader(const BlockDiagMat<KT,Complex>*) {return "CBLOCKDIAGMAT";}

template<typename KT>
string
GetHeader(const BlockMat<KT,Real>*) {return "RBLOCKMAT";}

template<typename KT>
string
GetHeader(const BlockMat<KT,Complex>*) {return "CBLOCKMAT";}

template<typename KT>
string
GetHeader(const BlockMatArray<KT,Real>*) {return "RBLOCKMATARRAY";}

template<typename KT>
string
GetHeader(const BlockMatArray<KT,Complex>*) {return "CBLOCKMATARRAY";}

template<typename KT>
string
GetHeader(const BlockDiagMatArray<KT,Real>*) {return "RBLOCKDIAGMATARRAY";}

template<typename KT>
string
GetHeader(const BlockDiagMatArray<KT,Complex>*) {return "CBLOCKDIAGMATARRAY";}


/** GENERIC SAVE/LOAD FUNCTIONS TO SAVE VECTORS OF BLOCKOBJS *********************************************************************/
template<typename BLKT>
bool
save(const std::vector<BLKT>& X, std::string name)
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay = save(X,file);
    if (!save_okay) cerr << "save<BLK>(const vector<BLKT>& X, string name): could not save "<<name<<endl;
    file.close();

    return save_okay;
}

template<typename BLKT>
bool
save(const std::vector<BLKT>& X, std::ofstream& file)
{
    bool save_okay = file.good();
    if (!save_okay)
    {
        cerr << "save<BLK>(const vector<BLKT>& X, ofstream& file): bad file handle"<<endl;
        return false;
    }
    uint N = X.size();
    file << GetHeader(&X.front()) << "VEC\n";
    file.write(reinterpret_cast<const char*>(&N), std::streamsize(sizeof(uint)));

    for (const auto& it : X) if(!it.save(file)) return false;
    return true;
}

#endif // BLOCK_OBJ_H
