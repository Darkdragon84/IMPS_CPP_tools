#ifndef OP_TYPES_H_
#define OP_TYPES_H_

//#include <map>
//#include <cmath>
//#include <iostream>
//#include <assert.h>
//
//#include "arma_typedefs.h"

using std::abs;
using std::cout;
using std::endl;

template<typename VT>
class
SparseOperator : public SpMat<VT>
{
public:
    SparseOperator() {}; /// std c'tor
    SparseOperator(uint d, uint NSites, const std::string& name=""):SpMat<VT>((uint)pow(d,NSites),(uint)pow(d,NSites)),d_(d),NSites_(NSites),name_(name){}

//    SparseOperator(const SparseOperator& other):SpMat<VT>(other),d_(other.GetLocalDim()),NSites_(other.GetNSites()){} /// copy c'tor
//    SparseOperator(SparseOperator&& other):SpMat<VT>(std::move(other)),d_(other.GetLocalDim()),NSites_(other.GetNSites()){} /// move c'tor

    SparseOperator(const SpMat<VT>& mat, uint d, uint NSites, const std::string& name=""):SpMat<VT>(mat),d_(d),NSites_(NSites),name_(name){} /// copy from SpMat
    SparseOperator(SpMat<VT>&& mat, uint d, uint NSites, const std::string& name=""):SpMat<VT>(std::move(mat)),d_(d),NSites_(NSites),name_(name){} /// move from SpMat
    SparseOperator(uint d, uint NSites, const umat& locations, const Col<VT>& values, const std::string& name="")
        :SpMat<VT>(locations,values,(uint)pow(d,NSites),(uint)pow(d,NSites)),d_(d),NSites_(NSites),name_(name){}

    inline void SetDims(uint d, uint NSites);
    inline uint GetLocalDim() const {return d_;}
    inline uint GetNSites() const {return NSites_;}
    inline uint GetTotalDim() const {return (uint)pow(d_,NSites_);}
    inline std::string GetName() const {return name_;}
    inline SparseOperator<VT>& SetName(const std::string& name) {name_ = name; return *this;}

    inline SparseOperator<VT>& operator+=(const SparseOperator<VT>& other);
    inline SparseOperator<VT>& operator-=(const SparseOperator<VT>& other);
    inline SparseOperator<VT>& operator*=(const SparseOperator<VT>& other);
    inline SparseOperator<VT>& operator*=(VT scalar);

    inline SparseOperator<VT> t() const {return SparseOperator<VT>(this->SpMat<VT>::t(),this->GetLocalDim(),this->GetNSites());}

    /// I don't remember why I declared these operators as friends, maybe in view of mixing template parameters
    friend inline SparseOperator<VT> operator*(VT scalar, SparseOperator<VT> op) {return op*=scalar;}
    friend inline SparseOperator<VT> operator*(SparseOperator<VT> op, VT scalar) {return op*=scalar;}

    friend inline SparseOperator<VT> operator*(SparseOperator<VT> lhs,const SparseOperator<VT>& rhs) {return lhs*=rhs;}
    friend inline SparseOperator<VT> operator+(SparseOperator<VT> lhs,const SparseOperator<VT>& rhs) {return lhs+=rhs;}
    friend inline SparseOperator<VT> operator-(SparseOperator<VT> lhs,const SparseOperator<VT>& rhs) {return lhs-=rhs;}
    friend inline SparseOperator<VT> operator-(SparseOperator<VT>& op){return op*=(-1.);}

    /// this is not necessary, we have a factory function down below:
//    static inline SparseOperator<VT> SpId(uint d, uint N) {return SparseOperator<VT>(speye((uint)pow(d,N),(uint)pow(d,N)),d,N,"Id");}

protected:
    uint d_,NSites_;
    std::string name_;
};

using RSpOp = SparseOperator<Real>;
using CSpOp = SparseOperator<Complex>;

template<typename VT>
inline void
SparseOperator<VT>::SetDims(uint d, uint NSites)
{
    d_=d;
    NSites_=NSites;
    this->resize(uint(pow(d,NSites)),uint(pow(d,NSites)));
}

template<typename VT>
inline
SparseOperator<VT>&
SparseOperator<VT>::operator+=(const SparseOperator<VT>& other)
{
    assert(this->GetLocalDim() == other.GetLocalDim() && this->GetNSites() == other.GetNSites());
    this->SpMat<VT>::operator+=(other);
    return *this;
}

template<typename VT>
inline
SparseOperator<VT>&
SparseOperator<VT>::operator-=(const SparseOperator<VT>& other)
{
    assert(this->GetLocalDim() == other.GetLocalDim() && this->GetNSites() == other.GetNSites());
    this->SpMat<VT>::operator-=(other);
    return *this;
}

template<typename VT>
inline
SparseOperator<VT>&
SparseOperator<VT>::operator*=(const SparseOperator<VT>& other)
{
    assert(this->GetLocalDim() == other.GetLocalDim() && this->GetNSites() == other.GetNSites());
    this->SpMat<VT>::operator*=(other);
    return *this;
}

template<typename VT>
inline
SparseOperator<VT>&
SparseOperator<VT>::operator*=(VT scalar)
{
    this->SpMat<VT>::operator*=(scalar);
    return *this;
}

/**< external functions ****************************************************************** */

template<typename VT>
SparseOperator<VT>
kron(const SparseOperator<VT>& lhs, const SparseOperator<VT>& rhs)
{
    assert(lhs.GetLocalDim() == rhs.GetLocalDim());
    uint rdim = rhs.GetTotalDim();
    uint newdim = lhs.n_nonzero * rhs.n_nonzero;
    umat locs(2,newdim);
    Col<VT> vals(newdim);

    uint ct=0;
    for (typename SparseOperator<VT>::const_iterator lhit=lhs.begin(); lhit!=lhs.end(); ++lhit)
    {
        for (typename SparseOperator<VT>::const_iterator rhit=rhs.begin(); rhit!=rhs.end(); ++rhit)
        {
            locs(0,ct)=lhit.row()*rdim + rhit.row();
            locs(1,ct)=lhit.col()*rdim + rhit.col();
            vals(ct)=(*lhit)*(*rhit);
            ++ct;
        }
    }
    SparseOperator<VT> out(lhs.GetLocalDim(),lhs.GetNSites() + rhs.GetNSites(),locs,vals);
    return out;
}

template<typename VT>
inline
SparseOperator<VT>
expmat(const SparseOperator<VT>& in)
{
    return SparseOperator<VT>(in.GetLocalDim(),in.GetNSites(),SpMat<VT>(expmat(Mat<VT>(in))));
}

template<typename VT>
inline
SparseOperator<VT>
SpId(uint d, uint N)
{
    int dim = pow(d,N);
    return SparseOperator<VT>(speye(dim,dim),d,N);
}

/**< SOME SPECIAL OPERATORS *************************************************************** */
/**< reduced operators, i.e. in a mixture between fully virtual and physical */

template<typename MatT>
class RedOp : public std::vector<std::vector<MatT> > /// WATCH OUT, THE LAYOUT IS ROW MAJOR (MADE A MISTAKE IN THE BEGINNING AND NOW TOO LAZY TO CHANGE IT BACK)
{
public:
    typedef typename std::vector<std::vector<MatT> >::const_iterator const_row_it;
    typedef typename std::vector<MatT>::const_iterator const_col_it;
    typedef typename std::vector<std::vector<MatT> >::iterator row_it;
    typedef typename std::vector<MatT>::iterator col_it;

    RedOp(uint d, uint N=1):std::vector<std::vector<MatT> >(std::pow(d,N)),dim_(d),NSites_(N){for (auto& it : *this) it.resize(std::pow(d,N));};
    RedOp(const std::vector<std::vector<MatT> >& other, uint d, uint N):std::vector<std::vector<MatT> >(other),dim_(d),NSites_(N){};
    RedOp(std::vector<std::vector<MatT> >&& other, uint d, uint N):std::vector<std::vector<MatT> >(std::move(other)),dim_(d),NSites_(N){};

    inline uint GetLocalDim() const {return dim_;};
    inline uint GetNSites() const {return NSites_;};
    inline void print(std::string name="") const;

    /// DISK-IO
    bool save(std::string name) const;
    bool save(std::ofstream& file) const;
    bool load(std::string name);
    bool load(std::ifstream& file);

    RedOp st() const;
    RedOp t() const;
protected:
    uint dim_,NSites_;
};

template<typename T>
RedOp<T>
RedOp<T>::st() const
{
    RedOp<T> out(this->size());
/// TODO (valentin#1#): figure out if this can be done with std::transform
    typename RedOp<T>::const_row_it inrow;
    typename RedOp<T>::const_col_it incol;
    typename RedOp<T>::row_it outrow;
    typename RedOp<T>::col_it outcol;
    for (inrow = this->begin(),outrow=out.begin();inrow!=this->end();++inrow,++outrow)
    {
        outrow->resize(inrow->size());
        for (incol = inrow->begin(),outcol=outrow->begin();incol!=inrow->end();++incol,outcol++)
        {
            *outcol = incol->st();
        }
    }
    return out;
}


template<typename T>
bool
RedOp<T>::save(std::string name) const
{
    std::ofstream file(name.c_str(), std::fstream::binary);
    bool save_okay =  this->save(file);
    if (!save_okay)
    {
        cerr << "RedOp<T>::save(): could not save "<<name<<endl;
    }
    file.close();

    return save_okay;
}

template<typename T>
bool
RedOp<T>::save(std::ofstream& file) const
{
    bool save_okay = file.good();
    if (!save_okay)
    {
        cerr<<"RedOp<T>::save(): bad file handle"<<endl;
        return false;
    }
    uint dim(dim_),N(NSites_);

    file << "REDUCEDOP" << endl;
    file.write(reinterpret_cast<const char*>(&dim), std::streamsize(sizeof(uint)));
    file.write(reinterpret_cast<const char*>(&N), std::streamsize(sizeof(uint)));

    uint i=0,j=0;
    for (const auto& rowit : *this)
    {
        j=0;
        for (const auto& colit : rowit)
        {
            save_okay = colit.save(file);
            if (!save_okay)
            {
                cerr << "RedOp<T>::save() saving failed at ["<<i<<","<<j<<"]"<<endl;
                break;
            }
        }
        if (!save_okay) break;
        ++i;
    }
    file.close();

    return save_okay;
}

template<typename T>
bool
RedOp<T>::load(std::string name)
{
    std::ifstream file(name.c_str(), std::fstream::binary);
    if (!file.good())
    {
        cerr<<"RedOp<T>::load(): could not open "<<name<<endl;
        return false;
    }
    bool load_okay = this->load(file);
    if (!load_okay)
    {
        cerr<<"RedOp<T>::load(): could not load "<<name<<endl;
    }
    file.close();
    return load_okay;
}


template<typename T>
bool
RedOp<T>::load(std::ifstream& file)
{
    this->clear();
    bool load_okay = file.good();
    if (!load_okay)
    {
        cerr<<"RedOp<T>::load(): bad file handle"<<endl;
        return false;
    }

    std::string header,err;
    file >> header;
    file.get(); /// somehow we need this to get over the newline stored to terminate header. WTF it took me 1 hr to find that out!!

    if (header != "REDUCEDOP")
    {
        cerr << "wrong header "<<header<<endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(&dim_), std::streamsize(sizeof(uint)));
    file.read(reinterpret_cast<char*>(&NSites_), std::streamsize(sizeof(uint)));

    uint n_elem = std::pow(dim_,NSites_);
    this->resize(n_elem);
    for (auto& it : *this) it.resize(n_elem);

    uint i=0,j=0;
    for (auto& rowit : *this)
    {
        j=0;
        for (auto& colit : rowit)
        {
            load_okay = colit.load(file);
            if (!load_okay)
            {
                cerr << "RedOp<T>::load() loading failed at ["<<i<<","<<j<<"]"<<endl;
                break;
            }
            ++j;
        }
        if (!load_okay) break;
        ++i;
    }
    return load_okay;
}

template<typename T>
RedOp<T>
RedOp<T>::t() const
{
    RedOp<T> out(this->size());
    typename RedOp<T>::const_row_it inrow;
    typename RedOp<T>::const_col_it incol;
    typename RedOp<T>::row_it outrow;
    typename RedOp<T>::col_it outcol;
    for (inrow = this->begin(),outrow=out.begin();inrow!=this->end();++inrow,++outrow)
    {
        outrow->resize(inrow->size());
        for (incol = inrow->begin(),outcol=outrow->begin();incol!=inrow->end();++incol,outcol++)
        {
            *outcol = incol->t();
        }
    }
    return out;
}
template<typename T>
inline
void
RedOp<T>::print(std::string name) const
{
    if (name != "") cout<<name<<":"<<endl;
    cout<<*this<<endl;
}

template<typename T>
std::ostream&
operator<<(std::ostream& os, const RedOp<T>& Op)
{
    for (uint i = 0; i<Op.size(); ++i)
    {
        for (uint j=0; j<Op.size(); ++j)
        {
            os<<"["<<i<<","<<j<<"]:"<<endl;
            os<<Op[i][j]<<endl;
        }
    }
    return os;
}

//
//template<typename T>
//class RedOp : public std::vector<std::vector<Mat<T> > >
//{
//public:
//    typedef typename std::vector<std::vector<Mat<T> > >::const_iterator const_row_it;
//    typedef typename std::vector<Mat<T> >::const_iterator const_col_it;
//    typedef typename std::vector<std::vector<Mat<T> > >::iterator row_it;
//    typedef typename std::vector<Mat<T> >::iterator col_it;
//
//    RedOp(uint d, uint N=1):std::vector<std::vector<Mat<T> > >(std::pow(d,1)),dim_(d),NSites_(N){for (auto& it : *this) it.resize(d);};
//    RedOp(const std::vector<std::vector<Mat<T> > >& other, uint d, uint N):std::vector<std::vector<Mat<T> > >(other),dim_(d),NSites_(N){};
//    RedOp(std::vector<std::vector<Mat<T> > >&& other, uint d, uint N):std::vector<std::vector<Mat<T> > >(std::move(other)),dim_(d),NSites_(N){};
//
//    inline void GetDim() const {return dim_;};
//    inline void GetNSites() const {return NSites_;};
//
//    RedOp st() const;
//    RedOp t() const;
//protected:
//    uint dim_,NSites_;
//};
//
//template<typename T>
//RedOp<T>
//RedOp<T>::st() const
//{
//    RedOp<T> out(this->size());
///// TODO (valentin#1#): figure out if this can be done with std::transform
//    typename RedOp<T>::const_row_it inrow;
//    typename RedOp<T>::const_col_it incol;
//    typename RedOp<T>::row_it outrow;
//    typename RedOp<T>::col_it outcol;
//    for (inrow = this->begin(),outrow=out.begin();inrow!=this->end();++inrow,++outrow)
//    {
//        outrow->resize(inrow->size());
//        for (incol = inrow->begin(),outcol=outrow->begin();incol!=inrow->end();++incol,outcol++)
//        {
//            *outcol = incol->st();
//        }
//    }
//    return out;
//}
//
//
//template<typename T>
//RedOp<T>
//RedOp<T>::t() const
//{
//    RedOp<T> out(this->size());
//    typename RedOp<T>::const_row_it inrow;
//    typename RedOp<T>::const_col_it incol;
//    typename RedOp<T>::row_it outrow;
//    typename RedOp<T>::col_it outcol;
//    for (inrow = this->begin(),outrow=out.begin();inrow!=this->end();++inrow,++outrow)
//    {
//        outrow->resize(inrow->size());
//        for (incol = inrow->begin(),outcol=outrow->begin();incol!=inrow->end();++incol,outcol++)
//        {
//            *outcol = incol->t();
//        }
//    }
//    return out;
//}


#endif //OP_TYPES_H_
