#ifndef KEY_TYPES_H
#define KEY_TYPES_H

//#include <functional>
//#include <vector>
//#include <algorithm>
//#include <iostream>

//#include "arma_typedefs.h"
//#include "symobj.hpp"
//#include <fstream>

/// TODO (valentin#1#): Maybe consider implementing a method in symobj.hpp to cast the values of vectors/KeyTypes in copy c'tors
/// into the correct interval (i.e. [0,N] for Z(N), but do nothing for U(1))


/**< This Keytype can only be constructed with a given group object (GO). It can be default constructed just from GO and will then be initialized to zero. */
template<typename VT>
class KeyType : public std::vector<VT>
{
public:
    typedef symobj<VT> GroupObj;

    /**< CTORS **********************************************************************************/
    KeyType() = delete; /// no standard c'tor, require an associated group object to perform group operation

    /// constructs a zero quantum number, VT has to be initializable with zero
    KeyType(const GroupObj& O):std::vector<VT>(O.Nsym_,VT{0}),GO_(O){};

    KeyType(const GroupObj& O, const std::vector<VT>& vec):std::vector<VT>(vec),GO_(O){ assert(this->size()==GO_.Nsym_);};
    KeyType(const GroupObj& O, const std::vector<VT>&& vec):std::vector<VT>(std::move(vec)),GO_(O){assert(this->size()==GO_.Nsym_);};

    KeyType(const KeyType& other):std::vector<VT>(other),GO_(other.GO_) {};
    KeyType(const KeyType&& other):std::vector<VT>(std::move(other)),GO_(std::move(other.GO_)) {};

    KeyType(std::ifstream& file, const GroupObj& GO); /// loading constructor

    KeyType& operator=(const KeyType& other);
    KeyType& operator=(const KeyType&& other)
    {
        std::vector<VT>::operator=(std::move(other));
        return *this;
    };

    /**< GETTER FUNCTIONS **********************************************************************************/
    inline const GroupObj& GetGroupObj () const {return GO_;};

    /**< EXTERNAL GROUP OPERATORS FOR FINAL USE ************************************************************/
    template<typename VT2>
    friend inline KeyType<VT2> operator+(const KeyType<VT2>& lhs, const KeyType<VT2>& rhs);
    template<typename VT2>
    friend inline KeyType<VT2> operator-(const KeyType<VT2>& lhs, const KeyType<VT2>& rhs);


    /**< IO ************************************************************************************************/
    bool save(std::ofstream& file) const;
protected:
    const GroupObj& GO_;
};

template<typename VT>
KeyType<VT>&
KeyType<VT>::operator=(const KeyType<VT>& other)
{
    /// do nothing for self assignment
    if (this != &other)
    {
        KeyType<VT> tmp(other);
        std::swap(*this,other);
    }
    return *this;
}

template<typename VT>
KeyType<VT>::KeyType(std::ifstream& file,const GroupObj& GO):GO_(GO)
{
    if (!file.good()) throw std::runtime_error("KeyType<VT>(std::ifstream& file): bad file handle");

    uint Nsym;
    file.read(reinterpret_cast<char*>(&Nsym),std::streamsize(sizeof(uint)));
    this->reserve(Nsym);

    VT tmp;
    for (uint i=0;i<Nsym;++i)
    {
        file.read(reinterpret_cast<char*>(&tmp),std::streamsize(sizeof(VT)));
        if (!file.good()) throw std::runtime_error("KeyType<VT>(std::ifstream& file): failed loading");
        this->emplace_back(tmp);
    }
}

template<typename VT>
bool
KeyType<VT>::save(std::ofstream& file) const
{
    if (!file.good())
    {
        cerr << "KeyType<VT>::save(): bad file handle"<<endl;
        return false;
    }
    /// DON'T save symmetry type, this should be done separately and only once.
    /// When loading, all keys should get a reference to the same Group Object!
//    GO_.save(file);

    const uint Nsym = this->size();
    file.write(reinterpret_cast<const char*>(&Nsym),std::streamsize(sizeof(const uint)));

    for (const auto& it : *this)
    {
        file.write(reinterpret_cast<const char*>(&it),std::streamsize(sizeof(const VT)));
        if (!file.good())
        {
            cerr << "KeyType<VT>::save(): failed saving "<<it<<endl;
            return false;
        }
    }
    return true;
}

using IKey = KeyType<int>;


template<typename VT>
inline
KeyType<VT> operator-(const KeyType<VT>& lhs, const KeyType<VT>& rhs)
{
    return KeyType<VT>(lhs.GetGroupObj(),lhs.GetGroupObj().Minus_(lhs,rhs));
}

template<typename VT>
inline
KeyType<VT>
operator+(const KeyType<VT>& lhs, const KeyType<VT>& rhs)
{
    return KeyType<VT>(lhs.GetGroupObj(),lhs.GetGroupObj().Plus_(lhs,rhs));
}

template<typename VT>
inline
KeyType<VT>
operator-(const KeyType<VT>& lhs)
{
    return KeyType<VT>(lhs.GetGroupObj(),lhs.GetGroupObj().Negate_(lhs));
}

//template<typename VT>
//inline
//bool
//operator==(const KeyType<VT>& lhs, const KeyType<VT>& rhs)
//{
//    return ;
//}


template<typename VT>
KeyType<VT> FlipK(const KeyType<VT>& K, const std::vector<bool>& which = std::vector<bool>())
{
    KeyType<VT> K0(K.GetGroupObj());
    KeyType<VT> Kout(K);

    /// if which is empty, it is assumed that all QN are flipped (this is the default for which)
    if (which.empty()) Kout = -K;
    else
    {
        assert(which.size() == K.size() && "The key K and the mask which need to have the same amount of entries");
        for (uint i=0;i<K.size();++i) if (which[i]) Kout[i] = -K[i];
    }
    return Kout;
}

/// Kout contains the QN of Kin permuted to the order given in perm (must contain a permutation of the numbers [0,Nsym-1])
/// if Kin = (2,6,1) and perm = (3,1,2), then the returned Kout is (1,2,6)
template<typename VT>
KeyType<VT> PermuteK(const KeyType<VT>& Kin, const std::vector<uint>& perm)
{
    uint N = Kin.size();
    assert(perm.size() == N && "Kin and perm need to be of same length");
    std::vector<VT> Kout(N,VT(0));

    for (uint i=0;i<N;++i) Kout[perm[i]] = Kin[i];

    return KeyType<VT>(Kin.GetGroupObj(),Kout);
}

template<typename ST,typename T>
ST& operator<<(ST& stream, const KeyType<T>& vec);

template<typename T>
std::stringstream& operator<<(std::stringstream& ss, const KeyType<T>& vec) /**< to be able to pass KeyTypes to stringstreams (such as cout) */
{
    bool first=true;
    for (const auto& it : vec)
    {
        if (first) first = false;
        else ss<<"_";
        ss<<std::to_string(it);
    }
    return ss;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const KeyType<T>& vec) /**< to be able to pass KeyTypes to output streams (such as cout) */
{
    bool first=true;
    os<<"(";
    for (const auto& it : vec)
    {
        if (first) first=false;
        else os<<",";
        os<<it;
    }
    os<<")";
    return os;
}


/**< GENERAL HELPER FUNCTIONS  **************************************************************************/
template<typename KT>
inline
dim_map<KT>
operator+(const dim_map<KT>& inmap, const KT& key)
{
    dim_map<KT> outmap;
    for (const auto& it : inmap) outmap.emplace_hint(outmap.end(),it.first + key,it.second);
    return outmap;
}


template<typename KT>
inline
dim_map<KT>
operator+(const KT& key, const dim_map<KT>& inmap){return inmap + key;}


template<typename KT>
inline
dim_map<KT>
operator-(const dim_map<KT>& inmap, const KT& key)
{
    dim_map<KT> outmap;
    for (const auto& it : inmap) outmap.emplace_hint(outmap.end(),it.first - key,it.second);
    return outmap;
}


template<typename KT>
std::ostream&
operator<<(std::ostream& os, const dim_map<KT>& dmap)
{
    for (const auto& it : dmap) os<<it.first<<": "<<it.second<<endl;
    return os;
}

namespace std
{
    template<typename T>
    std::string
    to_string(const std::vector<T>& vec)
    {
        std::stringstream os;

        bool first=true;
        for (const auto& it : vec)
        {
            if (first) first=false;
            else os<<"_";
            os<<it;
        }
        return os.str();
    }
}

#endif // KEY_TYPES_H

