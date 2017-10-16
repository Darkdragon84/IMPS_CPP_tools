#ifndef ITOKEY_H_
#define ITOKEY_H_

//#include <assert.h>
//#include <vector>
//#include <initializer_list>
//#include "helpers.h"

/// TODO (valentin#1#): THIS MUST BE RETHOUGHT, WE WOULD LIKE A KEY CLASS THAT HAS NO DEFAULT CONSTRUCTOR -> WE CANNOT INITIALIZE ItoKey TO ITS PROPER LENGTH -> just reserve and have to add with push_back

/** \brief container class for KeyTypes, that maps physical indices to corresponding group representations.
 * It is used to calculate the change of an ingoing quantum number as a function of the physical state: Qout = Qin + ItoKey[s]
 *
 *
 */

//template<uint N_,typename KT>
template<uint N_,typename KT>
class ItoKey : public std::vector<KT>
{
public:
    /// For now we don't need a default constructor, so delete it
    ItoKey() = delete;
    explicit ItoKey(uint d): d_(d) {this->reserve(pow(d,N_));};

    ItoKey(uint d, const std::vector<KT>& lst):
        std::vector<KT>(lst),d_(d)
    {
        assert(lst.size()==pow(d_,N_));
//        DOUT("I2K constructed from vec copy");
    };
    ItoKey(uint d, const std::initializer_list<KT>& lst):
        std::vector<KT>(lst),d_(d)
    {
        assert(lst.size()==pow(d_,N_));
//        DOUT("I2K constructed from list copy");
    };
/// TODO (valentin#1#): for some reason the keys still get copied if passed as an rvalue list or vector. Find out why
    ItoKey(uint d, std::vector<KT>&& lst):
        std::vector<KT>(std::move(lst)),d_(d)
    {
        assert(lst.size()==pow(d_,N_));
//        DOUT("I2K constructed from vec move");
    };
    ItoKey(uint d, std::initializer_list<KT>&& lst):
        std::vector<KT>(std::move(lst)),d_(d)
    {
        assert(lst.size()==pow(d_,N_));
//        DOUT("I2K constructed from list move");
    };

    inline uint GetLocalDim() const {return d_;};
    inline uint GetNSites() const {return N_;};
protected:
    uint d_;
};

template<uint N_,typename KT>
inline
std::ostream&
operator<<(std::ostream& os, const ItoKey<N_,KT>& i2k)
{
    if (N_>1)
    {
        std::vector<uint> inds;
        for (uint i=0;i<i2k.size();++i)
        {
            inds = num2ditvec(i,i2k.GetLocalDim(),N_);
            for (uint k=0;k<N_;++k)
            {
                if (k==0) os<<"(";
                else os<<",";
                os<<inds[k];
            }
            os<<"): "<<i2k[i]<<endl;
        }
    }
    else for (uint i=0;i<i2k.size();++i) os<<i<<": "<<i2k[i]<<endl;
    return os;
}

/**< CONCATENATION OF TWO I2K */
template<uint N1, uint N2, typename KT>
inline
ItoKey<N1+N2,KT>
operator*(const ItoKey<N1,KT>& lhs, const ItoKey<N2,KT>& rhs)
{
    assert(lhs.GetLocalDim()==rhs.GetLocalDim());
    ItoKey<N1+N2,KT> out(lhs.GetLocalDim());
//    typename ItoKey<N1>::const_iterator rit;
//    typename ItoKey<N2>::const_iterator rit;
//    typename ItoKey<N1+N2,KT>::iterator oit = out.begin();

//    for (const auto& lit : lhs) for (const auto& rit : rhs) *(oit++) = lit+rit;
    for (const auto& lit : lhs) for (const auto& rit : rhs) out.emplace_back(lit+rit);
    return out;
}


/**< some further derived types */

template<uint N, typename KT>
ItoKey<N,KT>
operator+(const ItoKey<N,KT> I2Kin, const KT& key)
{
    ItoKey<N,KT> I2Kout(I2Kin.GetLocalDim());
    for (const auto& init : I2Kin) I2Kout.push_back(init + key);
//    std::transform(I2Kin.begin(),I2Kin.end(),I2Kout.begin(),[&key](const KT& K1) -> KT {return K1 + key;});
    return I2Kout;
}


template<uint N, typename KT>
ItoKey<N,KT>
operator-(const ItoKey<N,KT> I2Kin, const KT& key)
{
    ItoKey<N,KT> I2Kout(I2Kin.GetLocalDim());
//    std::transform(I2Kin.begin(),I2Kin.end(),I2Kout.begin(),[&key](const IKey& K1) -> IKey {return K1 - key;});
    for (const auto& init : I2Kin) I2Kout.push_back(init - key);
    return I2Kout;
}


template<typename KT>
std::vector<uint>
validpath(uint N, const ItoKey<1,KT>& I2K)
{
    auto order = path_iter(N,I2K.front(),I2K,I2K.front());
    if (!order.first) throw std::domain_error("validpath: no valid path found for current ItoKey");
    return order.second;
}

template<typename KT>
std::pair<bool,std::vector<uint> >
path_iter(uint N, const KT& in, const ItoKey<1,KT>& I2K, const KT& zerokey)
{
    uint j=0;
    std::pair<bool,std::vector<uint> > res;

    if (N==1)
    {
        for (j=0;j<I2K.size();++j)
        {
            if (in + I2K[j] == zerokey)
            {
                res = std::make_pair(true,std::vector<uint>{j});
                break;
            }
        }

    }
    else
    {
        for (j=0;j<I2K.size();++j)
        {
            res = path_iter(N-1,in + I2K[j],I2K,zerokey);
            if (res.first)
            {
                res.second.push_back(j);
                break;
            }
        }
    }
    return res;
}

#endif // ITOKEY_H_
