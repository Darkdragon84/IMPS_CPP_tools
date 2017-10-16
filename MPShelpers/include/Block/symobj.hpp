#ifndef SYMOBJ_H
#define SYMOBJ_H
//
//#include <functional>

/**< helper elementwise vector operations *******************************************************************************/
template<typename VT>
std::vector<VT> VecPlus(const std::vector<VT>& lhs,const std::vector<VT>& rhs)
{
    assert(lhs.size()==rhs.size());
    std::vector<VT> out(lhs.size());
    std::transform(lhs.begin(),lhs.end(),rhs.begin(),out.begin(),std::plus<VT>());
    return out;
}

template<typename VT>
std::vector<VT> VecMinus(const std::vector<VT>& lhs,const std::vector<VT>& rhs)
{
    assert(lhs.size()==rhs.size());
    std::vector<VT> out(lhs.size());
    std::transform(lhs.begin(),lhs.end(),rhs.begin(),out.begin(),std::minus<VT>());
    return out;
}


template<typename VT>
std::vector<VT> VecNegate(const std::vector<VT>& lhs)
{
    std::vector<VT> out(lhs.size());
    std::transform(lhs.begin(),lhs.end(),out.begin(),std::negate<VT>());
    return out;
}

template<typename VT>
std::vector<VT> VecMod(const std::vector<VT>& lhs, const std::vector<VT>& rhs)
{
    assert(lhs.size()==rhs.size());
    std::vector<VT> out(lhs.size());
    std::transform(lhs.begin(),lhs.end(),rhs.begin(),out.begin(),std::modulus<VT>());
//    std::transform(lhs.begin(),lhs.end(),rhs.begin(),out.begin(),[](const VT lhs, const VT rhs){return lhs%rhs;});
    return out;
}

template<typename VT>
std::vector<VT> VecModSingle(const std::vector<VT>& lhs, VT mod)
{
    std::vector<VT> out(lhs.size());
    /// define "unary" modulus as a lambda function (with fixed y in x % y, i.e. f(x) = x % y)
    /// for possible negative values in lhs (e.g. as a result from a subtraction), make sure we are in the [0,mod] interval by adding mod to in
    /// (this assumes that inputs to the subtraction were also originally in the [0,mod] interval)
    std::transform(lhs.begin(),lhs.end(),out.begin(),[&](const VT& in){return (in + mod) % mod;});
    return out;
}

template<typename T>
struct symobj
{
    /// define the group operation as function handle
    typedef typename std::function<std::vector<T> (const std::vector<T>&, const std::vector<T>&)> groupop;
    typedef typename std::function<std::vector<T> (const std::vector<T>&)> singleop;

    symobj() = default;
    symobj(const std::vector<uint>& periods) {init(periods);};
    symobj(std::ifstream& file); /// construct from loading
    bool save(std::ofstream& file) const;
    void init(const std::vector<uint>& periods);

    std::string symstr_;
    std::vector<uint> periods_;
    uint Nsym_;
    groupop Plus_;
    groupop Minus_;
    singleop Negate_;
};



template<typename T>
void
symobj<T>::init(const std::vector<uint>& periods)
{
    periods_ = periods;
    Nsym_ = periods_.size();
    std::stringstream strbuf;

    /// the periods determine the cyclicity of the group, i.e. if we have a Z(N) symmetry (for 1<period<infty), or a U(1) symmetry (period=infty)
    /// concretely, period=0/1 corresponds to infty, i.e. U(1)
    uint firstelement = periods_.front();
    if (firstelement == 1) firstelement = 0; /// period one and zero are treated the same, i.e. as no period -> U(1) symmetry

    std::vector<std::pair<uint,T> > modpairvec;
    modpairvec.reserve(periods_.size());

    bool allequal = true;
    bool first = true;

    for (uint s=0;s<periods_.size();++s)
    {
        if (first) first = false;
        else strbuf<<"x";

        if (periods_[s]>1)
        {
            strbuf<<"Z"<<periods_[s];
            modpairvec.emplace_back(std::make_pair(s,periods_[s]));
        }
        else
        {
            if (periods_[s]>0) periods_[s]=0; /// period 1 does not make sense, interpret it as U(1) and set period to zero
            strbuf<<"U1";
        }
        allequal = allequal && periods_[s] == firstelement;
    }

    if (allequal) /// all symmetries are the same
    {
//        cout<<"all equal"<<endl;
        if (firstelement == 0) /// all U(1) symmetries
        {
            Plus_ = [](const std::vector<T>& lhs, const std::vector<T>& rhs) -> std::vector<T> { return VecPlus(lhs,rhs);};
            Minus_ = [](const std::vector<T>& lhs, const std::vector<T>& rhs) -> std::vector<T> { return VecMinus(lhs,rhs);};
            Negate_ = [](const std::vector<T>& lhs) -> std::vector<T> { return VecNegate(lhs);};
        }
        else /// all some Z(N) symmetry
        {
            T modval(firstelement);
            Plus_ = [modval](const std::vector<T>& lhs, const std::vector<T>& rhs) -> std::vector<T> { return VecModSingle(VecPlus(lhs,rhs),modval);};
            Minus_ = [modval](const std::vector<T>& lhs, const std::vector<T>& rhs) -> std::vector<T> { return VecModSingle(VecMinus(lhs,rhs),modval);};
            Negate_ = [modval](const std::vector<T>& lhs) -> std::vector<T> { return VecModSingle(VecNegate(lhs),modval);};
        }
    }
    else /// symmetries are mixed
    {
        /// we have to capture modpairvec by copy, as modvec goes out of scope after finishing constructor
        Plus_ = [modpairvec](const std::vector<T>& lhs, const std::vector<T>& rhs) -> std::vector<T>
        {
            std::vector<T> tmp = VecPlus(lhs,rhs);
            /// check for the stupid case of having too many symmetries
            for (const auto& it : modpairvec) if (it.first<tmp.size()) tmp[it.first]%=it.second;
            return tmp;
        };
        Minus_ = [modpairvec](const std::vector<T>& lhs, const std::vector<T>& rhs) -> std::vector<T>
        {
            std::vector<T> tmp = VecMinus(lhs,rhs);
            /// check for the stupid case of having too many symmetries
//            for (const auto& it : modpairvec) if (it.first<tmp.size()) tmp[it.first]%=it.second;
            for (const auto& it : modpairvec) if (it.first<tmp.size()) tmp[it.first] = (tmp[it.first] + it.second)%it.second;
            return tmp;
        };
        Negate_ = [modpairvec](const std::vector<T>& lhs) -> std::vector<T>
        {
            std::vector<T> tmp = VecNegate(lhs);
            /// check for the stupid case of having too many symmetries
//            for (const auto& it : modpairvec) if (it.first<tmp.size()) tmp[it.first]%=it.second;
            for (const auto& it : modpairvec) if (it.first<tmp.size()) tmp[it.first] = (tmp[it.first] + it.second)%it.second;
            return tmp;
        };
    }
    symstr_ = strbuf.str();
//    cout<<symstr_<<endl;
}

template<typename T>
symobj<T>::symobj(std::ifstream& file)
{
    if (!file.good())
    {
        throw std::runtime_error("symobj<T>(std::ifstream& file): bad file handle");
    }

    file.read(reinterpret_cast<char*>(&Nsym_), std::streamsize(sizeof(uint)));
    periods_.reserve(Nsym_);

    uint tmp;
    for (uint i=0;i<Nsym_;++i)
    {
        file.read(reinterpret_cast<char*>(&tmp), std::streamsize(sizeof(uint)));
        if (!file.good()) throw std::runtime_error("symobj<T>(std::ifstream& file): failed loading");
        periods_.emplace_back(tmp);
    }

    init();
}

template<typename T>
bool
symobj<T>::save(std::ofstream& file) const
{
    if (!file.good())
    {
        cerr << "symobj<T>::save(): bad file handle"<<endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(&Nsym_),std::streamsize(sizeof(uint)));
    for (const auto& it : periods_)
    {
        file.write(reinterpret_cast<const char*>(&it),std::streamsize(sizeof(uint)));
        if (!file.good())
        {
            cerr << "symobj<T>::save(): failed saving "<<it<<endl;
            return false;
        }
    }
    return true;
}


#endif // SYMOBJ_H
