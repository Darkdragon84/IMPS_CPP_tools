#ifndef DIM_MAPS_
#define DIM_MAPS_


/// fwd declarations
template<typename KT>
class dim_map;

template<typename KT>
class dim_vec;

template<typename KT>
class dimpair_vec;

template<typename KT>
class dimpair_vec;

template<typename KT>
class dimkeypair_map;

template<typename KT>
class dimkeypair_vec;


/// class definitions

/** vector and map of single sizes, e.g. for left and right dimensions of a BlockMat *****************/
//template<typename KT>
//using dim_map = std::map<KT,uint>;
template<typename KT>
class dim_map : public std::map<KT,uint>
{
public:
    dim_map() = default;
    dim_map(const dim_map& ) = default;
    dim_map(      dim_map&&) = default;
    dim_map& operator=(const dim_map& ) = default;
    dim_map& operator=(      dim_map&&) = default;

    dim_map(const dim_vec<KT>& vec)
    {
        for (const auto& it : vec) this->emplace_hint(this->end(),it);
    }

    inline uint GetNElem() const
    {
        uint m_tot = 0;
        for (const auto& it : *this) m_tot += it.second;
        return m_tot;
    }
};

//template<typename KT>
//using dim_vec = std::vector<std::pair<KT,uint> >;
template<typename KT>
class dim_vec : public std::vector<std::pair<KT,uint> >
{
public:
    dim_vec() = default;
    dim_vec(const dim_vec& ) = default;
    dim_vec(      dim_vec&&) = default;
    dim_vec& operator=(const dim_vec& ) = default;
    dim_vec& operator=(      dim_vec&&) = default;

    dim_vec(const dim_map<KT>& inmap)
    {
        this->reserve(inmap.size());
//        for (const auto& it : inmap) this->emplace_back(std::make_pair(it.first,it.second))
        for (const auto& it : inmap) this->emplace_back(it); /// it is of type std::pair<KT,uint>
    }

    inline uint GetNElem() const
    {
        uint m_tot = 0;
        for (const auto& it : *this) m_tot += it.second;
        return m_tot;
    }
};

/** vector and map of block-dimensions with only one key, within e.g. a BlockDiagMat *****************/
//template<typename KT>
//using dimpair_map = std::map<KT,std::pair<uint,uint> >;
template<typename KT>
class dimpair_map : public std::map<KT,std::pair<uint,uint> >
{
public:
    dimpair_map() = default; /// std ctor
    dimpair_map(const dimpair_map& ) = default; /// default copy ctor
    dimpair_map(      dimpair_map&&) = default; /// default move ctor
    dimpair_map& operator=(const dimpair_map& ) = default;
    dimpair_map& operator=(      dimpair_map&&) = default;

    dimpair_map(const dimpair_vec<KT>& vec)
    {
        for (const auto& it : vec) this->emplace_hint(this->end(),
                    std::get<0>(it),
                    std::make_pair(std::get<1>(it),
                                   std::get<2>(it)));
    }

    inline uint GetNElem() const
    {
        uint m_tot = 0;
        for (const auto& it : *this) m_tot += it.second.first*it.second.second;
        return m_tot;
    }
};

//template<typename KT>
//using dimpair_vec = std::vector<std::tuple<KT,uint,uint> >;
template<typename KT>
class dimpair_vec : public std::vector<std::tuple<KT,uint,uint> >
{
public:
    dimpair_vec() = default; /// std ctor
    dimpair_vec(const dimpair_vec& ) = default; /// default copy ctor
    dimpair_vec(      dimpair_vec&&) = default; /// default move ctor
    dimpair_vec& operator=(const dimpair_vec& ) = default;
    dimpair_vec& operator=(      dimpair_vec&&) = default;

    dimpair_vec(const dimpair_map<KT>& inmap)
    {
        this->reserve(inmap.size());
        for (const auto& it : inmap) this->emplace_back(std::make_tuple(it.first,
                    it.second.first,
                    it.second.second));
    }

    inline uint GetNElem() const
    {
        uint m_tot = 0;
        for (const auto& it : *this) m_tot += std::get<1>(it)*std::get<2>(it);
        return m_tot;
    }
};

template<typename KT>
class dimkeypair_map : public std::map<KT,std::tuple<KT,uint,uint> >
{
public:
    dimkeypair_map() = default;
    dimkeypair_map(const dimkeypair_map& ) = default;
    dimkeypair_map(      dimkeypair_map&&) = default;
    dimkeypair_map& operator=(const dimkeypair_map& ) = default;
    dimkeypair_map& operator=(      dimkeypair_map&&) = default;

    inline uint GetNElem() const
    {
        uint m_tot = 0;
        for (const auto& it : *this) m_tot += std::get<1>(it)*std::get<2>(it);
        return m_tot;
    }
};

/**< vector of block-dimensions within e.g. a BlockMat */
template<typename KT>
class dimkeypair_vec : public std::vector<std::tuple<KT,KT,uint,uint> >
{
public:
    dimkeypair_vec() = default;
    dimkeypair_vec(const dimkeypair_vec& ) = default;
    dimkeypair_vec(      dimkeypair_vec&&) = default;
    dimkeypair_vec(const dim_vec<KT>& vec)
    {
        this->reserve(vec.size());
        for (const auto& it : vec) this->emplace_back(std::make_tuple(it.first,
                    it.first,
                    it.second,
                    it.second));
    }
    dimkeypair_vec(const dimpair_vec<KT>& vec)
    {
        this->reserve(vec.size());
        for (const auto& it : vec) this->emplace_back(std::make_tuple(std::get<0>(it),
                    std::get<0>(it),
                    std::get<1>(it),
                    std::get<2>(it)));
    }
    dimkeypair_vec(const dim_map<KT>& lhs, const dim_map<KT>& rhs)
    {
        assert(lhs.size()==rhs.size() && "dimkeypair_vec(const dim_map& lhs, const dim_map& rhs): lhs and rhs need to be of same size");
        this->reserve(lhs.size());
        {
            typename dim_map<KT>::const_iterator itl=lhs.cbegin(), itr=rhs.cbegin();
            for ( ; itl!=lhs.cend(); itl++,itr++)
            {
                /// only add if both left and right dimension are nonzero
                if (itl->second > 0 && itr->second > 0) this->emplace_back(std::make_tuple(itl->first,itr->first,itl->second,itr->second));
            }
        }
    }

    inline uint GetNElem() const
    {
        uint m_tot = 0;
        for (const auto& it : *this) m_tot += std::get<2>(it)*std::get<3>(it);
        return m_tot;
    }
};

#endif
