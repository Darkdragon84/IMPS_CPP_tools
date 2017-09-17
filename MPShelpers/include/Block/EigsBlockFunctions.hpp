#ifndef EIGS_BLOCK_FUN_H_
#define EIGS_BLOCK_FUN_H_

//#include <string>
//#include <typeinfo>

//#include "arma_typedefs.h"
//#include "MPSBlockMat.hpp"
#include "eigs.h"

using std::map;
using std::cout;
using std::endl;
using std::string;

//template<typename VT>
//int eigs(std::function<void (VT*,VT*)> MultOPx, int n, CVecType& vals, CMatType& vecs, int nev,
//         std::string whch="LM", double tol=1e-14, const Col<VT>& x0=Col<VT>(), int maxit=0, int ncv=0);
//
//template<>
//int eigs(std::function<void (Real*,Real*)> MultOPx, int n, CVecType& vals, CMatType& vecs, int nev,
//         std::string whch, double tol, const Col<Real>& x0, int maxit, int ncv)
//         {
//             return eigs_rn(MultOPx, n, vals, vecs, nev, whch, tol, x0, maxit, ncv);
//         }
//
//template<>
//int eigs(std::function<void (Complex*,Complex*)> MultOPx, int n, CVecType& vals, CMatType& vecs, int nev,
//         std::string whch, double tol, const Col<Complex>& x0, int maxit, int ncv)
//         {
//             return eigs_cn(MultOPx, n, vals, vecs, nev, whch, tol, x0, maxit, ncv);
//         }

template<typename VT>
inline
Col<VT> CVectoVTVec(const CVecType& cvec);

template<>
inline
Col<Real> CVectoVTVec(const CVecType& cvec) {return real(cvec);};

template<>
inline
Col<Complex> CVectoVTVec(const CVecType& cvec) {return cvec;};

/// Multiplication routines for eigs, which work just with C-style arrays of double.
/// Block(Diag)Mats therefore get vectorized by successively vectorizing all matrices for every symmetry sector
/// and concatenating them consecutively. To avoid copying as much as possible, Arma matrices get binded to the
/// respective memory sectors within these arrays to enable high level routines with quantum numbers for applying the TM
template<typename KT,typename VT>
void
TMMultDiag(VT* invec, VT* outvec, const std::function<void (const BlockDiagMat<KT,VT>&, BlockDiagMat<KT,VT>&)>& TMfun, const dim_map<KT>& dims, uint D_tot)
{
    BlockDiagMat<KT,VT> in,out;
    memset(outvec,0,D_tot*(sizeof(*outvec)));

    uint pos=0;
    for (const auto& dimit : dims)
    {
        uint m = dimit.second;
        in.emplace_hint(in.end(),dimit.first,Mat<VT>(&invec[pos],m,m,false,true));
        out.emplace_hint(out.end(),dimit.first,Mat<VT>(&outvec[pos],m,m,false,true));
        pos += m*m;
    }
    TMfun(in,out);
}

template<typename KT,typename VT>
void
TMMult(VT* invec, VT* outvec, const std::function<void (const BlockMat<KT,VT>&, BlockMat<KT,VT>&)>& TMfun, const dimkeypair_vec<KT>& dims, uint D_tot)
{
    BlockMat<KT,VT> in,out;
    memset(outvec,0,D_tot*(sizeof(*outvec)));

    uint pos=0,ml=0,mr=0;
    for (const auto& dimit : dims)
    {
        ml = get<2>(dimit);
        mr = get<3>(dimit);
        in.emplace_hint(in.end(),get<0>(dimit),std::make_pair(get<1>(dimit),Mat<VT>(&invec[pos], ml, mr, false, true)));
        out.emplace_hint(out.end(),get<0>(dimit),std::make_pair(get<1>(dimit),Mat<VT>(&outvec[pos], ml, mr, false, true)));
        pos += ml*mr;
    }
    TMfun(in,out);
}

/** CALCULATE DOMINANT EIGENPAIR OF TM ********************************************************************************************************
 * the dominant eigenvalue is guaranteed to be real and positive (otherwise the MPS is quite screwed up...)
 */

template<typename KT, typename MPStype>
inline Real
TMDominantEig(const MPStype& MPS,
              BlockDiagMat<KT,typename MPStype::scalar_type>& V,
              dirtype dir,
              Real tol=1e-14,
              const BlockDiagMat<KT,typename MPStype::scalar_type>& x0=BlockDiagMat<KT,typename MPStype::scalar_type>(),
              int maxit=0,
              string mode="LR",
              bool verbose=false)
{
    dim_map<KT> dims = MPS.GetMl(); /// get all possible symmetry sectors present as ingoing or outgoing from MPS and their dimensions
    assert(MPS.GetMr() == dims && "left and right end of UC need to have same dimensions");
    uint D_tot=0;
    for (const auto& it : dims) D_tot += it.second*it.second; /// sum up all dimensions to get maximum dimension (for linear vector)
    return TMDominantEig(MPS,dims,D_tot,V,dir,tol,x0,maxit,mode,verbose);
}

/**< generic version for both single and multi site (once the boundary dims are known, it doesn't matter if it's single or multi-site) */
template<typename KT, typename MPStype>
Real
TMDominantEig(const MPStype& MPS,
              const dim_map<KT>& dims,
              uint D_tot,
              BlockDiagMat<KT,typename MPStype::scalar_type>& V,
              dirtype dir,
              Real tol=1e-14,
              const BlockDiagMat<KT,typename MPStype::scalar_type>& x0=BlockDiagMat<KT,typename MPStype::scalar_type>(),
              int maxit=0,
              string mode="LR",
              bool verbose=false)
{
    using VT = typename MPStype::scalar_type;
    uint pos = 0;
    if (mode!="LR" && mode!="LM")
    {
        cerr<<"wrong mode "<<mode<<endl;
        abort();
    }
    /// function handle for the actual routine for applying the TM frm the left/right onto some BlockDiagMat
    std::function<void (const BlockDiagMat<KT,VT>&, BlockDiagMat<KT,VT>&)> TMfun;
    if (dir==l) TMfun=[&MPS](const BlockDiagMat<KT,VT>& in, BlockDiagMat<KT,VT>& out) -> void {ApplyTMLeft(MPS,in,out);};
    else if (dir==r) TMfun=[&MPS](const BlockDiagMat<KT,VT>& in, BlockDiagMat<KT,VT>& out) -> void {ApplyTMRight(MPS,in,out);};
    else throw std::logic_error("wrong direction specified");

    /// actual calculation of the dominant eigenpair of the TM
    CVecType valtmp;
    CMatType Vtmp;
    Col<VT> x0v;

    if (!x0.empty() && x0.GetUniformSizes() == dims) x0v = x0.Vectorize();

//    eigs_rn([&dims,D_tot,&TMfun](Real* invec, Real* outvec)->void {TMMultDiag(invec,outvec,TMfun,dims,D_tot);},D_tot,valtmp,Vtmp,1,mode,tol,x0v,maxit);
//    eigs<VT>([&dims,D_tot,&TMfun](VT* invec, VT* outvec)->void {TMMultDiag(invec,outvec,TMfun,dims,D_tot);},D_tot,valtmp,Vtmp,1,mode,tol,x0v,maxit);
    eigs_n([&dims,D_tot,&TMfun](VT* invec, VT* outvec)->void {TMMultDiag(invec,outvec,TMfun,dims,D_tot);},D_tot,valtmp,Vtmp,1,mode,tol,x0v,maxit);

    /// analyze and postedit the dominant eigenpair
    V.clear();
    if (imag(valtmp(0))>2*tol) cerr<<"Warning: dominant eigenvalue is complex: "<<valtmp(0)<<endl;
//    if (norm(imag(Vtmp.col(0)))>2*tol*D_tot) cerr<<"Warning: dominant eigenvector is complex"<<endl;

    Real val = real(valtmp(0));
//    Col<VT> Vvec = Vtmp.col(0);
    Col<VT> Vvec = CVectoVTVec<VT>(Vtmp.col(0));

    pos=0;
    for (const auto& it : dims)
    {
        V.emplace_hint(V.end(),it.first,Mat<VT>(&Vvec.memptr()[pos],it.second,it.second));
        pos += it.second*it.second;
    }
    V /= trace(V); /// compute overall trace to make eigenmat. hermitian and positive in the end (we could also divide by just one element, but that might be very small, so trace is better)
//    V /= V.begin()->second(0,0); /// compute overall trace to make eigenmat. hermitian and positive in the end

    if (verbose)
    {
        if (dir==l) cout<<"left: "<<abs(val)<<", "<<val<<": "<<norm(ApplyTMLeft(MPS,V) - val*V)<<endl;
        else if (dir==r) cout<<"right: "<<abs(val)<<", "<<val<<": "<<norm(ApplyTMRight(MPS,V) - val*V)<<endl;
    }

    return val;
}

/**< CALCULATE FIRST FEW EIGENPAIRS OF TM WITH A PARTICULAR QUANTUM NUMBER *****************************************************************************************************
 * available modes are LM and LR
 */
template<typename MPStype>
inline CVecType
TMEigs(const MPStype& MPS,
       std::vector<BlockMat<typename MPStype::key_type,Complex> >& V,
       dirtype dir,
       const typename MPStype::key_type& K,
       uint nev,
       string mode="LM",
       Real tol=1e-14,
       const BlockMat<typename MPStype::key_type,typename MPStype::scalar_type>& x0 = BlockMat<typename MPStype::key_type,typename MPStype::scalar_type>(),
       int maxit=0,
       bool verbose=false)
{
    using KT = typename MPStype::key_type;
//    dim_map<KT> dim0 = MPS.GetUniformSizes(); /// get all possible symmetry sectors present as ingoing or outgoing from MPS and their dimensions
    dim_map<KT> dim0 = MPS.GetMl(); /// get all possible symmetry sectors present as ingoing or outgoing from MPS and their dimensions
    if (MPS.GetMr() != dim0) throw std::domain_error("left and right end of UC need to have same dimensions");

    dimkeypair_vec<KT> dims;
    uint D_tot=0;
    /// determine all contributing symmetry sectors and sum up all dimensions to get maximum dimension (for linear vector)
    for (const auto& mlit : dim0)
    {
        auto mrit = dim0.find(mlit.first + K);
        if (mrit != dim0.end())
        {
            dims.emplace_back(std::make_tuple(mlit.first,mrit->first,mlit.second,mrit->second));
            D_tot += mlit.second*mrit->second;
        }
    }
    if (dims.empty()) throw std::domain_error("TMEigs: no TM eigenvalues for this quantum number");
    return TMEigs(MPS,dims,D_tot,V,dir,K,nev,mode,tol,x0,maxit,verbose);
}

/**< generic version for both single and multi site (once the boundary dims are known, it doesn't matter if it's single or multi-site) */
template<typename KT, typename MPStype>
CVecType
TMEigs(const MPStype& MPS,
       const dimkeypair_vec<KT>& dims,
       uint D_tot,
       std::vector<BlockMat<KT,Complex> >& V,
       dirtype dir,
       const KT& K,
       uint nev,
       string mode="LM",
       Real tol=1e-14,
       const BlockMat<KT,typename MPStype::scalar_type>& x0 = BlockMat<KT,typename MPStype::scalar_type>(),
       int maxit=0,
       bool verbose=false)
{
    using VT = typename MPStype::scalar_type;

    if (mode!="LR" && mode!="LM") throw std::domain_error("TMEigs: wrong mode "+mode);
    if (dims.empty()) throw std::domain_error("TMEigs: no TM eigenvalues for this quantum number");

    /// function handle for the actual routine for applying the TM frm the left/right onto some BlockDiagMat
    std::function<void (const BlockMat<KT,VT>&, BlockMat<KT,VT>&)> TMfun;

    if (dir==l) TMfun=[&MPS](const BlockMat<KT,VT>& in, BlockMat<KT,VT>& out) -> void {ApplyTMLeft(MPS,in,out);};
    else if (dir==r) TMfun=[&MPS](const BlockMat<KT,VT>& in, BlockMat<KT,VT>& out) -> void {ApplyTMRight(MPS,in,out);};
    else throw std::domain_error("wrong direction specified");

    /// actual calculation of eigenpairs of the TM
    CVecType vals;
    CMatType Vtmp;
    Col<VT> x0v;

    if (!x0.empty() && x0.GetSizesVector() == dims) x0v = x0.Vectorize();
//    eigs<VT>([&dims,D_tot,&TMfun](VT* invec, VT* outvec)->void {TMMult(invec,outvec,TMfun,dims,D_tot);},D_tot,vals,Vtmp,nev,mode,tol,x0v,maxit);
    eigs_n([&dims,D_tot,&TMfun](VT* invec, VT* outvec)->void {TMMult(invec,outvec,TMfun,dims,D_tot);},D_tot,vals,Vtmp,nev,mode,tol,x0v,maxit);

    nev = vals.size();
    /// analyze and postedit the dominant eigenpair
    V.clear();
    V.resize(nev);

    /// reorder
    if (nev>1)
    {
        uvec order;
        if (mode=="LR") order = sort_index(real(vals),"descend");
        else if (mode=="LM") order = sort_index(abs(vals),"descend");
        vals = vals(order);
        Vtmp = Vtmp.cols(order);
    }

    for (uint n=0; n<nev; ++n)
    {
        uint pos=0,ml=0,mr=0;
        for (const auto& it : dims)
        {
            ml = get<2>(it);
            mr = get<3>(it);
            V[n].emplace_hint(V[n].end(),get<0>(it),std::make_pair(get<1>(it),CMatType(&(Vtmp.colptr(n))[pos],ml,mr)));
            pos += ml*mr;
        }
        if (verbose)
        {
            if (dir==l) cout<<K<<" left: "<<abs(vals(n))<<", "<<vals(n)<<": "<<norm(ApplyTMLeft(MPS,V[n]) - vals(n)*V[n])<<endl;
            else if (dir==r) cout<<K<<" right: "<<abs(vals(n))<<", "<<vals(n)<<": "<<norm(ApplyTMRight(MPS,V[n]) - vals(n)*V[n])<<endl;
        }
    }
    return vals;
}

template<typename MPStype>
inline CVecType
TMmixedEigs(const MPStype& A,
            const MPStype& B,
            std::vector<BlockMat<typename MPStype::key_type,Complex> >& V,
            dirtype dir,
            const typename MPStype::key_type& K,
            uint nev,
            string mode="LM",
            Real tol=1e-14,
            const BlockMat<typename MPStype::key_type,typename MPStype::scalar_type>& x0 = BlockMat<typename MPStype::key_type,typename MPStype::scalar_type>(),
            int maxit=0,
            bool verbose=false)
{
    using KT = typename MPStype::key_type;
//    dim_map<KT> dim0 = MPS.GetUniformSizes(); /// get all possible symmetry sectors present as ingoing or outgoing from MPS and their dimensions
    dim_map<KT> diml = A.GetMl();
    if (A.GetMr() != diml) throw std::domain_error("left and right end of A need to have same dimensions");
    dim_map<KT> dimr = B.GetMl();
    if (B.GetMr() != dimr) throw std::domain_error("left and right end of B need to have same dimensions");

    if (dir==l) swap(diml,dimr);

    dimkeypair_vec<KT> dims;
    uint D_tot=0;
    /// determine all contributing symmetry sectors and sum up all dimensions to get maximum dimension (for linear vector)
    for (const auto& mlit : diml)
    {
        auto mrit = dimr.find(mlit.first + K);
        if (mrit != dimr.end())
        {
            dims.emplace_back(std::make_tuple(mlit.first,mrit->first,mlit.second,mrit->second));
            D_tot += mlit.second*mrit->second;
        }
    }
    if (dims.empty()) throw std::domain_error("TMmixedEigs: no TM eigenvalues for this quantum number");

    return TMmixedEigs(A,B,dims,D_tot,V,dir,K,nev,mode,tol,x0,maxit,verbose);
}

template<typename KT, typename MPStype>
CVecType
TMmixedEigs(const MPStype& A,
            const MPStype& B,
            const dimkeypair_vec<KT>& dims,
            uint D_tot,
            std::vector<BlockMat<KT,Complex> >& V,
            dirtype dir,
            const KT& K,
            uint nev,
            string mode="LM",
            Real tol=1e-14,
            const BlockMat<KT,typename MPStype::scalar_type>& x0 = BlockMat<KT,typename MPStype::scalar_type>(),
            int maxit=0,
            bool verbose=false)
{
    using VT = typename MPStype::scalar_type;

    if (mode!="LR" && mode!="LM") throw std::domain_error("TMmixedEigs: wrong mode "+mode);
    if (dims.empty()) throw std::domain_error("TMmixedEigs: no TM eigenvalues for this quantum number");

    /// function handle for the actual routine for applying the TM frm the left/right onto some BlockDiagMat
    std::function<void (const BlockMat<KT,VT>&, BlockMat<KT,VT>&)> TMfun;

    if (dir==l) TMfun=[&A,&B](const BlockMat<KT,VT>& in, BlockMat<KT,VT>& out) -> void {ApplyTMmixedLeft(A,B,in,out);};
    else if (dir==r) TMfun=[&A,&B](const BlockMat<KT,VT>& in, BlockMat<KT,VT>& out) -> void {ApplyTMmixedRight(A,B,in,out);};
    else throw std::domain_error("wrong direction specified");

    /// actual calculation of eigenpairs of the TM
    CVecType vals;
    CMatType Vtmp;
    Col<VT> x0v;

    if (!x0.empty() && x0.GetSizesVector() == dims) x0v = x0.Vectorize();
//    eigs<VT>([&dims,D_tot,&TMfun](VT* invec, VT* outvec)->void {TMMult(invec,outvec,TMfun,dims,D_tot);},D_tot,vals,Vtmp,nev,mode,tol,x0v,maxit);
    eigs_n([&dims,D_tot,&TMfun](VT* invec, VT* outvec)->void {TMMult(invec,outvec,TMfun,dims,D_tot);},D_tot,vals,Vtmp,nev,mode,tol,x0v,maxit);

    nev = vals.size();
    /// analyze and postedit the dominant eigenpair
    V.clear();
    V.resize(nev);

    /// reorder
    if (nev>1)
    {
        uvec order;
        if (mode=="LR") order = sort_index(real(vals),"descend");
        else if (mode=="LM") order = sort_index(abs(vals),"descend");
        vals = vals(order);
        Vtmp = Vtmp.cols(order);
    }

    for (uint n=0; n<nev; ++n)
    {
        uint pos=0,ml=0,mr=0;
        for (const auto& it : dims)
        {
            ml = get<2>(it);
            mr = get<3>(it);
            V[n].emplace_hint(V[n].end(),get<0>(it),std::make_pair(get<1>(it),CMatType(&(Vtmp.colptr(n))[pos],ml,mr)));
            pos += ml*mr;
        }
        if (verbose)
        {
            if (dir==l) cout<<K<<" left: "<<abs(vals(n))<<", "<<vals(n)<<": "<<norm(ApplyTMmixedLeft(A,B,V[n]) - vals(n)*V[n])<<endl;
            else if (dir==r) cout<<K<<" right: "<<abs(vals(n))<<", "<<vals(n)<<": "<<norm(ApplyTMmixedRight(A,B,V[n]) - vals(n)*V[n])<<endl;
        }
    }
    return vals;
}

/// old division into single site and multi site version. Now, through the declaration of MPStype::scalar_type and ::key_type,
/// it is possible to extract KT and VT from a generic MPS template (be it a single tensor or a UC thereof)

/**< single site version */
//template<typename KT>
//inline CVecType
//TMEigs(const MPSBlockMat<KT,Real>& MPS,
//       std::vector<BlockMat<KT,Complex> >& V,
//       dirtype dir,
//       const KT& K,
//       uint nev,
//       string mode="LM",
//       Real tol=1e-14,
//       const BlockMat<KT,Real>& x0 = BlockMat<KT,Real>(),
//       int maxit=0)
//template<typename KT, typename VT>
//inline CVecType
//TMEigs(const MPSBlockMat<KT,VT>& MPS,
//       std::vector<BlockMat<KT,Complex> >& V,
//       dirtype dir,
//       const KT& K,
//       uint nev,
//       string mode="LM",
//       Real tol=1e-14,
//       const BlockMat<KT,VT>& x0 = BlockMat<KT,VT>(),
//       int maxit=0)
//{
//    dim_map<KT> dim0 = MPS.GetUniformSizes(); /// get all possible symmetry sectors present as ingoing or outgoing from MPS and their dimensions
////    dim_map<KT> dim0 = MPS.GetMl(); /// get all possible symmetry sectors present as ingoing or outgoing from MPS and their dimensions
//////    assert(MPS.GetMr() == dims && "left and right end of UC need to have same dimensions");
////    if (MPS.GetMr() != dim0) throw std::domain_error("left and right end of UC need to have same dimensions");
//
//    dimkeypair_vec<KT> dims;
//    uint D_tot=0;
//    /// determine all contributing symmetry sectors and sum up all dimensions to get maximum dimension (for linear vector)
//    for (const auto& mlit : dim0)
//    {
//        auto mrit = dim0.find(mlit.first + K);
//        if (mrit != dim0.end())
//        {
//            dims.emplace_back(std::make_tuple(mlit.first,mrit->first,mlit.second,mrit->second));
//            D_tot += mlit.second*mrit->second;
//        }
//    }
//    return TMEigs(MPS,dims,D_tot,V,dir,K,nev,mode,tol,x0,maxit);
//}


///**< multi site version */
//template<typename KT, typename VT>
//inline CVecType
//TMEigs(const MPSBlockMatArray<KT,VT>& MPSvec,
//       std::vector<BlockMat<KT,Complex> >& V,
//       dirtype dir,
//       const KT& K,
//       uint nev,
//       string mode="LM",
//       Real tol=1e-14,
//       const BlockMat<KT,VT>& x0 = BlockMat<KT,VT>(),
//       int maxit=0)
//{
///// TODO (valentin#1#2016-11-03): consider additional loop to determine intersection of ml and mr sectors, instead of just directly comparing them.
//    dim_map<KT> dim0 = MPSvec.front().GetMl(); /// get all possible symmetry sectors present as ingoing or outgoing from MPS and their dimensions
//    if (MPSvec.back().GetMr() != dim0) throw std::domain_error("left and right end of UC need to have same dimensions");
//
//    dimkeypair_vec<KT> dims;
//    uint D_tot=0;
//    /// determine all contributing symmetry sectors and sum up all dimensions to get maximum dimension (for linear vector)
//    for (const auto& mlit : dim0)
//    {
//        auto mrit = dim0.find(mlit.first + K);
//        if (mrit != dim0.end())
//        {
//            dims.emplace_back(std::make_tuple(mlit.first,mrit->first,mlit.second,mrit->second));
//            D_tot += mlit.second*mrit->second;
//        }
//    }
//    return TMEigs(MPSvec,dims,D_tot,V,dir,K,nev,mode,tol,x0,maxit);
//}


//
///**< single site version */
//template<typename KT>
//inline CVecType
//TMEigs(const MPSBlockMat<KT,Real>& MPS,
//       std::vector<BlockDiagMat<KT,Complex> >& V,
//       dirtype dir,
//       uint nev,
//       Real tol=1e-14,
//       const BlockDiagMat<KT,Real>& x0 = BlockDiagMat<KT,Real>(),
//       int maxit=0,
//       string mode="LM")
//{
//    dim_map<KT> dims = MPS.GetMl(); /// get all possible symmetry sectors present as ingoing or outgoing from MPS and their dimensions
//    assert(MPS.GetMr() == dims && "left and right end of UC need to have same dimensions");
//    uint D_tot=0;
//    for (const auto& it : dims) D_tot += it.second*it.second; /// sum up all dimensions to get maximum dimension (for linear vector)
//    return TMEigs(MPS,dims,D_tot,V,dir,nev,tol,x0,maxit,mode);
//}
//
///**< multi site version */
//template<typename KT>
//inline CVecType
//TMEigs(const MPSBlockMatArray<KT,Real>& MPSvec,
//       std::vector<BlockDiagMat<KT,Complex> >& V,
//       dirtype dir,
//       uint nev,
//       Real tol=1e-14,
//       const BlockDiagMat<KT,Real>& x0 = BlockDiagMat<KT,Real>(),
//       int maxit=0,
//       string mode="LM")
//{
//    dim_map<KT> dims = MPSvec.front().GetMl(); /// get all possible symmetry sectors present as ingoing or outgoing from MPS and their dimensions
//    assert(MPSvec.back().GetMr() == dims && "left and right end of UC need to have same dimensions");
//    uint D_tot=0;
//    for (const auto& it : dims) D_tot += it.second*it.second; /// sum up all dimensions to get maximum dimension (for linear vector)
//    return TMEigs(MPSvec,dims,D_tot,V,dir,nev,tol,x0,maxit,mode);
//}
//
///**< generic version for both single and multi site (once the boundary dims are known, it doesn't matter if it's single or multi-site) */
//template<typename KT, typename MPStype>
//CVecType
//TMEigs(const MPStype& MPS,
//       const dim_map<KT>& dims,
//       uint D_tot,
//       std::vector<BlockDiagMat<KT,Complex> >& V,
//       dirtype dir,
//       uint nev,
//       Real tol=1e-14,
//       const BlockDiagMat<KT,Real>& x0 = BlockDiagMat<KT,Real>(),
//       int maxit=0,
//       string mode="LM")
//{
//    if (mode!="LR" && mode!="LM")
//    {
//        cerr<<"wrong mode "<<mode<<endl;
//        abort();
//    }
//    /// function handle for the actual routine for applying the TM frm the left/right onto some BlockDiagMat
//    std::function<void (const BlockDiagMat<KT,Real>&, BlockDiagMat<KT,Real>&)> TMfun;
//    if (dir==l) TMfun=[&MPS](const BlockDiagMat<KT,Real>& in, BlockDiagMat<KT,Real>& out) -> void {ApplyTMLeft(MPS,in,out);};
//    else if (dir==r) TMfun=[&MPS](const BlockDiagMat<KT,Real>& in, BlockDiagMat<KT,Real>& out) -> void {ApplyTMRight(MPS,in,out);};
//    else
//    {
//        cerr<<"wrong direction specified"<<endl;
//        abort();
//    }
//
//
//    /// actual calculation of the dominant eigenpair of the TM
//    CVecType vals;
//    CMatType Vtmp;
//    RVecType x0v;
//
//    if (!x0.empty() && x0.GetUniformSizes() == dims) x0v = x0.Vectorize();
//    eigs_rn([&dims,D_tot,&TMfun](Real* invec, Real* outvec)->void {TMMult(invec,outvec,TMfun,dims,D_tot);},D_tot,vals,Vtmp,nev,mode,tol,x0v,maxit);
//
//    nev = vals.size();
//    /// analyze and postedit the dominant eigenpair
//    V.clear();
//    V.resize(nev);
//
//    /// reorder
//    if (nev>1)
//    {
//        uvec order;
//        if (mode=="LR") order = sort_index(real(vals),"descend");
//        else if (mode=="LM") order = sort_index(abs(vals),"descend");
//        vals = vals(order);
//        Vtmp = Vtmp.cols(order);
//    }
//
//    for (uint n=0; n<nev; ++n)
//    {
//        uint pos=0;
//        for (const auto& it : dims)
//        {
//            CMatType tmp(&(Vtmp.colptr(n))[pos],it.second,it.second);
//            V[n].emplace_hint(V[n].end(),it.first,tmp);
//            pos += it.second*it.second;
//        }
//        if (dir==l) DOUT("left: "<<abs(vals(n))<<", "<<vals(n)<<": "<<norm(ApplyTMLeft(MPS,V[n]) - vals(n)*V[n]));
//        else if (dir==r) DOUT("right: "<<abs(vals(n))<<", "<<vals(n)<<": "<<norm(ApplyTMRight(MPS,V[n]) - vals(n)*V[n]));
//    }
//    DOUT("");
//    return vals;
//}

#endif // EIGS_BLOCK_FUN_H_
