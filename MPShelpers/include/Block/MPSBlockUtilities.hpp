#ifndef MPS_BLOCK_UTIL_H
#define MPS_BLOCK_UTIL_H

//#include <utility>
//
//#include "MPSBlockMat.hpp"
//#include "OperatorTypes.hpp"
//#include "TMBlockFunctions.hpp"
//#include "helpers.hpp"

using std::cout;
using std::endl;
using std::map;

/// TODO (valentin#1#2016-04-08): implement explicit use of openmp clauses

/**< Apply Operator onto MPS *************************************************************************************************************++*/
template<typename KT,typename VTMPS, typename VTO>
MPSBlockMat<KT,typename promote_type<VTMPS,VTO>::result>
ApplyOperator(const MPSBlockMat<KT,VTMPS>& MPSin, const SparseOperator<VTO>& op)
{
    assert(MPSin.GetLocalDim() == op.GetLocalDim() && "MPS and op need to have the same physical dimension");
    assert(MPSin.GetNSites() == op.GetNSites() && "MPS and op need to span the same amount of sites");
    MPSBlockMat<KT,typename promote_type<VTMPS,VTO>::result> MPSout(MPSin.GetLocalDim(),MPSin.GetNSites());
    uint ii,jj;

    for (auto opit=op.begin(); opit!=op.end(); opit++)
    {
        ii=opit.row();
        jj=opit.col();
        MPSout[ii] += (*opit) * MPSin[jj];
    }
    return MPSout;
}


/**< generate reduced operators */

template<typename KT, typename VTMPS, typename VTO>
RedOp<BlockMat<KT,typename promote_type<VTMPS,VTO>::result> >
ReducedOp(const SparseOperator<VTO>& O, const MPSBlockMat<KT,VTMPS>& A, dirtype dir)
{
    uint d = A.GetLocalDim();
    assert(O.GetLocalDim() == d && "H and A need to be of same physical dimension");
    assert(O.GetNSites() == 2 && "H needs to be two-site (for now)");
    RedOp<BlockMat<KT,typename promote_type<VTMPS,VTO>::result> > Ored(d,1);
    if (dir == r)
    {
        for (auto opit = O.begin(); opit!=O.end(); ++opit)
        {
            std::vector<uint> ii = num2ditvec(opit.row(),d,2);
            std::vector<uint> jj = num2ditvec(opit.col(),d,2);
            if (Ored[ii[0]][jj[0]].empty()) Ored[ii[0]][jj[0]] = (*opit)*A[jj[1]]*A[ii[1]].t();
            else Ored[ii[0]][jj[0]] += (*opit)*(A[jj[1]]*A[ii[1]].t());
        }
    }
    else if (dir == l)
    {
        for (auto opit = O.begin(); opit!=O.end(); ++opit)
        {
            std::vector<uint> ii = num2ditvec(opit.row(),d,2);
            std::vector<uint> jj = num2ditvec(opit.col(),d,2);
            if (Ored[ii[1]][jj[1]].empty()) Ored[ii[1]][jj[1]] = (*opit)*A[ii[0]].t()*A[jj[0]];
            else Ored[ii[1]][jj[1]] += (*opit)*(A[ii[0]].t()*A[jj[0]]);
        }
    }
    else
    {
        cerr<<"wrong direction specified"<<endl;
        abort();
    }

    return Ored;
}

/**< DECOMPOSITIONS AND TRUNCATION =========================================================================================================== */

/**< single site decompositions */

template<typename KT, typename VT>
MPSBlockMat<KT,VT>
svd(BlockLam<KT>& S, BlockDiagMat<KT,VT>& DiagIso, const MPSBlockMat<KT,VT>& MPS, dirtype dir)
{
    /// This svd wrapper already hermitian conjugates V
    /// for dir = l: MPS[s] = MPSIso[s] * S * Diagiso
    /// for dir = r: MPS[s] = DiagIso * S * MPSIso[s]
    auto DenseMat = MPS.GetDenseMat(dir);
    MPSBlockMat<KT,VT> MPSIso(MPS.GetLocalDim(),MPS.GetNSites());

    S.clear();
    DiagIso.clear();

    Mat<VT> Utmp,Vtmp;
    RVecType Stmp;

    if (dir==l)
    {
        for (const auto& qnit : DenseMat) /// qnit.first = Qout, qnit.second = singlesecdata
        {
            /// qnit.first = current Qout
            /// qnit.second.first = actual dense matrix to be composed from MPS
            /// qnit.second.second = singlesecdata object to access sizes, spans and keys for current Qout

            svd_econ(Utmp,Stmp,Vtmp,qnit.second.first);

            DiagIso.emplace_hint(DiagIso.end(),qnit.first,Vtmp.t());
            S.emplace_hint(S.end(),qnit.first,Stmp);

            for (const auto& mit : qnit.second.second.sizes_spans)
            {
                /// mit.first = phys. ind. s
                /// mit.second = tuple<uint,span,KT>
                /// get<1>(mit.second) = span on the other (left) side of s
                /// get<2>(mit.second) = corresponding QN on the other (left) side of s
                /// qnit.first = quantum number on this (right) side of s -> goes into QMatPair!!
                if (!MPSIso[mit.first].emplace(*get<2>(mit.second),std::make_pair(qnit.first,Utmp(get<1>(mit.second),span::all))).second)
                    cerr<<"svd(): could not insert ("<<*get<2>(mit.second)<<","<<qnit.first<<") into MPSIso["<<mit.first<<"]"<<endl;
            }
        }
    }
    else if (dir==r)
    {
        for (const auto& qnit : DenseMat) /// qnit.first = Qout, qnit.second = singlesecdata
        {
            /// qnit.first = current Qout
            /// qnit.second.first = actual dense matrix to be composed from MPS
            /// qnit.second.second = singlesecdata object to access sizes, spans and keys for current Qout

            svd_econ(Utmp,Stmp,Vtmp,qnit.second.first);

            DiagIso.emplace_hint(DiagIso.end(),qnit.first,Utmp);
            S.emplace_hint(S.end(),qnit.first,Stmp);

            for (const auto& mit : qnit.second.second.sizes_spans)
            {
                /// mit.first = phys. ind. s
                /// mit.second = tuple<uint,span,KT>
                /// get<1>(mit.second) = span on the other (right) side of s
                /// get<2>(mit.second) = corresponding QN on the other (right) side of s -> goes into QMatPair!!
                /// qnit.first = quantum number on this (left) side of s
                if (!MPSIso[mit.first].emplace(qnit.first,std::make_pair(*get<2>(mit.second),Vtmp(get<1>(mit.second),span::all).t())).second)
                    cerr<<"svd(): could not insert ("<<qnit.first<<","<<*get<2>(mit.second)<<") into MPSIso["<<mit.first<<"]"<<endl;
            }
        }
    }
    else
    {
        cerr<<"svd(): wrong direction specified!"<<endl;
        abort();
    }

    return MPSIso;
}

//template<typename KT, typename VT>
//BlockDiagMat<KT,VT>
//qr(BlockDiagMat<KT,VT>& R, const BlockDiagMat<KT,VT>& X, dirtype dir)
//{
//    R.clear();
//    BlockDiagMat<KT,VT> Q;
////    Mat<VT> Qmat,Rmat;
//    if (dir == l)
//    {
//        for (const auto& it : X)
//        {
////            qr_pos(Qmat,Rmat,it.second);
//            auto qit = Q.emplace_hint(Q.end(),it.first,Mat<VT>);
//            auto rit = R.emplace_hint(R.end(),it.first,Mat<VT>);
//            qr_pos(qit->second,rit->second,it.second);
//        }
//    }
//    else if (dir == r)
//    {
//
//    }
//    else throw std::logic_error("qr(): wrong direction specified");
//
//    return Q;
//}

template<typename KT, typename VT>
MPSBlockMat<KT,VT>
qr(BlockDiagMat<KT,VT>& R, const MPSBlockMat<KT,VT>& MPS, dirtype dir)
{
    auto DenseMat = MPS.GetDenseMat(dir);
    MPSBlockMat<KT,VT> Q(MPS.GetLocalDim(),MPS.GetNSites());
    R.clear();

    Mat<VT> Qmat,Rmat;

    if (dir == l)
    {
        for (const auto& qnit : DenseMat) /// qnit.first = Qout, qnit.second = singlesecdata
        {
            /// qnit.first = current Qout
            /// qnit.second.first = actual dense matrix to be composed from MPS
            /// qnit.second.second = singlesecdata object to access sizes, spans and keys for current Qout
//        cout<<qnit.first<<endl;
            const typename MPSBlockMat<KT,VT>::singlesecdata& sec(qnit.second.second); /// just as a placeholder name

            qr_pos(Qmat,Rmat,qnit.second.first);

            R.emplace_hint(R.end(),qnit.first,Rmat);
            for (const auto& mit : sec.sizes_spans)
            {
                /// mit.first = phys. ind. s
                /// mit.second = tuple<uint,span,KT>
                /// get<1>(mit.second) = span on the other (left) side of s
                /// get<2>(mit.second) = corresponding QN on the other (left) side of s
                /// qnit.first = quantum number on this (right) side of s -> goes into QMatPair!!
                if (!Q[mit.first].emplace(*get<2>(mit.second),std::make_pair(qnit.first,Qmat(get<1>(mit.second),span::all))).second)
                    cerr<<"qr(): could not insert ("<<*get<2>(mit.second)<<","<<qnit.first<<") into Q["<<mit.first<<"]"<<endl;
            }
        }
    }
    else if (dir == r)
    {
        for (const auto& qnit : DenseMat) /// qnit.first = Qout, qnit.second = singlesecdata
        {
            /// qnit.first = current Qout
            /// qnit.second.first = actual dense matrix to be composed from MPS
            /// qnit.second.second = singlesecdata object to access sizes, spans and keys for current Qout
//        cout<<qnit.first<<endl;
            const typename MPSBlockMat<KT,VT>::singlesecdata& sec(qnit.second.second); /// just as a placeholder name

            qr_pos(Qmat,Rmat,qnit.second.first.t().eval());

            R.emplace_hint(R.end(),qnit.first,Rmat.t());
            for (const auto& mit : sec.sizes_spans)
            {
                /// mit.first = phys. ind. s
                /// mit.second = tuple<uint,span,KT>
                /// get<1>(mit.second) = span on the other (right) side of s
                /// get<2>(mit.second) = corresponding QN on the other (right) side of s
                /// qnit.first = quantum number on this (left) side of s -> is key for Q
                if (!Q[mit.first].emplace(qnit.first,std::make_pair(*get<2>(mit.second),Qmat(get<1>(mit.second),span::all).t())).second)
                    cerr<<"qr(): could not insert ("<<qnit.first<<","<<*get<2>(mit.second)<<") into Q["<<mit.first<<"]"<<endl;
            }
        }
    }
    else throw std::logic_error("qr(): wrong direction specified!");

    return Q;
}


template<typename KT, typename VT>
MPSBlockMat<KT,VT>
Nullspace(const MPSBlockMat<KT,VT>& MPS, dirtype dir, bool markfullrank=false)
{
    auto DenseMat = MPS.GetDenseMat(dir);
    MPSBlockMat<KT,VT> NS(MPS.GetLocalDim(),MPS.GetNSites());

    if (dir == l)
    {
        for (const auto& qnit : DenseMat)
        {
            /// qnit.first = current Qout
            /// qnit.second.first = actual dense matrix to be composed from MPS
            /// qnit.second.second = singlesecdata object to access sizes, spans and keys for current Qout
            const typename MPSBlockMat<KT,VT>::singlesecdata& sec(qnit.second.second);

//            cout<<qnit.first<<":"<<endl;
//            cout<<sec.ml_tot<<" x "<<sec.mr_tot<<endl;
            if (sec.ml_tot > sec.mr_tot)
            {
                Mat<VT> B = null(qnit.second.first.t());

                for (const auto& mit : sec.sizes_spans)
                {
                    /// mit.first = phys. ind. s
                    /// mit.second = tuple<uint,span,KT>
                    /// get<1>(mit.second) = span on the other (left) side of s
                    /// get<2>(mit.second) = corresponding QN on the other (left) side of s
                    /// qnit.first = quantum number on this (right) side of s -> goes into QMatPair!!
                    if (!NS[mit.first].emplace(*get<2>(mit.second),std::make_pair(qnit.first,B(get<1>(mit.second),span::all))).second)
                        cerr<<"nullspace(left): could not insert ("<<*get<2>(mit.second)<<","<<qnit.first<<") into NS["<<mit.first<<"]"<<endl;
                }
            }
            else
            {
                DOUT("no left null space for "<<qnit.first<<endl);
                /// we insert zero contribs to mark that this QN sector has no null space as it is already full rank
                if (markfullrank)
                {
                    for (const auto& mit : sec.sizes_spans)
                    {
                        if (!NS[mit.first].emplace(*get<2>(mit.second),std::make_pair(qnit.first,Mat<VT>(get<0>(mit.second),0))).second)
                            cerr<<"nullspace(left): could not insert zero nullspace for ("<<*get<2>(mit.second)<<","<<qnit.first<<") into NS["<<mit.first<<"]"<<endl;
                    }
                }
            }
        }
    }
    else if (dir == r)
    {
        for (const auto& qnit : DenseMat)
        {
            /// qnit.first = current Qout
            /// qnit.second.first = actual dense matrix to be composed from MPS
            /// qnit.second.second = singlesecdata object to access sizes, spans and keys for current Qout
            const typename MPSBlockMat<KT,VT>::singlesecdata& sec(qnit.second.second);

            if (sec.mr_tot > sec.ml_tot)
            {
                Mat<VT> B = null(qnit.second.first);

                for (const auto& mit : sec.sizes_spans)
                {
                    /// mit.first = phys. ind. s
                    /// mit.second = tuple<uint,span,KT>
                    /// get<1>(mit.second) = span on the other (right) side of s
                    /// get<2>(mit.second) = corresponding QN on the other (right) side of s
                    /// qnit.first = quantum number on this (left) side of s -> is key for NS
                    if (!NS[mit.first].emplace(qnit.first,std::make_pair(*get<2>(mit.second),B(get<1>(mit.second),span::all).t())).second)
                        cerr<<"nullspace(right): could not insert ("<<qnit.first<<","<<*get<2>(mit.second)<<") into NS["<<mit.first<<"]"<<endl;
                }
            }
            else
            {
                DOUT("no right null space for "<<qnit.first<<endl);
                if (markfullrank)
                {
                    for (const auto& mit : sec.sizes_spans)
                    {
                        if (!NS[mit.first].emplace(qnit.first,std::make_pair(*get<2>(mit.second),Mat<VT>(0,get<0>(mit.second) ))).second)
                            cerr<<"nullspace(right): could not insert ("<<qnit.first<<","<<*get<2>(mit.second)<<") into NS["<<mit.first<<"]"<<endl;
                    }
                }
            }
        }
    }
    else
    {
        cerr<<"wrong direction specified"<<endl;
        abort();
    }

    return NS;
}


template<typename KT,typename VT,uint NL>
void
Split(const MPSBlockMat<KT,VT>& psi, MPSBlockMat<KT,VT>& A, MPSBlockMat<KT,VT>& B, BlockLam<KT>& lam, const ItoKey<NL,KT>& I2K)
{
    assert(psi.GetNSites()>NL && "psi needs to span more than NL sites");
    assert(psi.GetLocalDim() == I2K.GetLocalDim() && "psi and I2K need to have the same physical dimension");

    auto DenseMat = psi.GetDenseSplitMat(I2K);

    Mat<VT> U,V;
    RVecType lamtmp;
    uint lamct;

    lam.clear();
    A = MPSBlockMat<KT,VT>(I2K.GetLocalDim(),NL);
    B = MPSBlockMat<KT,VT>(I2K.GetLocalDim(),psi.GetNSites() - NL);

    for (const auto& qnit : DenseMat)
    {

        const typename MPSBlockMat<KT,VT>::doublesecdata& sec(qnit.second.second);
        /// actually perform SVD (which is the main reason for all this hustle :-) )
        svd_econ(U,lamtmp,V,qnit.second.first);

        /// determine if some of the Schmidt-Values are already ~zero and discard (when there has been no mixing between left and right due to e.g. a quantum gate)
        lamct = lamtmp.n_elem;
/// TODO (valentin#1#2015-05-13): Which threshold to use here?
        while (lamct>1 && lamtmp(lamct-1)<1e-14) --lamct;
        if (lamct<lamtmp.n_elem)
        {
/// TODO (valentin#1#2015-05-13): consider resize instead of recreating with head and head_cols
            lamtmp = lamtmp.head(lamct);
            U = U.head_cols(lamct);
            V = V.head_cols(lamct);
        }

        /// put results in respective left and right MPSBlockMatrices A and B and BlockLambda lam
        lam.emplace_hint(lam.end(),qnit.first,lamtmp);

        for (const auto& mlit : sec.sizes_spans_left)
        {
            /// qnit.first = current center QN
            /// mlit.first = left phys. index
            /// get<1>(mlit.second) = span corresponding to left phys. index
            /// get<2>(mlit.second) = pointer to corresponding left QN
            if (!A[mlit.first].emplace(*get<2>(mlit.second),std::make_pair(qnit.first,U(get<1>(mlit.second),span::all))).second )
            {
                cerr<<"could not insert ("<<*get<2>(mlit.second)<<","<<qnit.first<<") into A["<<mlit.first<<"]"<<endl;
            }
        }
        for (const auto& mrit : sec.sizes_spans_right)
        {
            /// qnit.first = current center QN
            /// mrit.first = right phys. index
            /// get<1>(mrit.second) = span corresponding to right phys. index
            /// get<2>(mrit.second) = pointer to corresponding right QN
            if (!B[mrit.first].emplace(qnit.first,std::make_pair(*get<2>(mrit.second),V(get<1>(mrit.second),span::all).t())).second)
            {
                cerr<<"could not insert ("<<qnit.first<<","<<*get<2>(mrit.second)<<") into B["<<mrit.first<<"]"<<endl;
            }
        }
    }
}

template<typename KT, typename VTMPS, typename VTH, uint N>
dim_map<KT>
ExpandFromH(const SparseOperator<VTH>& H, const ItoKey<N,KT>& I2K, MPSBlockMat<KT,VTMPS>& AL, MPSBlockMat<KT,VTMPS>& AR, BlockDiagMat<KT,VTMPS>& C, uint dmmax, Real lamthresh)
{
    using VTres = typename promote_type<VTMPS,VTH>::result;
    typedef struct svdstruct
    {
        svdstruct(const KT& CK, bool isnew, const typename MPSBlockMat<KT,VTres>::doublesecdata& sec): CK(CK), isnew(isnew), sec(sec), dm(0) {};
        const KT& CK; /// center QN
        const bool isnew;
        const typename MPSBlockMat<KT,VTres>::doublesecdata& sec; /// doublesec data for M(=U*lam*V') for that center QN

        Mat<VTres> U, V;
        RVecType lam;
        uint dm;
    } svdstruct;

    assert(AL.GetNSites() + AR.GetNSites() > N);


    uint dmtot = 0;
    /// the third argument makes NL and NR contain zero matrices for sectors, which are already full rank. We need this later to distinguish between
    /// symmetry sectors that don't show up in M due to them
    /// a) being new (hence to be added without projection) or
    /// b) being zero bc/ AL and/or AR are already full rank there (hence nothing to be done)
    auto NL = Nullspace(AL,l,true);
    auto NR = Nullspace(AR,r,true);
    auto phi = ApplyOperator(AL*C*AR,H);
    auto PhiDense = phi.GetDenseSplitMat(I2K);
    auto oldlammin = min(svd(C));

    uint dimleft = AL.size();
    uint dimright = AR.size();
    assert(I2K.size()==dimleft && "AL and I2K need to be of same size");
//    uint dimleft = I2K.size();
//    assert(phi.size()%dimleft==0 && "dimension of I2K is not a divisor of the dimension of Phi");
//
//    uint dimright = phi.size()/dimleft;

    BlockDiagMat<KT,VTres> M;
    for (uint sl=0; sl<dimleft; ++sl)
    {
        for (uint sr=0; sr<dimright; ++sr)
        {
            M += NL[sl].t()*phi[sl*dimleft+sr]*NR[sr].t();
        }
    }
//    M.ShowDims("M");
//    cin.get();

    typedef std::vector<std::unique_ptr<svdstruct> > svec_type;
    typedef std::pair<Real, typename svec_type::iterator> lampair;

    svec_type svd_vec;
    svd_vec.reserve(PhiDense.size());

    std::vector<lampair> lamvec;

    for (const auto& qnit : PhiDense)
    {
        const Mat<VTres> * pmat = NULL;
        auto mit = M.find(qnit.first);
        bool isnew = false;

        if(mit==M.end()) /// newly generated QN sector
        {
            pmat = &qnit.second.first;
            isnew = true;
        }
        else /// already present sectors, check if there is maybe no left or right null space
        {
            /// empty mat in Null space causes empty mats in M
            /// empty mat in Null space means that there is no null space for this QN (sector already at full rank)
            if (mit->second.n_elem > 0) pmat = &mit->second;
            else continue;
        }

        std::unique_ptr<svdstruct> psvdtmp(new svdstruct(qnit.first,isnew,qnit.second.second));
        svd_econ(psvdtmp->U,psvdtmp->lam,psvdtmp->V,*pmat);
        dmtot += psvdtmp->lam.n_elem;

        svd_vec.emplace_back(std::move(psvdtmp));
    }


    /// if truncation necessary, fill all newly generated singular values in a big vector,
    /// but remember from which central QN they come via an iterator to the relevant element in svd_vec
    if (dmtot > dmmax)
    {
        lamvec.reserve(dmtot);
        for (auto svit = svd_vec.begin(); svit != svd_vec.end(); ++svit) for (const auto& lamit : (*svit)->lam) lamvec.emplace_back(lamit,svit);

        /// sort all singular values and throw away the smallest ones if necessary
        std::sort(lamvec.begin(),lamvec.end(),[](const lampair& lhs, const lampair& rhs)
        {
            return lhs.first > rhs.first;
        } );

        lamvec.resize(dmmax);
        for (const auto& lamit : lamvec) ++(*lamit.second)->dm;
    }
    else for (auto& svit : svd_vec) svit->dm = svit->lam.n_elem;

    dim_map<KT> dm;
    uint dmcurr;
    bool to_truncate = false;

    BlockDiagMat<KT,VTMPS> UNL,VNR;

    for (const auto& qnit : svd_vec)
    {
        dmcurr = qnit->dm;
        dm.emplace_hint(dm.end(),qnit->CK,dmcurr);

        if (dmcurr == 0) continue; /// nothing to enlarge for this QN
        dmtot += dmcurr;
        to_truncate = dmcurr < qnit->lam.n_elem;

        /// enlarge C, but only fill with zeros in order not to alter the state itself
        auto Cit = C.lower_bound(qnit->CK);
        if (Cit != C.end() && Cit->first == qnit->CK) Cit->second.resize(Cit->second.n_rows + dmcurr,Cit->second.n_cols + dmcurr);
        else C.emplace_hint(Cit,qnit->CK,Mat<VTMPS>(dmcurr,dmcurr,fill::zeros));

        if (qnit->isnew)
        {

            for (const auto& lit : qnit->sec.sizes_spans_left)
            {
                /// qnit->CK = current center QN
                /// lit.first = left phys. index
                /// get<0>(lit.second) = n_rows of corresponding block mat
                /// get<1>(lit.second) = span corresponding to left phys. index
                /// get<2>(lit.second) = pointer to corresponding left QN
                auto ALit = AL[lit.first].lower_bound(*get<2>(lit.second));
                assert(ALit == AL[lit.first].end() || ALit->first != *get<2>(lit.second));

                if (to_truncate) AL[lit.first].emplace_hint(ALit,*get<2>(lit.second),std::make_pair(qnit->CK,qnit->U(get<1>(lit.second),span(0,dmcurr-1))));
                else AL[lit.first].emplace_hint(ALit,*get<2>(lit.second),std::make_pair(qnit->CK,qnit->U(get<1>(lit.second),span::all)));
            }
            /// enlarge AR
            for (const auto& rit : qnit->sec.sizes_spans_right)
            {
                /// qnit->CK = current center QN
                /// rit.first = right phys. index
                /// get<0>(rit.second) = n_cols of corresponding block mat
                /// get<1>(rit.second) = span corresponding to right phys. index
                /// get<2>(rit.second) = pointer to corresponding right QN

                auto ARit = AR[rit.first].lower_bound(qnit->CK);
                assert(ARit == AR[rit.first].end() || ARit->first != qnit->CK);

                if (to_truncate) AR[rit.first].emplace_hint(ARit,qnit->CK,std::make_pair(*get<2>(rit.second),qnit->V(get<1>(rit.second),span(0,dmcurr-1)).t()));
                else AR[rit.first].emplace_hint(ARit,qnit->CK,std::make_pair(*get<2>(rit.second),qnit->V(get<1>(rit.second),span::all).t()));

            }

        }
        else
        {
            if (to_truncate)
            {
                UNL.emplace_hint(UNL.end(),qnit->CK,qnit->U.head_cols(dmcurr));
                VNR.emplace_hint(VNR.end(),qnit->CK,qnit->V.head_cols(dmcurr).t());
            }
            else
            {
                UNL.emplace_hint(UNL.end(),qnit->CK,qnit->U);
                VNR.emplace_hint(VNR.end(),qnit->CK,qnit->V.t());
            }
        }
    }


    auto ALn = NL*UNL;
    auto ARn = VNR*NR;

    for (uint s = 0; s < ALn.size(); ++s)
    {
        for (const auto& newit : ALn[s])
        {
/// TODO (valentin#1#2016-12-13): Why use lower_bound? find should do!
            auto ALit = AL[s].lower_bound(newit.first);
            assert(ALit != AL[s].end() && ALit->first == newit.first);

            QMat(*ALit) = join_rows(QMat(*ALit),QMat(newit)); /// U in ALn is already truncated
        }
    }

    for (uint s = 0; s < ARn.size(); ++s)
    {
        for (const auto& newit : ARn[s])
        {
/// TODO (valentin#1#2016-12-13): Why use lower_bound? find should do!
            auto ARit = AR[s].lower_bound(newit.first);
            assert(ARit != AR[s].end() && ARit->first == newit.first);

            QMat(*ARit) = join_cols(QMat(*ARit),QMat(newit)); /// V in ARn is already truncated
        }
    }

    return dm;
}

template<typename KT, typename VTMPS, typename VTH, uint N>
dim_map<KT>
ExpandFromH_old(const SparseOperator<VTH>& H, const ItoKey<N,KT>& I2K, MPSBlockMat<KT,VTMPS>& AL, MPSBlockMat<KT,VTMPS>& AR, BlockDiagMat<KT,VTMPS>& C, uint dmmax, Real lamthresh)
{
    uint dmtot = 0;
    auto NL = Nullspace(AL,l);
    auto NR = Nullspace(AR,r);
    auto phi = ApplyOperator(AL*C*AR,H);
    auto PhiDense = phi.GetDenseSplitMat(I2K);
    auto NLdense = NL.GetDenseMat(l);
    auto NRdense = NR.GetDenseMat(r);
    auto oldlammin = min(svd(C));

//    phi.ShowDims("phi");
//    AL.ShowDims("AL");
//    AR.ShowDims("AR");
//    NL.ShowDims("NL");
//    NR.ShowDims("NR");
//    cin.get();



/// TODO (valentin#1#2016-04-06): consider only using ALdense and ARdense to construct phi

    /// define a struct that holds all necessary information to perform the enlargement of AL and AR after the SVD has been calculated.
    /// We need to postpone the enlargement itself until after ALL new singular values have been calculated in order to be able to discard
    /// the smallest ones if necessary.
    typedef struct svdstruct
    {
        svdstruct(const KT& CK, const typename MPSBlockMat<KT,VTMPS>::doublesecdata& sec): CK(CK), sec(sec), dm(0) {};
        const KT& CK; /// center QN
        Mat<VTMPS> U, V;
        RVecType lam;
        typename MPSBlockMat<KT,VTMPS>::dense_map::const_iterator nlit, nrit; /// iterators to dense blocks of left and right null spaces for that center QN
        const typename MPSBlockMat<KT,VTMPS>::doublesecdata& sec; /// doublesec data for M(=U*lam*V') for that center QN
        uint dm;
    } svdstruct;


    typedef std::vector<std::unique_ptr<svdstruct> > svec_type;
    typedef std::pair<Real, typename svec_type::iterator> lampair;

    svec_type svd_vec;
    std::vector<lampair> lamvec;

    svd_vec.reserve(PhiDense.size()); /// we need at most that many entries

    for (auto& qnit : PhiDense)
    {
        cout<<qnit.first<<":"<<endl;
        /// qnit.first = current center QN
        /// qnit.second = pair(Mat,doublesecdata)
        /// qnit.second.first = actual dense matrix for the current center QN
        /// qnit.second.second = doublesecdata with info on contributing physical indices and the left and right QN

        cout<<"M:"<<endl;
        for (const auto& phys : qnit.second.second.v_phys_ind)
        {
            uint left=get<0>(phys),right=get<1>(phys);
            cout<<"left: ["<<left<<"]: "<<*get<2>(qnit.second.second.sizes_spans_left[left])<<",\t";
            cout<<"right: ["<<right<<"]: "<<*get<2>(qnit.second.second.sizes_spans_right[right])<<endl;
        }
        /// check whether the current symmetry sector has already small Schmidt-values anyways;
        auto lamit = oldlammin.find(qnit.first);
        if (lamit!=oldlammin.end() && lamit->second < lamthresh) continue;

        Mat<VTMPS>& M(qnit.second.first); /// use reference just as a placeholder, (s.t. we don't have to write qnit.second.first all the time)

        /// we need to use a unique_ptr, as svdstruct is not default constructible and we need to move the generated
        /// pointer into svd_vec after filling it (e.g. shared_ptr is not movable)
        std::unique_ptr<svdstruct> psvdtmp(new svdstruct(qnit.first,qnit.second.second));

        psvdtmp->nlit = NLdense.find(qnit.first);
        /// nlit->first = current QN (right QN on NL)
        /// nlit->second = pair(mat,singlesecdata)
        /// nlit->second.first = dense matrix representation for current right QN
        /// nlit->second.second = singlesecdata containing contributing phys. indices, ranges and QN of left side, etc.

        if (psvdtmp->nlit != NLdense.end())
        {
            cout<<"NL"<<endl;
            for (const auto& phys : psvdtmp->nlit->second.second.v_phys_ind)
            {
                uint left=phys.first;
                auto tmp = psvdtmp->nlit->second.second.sizes_spans.find(left);
                if (tmp!=psvdtmp->nlit->second.second.sizes_spans.end()) cout<<"left: ["<<left<<"]: "<<*get<2>(tmp->second)<<endl;
                else cout<<"left: ["<<left<<"] not contributing"<<endl;
            }
            /// if the nullspace has zero dimension (i.e. corresponding dense matrix has full rank), we cannot perform any enlargement
            /// do nothing in this case and continue
            if (psvdtmp->nlit->second.first.n_cols==0) continue; //{cout<<"NL continuing"<<endl; continue;}
            M = psvdtmp->nlit->second.first.t()*M;
        }
        else cerr<<qnit.first<<" not found as right QN of NL"<<endl;

        psvdtmp->nrit = NRdense.find(qnit.first);
        /// nrit->first = current QN (left QN on NR)
        /// nrit->second = pair(mat,singlesecdata)
        /// nrit->second.first = dense matrix representation for current left QN
        /// nrit->second.second = singlesecdata containing contributing phys. indices, ranges and QN of right side, etc.

        if (psvdtmp->nrit != NRdense.end())
        {
            cout<<"NR"<<endl;
            for (const auto& phys : psvdtmp->nrit->second.second.v_phys_ind)
            {
                uint right=phys.first;
                auto tmp = psvdtmp->nrit->second.second.sizes_spans.find(right);
                if (tmp!=psvdtmp->nrit->second.second.sizes_spans.end()) cout<<"right: ["<<right<<"]: "<<*get<2>(tmp->second)<<endl;
                else cout<<"right: ["<<right<<"] not contributing"<<endl;
            }
            /// if the nullspace has zero dimension (i.e. corresponding dense matrix has full rank), we cannot perform any enlargement
            /// do nothing in this case and continue
            if (psvdtmp->nrit->second.first.n_rows==0) continue; //{cout<<"NR continuing"<<endl; continue;}
            M = M*psvdtmp->nrit->second.first.t();
        }
        else cerr<<qnit.first<<" not found as left QN of NR"<<endl;

        svd_econ(psvdtmp->U,psvdtmp->lam,psvdtmp->V,M);

//        uvec inds = find(psvdtmp->lam > lamthresh,1,"last");
//        if (inds.is_empty()) {cout<<"skipping "<<qnit.first<<endl;continue;} /// if all new lambdas are smaller than lamthresh, disregard and immediately move on
//
//        cout<<psvdtmp->lam.n_elem<<", "<<inds(0)<<endl;
//        cin.get();
//
//        if (inds(0) < psvdtmp->lam.n_elem - 1)
//        {
////            psvdtmp->lam.print("lam");
////            cout<<"new length: "<<inds(0)+1<<endl;
////            cin.get();
//            psvdtmp->lam.resize(inds(0) + 1);
//            psvdtmp->U.resize(psvdtmp->U.n_rows,inds(0) + 1);
//            psvdtmp->V.resize(psvdtmp->V.n_rows,inds(0) + 1);
//        }

        dmtot += psvdtmp->lam.n_elem;

        svd_vec.emplace_back(std::move(psvdtmp));
    }

    /// if truncation necessary, fill all newly generated singular values in a big vector,
    /// but remember from which central QN they come via an iterator to the relevant element in svd_vec
    if (dmtot > dmmax)
    {
        lamvec.reserve(dmtot);
        for (auto svit = svd_vec.begin(); svit != svd_vec.end(); ++svit) for (const auto& lamit : (*svit)->lam) lamvec.emplace_back(lamit,svit);

        /// sort all singular values and throw away the smallest ones if necessary
        std::sort(lamvec.begin(),lamvec.end(),[](const lampair& lhs, const lampair& rhs)
        {
            return lhs.first > rhs.first;
        } );

        lamvec.resize(dmmax);
        for (const auto& lamit : lamvec) ++(*lamit.second)->dm;
    }
    else for (auto& svit : svd_vec) svit->dm = svit->lam.n_elem;

    cout<<"From old scheme ("<<lamvec.size()<<" elements)"<<endl;
    uint ct=0;
    for (const auto& lamit: lamvec) cout<<++ct<<": "<<(*(lamit.second))->CK<<": "<<lamit.first<<endl;

    dim_map<KT> dm;
    uint dmcurr;
//    dmtot = 0;
    bool to_truncate = false;

    for (const auto& qnit : svd_vec)
    {
        dmcurr = qnit->dm;
        dm.emplace_hint(dm.end(),qnit->CK,dmcurr);

        if (dmcurr == 0) continue; //{cout<<"nothing to enlarge for "<<qnit->CK<<endl;cin.get(); continue;}
        dmtot += dmcurr;
        to_truncate = dmcurr < qnit->lam.n_elem;

        /// enlarge C, but only fill with zeros in order not to alter the state itself
        auto Cit = C.lower_bound(qnit->CK);
        if (Cit != C.end() && Cit->first == qnit->CK) Cit->second.resize(Cit->second.n_rows + dmcurr,Cit->second.n_cols + dmcurr);
        else C.emplace_hint(Cit,qnit->CK,Mat<VTMPS>(dmcurr,dmcurr,fill::zeros));

        /// enlarge AL
        for (const auto& lit : qnit->sec.sizes_spans_left)
        {
            /// qnit->CK = current center QN
            /// lit.first = left phys. index
            /// get<0>(lit.second) = n_rows of corresponding block mat
            /// get<1>(lit.second) = span corresponding to left phys. index
            /// get<2>(lit.second) = pointer to corresponding left QN

            auto ALit = AL[lit.first].lower_bound(*get<2>(lit.second));
            if (ALit != AL[lit.first].end() && ALit->first==*get<2>(lit.second))
            {
                assert(qnit->nlit != NLdense.end());
                if (to_truncate) QMat(*ALit) = join_rows(QMat(*ALit),qnit->nlit->second.first(get<1>(lit.second),span::all)*qnit->U.head_cols(dmcurr));
                else QMat(*ALit) = join_rows(QMat(*ALit),qnit->nlit->second.first(get<1>(lit.second),span::all)*qnit->U);
            }
            else
            {
                assert(qnit->nlit == NLdense.end());
                if (to_truncate) AL[lit.first].emplace_hint(ALit,*get<2>(lit.second),std::make_pair(qnit->CK,qnit->U(get<1>(lit.second),span(0,dmcurr-1))));
                else AL[lit.first].emplace_hint(ALit,*get<2>(lit.second),std::make_pair(qnit->CK,qnit->U(get<1>(lit.second),span::all)));
            }
        }

        /// enlarge AR
        for (const auto& rit : qnit->sec.sizes_spans_right)
        {
            /// qnit->CK = current center QN
            /// rit.first = right phys. index
            /// get<0>(rit.second) = n_cols of corresponding block mat
            /// get<1>(rit.second) = span corresponding to right phys. index
            /// get<2>(rit.second) = pointer to corresponding right QN

            auto ARit = AR[rit.first].lower_bound(qnit->CK);
            if (ARit != AR[rit.first].end() && ARit->first==qnit->CK)
            {
                assert(qnit->nrit != NRdense.end());
                if (to_truncate) QMat(*ARit) = join_cols(QMat(*ARit),qnit->V.head_cols(dmcurr).t() * qnit->nrit->second.first(span::all,get<1>(rit.second)));
                else QMat(*ARit) = join_cols(QMat(*ARit),qnit->V.t() * qnit->nrit->second.first(span::all,get<1>(rit.second)));
            }
            else
            {
                assert(qnit->nrit == NRdense.end());
                if (to_truncate) AR[rit.first].emplace_hint(ARit,qnit->CK,std::make_pair(*get<2>(rit.second),qnit->V(get<1>(rit.second),span(0,dmcurr-1)).t()));
                else AR[rit.first].emplace_hint(ARit,qnit->CK,std::make_pair(*get<2>(rit.second),qnit->V(get<1>(rit.second),span::all).t()));
            }
        }
    }

    return dm;
}

template<typename KT,typename VT, uint N>
void
Truncate(MPSBlockMat<KT,VT>& A, MPSBlockMat<KT,VT>& B, BlockLam<KT>& lam, const ItoKey<N,KT>& I2K, uint mmax, Real lamthresh=1e-10)
{
    /** we will have to distinguish two cases where we have to do something:
     *
     * 1.) Schmidt values are smaller than lamthresh and are thrown out any way
     * 2.) After considering 1.) there are more than mmax remaining Schmidt values and we can only keep the mmax largest ones
     *
     * For this we will fill all Schmidt values that are larger than lamthresh from the BlockLam lam into a single vector. If the size of this
     * vector is then still larger than mmax we will have to throw away additional ones. If any of 1.) or 2.) applies we will have to perform some truncation,
     * otherwise nothing needs to be done.
     */


    assert(A.GetNSites()==N);
    bool vectoobig=false,valtoosmall=false; /// if any of these two are true we need to do some truncation

    /// necessary local typedefs
    typedef typename BlockLam<KT>::iterator lam_iter_type;
    std::vector<std::pair<uint,lam_iter_type> > size_vec;
    typedef typename std::vector<std::pair<uint,lam_iter_type> >::iterator size_iter_type;

    lam_iter_type lamit;
    size_iter_type sizit;

    typedef std::pair<Real,size_iter_type> literpair;
    std::vector<literpair> lamvec;
    lamvec.reserve(lam.GetNElem());

/// TODO (valentin#1#2015-05-13): try not to use two loops through lam and to grow size_vec along with looping through lam
    size_vec.reserve(lam.size());
    for (lamit=lam.begin(); lamit!=lam.end(); ++lamit) size_vec.emplace_back(0,lamit);
    for (lamit=lam.begin(),sizit = size_vec.begin(); lamit!=lam.end(); ++lamit,++sizit)
    {
        for(const auto& vit : lamit->second)
        {
            if (vit>lamthresh) lamvec.emplace_back(vit,sizit);
            else valtoosmall=true;
        }
    }
    vectoobig = lamvec.size() > mmax;

    if (vectoobig || valtoosmall)
    {
        /// if there are too many remaining Schmidt-values, we have to determine the smallest ones to throw away.
        /// For this we needed to put all of them in a single vector, such that we can now sort it and then only keep the largest mmax ones.
        if (vectoobig)
        {
            std::sort(lamvec.begin(),lamvec.end(),[](const literpair& lhs, const literpair& rhs) -> bool {return lhs.first > rhs.first;});
            lamvec.resize(mmax);
        }

        /// determine new bond dimensions for each QN. It can either be that we have truncated some Schmidt values away in the above sort/truncate process
        /// or some Schmidt values were too small to begin with and we didn't even put them in the single vector.
        /// Since we also already stored the iterators to the QN/size map, we can easily loop through all the elements of the large lam vector and
        /// increase the size for the corresponding QN without searching the map!
        for (const auto& vit : lamvec)
        {
            /// vit.first = lam value
            /// vit.second = size_vec iterator
            /// vit.second->first = size
            /// vit.second->second = lam_iter_type
            ++(vit.second->first);
        }

        uint siz=0;
        for (const auto& vit : size_vec)
        {
            /// vit.first = size
            /// vit.second = lam_iter_type
            /// vit.second->first = KT
            /// vit.second->second = lam in sector of KT
            siz = vit.first;
            const KT& QN = vit.second->first;

            assert(siz <= vit.second->second.n_elem);
            if (siz < vit.second->second.n_elem) /// only do something if the size has shrunk
            {
                if (siz > 0) /// if smaller, but still finite, perform truncation
                {
                    vit.second->second.resize(siz);
                    for (uint s=0; s<A.size(); ++s)
                    {
                        auto it = A[s].find(QN - I2K[s]);
                        if (it!=A[s].end()) QMat(*it).resize(QMat(*it).n_rows,siz);
                    }
                    for (uint s=0; s<B.size(); ++s)
                    {
                        auto it = B[s].find(QN);
                        if (it!=B[s].end()) QMat(*it).resize(siz,QMat(*it).n_cols);
                    }
                }
                else /// if completely gone, erase from containers
                {
                    /// unfortunately we have to search for QN for every s in both A and B. Idk if this could be avoided?
                    for (uint s=0; s<A.size(); ++s) A[s].erase(QN - I2K[s]);
                    for (uint s=0; s<B.size(); ++s) B[s].erase(QN);

                    /// IMPORTANT: erase from lambda only after erasing from A and B as otherwise the reference to QN becomes invalid (iterator vit.second does not exist anymore)!
                    if(lam.erase(vit.second)==vit.second)cerr<<QN<<" should have been erased from lam!"<<endl;
//                    else cout<<"erased "<<QN<<" from lam"<<endl;

                }
            }
        }
    }
//    else cout<<"nothing to do"<<endl;
}

/// TODO (valentin#1#): Check how to do symmetric gauge
template<typename VT,typename KT>
void
CheckOrthoSingle(const MPSBlockMat<KT,VT>& A, const BlockDiagMat<KT,VT>& CL, const BlockDiagMat<KT,VT>& CR, dirtype dir)
{
    Real chk1=0,chk2=0;
//    BlockDiagMat<KT,VT> L,R,lammat;
    std::string str;
    switch (dir)
    {
    case l:
        str = "left";
        chk1 = norm(ApplyTMLeft(A) - eye<VT>(A.GetMr()));
        chk2 = norm(ApplyTMRight(A,CR*CR.t()) - CL*CL.t());
        break;
    case r:
        str = "right";
        chk1 = norm(ApplyTMLeft(A,CL.t()*CL)-CR.t()*CR);
        chk2 = norm(ApplyTMRight(A) - eye<VT>(A.GetMl()));
        break;
//    case s:
//        str="symmetric";
//        lammat = lam;
//        L = ApplyTMmixedLeftDiag(lam>>A,A) - lammat;
//        R = ApplyTMmixedRightDiag(A<<lam,A) - lammat;
//        for (const auto& it : L) chk1 += pow(norm(it.second,"fro"),2)/it.second.n_elem;
//        for (const auto& it : R) chk2 += pow(norm(it.second,"fro"),2)/it.second.n_elem;
//        for (const auto& it : L) chk1 += dot(it.second)/it.second.n_elem;
//        for (const auto& it : R) chk2 += dot(it.second)/it.second.n_elem;
//        break;
    case c:
        str="canonical";
        chk1 = norm(ApplyTMLeft(A,CL.t()*CL) - CR.t()*CR);
        chk2 = norm(ApplyTMRight(A,CR*CR.t()) - CL*CL.t());
        break;
    default:
        cerr<<"wrong direction specified"<<endl;
    }
    cout<<"check "<<str<<" gauge:"<<endl;
    cout<<"left: "<<chk1<<", right: "<<chk2<<endl;
}

template<typename VT,typename KT>
void
CheckOrthoLRSqrt(const MPSBlockMatArray<KT,VT>& AL,
                 const MPSBlockMatArray<KT,VT>& AR,
                 const BlockDiagMatArray<KT,VT>& C)
{
    uint N=AL.size();
    assert(AR.size()==N && "AL and AR need to be of same length");
    assert(C.size()==N && "AL and C need to be of same length");
    auto PBC = [N](int x) -> int {return (x + N)%N;};

    cout<<"Checking single layer gauge |AL(n)*C(n) - C(n-1)*AR(n)|"<<endl;
    for (uint n=0;n<N;++n)
    {
//        cout<<n+1<<": "<<norm(AL[n]*C[n] - C[PBC(n-1)]*AR[n])<<endl;
//        cout<<n+1<<": "<<norm_inf(AL[n]*C[n] - C[PBC(n-1)]*AR[n])<<endl;
        cout<<n+1<<": "<<norm_scaled(AL[n]*C[n] - C[PBC(n-1)]*AR[n])<<endl;
    }
    cout<<std::string(100,'-')<<endl;
}

template<typename KT, typename MPST, typename OT>
std::map<std::string,RVecType>
//std::vector<RVecType>
MeasureObservables(const std::vector<SparseOperator<OT> >& obs,
                   const MPSBlockMatArray<KT,MPST>& A,
                   const BlockDiagMatArray<KT,MPST>& L=BlockDiagMatArray<KT,MPST>(),
                   const BlockDiagMatArray<KT,MPST>& R=BlockDiagMatArray<KT,MPST>(),
                   bool verbose=false)
{
    uint N = A.size();
    auto PBC = [N](int x) -> int {return (x + N)%N;};

    if (!L.empty()) assert(L.size()==N && "A and L need to be of same size");
    if (!R.empty()) assert(R.size()==N && "A and R need to be of same size");

    std::map<std::string,RVecType> expval;
//    std::vector<RVecType> expval;
//    expval.reserve(obs.size());

    dirtype dir = s; /// A has no particular gauge
    if (L.empty()) /// A = AL or AC
    {
        if (R.empty()) throw std::logic_error("MeasureObservables(): A needs either left or right RDM"); /// A = AC, center site gauge, would only work for single site observables
        else dir = l; /// A = AL, left gauge
    }
    else if (R.empty()) dir = r; /// A = AR, right gauge
    /// else: no gauge, i.e. both L and R are needed.

    for (const auto& obsit : obs)
    {
        uint NS = obsit.GetNSites();
        /// do not use expval[name] here, as in the subsequent for-loop, this call would search the map expval for every 0<=n<N!
        auto iexp = expval.emplace(std::make_pair(obsit.GetName(),RVecType(N))).first; /// emplace returns a pair (iterator,bool), we only need the iterator
//        RVecType tmpval(N);
        for (uint n=0; n<N; ++n)
        {
            /// TODO (valentin#1#): the copying to Atmp is not necessary if NS=1, account for that?
            MPSBlockMat<KT,MPST> Atmp(A[n]);
            if (obsit.GetNSites() > 1) for (uint l=1; l<NS; ++l) Atmp = Atmp*A[PBC(n+l)];

                 if (dir == l) iexp->second(n) = trace(ApplyOpTMLeftGen(obsit,Atmp)*R[PBC(n+NS-1)]);
            else if (dir == r) iexp->second(n) = trace(L[PBC(n-1)]*ApplyOpTMRightGen(obsit,Atmp));
            else               iexp->second(n) = trace(ApplyOpTMLeftGen(obsit,L[PBC(n-1)]*Atmp,Atmp)*R[PBC(n+NS-1)]); /// dir = s
//                 if (dir == l) tmpval(n) = trace(ApplyOpTMLeftGen(obsit,Atmp)*R[PBC(n+NS-1)]);
//            else if (dir == r) tmpval(n) = trace(L[PBC(n-1)]*ApplyOpTMRightGen(obsit,Atmp));
//            else               tmpval(n) = trace(ApplyOpTMLeftGen(obsit,L[PBC(n-1)]*Atmp,Atmp)*R[PBC(n+NS-1)]); /// dir = s
//            expval.emplace_back(tmpval);
        }

        if (verbose)
        {
            iexp->second.raw_print(obsit.GetName());
            cout<<"mean: "<<mean(iexp->second)<<endl;
//            tmpval.raw_print(obsit.GetName());
//            cout<<"mean: "<<mean(tmpval)<<endl;
            cout<<std::string(100,'-')<<endl;
        }
    }

    return expval;
}

/// single operator case
template<typename KT, typename MPST, typename OT>
inline
RVecType
MeasureObservables(const SparseOperator<OT>& obs,
                   const MPSBlockMatArray<KT,MPST>& A,
                   const BlockDiagMatArray<KT,MPST>& L=BlockDiagMatArray<KT,MPST>(),
                   const BlockDiagMatArray<KT,MPST>& R=BlockDiagMatArray<KT,MPST>(),
                   bool verbose=false)
{
    return MeasureObservables(std::vector<SparseOperator<OT> >({obs}),A,L,R,verbose).begin()->second;
//    return MeasureObservables(std::vector<SparseOperator<OT> >({obs}),A,L,R,verbose).front();
}

template<typename KT, typename VT>
void
randMPS_LR(MPSBlockMatArray<KT,VT>& ALvec,
           MPSBlockMatArray<KT,VT>& ARvec,
           BlockDiagMatArray<KT,VT>& Cvec,
           BlockLamArray<KT>& Lamvec,
           uint N, const ItoKey<1,KT>& I2K,
           const std::vector<dim_map<KT> >& dimvec,
           bool verbose=false)
{
    ALvec.clear();
    ARvec.clear();
    Cvec.clear();

    BlockDiagMat<KT,VT> Rtmp;

    for (uint n=1; n<N; ++n) ALvec.emplace_back(qr(Rtmp,MPSBlockMat<KT,VT>(I2K,dimvec[n-1],dimvec[n],fill::randn),l));
    ALvec.emplace_back(qr(Rtmp,MPSBlockMat<KT,VT>(I2K,dimvec.back(),dimvec.front(),fill::randn),l));

    BlockDiagMat<KT,VT> L,R,U,C,V;
    TMDominantEig(ALvec,R,r);

    BlockLam<KT> DR;
//    eig_sym(U,DR,R);
    svd(U,DR,V,R,"left");
//    DR.print("DR");

    BlockLam<KT> lam = sqrt(DR);
    qr(C,U<<lam,r); /// transform C into lower triangular form (no adaptation to ALvec necessary!)

    BlockDiagMat<KT,VT> Ctmp(C);
    for (auto iter = ALvec.crbegin(); iter!=ALvec.crend(); ++iter)
    {
        Cvec.emplace_front(Ctmp);
        Lamvec.emplace_front(svd(Ctmp));
        ARvec.emplace_front(qr(Ctmp,(*iter)*Ctmp,r));
    }
    assert(ALvec.size()==N);
    assert(ALvec.size()==ARvec.size());
    assert(ALvec.size()==Cvec.size());

    if (verbose) CheckOrthoLRSqrt(ALvec,ARvec,Cvec);
}


template<typename AT>
inline
void
rotate(AT& arr, uint middle)
{
    /// rotates the array such that middle is the first position within the array.
    /// Counting starts at 1, so 1 <= middle <= N (where middle=1 is trivial)
    /// e.g. rotate({1,2,3,4,5},3) = {3,4,5,1,2}
    /// for stl containers, std::rotate calls the container's inherent swap operation, which essentially
    /// just swaps dynamic pointers, so no actual data is copied/swapped. This can therefore safely be called on vectors/deques etc. without speed impact
    assert(middle <= arr.size() && middle>0 && "middle needs to be a valid position within arr.");
    if (middle > 1) std::rotate(arr.begin(),arr.begin()+(middle-1),arr.end());
}


template<typename AT>
inline
void
shift(AT& arr, int shift)
{
    /// shifts the array 'arr' by 'shift' sites.
    /// If shift > 0 the array is rotated such that all original sites are shifted to the RIGHT by 'shift' sites afterwards, e.g. shift({1,2,3},1) = {3,1,2}
    /// If shift < 0 the array is rotated such that all original sites are shifted to the LEFT by 'shift' sites afterwards, e.g. shift({1,2,3},-1) = {2,3,1}
    int N = arr.size();
    if (abs(shift) >= N)
    {
        shift = shift%N;
        cerr<<"shift(): shifting by more than "<<N-1<<", setting to "<<shift<<endl;
    }
    if (shift > 0) rotate(arr,static_cast<uint>(N - shift + 1));
    else if (shift < 0) rotate(arr,static_cast<uint>(1 - shift));
}

template<typename KT, typename VT, uint N>
bool
operator==(const MPSBlockMat<KT,VT>& A, const ItoKey<N,KT>& I2K)
{
    bool is_equal = A.GetNSites() == N && (A.GetLocalDim() == I2K.GetLocalDim());
    if (is_equal)
    {
        for (uint s=0; s<A.size(); ++s)
        {
            for (const auto& Ait : A[s]) is_equal = is_equal && (Qout(Ait) == Qin(Ait) + I2K[s]);
        }
    }
    return is_equal;
}

template<typename KT, typename VT, uint N>
bool
operator==(const MPSBlockMatArray<KT,VT>& A, const ItoKey<N,KT>& I2K)
{
    bool is_equal = true;
    for (const auto& Ait : A) is_equal = is_equal && Ait == I2K;
    return is_equal;
}


template<typename KT, typename VT, uint N>
inline
bool
operator!=(const MPSBlockMat<KT,VT>& A, const ItoKey<N,KT>& I2K)
{
    return !(A==I2K);
}

template<typename KT, typename VT, uint N>
inline
bool
operator!=(const MPSBlockMatArray<KT,VT>& A, const ItoKey<N,KT>& I2K)
{
    return !(A==I2K);
}

/**< SAVING AND LOADING IMPS */
template<typename KT, typename VT>
bool
saveIMPS(const BlockLamArray<KT>& Lamvec, const BlockDiagMatArray<KT,VT>& Cvec, const MPSBlockMatArray<KT,VT>& ALvec, const MPSBlockMatArray<KT,VT>& ARvec, std::string name, std::string ending="bin", std::string folder=".")
{
    bool success = true;

    /// check for file existence and append number if necessary
    auto tmpname = GetUniqueFileName(name,ending,folder);

    /// open and save to file
    std::ofstream file(tmpname, std::fstream::binary);

    success = success && Lamvec.save(file);
    if (!success)
    {
        cerr<<"failed to save LAM"<<endl;
        file.close();
        return false;
    }
    success = success && Cvec.save(file);
    if (!success)
    {
        cerr<<"failed to save C"<<endl;
        file.close();
        return false;
    }
    success = success && ALvec.save(file);
    if (!success)
    {
        cerr<<"failed to save AL"<<endl;
        file.close();
        return false;
    }
    success = success && ARvec.save(file);
    if (!success)
    {
        cerr<<"failed to save AR"<<endl;
        file.close();
        return false;
    }
    cout<<"saved UMPS to "<<tmpname<<endl;
    return success;
}

template<typename KT, typename VT, typename GO>
bool
loadIMPS(BlockLamArray<KT>& Lamvec, BlockDiagMatArray<KT,VT>& Cvec, MPSBlockMatArray<KT,VT>& ALvec, MPSBlockMatArray<KT,VT>& ARvec, std::string filename, const GO& GroupObj)
{
    bool success = true;
    if (!RegFileExist(filename))
    {
        cerr<<"file "<<filename<<" not found"<<endl;
        return false;
    }

    std::ifstream file(filename,std::ifstream::binary);
    success = success && Lamvec.load(file,GroupObj);
    if (!success)
    {
        cerr<<"failed to load LAM"<<endl;
        return false;
    }
    success = success && Cvec.load(file,GroupObj);
    if (!success)
    {
        cerr<<"failed to load C"<<endl;
        return false;
    }
    success = success && ALvec.load(file,GroupObj);
    if (!success)
    {
        cerr<<"failed to load AL"<<endl;
        return false;
    }
    success = success && ARvec.load(file,GroupObj);
    if (!success)
    {
        cerr<<"failed to load AR"<<endl;
        return false;
    }

    cout<<"loaded "<<filename<<endl;
    file.close();
    return success;
}

#endif // MPS_BLOCK_UTIL_H

