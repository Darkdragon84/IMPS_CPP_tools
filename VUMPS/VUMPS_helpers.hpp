#ifndef VUMPS_HELPERS_H
#define VUMPS_HELPERS_H
template<typename KT, typename VT>
BlockDiagMat<KT,VT>
GetHL(const MPSBlockMatArray<KT,VT>& AL, const RSpOp& H)
{
//    MPSBlockMat<KT,VT> AA = AL.back()*AL.front();
//    BlockDiagMat<KT,VT> HL = ApplyTMmixedLeftDiag(ApplyOperator(AA,H),AA);
    BlockDiagMat<KT,VT> HL = ApplyOpTMLeftDiag(H,AL.back()*AL.front());

    for (auto iter = AL.cbegin()+1; iter!=AL.cend(); ++iter)
    {
//        AA = (*(iter-1))*(*iter);
//        HL = ApplyTMLeft(*iter,HL) + ApplyTMmixedLeftDiag(ApplyOperator(AA,H),AA);
        HL = ApplyTMLeft(*iter,HL) + ApplyOpTMLeftDiag(H,(*(iter-1))*(*iter));
    }
    return HL;
}

template<typename KT, typename VT>
BlockDiagMat<KT,VT>
GetHR(const MPSBlockMatArray<KT,VT>& AR, const RSpOp& H)
{
//    MPSBlockMat<KT,VT> AA = AR.back()*AR.front();
//    BlockDiagMat<KT,VT> HR = ApplyTMmixedRight(ApplyOperator(AA,H),AA);
    BlockDiagMat<KT,VT> HR = ApplyOpTMRightDiag(H,AR.back()*AR.front());

    /// for backward iterators, iter+1 is actually the element BEFORE iter!
    for (auto iter = AR.crbegin()+1; iter!=AR.crend(); ++iter)
    {
//        AA = (*iter)*(*(iter-1));
//        HR = ApplyTMRight(*iter,HR) + ApplyTMmixedRight(ApplyOperator(AA,H),AA);
        HR = ApplyTMRight(*iter,HR) + ApplyOpTMRightDiag(H,(*iter)*(*(iter-1)));
    }
    return HR;
}

template<typename KT, typename VT>
void
ApplyHA(const MPSBlockMat<KT,VT>& in, MPSBlockMat<KT,VT>& out, const BlockDiagMat<KT,VT>& EHL, const BlockDiagMat<KT,VT>& EHR, const RedOp<BlockMat<KT,VT> >& HL, const RedOp<BlockMat<KT,VT> >& HR)
{
    /// MAKE SURE OUT IS INITIALIZED TO ZERO
    assert(HL.GetLocalDim() == in.GetLocalDim() && "HL and MPSin need to be of same physical dimension");
    assert(HR.GetLocalDim() == in.GetLocalDim() && "HR and MPSin need to be of same physical dimension");
    assert(HL.GetNSites() == in.GetNSites() && "HL and MPSin need to span the same amount of sites");
    assert(HR.GetNSites() == in.GetNSites() && "HR and MPSin need to span the same amount of sites");

    out += EHL*in + in*EHR;
    for (uint i=0; i<in.size(); ++i)
    {
        for (uint j=0; j<in.size(); ++j) out[i] += HL[i][j]*in[j] + in[j]*HR[i][j];
    }
}

template<typename KT, typename VT>
void
ApplyHAvec(VT* in,
           VT* out,
           const std::vector<dimkeypair_vec<KT> >& Adims,
           uint d,
           uint Am_tot,
           const BlockDiagMat<KT,VT>& EHL,
           const BlockDiagMat<KT,VT>& EHR,
           const RedOp<BlockMat<KT,VT> >& HL,
           const RedOp<BlockMat<KT,VT> >& HR)
{
    assert(Adims.size() == d && "sizesvector Adims must be of length d");
    MPSBlockMat<KT,VT> MPSin(d),MPSout(d);
    memset(out,0.,Am_tot*(sizeof(*out)));

    uint pos=0,ml=0,mr=0;
    for (uint s=0; s<d; ++s)
    {
        for (const auto& dimit : Adims[s])
        {
            /// get<0>(dimit) = Qin
            /// get<1>(dimit) = Qout
            /// get<2>(dimit) = ml
            /// get<3>(dimit) = mr
            ml = get<2>(dimit);
            mr = get<3>(dimit);
            MPSin[s].emplace_hint(MPSin[s].end(),get<0>(dimit),std::make_pair(get<1>(dimit),MatType(&in[pos],ml,mr,false,true)));
            MPSout[s].emplace_hint(MPSout[s].end(),get<0>(dimit),std::make_pair(get<1>(dimit),MatType(&out[pos],ml,mr,false,true)));
            pos += ml*mr;
        }
    }
    ApplyHA(MPSin,MPSout,EHL,EHR,HL,HR);
}


template<typename KT, typename VT>
void
ApplyHC(const BlockDiagMat<KT,VT>& in,
        BlockDiagMat<KT,VT>& out,
        const BlockDiagMat<KT,VT>& EHL,
        const BlockDiagMat<KT,VT>& EHR,
        const SparseOperator<VT>& H,
        const MPSBlockMat<KT,VT>& AL,
        const MPSBlockMat<KT,VT>& AR)
{
    /// MAKE SURE OUT IS INITIALIZED TO ZERO
    out += EHL*in + in*EHR;
    std::vector<uint> ii,jj;
    uint d = H.GetLocalDim();
    for (auto opit = H.begin(); opit != H.end(); ++opit)
    {
        ii = num2ditvec(opit.row(),d,2);
        jj = num2ditvec(opit.col(),d,2);
        out += (*opit)*AL[ii[0]].t()*AL[jj[0]]*in*AR[jj[1]]*AR[ii[1]].t();
    }
}


template<typename KT, typename VT>
void
ApplyHCvec(VT* in, VT* out, const dimpair_vec<KT>& Cdims, uint Cm_tot, const BlockDiagMat<KT,VT>& EHL, const BlockDiagMat<KT,VT>& EHR, const SparseOperator<VT>& H, const MPSBlockMat<KT,VT>& AL, const MPSBlockMat<KT,VT>& AR)
{
    memset(out,0.,Cm_tot*(sizeof(*out)));

    BlockDiagMat<KT,VT> inmat(in,Cdims,false,true);
    BlockDiagMat<KT,VT> outmat(out,Cdims,false,true);

//    BlockDiagMat<KT,VT> inmat,outmat;
//    uint pos=0;
//    for (const auto& dimit : Cdims)
//    {
//        inmat.emplace_hint(inmat.end(),dimit.first,MatType(&in[pos],dimit.second,dimit.second,false,true));
//        outmat.emplace_hint(outmat.end(),dimit.first,MatType(&out[pos],dimit.second,dimit.second,false,true));
//        pos += dimit.second*dimit.second;
//    }

    ApplyHC(inmat,outmat,EHL,EHR,H,AL,AR);
}

template<typename KT, typename VT, typename FT>
inline
Real
GradNormLeft(const MPSBlockMat<KT,VT>& AL, const BlockDiagMat<KT,VT>& C, FT&& HAfun)
{
//    return norm(ApplyTMmixedLeft(HAfun(AL*C),Nullspace(AL,l)));
//    return norm_inf(ApplyTMmixedLeft(HAfun(AL*C),Nullspace(AL,l)));
    return norm_scaled(ApplyTMmixedLeft(HAfun(AL*C),Nullspace(AL,l)));
}

template<typename KT, typename VT, typename FT>
inline
Real
GradNormRight(const MPSBlockMat<KT,VT>& AR, const BlockDiagMat<KT,VT>& C, FT&& HAfun)
{
//    return norm(ApplyTMmixedRight(HAfun(C*AR),Nullspace(AR,r)));
//    return norm_inf(ApplyTMmixedRight(HAfun(C*AR),Nullspace(AR,r)));
    return norm_scaled(ApplyTMmixedRight(HAfun(C*AR),Nullspace(AR,r)));
}

template<typename KT, typename VT, typename FT>
inline
Real
GradNorm(const MPSBlockMat<KT,VT>& A, const BlockDiagMat<KT,VT>& C, FT&& HAfun, dirtype dir)
{
    Real F = 0;
    if (dir==l) F = GradNormLeft(A,C,HAfun);
    else if (dir==r) F = GradNormRight(A,C,HAfun);
    else throw std::domain_error("GradNorm: wrong direction specified");
    return F;
}

//template<typename KT, typename VT, typename FT>
//Real
//GradNorm(const MPSBlockMat<KT,VT>& AR, const BlockDiagMat<KT,VT>& C, FT&& HAfun)
//{
////    MPSBlockMat<KT,VT> dAC = HAfun(AL*C);
////    BlockDiagMat<KT,VT> dC = ApplyTMmixedLeft(dAC,AL);
////    return norm(dAC - AL*dC);
//    MPSBlockMat<KT,VT> dAC = HAfun(C*AR);
//    BlockDiagMat<KT,VT> dC = ApplyTMmixedRight(dAC,AR);
//    return norm(dAC - dC*AR);
//}

#endif // VUMPS_HELPERS_H
