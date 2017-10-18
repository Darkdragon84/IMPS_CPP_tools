#ifndef EXCITATIONS_HELPERS
#define EXCITATIONS_HELPERS


//template<typename KT, typename VTA, typename VTX>
//inline
//BlockMat<KT,VTX>
//GetEBR(const MPSBlockMatArray<KT,VTA>& AL,
//       const MPSBlockMatArray<KT,VTA>& AR,
//       const BlockMat<KT,VTX>& x,
//       Complex kfac,
//       Real tol,
//       bool verbose = false)
//{
//    return InvertE_fac(AL,AR,x,r,kfac,tol,0,BlockMat<KT,VTX>(),verbose);
//}


/// TODO (valentin#1#2016-10-31): implement BlockDiagMat version, that also needs to project for EBR and EHBL
template<typename KT, typename VTA, typename VTX, typename VTH>
void
ApplyHeff(const BlockMatArray<KT,VTX>& Xin,
                BlockMatArray<KT,VTX>& Xout,
          Complex kfac,
          const MPSBlockMatArray<KT,VTA>& AL,
          const MPSBlockMatArray<KT,VTA>& AR,
          const MPSBlockMatArray<KT,VTA>& NL,
          const BlockMat<KT,VTA>& LM, /// left dominant eigenmatrix of mixed TM \sum_s AL[s] \otimes conj(AR[r])
          const BlockMat<KT,VTA>& RM, /// right dominant eigenmatrix of mixed TM \sum_s AL[s] \otimes conj(AR[r])
          const SparseOperator<VTH>& H,
          const BlockDiagMatArray<KT,VTA>& HLtot,
          const BlockDiagMatArray<KT,VTA>& HRtot,
          Real InvETol=1e-14,
          bool verbose=false)//,
//          BlockMat<KT,VTX>* pEBR,
//          BlockMat<KT,VTX>* pEHBL)
{
    /// MAKE SURE XOUT IS INITIALIZED TO ZERO AND CONTAINS THE CORRECT SYMMETRY SECTORS
    /// CHECK OUTSIDE IF ALL ARRAYS HAVE THEIR PROPER LENGTHS

/// TODO (valentin#1#): further optimize the generation of ABR, EBR, HBL, EHBL and see if we can generate the respective versions per site on the fly
/// TODO (valentin#1#): Add functionality to recycle EBR and EHBL from last iteration. Possible solution: pass along pointers to these objects from the heap and have them allocated at the beginning before calling eigs
    bool proj = (!LM.empty() && !RM.empty());
//    if (proj) cout<<"projecting"<<endl;

    uint N = Xin.size();
    assert(Xout.size() == N);
    for (uint n=0;n<N;++n) assert(Xin[n].GetSizesVector() == Xout[n].GetSizesVector());
    for (uint n=0;n<N;++n) assert(Xin[n].GetSizesVector() == Xout[n].GetSizesVector());

    uint d = H.GetLocalDim();
    Complex ckfac = std::conj(kfac);

    auto PBC = [N](int x) -> int {return (x + N)%N;};

    MPSBlockMatArray<KT,VTX> Bin,Bout;
    BlockMatArray<KT,VTX> ABR,HBL,ABRtot;
    BlockMatArray<KT,VTX> ABR2(N),HBL2(N),ABRtot2(N);
    BlockMat<KT,VTX> EHBL,EBR;
//    BlockMatArray<KT,VTX> EBR;

    /// create B mats from current X vector (Xin)
    for (uint n=0;n<N;++n) Bin.emplace_back(NL[n]*Xin[n]);

    /// constants for terms where Bin is right of Bout

    /// cumulative contribution of AB overlaps to the right within same UC (checked)
    ABR.emplace_front(ApplyTMmixedRight(Bin.back(),AR.back()));
    {
        auto ALit = AL.crbegin() + 1;
        auto ARit = AR.crbegin() + 1;
        auto Bit = Bin.crbegin() + 1;
        for ( ;ALit != AL.crend(); ++ALit, ++ARit, ++Bit)
        {
            ABR.emplace_front(ApplyTMmixedRight(*Bit + (*ALit)*ABR.front(),*ARit));
        }
    }

    /// contributions from all other UC to the right (checked)
    if (proj) EBR = InvertE_proj_fac(AL,AR,ABR.front(),LM,RM,r,kfac,InvETol,0,BlockMat<KT,VTX>(),verbose);
    else EBR = InvertE_fac(AL,AR,ABR.front(),r,kfac,InvETol,0,BlockMat<KT,VTX>(),verbose);
//    if (pEBR == nullptr)
//    {
//        EBR = InvertE_fac(AL,AR,ABR.front(),r,kfac,max(tol/100,InvETol),0,BlockMat<KT,VTX>(),verbose);
//    }
//    else
//    {
//        *pEBR = InvertE_fac(AL,AR,ABR.front(),r,kfac,max(tol/100,InvETol),0,*pEBR,verbose);
//        EBR = *pEBR;
//    }



    /// combined AB overlap contribs of both within and right of current UC (checked)
    ABRtot.emplace_front(kfac*EBR);
//    ABRtot.emplace_front(kfac*(*pEBR));
//    ABRtot.emplace_front(kfac*EBR.front());
    {
        auto ALit = AL.crbegin();
        auto ARit = AR.crbegin();
        auto ABRit = ABR.crbegin();
        auto Aend = AL.crend()-1;
        for ( ; ALit!=Aend; ++ALit,++ARit,++ABRit)
        {
            EBR = ApplyTMmixedRight(*ALit,*ARit,EBR);
            ABRtot.emplace_front(*ABRit + kfac*EBR);
//            (*pEBR) = ApplyTMmixedRight(*ALit,*ARit,(*pEBR));
//            ABRtot.emplace_front(*ABRit + kfac*(*pEBR));
//            EBR.emplace_front(ApplyTMmixedRight(*ALit,*ARit,EBR.front()));
//            ABRtot.emplace_front(*ABRit + kfac*EBR.front());
        }
    }
    shift(ABRtot,1);/// bring last element to the front


    /// constants for terms where Bin is left of Bout (checked)
    /// these are all terms where H is left of or on Bin

    /// cumulative contributions from same UC (checked)
    /// we also need to include one term from next left unit cell, to which the Hamiltonian connects (term with ckfac below)
    HBL.emplace_back(ApplyTMmixedLeft(HLtot.back()*Bin.front(),AL.front()) +
                     ApplyOpTMLeftGen(H,AL.back()*Bin.front() + ckfac*(Bin.back()*AR.front()),AL.back()*AL.front()));

    {
        auto ALit = AL.cbegin() + 1;
        auto ARit = AR.cbegin() + 1;
        auto Bit = Bin.cbegin() + 1;
        auto HLit = HLtot.cbegin();
        for ( ; ALit!=AL.cend(); ++ALit,++ARit,++Bit,++HLit)
        {
            HBL.emplace_back(ApplyTMmixedLeft(HBL.back()*(*ARit) + (*HLit)*(*Bit),*ALit) +
                             ApplyOpTMLeftGen(H,(*(ALit-1))*(*Bit) + (*(Bit-1))*(*ARit),(*(ALit-1))*(*ALit) ) );
        }
    }

    /// contribs from all other UC to the left (checked)
    if (proj) EHBL = InvertE_proj_fac(AR,AL,HBL.back(),LM.t(),RM.t(),l,ckfac,InvETol,0,BlockMat<KT,VTX>(),verbose);
    else EHBL = InvertE_fac(AR,AL,HBL.back(),l,ckfac,InvETol,0,BlockMat<KT,VTX>(),verbose);

//    if (pEHBL == nullptr)
//    {
//        EHBL = InvertE_fac(AR,AL,HBL.back(),l,ckfac,max(tol/100,InvETol),0,BlockMat<KT,VTX>(),verbose);
//        pEHBL = &EHBL;
//    }
//    else
//    {
//        *pEHBL = InvertE_fac(AR,AL,HBL.back(),l,ckfac,max(tol/100,InvETol),0,*pEHBL,verbose);
//        EHBL = *pEHBL;
//    }

    /************************************************************************************************/
    /** terms where Bin and Bout are on top of each other (checked)                                **/
    /************************************************************************************************/

    /// h left and right of B's (checked)
    /// first fill and expand Bout to length N here
    for (uint n=0; n<N; ++n)
    {
        Bout.emplace_back(HLtot[PBC(n-1)]*Bin[n] + Bin[n]*HRtot[PBC(n+1)]);
    }

    /// h on B's (checked)
    for (uint n=0; n<N; ++n)
    {
        auto HAB = ApplyOperator(AL[PBC(n-1)]*Bin[n],H);
        auto HBA = ApplyOperator(Bin[n]*AR[PBC(n+1)],H);
        for (uint s=0; s<d; ++s)
        {
            for (uint k=0; k<d; ++k)
            {
                Bout[n][s] += AL[PBC(n-1)][k].t()*HAB[k*d+s] + HBA[s*d+k]*AR[PBC(n+1)][k].t();
            }
        }
    }

    /************************************************************************************************/
    /** terms where Bin is left of Bout  (checked)                                                 **/
    /************************************************************************************************/

    /// n=0 has only contribs from next left UC (checked)
    Bout.front() += ckfac*EHBL*AR.front();
//    Bout.front() += ckfac*(*pEHBL)*AR.front();

    auto HBA = ApplyOperator(Bin.back()*AR.front(),H);
    for (uint s=0; s<d; ++s)
    {
        for (uint k=0; k<d; ++k)
        {
            Bout.front()[s] += ckfac*AL.back()[k].t()*HBA[k*d+s];
        }
    }


    /// n>0 also has contribs from within same UC (checked)
    for (uint n=1; n<N; ++n)
    {
        EHBL = ApplyTMmixedLeft(AR[n-1],AL[n-1],EHBL);
        Bout[n] += (HBL[n-1] + ckfac*EHBL)*AR[n];
//        (*pEHBL) = ApplyTMmixedLeft(AR[n-1],AL[n-1],*pEHBL);
//        Bout[n] += (HBL[n-1] + ckfac*(*pEHBL))*AR[n];

        auto HBA = ApplyOperator(Bin[n-1]*AR[n],H);
        for (uint s=0; s<d; ++s)
        {
            for (uint k=0; k<d; ++k)
            {
                Bout[n][s] += AL[n-1][k].t()*HBA[k*d+s];
            }
        }
    }

    /************************************************************************************************/
    /** terms where Bin is right of Bout (checked)                                                 **/
    /************************************************************************************************/
    for (uint n=0; n<N; ++n)
    {
        auto Atmp = Bin[PBC(n+1)] + AL[PBC(n+1)]*ABRtot[PBC(n+2)];
        if (n==N-1) Atmp*=kfac;

        Bout[n] += HLtot[PBC(n-1)]*AL[n]*ABRtot[PBC(n+1)];

        auto HAA = ApplyOperator(AL[PBC(n-1)]*(AL[n]*ABRtot[PBC(n+1)]),H);
        auto HAB = ApplyOperator(AL[n]*Atmp,H);

        for (uint s=0; s<d; ++s)
        {
            for (uint k=0; k<d; ++k)
            {
                Bout[n][s] += AL[PBC(n-1)][k].t()*HAA[k*d+s];
                Bout[n][s] += HAB[s*d+k]*AR[PBC(n+1)][k].t();
            }
        }
    }

    /************************************************************************************************/
    /** calculate Xout from Bout                                                                   **/
    /************************************************************************************************/
    for (uint n=0; n<N; ++n) Xout[n] += ApplyTMmixedLeft(Bout[n],NL[n]);
}

template<typename KT, typename VTA, typename VTX, typename VTH>
void
ApplyHeff(const VTX* in,
                VTX* out,
          const std::vector<dimkeypair_vec<KT> >& xdims,
          uint mtot,
          Complex kfac,
          const MPSBlockMatArray<KT,VTA>& AL,
          const MPSBlockMatArray<KT,VTA>& AR,
          const MPSBlockMatArray<KT,VTA>& NL,
          const BlockMat<KT,VTA>& LM, /// left dominant eigenmatrix of mixed TM \sum_s AL[s] \otimes conj(AR[r])
          const BlockMat<KT,VTA>& RM, /// right dominant eigenmatrix of mixed TM \sum_s AL[s] \otimes conj(AR[r])
          const SparseOperator<VTH>& H,
          const BlockDiagMatArray<KT,VTA>& HLtot,
          const BlockDiagMatArray<KT,VTA>& HRtot,
          Real InvETol=1e-14,
          bool verbose=false)//,
//          BlockMat<KT,VTX>* pEBR,
//          BlockMat<KT,VTX>* pEHBL)
{
    std::fill(out,out+mtot,0.);

    BlockMatArray<KT,VTX> Xin(in,xdims,false,true);
    BlockMatArray<KT,VTX> Xout(out,xdims,false,true);

//    memset(out,0.,mtot*sizeof(*out));
//    BlockMatArray<KT,VTX> Xin,Xout;
//    uint pos=0;
//    for (const auto& xvit : xdims)
//    {
//        Xin.emplace_back(BlockMat<KT,VTX>(&in[pos],xvit,false,true));
//        Xout.emplace_back(BlockMat<KT,VTX>(&out[pos],xvit,false,true));
//        pos += xvit.GetNElem();
//
////        Xin.emplace_back(BlockMat<KT,VTX>());
////        Xout.emplace_back(BlockMat<KT,VTX>());
////
////        for (const auto& dimit : xvit)
////        {
////            uint ml = get<2>(dimit);
////            uint mr = get<3>(dimit);
////
////             Xin.back().emplace_hint(Xin.back().end(),
////                                     get<0>(dimit),
////                                     std::make_pair(get<1>(dimit),
////                                                    Mat<VTX>(const_cast<VTX*>(&in[pos]),ml,mr,false,true)
////                                                    )
////                                     );
////
////            Xout.back().emplace_hint(Xout.back().end(),
////                                     get<0>(dimit),
////                                     std::make_pair(get<1>(dimit),
////                                                    Mat<VTX>(&out[pos],ml,mr,false,true)
////                                                    )
////                                     );
////            pos += ml*mr;
////        }
//
//    }

//    ApplyHeff(Xin,Xout,kfac,AL,AR,NL,H,HLtot,HRtot,tol,InvETol,verbose,pEBR,pEHBL);
    ApplyHeff(Xin,Xout,kfac,AL,AR,NL,LM,RM,H,HLtot,HRtot,InvETol,verbose);
//    ApplyHeff(Xin,Xout,kfac,AL,AR,NL,H,HLtot,HRtot,tol,InvETol,verbose);
}


template<typename KT, typename VTA, typename VTO, typename VTH>
void
HeffConstants(BlockDiagMatArray<KT,VTO>& HLtot,
              BlockDiagMatArray<KT,VTO>& HRtot,
              const MPSBlockMatArray<KT,VTA>& AL,
              const MPSBlockMatArray<KT,VTA>& AR,
              const BlockDiagMat<KT,VTA>& L,
              const BlockDiagMat<KT,VTA>& R,
              const SparseOperator<VTH>& H,
              double tol=1e-14,
              int maxit=0,
              bool verbose=false)
{
    /// MAKE SURE THAT E0 IS ALREADY SUBTRACTED FROM H!
    uint N = AL.size();
    auto PBC = [N](int x) -> int {return (x + N)%N;};

    BlockDiagMatArray<KT,VTO> HL,HR;
    BlockDiagMatArray<KT,VTO> EHL(N),EHR(N);
    HLtot.resize(N);
    HRtot.resize(N);

    HL.emplace_back(ApplyOpTMLeftDiag(H,AL.back()*AL.front()));
    for (auto iter = AL.cbegin() + 1; iter != AL.cend(); ++iter)
    {
        HL.emplace_back(ApplyOpTMLeftDiag(H,(*(iter-1))*(*iter))  + ApplyTMLeft(*iter,HL.back()) ); /// here, *iter is AL[n] and *(iter-1) is AL[n-1]
    }
    if (abs(trace(HL.back()*R)) > 1e-14)
    {
        cerr<<"tr(HL*R)="<<trace(HL.back()*R)<<endl;
//        throw std::domain_error("it seems E0 is not subtracted from H");
    }


    HR.emplace_front(ApplyOpTMRightDiag(H,AR.back()*AR.front()));
    for (auto iter = AR.crbegin() + 1; iter != AR.crend(); ++iter)
    {
        HR.emplace_front(ApplyOpTMRightDiag(H,(*iter)*(*(iter-1))) + ApplyTMRight(*iter,HR.front())); /// here, *iter is AR[n] and *(iter-1) is AR[n+1]
    }
    if (abs(trace(L*HR.front())) > 1e-14)
    {
        cerr<<"tr(L*HR)="<<trace(L*HR.front())<<endl;
//        throw std::domain_error("it seems E0 is not subtracted from H");
    }

/// TODO (valentin#1#): switch to iterators

    EHL.back() = InvertE_proj(AL,HL.back(),eye<VTO>(R.GetMr()),R,l,tol,maxit,BlockDiagMat<KT,VTO>(),verbose);
    HLtot.back() = EHL.back();

    for (uint n=0;n<N-1;++n)
    {
        EHL[n] = ApplyTMLeft(AL[n],EHL[PBC(n-1)]);
        HLtot[n] = HL[n] + EHL[n];
    }

    EHR.front() = InvertE_proj(AR,HR.front(),L,eye<VTO>(L.GetMr()),r,tol,maxit,BlockDiagMat<KT,VTO>(),verbose);
    HRtot.front() = EHR.front();

    for (uint n=N-1;n>0;--n)
    {
        EHR[n] = ApplyTMRight(AR[n],EHR[PBC(n+1)]);
        HRtot[n] = EHR[n] + HR[n];
    }

/// iterator part
//    EHL.emplace_back(InvertE_proj(AL,HL.back(),eye<VTO>(C.back().GetMr()),R,l,tol));
//    cout<<trace(EHL.back()*R)<<endl;
//
//    HLtot.emplace_back(EHL.back());
////    auto citer = C.cbegin();
//    auto HLiter = HL.cbegin();
//
//    for (auto iter = AL.cbegin(); iter != AL.cend() - 1; ++iter, ++HLiter)//, ++citer)
//    {
//        EHL.emplace_back(ApplyTMLeft(*iter,EHL.back()));
//        HLtot.emplace_back(*HLiter + EHL.back());
////        cout<<trace(EHL.back()*(*citer)*citer->t())<<endl;
////        cout<<trace(HLtot.back()*(*citer)*citer->t())<<endl;
//    }
//    shift(EHL,-1);
//    shift(HLtot,-1);
//
//
//    EHR.emplace_back(InvertE_proj(AR,HR.front(),L,eye<VTO>(C.back().GetMr()),r,tol));
//    cout<<trace(L*EHR.back())<<endl;

}


template<typename KT, typename VT>
uint
XDims(std::vector<dimkeypair_vec<KT> >& dimvec, const MPSBlockMatArray<KT,VT>& NL, const MPSBlockMatArray<KT,VT>& AR, const KT& K)
{

    uint N = NL.size();
    assert(AR.size() == N && "NL and AR need to be of same length");

    auto PBC = [N](int x) -> int {return (x + N)%N;};

//    std::vector<dimkeypair_vec<KT> > dimvec;
    dimvec.clear();
    dimvec.reserve(N);

    uint mtot = 0;

    for (uint n=0;n<N;++n)
    {
        auto ml = NL[n].GetMr();
        auto mr = AR[PBC(n+1)].GetMl();

        dimkeypair_vec<KT> dims;
        for (const auto& mlit : ml)
        {
            auto mrit = mr.find(mlit.first + K);
            /// actually, we shouldn't need the size checks anymore, but keep them for safety (adapted Nullspace with a switch to contain null matrices where full rank)
            if (mrit != mr.end() && mlit.second > 0 && mrit->second > 0)
            {
                dims.emplace_back(std::make_tuple(mlit.first,mrit->first,mlit.second,mrit->second));
                mtot += mlit.second*mrit->second;
            }
//            else cerr<<mlit.first + K<<" not found in mr"<<endl;
//            if (mrit != mr.end()) dims.emplace_back(std::make_tuple(mlit.first,mrit->first,mlit.second,mrit->second));
        }

        dimvec.emplace_back(dims);
    }

    return mtot;
}

template<typename KT, typename VT>
void
ApplyTransformation(MPSBlockMatArray<KT,VT>& A, BlockDiagMatArray<KT,VT>& C, const std::string opstring, const ItoKey<1,KT> I2K, const modptr& pmod)
{
    typename MPSBlockMatArray<KT,VT>::fun_type MPSFun;
    typename BlockDiagMatArray<KT,VT>::fun_type CFun;

    uint N = A.size();
    uint d = A.front().size();
    std::vector<std::vector<bool> > signvec(N);

    switch (pmod->GetModelType())
    {
    case ModelBase::XXZ:
        if (opstring == "FS")
        {
            MPSFun = [&](const MPSBlockMat<KT,VT>& in) -> MPSBlockMat<KT,VT> {return in.FlipQN(I2K);};
            CFun = [&](const BlockDiagMat<KT,VT>& in) -> BlockDiagMat<KT,VT> {return in.FlipQN();};
        }
        else throw std::domain_error("ApplyTransformation: transformation for XXZ not recognized");

        A = A.ApplyFun(MPSFun);
        C = C.ApplyFun(CFun);
        break;
    case ModelBase::FHUB:

        if (opstring=="FS")
        {
            #ifdef FHUB_NM_REP
            MPSFun = [&](const MPSBlockMat<KT,VT>& in) -> MPSBlockMat<KT,VT> {return in.FlipQN(I2K,{false,true});};
            CFun = [&](const BlockDiagMat<KT,VT>& in) -> BlockDiagMat<KT,VT> {return in.FlipQN({false,true});};
            #else
            MPSFun = [&](const MPSBlockMat<KT,VT>& in) -> MPSBlockMat<KT,VT> {return in.PermuteQN(I2K,{1,0});};
            CFun = [&](const BlockDiagMat<KT,VT>& in) -> BlockDiagMat<KT,VT> {return in.PermuteQN({1,0});};
            #endif // HUB_NM_REP
            std::fill(signvec.begin(),signvec.end(),std::vector<bool>({false,false,false,true}));
        }
        else if (opstring == "FC")
        {
            if (N % 2 != 0) throw std::logic_error("transform_state: Unit cell size needs to be even for charge flip");

            #ifdef FHUB_NM_REP
            MPSFun = [&](const MPSBlockMat<KT,VT>& in) -> MPSBlockMat<KT,VT> {return in.FlipQN(I2K,{true,false});};
            CFun = [&](const BlockDiagMat<KT,VT>& in) -> BlockDiagMat<KT,VT> {return in.FlipQN({true,false});};
            #else
            MPSFun = [&](const MPSBlockMat<KT,VT>& in) -> MPSBlockMat<KT,VT> {return in.FlipQN(I2K).PermuteQN(I2K,{1,0});};
            CFun = [&](const BlockDiagMat<KT,VT>& in) -> BlockDiagMat<KT,VT> {return in.FlipQN().PermuteQN({1,0});};
            #endif // HUB_NM_REP
            for (uint n=0; n<N; n+=2)
            {
                signvec[n]   = std::vector<bool>({false,false,true,false});
                signvec[n+1] = std::vector<bool>({true,false,true,true});
            }

        }
        else throw std::domain_error("transform_state: transformation for FHUB not recognized");

        A = A.ApplyFun(MPSFun);
        C = C.ApplyFun(CFun);

        /// fermionic signs
        assert(signvec.size()==N);
        for (uint n=0; n<N; ++n)
        {
            assert(signvec[n].size()==d);
            for (uint s=0;s<d;++s) if (signvec[n][s]) A[n][s] *= -1;
        }

        break;
    default:
        throw std::domain_error("transform_state: transformations for this model not implemented");
    }
}


template<typename KT, typename VT>
void
teststuff(const MPSBlockMatArray<KT,VT>& ALvec,
          const MPSBlockMatArray<KT,VT>& ARvec,
          const BlockDiagMatArray<KT,VT>& Lvec,
          const BlockDiagMatArray<KT,VT>& Rvec,
          const BlockMat<KT,VT>& LM,
          const BlockMat<KT,VT>& RM,
          Real OLL, Real OLR)
{

    BlockDiagMat<KT,VT> L = Lvec.back();
    BlockDiagMat<KT,VT> R = Rvec.back();

    cout<<"check L: "<<norm(ApplyTMLeft(ARvec,L) - L)<<endl;
    cout<<"check R: "<<norm(ApplyTMRight(ALvec,R) - R)<<endl;

    cout<<"check LM: "<<norm(ApplyTMmixedLeft(ALvec,ARvec,LM) - OLL*LM)<<endl;
    cout<<"check RM: "<<norm(ApplyTMmixedRight(ALvec,ARvec,RM) - OLR*RM)<<endl;

    cout<<"check LM': "<<norm(ApplyTMmixedLeft(ARvec,ALvec,LM.t()) - OLL*LM.t())<<endl;
    cout<<"check RM': "<<norm(ApplyTMmixedRight(ARvec,ALvec,RM.t()) - OLR*RM.t())<<endl;


    BlockDiagMat<IKey,Scalar> xd,yd1,yd2,tmpd1,tmpd2,ID;
    /// left version
    xd = BlockDiagMat<IKey,Scalar>(R.GetSizesVector(),fill::randn);
    yd1 = InvertE_proj(ALvec,xd,eye<Scalar>(R.GetMr()),R,l,1e-14,0,BlockDiagMat<IKey,Scalar>(),true);
    yd2 = InvertE_proj_fac(ALvec,ALvec,xd,eye<Scalar>(R.GetMr()),R,l,1.0,1e-14,0,BlockDiagMat<IKey,Scalar>(),true);

    tmpd1 = yd1;
    tmpd2 = yd2;

    ID = eye<Scalar>(R.GetSizesVector());

    for (auto Ait=ALvec.cbegin(); Ait!=ALvec.cend();++Ait)
    {
        tmpd1 = ApplyTMLeft(*Ait,tmpd1);
        tmpd2 = ApplyTMLeft(*Ait,tmpd2);
    }
//
    (tmpd1*R).ShowDims("tmp1*R");
    (tmpd2*R).ShowDims("tmp2*R");
    cout<<"tr(x*R) = "<<trace(xd*R)<<endl;
    cout<<"tr(yd1*R) = "<<trace(yd1*R)<<endl;
    cout<<"tr(yd2*R) = "<<trace(yd2*R)<<endl;
    cout<<"tr(tmp1*R) = "<<trace(tmpd1*R)<<endl;
    cout<<"tr(tmp2*R) = "<<trace(tmpd2*R)<<endl;
    cout<<"tmp1: "<<norm(yd1 - tmpd1 - xd + (trace(xd*R) + trace(yd1*R))*ID)/norm(xd)<<endl;
    cout<<"tmp2: "<<norm(yd2 - tmpd2 - xd + (trace(xd*R) + trace(yd2*R))*ID)/norm(xd)<<endl;

    /// right version
    xd = BlockDiagMat<IKey,Scalar>(L.GetSizesVector(),fill::randn);
    yd1 = InvertE_proj(ARvec,xd,L,eye<Scalar>(L.GetMr()),r,1e-14,0,BlockDiagMat<IKey,Scalar>(),true);
    yd2 = InvertE_proj_fac(ARvec,ARvec,xd,L,eye<Scalar>(L.GetMr()),r,1.0,1e-14,0,BlockDiagMat<IKey,Scalar>(),true);

    tmpd1 = yd1;
    tmpd2 = yd2;

    ID = eye<Scalar>(L.GetSizesVector());

    for (auto Ait=ARvec.crbegin(); Ait!=ARvec.crend();++Ait)
    {
        tmpd1 = ApplyTMRight(*Ait,tmpd1);
        tmpd2 = ApplyTMRight(*Ait,tmpd2);
    }
    (L*tmpd1).ShowDims("L*tmp1");
    (L*tmpd2).ShowDims("L*tmp2");
    cout<<"tr(L*x) = "<<trace(L*xd)<<endl;
    cout<<"tr(L*yd1) = "<<trace(L*yd1)<<endl;
    cout<<"tr(L*yd12 = "<<trace(L*yd2)<<endl;
    cout<<"tr(L*tmp1) = "<<trace(L*tmpd1)<<endl;
    cout<<"tr(L*tmp2) = "<<trace(L*tmpd2)<<endl;
    cout<<"tmp1: "<<norm(yd1 - tmpd1 - xd + (trace(L*xd) + trace(L*yd1))*ID)/norm(xd)<<endl;
    cout<<"tmp2: "<<norm(yd2 - tmpd2 - xd + (trace(L*xd) + trace(L*yd2))*ID)/norm(xd)<<endl;

    /// left version
//    LM.ShowDims("LM");
//    RM.ShowDims("RM");
//    auto xnd = BlockMat<IKey,Scalar>(LM.GetSizesVector(),fill::randn);
//    auto ynd = InvertE_proj_fac(ALvec,ARvec,xnd,LM,RM,l,-1.0,1e-14,0,BlockMat<IKey,Scalar>(),true);
//    auto ynd = InvertE_fac(ALvec,ARvec,xnd,l,0.8,1e-14,0,BlockMat<IKey,Scalar>(),true);
//    BlockMat<IKey,Scalar> tmpnd(ynd);
//    for (uint n=0; n<N; ++n) tmpnd = ApplyTMmixedLeft(ALvec[n],ARvec[n],tmpnd);
//    cout<<norm(ynd - 0.8*tmpnd - xnd)/norm(xnd)<<endl;
}

#endif // EXCITATIONS_HELPERS