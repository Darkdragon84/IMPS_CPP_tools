#ifndef INV_E_BLOCK_FUNCTIONS
#define INV_E_BLOCK_FUNCTIONS

//#include "MPSBlockUtilities.hpp"
//#include "IterativeSolvers.hpp"
//#include "helpers.hpp"

/**< GEOMETRIC SUM OF TRANSFER MATRICES ====================================================================================================== */

/// TODO (valentin#1#): implement BlockMat version of InvEfun_proj

 /// implements y = x - TM(x) + trace(x*rho_in)*rho_out
template<typename KT, typename VT>
Col<VT>
InvEfun_proj(const Col<VT>& invec,
             const std::function<BlockDiagMat<KT,VT> (const BlockDiagMat<KT,VT>&)>& TMfun,
             const BlockDiagMat<KT,VT>& rho_inside,
             const BlockDiagMat<KT,VT>& rho_outside,
             const dimpair_vec<KT>& dims,
             uint m_tot)
{
    Col<VT> outvec(invec); /// this already handles the y = x part

    BlockDiagMat<KT,VT> in(invec,dims,false,true);
    BlockDiagMat<KT,VT> out(outvec,dims,false,true);

    VT trtmp{0};
    {
        auto rhoit = rho_inside.cbegin();
        auto init = in.cbegin();
        for ( ;init!=in.end(); ++init, ++rhoit) trtmp += trace(init->second * rhoit->second); /// trace(x*rho_in) part
//        for ( ;init!=in.cend() && rhoit!=rho_inside.cend(); ++init, ++rhoit)
//            trtmp += trace(init->second * rhoit->second); /// trace(x*rho_in) part
    }

    BlockDiagMat<KT,VT> Tx = TMfun(in); /// TM(x) part
    { /// use extra brackets to limit scope of iterators
        auto outit = out.begin();
        auto Txit = Tx.cbegin();
        auto rhoit = rho_outside.cbegin();
        /// combine all three parts to y = x - TM(x) + trace(x*rho_in)*rho_out
        for (; outit!=out.end(); ++outit, ++Txit, ++rhoit) outit->second += trtmp*rhoit->second - Txit->second;
//        for (; outit!=out.end() && Txit!=Tx.cend() && rhoit!=rho_outside.cend(); ++outit, ++Txit, ++rhoit)
//            outit->second += trtmp*rhoit->second - Txit->second;
    }
    return outvec;
}


 /// implements y = x - fac*TM(x) + trace(x*rho_in)*rho_out
 /// this pretty much only makes sense when |fac|=1 and arg(fac) is equal to an argument of an eigenvalue of the mixed TM with magnitude ~ 1
template<typename KT, typename VT, typename VTR>
Col<VT>
InvEfun_proj_fac(const Col<VT>& invec,
                 const std::function<BlockDiagMat<KT,VT> (const BlockDiagMat<KT,VT>&)>& TMfun,
                 const BlockDiagMat<KT,VTR>& rho_inside,
                 const BlockDiagMat<KT,VTR>& rho_outside,
                 const dimpair_vec<KT>& dims,
                 VT fac,
                 uint m_tot)
{
    Col<VT> outvec(invec); /// this already handles the y = x part

    BlockDiagMat<KT,VT> in(invec,dims,false,true);
    BlockDiagMat<KT,VT> out(outvec,dims,false,true);

    VT trtmp{0};
    {
        auto rhoit = rho_inside.cbegin();
        auto init = in.cbegin();
        for ( ; init!=in.cend() && rhoit != rho_inside.cend() ; ++init, ++rhoit)
            trtmp += trace(init->second * rhoit->second); /// trace(x*rho_in) part
    }

    BlockDiagMat<KT,VT> Tx = TMfun(in); /// TM(x) part
    { /// use extra brackets to limit scope of iterators
        auto outit = out.begin();
        auto Txit = Tx.cbegin();
        auto rhoit = rho_outside.cbegin();
        /// combine all three parts to y = x - fac*TM(x) + trace(x*rho_in)*rho_out
        for ( ; outit!=out.end() && Txit!=Tx.cend() && rhoit!=rho_inside.cend(); ++outit, ++Txit, ++rhoit) outit->second += trtmp*rhoit->second - fac*Txit->second;
    }
    return outvec;
}

 /// implements y = x - fac*TM(x) + trace(x*rho_in)*rho_out
 /// this pretty much only makes sense when |fac|=1 and arg(fac) is equal to an argument of an eigenvalue of the mixed TM with magnitude ~ 1
template<typename KT, typename VT, typename VTR>
Col<VT>
InvEfun_proj_fac(const Col<VT>& invec,
                 const std::function<BlockMat<KT,VT> (const BlockMat<KT,VT>&)>& TMfun,
                 const BlockMat<KT,VTR>& rho_inside,
                 const BlockMat<KT,VTR>& rho_outside,
                 const dimkeypair_vec<KT>& dims,
                 VT fac,
                 uint m_tot)
{
    Col<VT> outvec(invec);

    BlockMat<KT,VT> in(invec,dims,false,true);
    BlockMat<KT,VT> out(outvec,dims,false,true);;

    VT trtmp{0}; /// trace(x*rho_in) part
    {
        auto rhoit = rho_inside.cbegin();
        auto init = in.cbegin();
        /// here it is assumed, that in and rho_inside have the same number of sectors
        for ( ; init != in.cend(); ++init, ++rhoit) trtmp += trace(QMat(*init) * QMat(*rhoit));
//        for ( ; init != in.cend() && rhoit != rho_inside.cend() ; ++init, ++rhoit)
//            trtmp += trace(init->second * rhoit->second); /// trace(x*rho_in) part
    }

    BlockMat<KT,VT> Tx = TMfun(in); /// TM(x) part
    { /// use extra brackets to limit scope of iterators
        auto outit = out.begin();
        auto Txit = Tx.cbegin();
        auto rhoit = rho_outside.cbegin();
        /// combine all three parts to y = x - fac*TM(x) + trace(x*rho_in)*rho_out
        /// here it is assumed, that out, Tx and rho_outside have the same number of sectors
        for ( ; outit!=out.end(); ++outit, ++Txit, ++rhoit) QMat(*outit) += trtmp*QMat(*rhoit) - fac*QMat(*Txit);
//        for ( ; outit!=out.end() && Txit!=Tx.cend() && rhoit!=rho_inside.cend(); ++outit, ++Txit, ++rhoit)
//            outit->second += trtmp*rhoit->second - fac*Txit->second;
    }
    return outvec;
}

 /// implements y = x - fac*TM(x) for BlockDiagMat
template<typename KT, typename VT>
Col<VT>
InvEfun_fac(const Col<VT>& invec,
            const std::function<BlockDiagMat<KT,VT> (const BlockDiagMat<KT,VT>&)>& TMfun,
            const dimpair_vec<KT>& dims,
            VT fac,
            uint m_tot)
{
    Col<VT> outvec(invec); /// this implements the y = x part, - TM(x) is subtracted below
    const VT* inmem = invec.memptr();
    VT* outmem = outvec.memptr();

    BlockDiagMat<KT,VT> in,out;
    uint pos=0,ml,mr;
    for (const auto& dimit : dims)
    {
        ml = get<1>(dimit);
        mr = get<2>(dimit);
        in.emplace_hint(in.end(),get<0>(dimit),Mat<VT>(const_cast<VT*>(&inmem[pos]),ml,mr,false,true)); /// we have to cheat here and pretend that in is const.
        out.emplace_hint(out.end(),get<0>(dimit),Mat<VT>(&outmem[pos],ml,mr,false,true));
        pos += ml*mr;
    }

    /// we will not use -= here, as we already know that the sectors of Tx and out are the same, so we just loop over both at the same time
    BlockDiagMat<KT,VT> Tx = TMfun(in); /// TM(x) part

    { /// use extra brackets to limit scope of iterators
        auto Txit = Tx.cbegin();
        /// combine both parts to y = x - TM(x)
        for (auto outit = out.begin() ; outit!=out.end(); ++outit, ++Txit) outit->second -= fac*Txit->second;
    }
    return outvec;
}


 /// implements y = x - fac*TM(x) for BlockMat
template<typename KT, typename VT>
Col<VT>
InvEfun_fac(const Col<VT>& invec,
            const std::function<BlockMat<KT,VT> (const BlockMat<KT,VT>&)>& TMfun,
            const dimkeypair_vec<KT>& dims,
            VT fac,
            uint m_tot)
{
    Col<VT> outvec(invec); /// this implements the y = x part, TM(x) is subtracted below
    const VT* inmem = invec.memptr();
    VT* outmem = outvec.memptr();

    BlockMat<KT,VT> in,out;
    uint pos=0,ml=0,mr=0;
    for (const auto& dimit : dims)
    {
        ml = get<2>(dimit);
        mr = get<3>(dimit);
        in.emplace_hint(in.end(),get<0>(dimit),std::make_pair(get<1>(dimit),Mat<VT>(const_cast<VT*>(&inmem[pos]),ml,mr,false,true)));
        out.emplace_hint(out.end(),get<0>(dimit),std::make_pair(get<1>(dimit),Mat<VT>(&outmem[pos],ml,mr,false,true)));
        pos += ml*mr;
    }

    /// we will not use -= here, as we already know that the sectors of Tx and out are the same, so we just loop over both at the same time
    /// MAKE SURE THAT TX HAS THE SAME BLOCK STRUCTURE AS IN!!
    BlockMat<KT,VT> Tx = TMfun(in); /// TM(x) part

    { /// use extra brackets to limit scope of iterators
        auto Txit = Tx.cbegin();
        /// combine both parts to y = x - TM(x)
        for (auto outit = out.begin() ; outit!=out.end(); ++outit, ++Txit) outit->second.second -= fac*(Txit->second.second);
    }
    return outvec;
}

/**< Actual inversion functions, i.e. iterative application of [1 - T]^{-1} or [1 - fac*T]^{-1} */

/// solves (y|[1 - T + |R)(L|]    = (x|[1 - |R)(L|]    or
///           [1 - T + |R)(L|]|y) =    [1 - |R)(L|]|x) iteratively for y
/// MPStype can be both a single MPS matrix or an MPSArray
/// BMatType can be both BlockMat and BlockDiagMat
template<typename MPStype, typename BMatType>
BMatType
InvertE_proj(const MPStype& MPS,
             const BMatType& x,
             const BMatType& L,
             const BMatType& R,
             dirtype dir,
             Real tol=1e-14,
             uint maxit=0,
             BMatType y0 = BMatType(),
             bool verbose=false)
{
    using VT = typename BMatType::scalar_type;
/// TODO (valentin#1#): Add functionality to pass empty BlockDiagMat to represent the identity

    auto dims = x.GetSizesVector();
    assert(x.size() == L.size() && dims == L.GetSizesVector() && "x and L must have same QN and sizes");
    assert(x.size() == R.size() && dims == R.GetSizesVector() && "x and R must have same QN and sizes");

    /// determine problem size, i.e. the total size of the concatenation of the vectorizations of all blocks
    uint m_tot = x.GetNElem();
    if (maxit==0) maxit = m_tot;

    /// it is not necessary to project out |R)(L| from y0, as this will happen in the very first iteration of gmres anyway
    if (y0.empty() || y0.GetSizesVector() != dims)
    {
        if (verbose) cout<<"InvertE_proj(A,x): randomly initializing y0"<<endl;
        y0 = BMatType(dims,fill::randn);
    }


    std::function<Col<VT> (const Col<VT>&)> Afun;
    std::function<BMatType (const BMatType&)> TMfun;

    Col<VT> xpv = x.Vectorize(); /// prepare input vector
    if (dir == l)
    {
        /// project out dominant subspace: (x| -> (x|[1 - |R)(L|]
        xpv -= trace(x*R)*L.Vectorize();

        /// implements (y| = (x|[1 - T + |R)(L|]
        TMfun = [&MPS] (const BMatType& in) -> BMatType {return ApplyTMLeft(MPS,in);};
        Afun = [&TMfun,&L,&R,&dims,m_tot](const Col<VT>& invec) -> Col<VT>{return InvEfun_proj(invec,TMfun,R,L,dims,m_tot);}; /// there is a slight difference to dir==r, keep separate!
    }
    else if(dir == r)
    {
        /// project out dominant subspace: |x) -> [1 - |R)(L|]|x)
        xpv -= trace(L*x)*R.Vectorize();

        /// implements |y) = [1 - T + |R)(L|]|x)
        TMfun = [&MPS] (const BMatType& in) -> BMatType {return ApplyTMRight(MPS,in);};
        Afun = [&TMfun,&L,&R,&dims,m_tot](const Col<VT>& invec) -> Col<VT>{return InvEfun_proj(invec,TMfun,L,R,dims,m_tot);};/// there is a slight difference to dir==l, keep separate!
    }
    else throw std::logic_error("InvertE_proj: wrong direction specified");

    Col<VT> y(m_tot);

    int flag = gmres(Afun,xpv,y,y0.Vectorize(),tol,maxit,0,verbose);
    if (flag<0) cerr<<"InvertE_proj did not converge"<<endl;

    return BMatType(y,dims);
}


/// solves (y|[1 - T + |R)(L|] = (x|[1 - |R)(L|] or [1 - T + |R)(L|]|y) = [1 - |R)(L|]|x) iteratively for y
/// this pretty much only makes sense when |fac|=1 and arg(fac) is equal to an argument of an eigenvalue of the mixed TM with magnitude ~ 1, otherwise we can just use InvertE_fac (t.b.i. for BlockDiagMats though)
/// MPStype can be both a single MPS matrix or an MPSArray
/// BMatType and BMatTypeX can be both BlockMat and BlockDiagMat
/// (this is not optimal. In principle it could happen that x, L & R and the result have different scalar_types,
/// but I don't know how to account for that... For now the two actual possible cases of L,R real, x, result complex, and everything complex both work (x is always complex)
template<typename FVT, typename MPStype, typename BMatType, typename BMatTypeX>
BMatTypeX
InvertE_proj_fac(const MPStype& MPSA,
                 const MPStype& MPSB,
                 const BMatTypeX& x,
                 const BMatType& L,
                 const BMatType& R,
                 dirtype dir,
                 FVT fac,
                 Real tol = 1e-14,
                 uint maxit = 0,
                 BMatTypeX y0 = BMatTypeX(),
                 bool verbose = false)
{
/// TODO (valentin#1#): Add functionality to pass empty BlockDiagMat to represent the identity
    using VT = typename BMatTypeX::scalar_type;

    auto dims = x.GetSizesVector();
    assert(x.size() == L.size() && "InvertE_proj_fac: x and L must have same sizes");
    assert(x.size() == R.size() && "InvertE_proj_fac: x and R must have same sizes");

    /// determine problem size, i.e. the total size of the concatenation of the vectorizations of all blocks
    uint m_tot = x.GetNElem();
    if (maxit==0) maxit = m_tot;

    /// it is not necessary to project out |R)(L| from y0, as this will happen in the very first iteration of gmres anyway
    if (y0.empty() || y0.GetSizesVector() != dims)
    {
        if (verbose) cout<<"InvertE_proj_fac(A,B,x): randomly initializing y0"<<endl;
        y0 = BMatTypeX(dims,fill::randn);
    }


    std::function<Col<VT> (const Col<VT>&)> Afun;
    std::function<BMatTypeX (const BMatTypeX&)> TMfun;

    Col<VT> xpv = x.Vectorize(); /// prepare input vector
    if (dir == l)
    {
        assert(x.dK() == L.dK() && "x and L need to have same dK");
        assert(x.dK() == -(R.dK()) && "x and R need to have opposite dK");

        /// project out dominant subspace: (x| -> (x|[1 - |R)(L|]
        xpv -= trace(x*R)*L.Vectorize();

        /// implements (y| = (x|[1 - T + |R)(L|]
        TMfun = [&MPSA,&MPSB] (const BMatTypeX& in) -> BMatTypeX {return ApplyTMmixedLeft(MPSA,MPSB,in); };
        Afun = [&TMfun,&L,&R,&dims,fac,m_tot](const Col<VT>& invec) -> Col<VT> {return InvEfun_proj_fac(invec,TMfun,R,L,dims,fac,m_tot);}; /// there is a slight difference to dir==r, keep separate!
    }
    else if(dir == r)
    {
        assert(x.dK() == R.dK() && "x and R need to have same dK");
        assert(x.dK() == -(L.dK()) && "x and L need to have opposite dK");
        /// project out dominant subspace: |x) -> [1 - |R)(L|]|x)
        xpv -= trace(L*x)*R.Vectorize();

        /// implements |y) = [1 - T + |R)(L|]|x)
        TMfun = [&MPSA,&MPSB] (const BMatTypeX& in) -> BMatTypeX {return ApplyTMmixedRight(MPSA,MPSB,in);};
        Afun = [&TMfun,&L,&R,&dims,fac,m_tot](const Col<VT>& invec) -> Col<VT> {return InvEfun_proj_fac(invec,TMfun,L,R,dims,fac,m_tot);};/// there is a slight difference to dir==l, keep separate!
    }
    else throw std::logic_error("InvertE_proj_fac: wrong direction specified");

    Col<VT> y(m_tot);

    int flag = gmres(Afun,xpv,y,y0.Vectorize(),tol,maxit,0,verbose);
    if (flag<0) cerr<<"InvertE_proj_fac did not converge"<<endl;

    return BMatTypeX(y,dims);
}


/// solves (y|[1 - fac*T] = (x| or [1 - fac*T]|y) = |x) iteratively for y with |fac|<1
/// MPStype can be both a single MPS matrix or an MPSArray
/// BMatType can be both a BlockMat or a BlockDiagMat
template<typename FVT, typename MPStype, typename BMatType>
BMatType
InvertE_fac(const MPStype& MPS,
            const BMatType& x,
            dirtype dir,
            FVT fac,
            Real tol=1e-14,
            uint maxit=0,
            BMatType y0 = BMatType(), /// pass by copy, as we might have to alter it if it is empty or has wrong block structure
            bool verbose=false)

{
    using VT = typename BMatType::scalar_type;
    /// determine problem size, i.e. the total size of the concatenation of the vectorizations of all blocks
    auto dims = x.GetSizesVector();
    uint m_tot = x.GetNElem();

    if (maxit==0) maxit = m_tot;

    /// if y0.size() == 0 the second condition will not even be evaluated (short-circuiting)
    if (y0.empty() || y0.GetSizesVector() != dims)
    {
        if (verbose) cout<<"InvertE_fac(A,x): randomly initializing y0"<<endl;
        y0 = BMatType(dims,fill::randn);
    }

    std::function<BMatType (const BMatType&)> TMfun;

    if (dir ==r) TMfun = [&MPS] (const BMatType& in) -> BMatType {return ApplyTMRight(MPS,in);};
    else if (dir == l) TMfun = [&MPS] (const BMatType& in) -> BMatType {return ApplyTMLeft(MPS,in);};
    else throw std::logic_error("InvertE_fac(): wrong direction specified");

    std::function<Col<VT> (const Col<VT>&)> Afun = [&TMfun,&dims,fac,m_tot](const Col<VT>& invec) -> Col<VT>{return InvEfun_fac(invec,TMfun,dims,fac,m_tot);};

    Col<VT> y(m_tot);
    int flag = gmres(Afun,x.Vectorize(),y,y0.Vectorize(),tol,maxit,0,verbose);
    if (flag<0) cerr<<"InvertE_fac did not converge"<<endl;

    return BMatType(y,dims);
}

//
///// solves (y|[1 - fac*T] = (x| or [1 - fac*T]|y) = |x) iteratively for y with |fac|<1
//template<typename KT, typename VT, typename MPStype> /// MPStype can be both a single MPS matrix or an MPSArray
//BlockDiagMat<KT,VT>
//InvertE_fac(const MPStype& MPS,
//            const BlockDiagMat<KT,VT>& x,
//            dirtype dir,
//            VT fac,
//            Real tol=1e-14,
//            uint maxit=0,
//            BlockMat<KT,VT> y0 = BlockMat<KT,VT>(), /// pass by copy, as we might have to alter it if it is empty or has wrong block structure
//            bool verbose=false)
//{
//    /// determine problem size, i.e. the total size of the concatenation of the vectorizations of all blocks
//
//    dim_vec<KT> dims = x.GetUniformSizesVector();
//    uint m_tot = x.GetNElem();
//
//    if (maxit==0) maxit = m_tot;
//
//    /// if y0.size() == 0 the second condition will not even be evaluated (short-circuiting)
//    if (y0.empty() || y0.GetSizesVector() != dims) y0 = BlockMat<KT,VT>(dims,fill::randn);
//
//    std::function<BlockMat<KT,VT> (const BlockMat<KT,VT>&)> TMfun;
//
//    if (dir ==r) TMfun = [&MPS] (const BlockMat<KT,VT>& in) -> BlockMat<KT,VT> {return ApplyTMRight(MPS,in);};
//    else if (dir == l) TMfun = [&MPS] (const BlockMat<KT,VT>& in) -> BlockMat<KT,VT> {return ApplyTMLeft(MPS,in);};
//    else throw std::logic_error("InvertE_fac(): wrong direction specified");
//
//    std::function<Col<VT> (const Col<VT>&)> Afun = [&TMfun,&dims,fac,m_tot](const Col<VT>& invec) -> Col<VT>{return InvEfun_fac(invec,TMfun,dims,fac,m_tot);};
//
//    Col<VT> y(m_tot);
//    int flag = gmres(Afun,x.Vectorize(),y,y0.Vectorize(),tol,maxit,0,verbose);
//    if (flag<0) cerr<<"InvertE_fac did not converge"<<endl;
//
//    return BlockMat<KT,VT>(y,dims);
//}


/// solves (y|[1 - fac*T] = (x| or [1 - fac*T]|y) = |x) iteratively for y
/// If (L| and |R) are the left and right dominant eigenvectors of the mixed TM formed by A and B, then it must hold (x|R) = (L|x) = 0, or otherwise |fac|<1 !!!
/// Usually, L and R are block diagonal. In that case x should NOT be Block DIAGONAL (or |fac| < 1)
/// MPStype can be both a single MPS matrix or an MPSArray
/// BMatType can be both a BlockMat or a BlockDiagMat
template<typename FVT, typename MPStype, typename BMatType>
BMatType
InvertE_fac(const MPStype& A,
            const MPStype& B,
            const BMatType& x,
            dirtype dir,
            FVT fac,
            Real tol=1e-14,
            uint maxit=0,
            BMatType y0 = BMatType(), /// pass by copy, as we might have to alter it if it is empty or has wrong block structure
            bool verbose=false)
{
    using VT = typename BMatType::scalar_type;

    assert(A.size() == B.size() && "A and B need to be of the same size");
    /// determine problem size, i.e. the total size of the concatenation of the vectorizations of all blocks

    auto dims = x.GetSizesVector();
    uint m_tot = x.GetNElem();

    if (maxit==0) maxit = m_tot;

    /// if y0.size() == 0 the second condition will not even be evaluated (short-circuiting)
    if (y0.empty() || y0.GetSizesVector() != dims)
    {
        if (verbose) cout<<"InvertE_proj_fac(A,B,x): randomly initializing y0"<<endl;
        y0 = BMatType(dims,fill::randn);
    }

    std::function<BMatType (const BMatType&)> TMfun;

    if (dir ==r) TMfun = [&A,&B] (const BMatType& in) -> BMatType {return ApplyTMmixedRight(A,B,in);};
    else if (dir == l) TMfun = [&A,&B] (const BMatType& in) -> BMatType {return ApplyTMmixedLeft(A,B,in);};
    else throw std::logic_error("InvertE_fac(): wrong direction specified");

    std::function<Col<VT> (const Col<VT>&)> Afun = [&TMfun,&dims,fac,m_tot](const Col<VT>& invec) -> Col<VT>{return InvEfun_fac(invec,TMfun,dims,fac,m_tot);};

    Col<VT> y(m_tot);
    int flag = gmres(Afun,x.Vectorize(),y,y0.Vectorize(),tol,maxit,0,verbose);
    if (flag<0) cerr<<"InvertE_fac did not converge"<<endl;

    return BMatType(y,dims);
}

///// solves (y|[1 - fac*T] = (x| or [1 - fac*T]|y) = |x) iteratively for y
///// If (L| and |R) are the left and right dominant eigenvectors of the mixed TM formed by A and B, then it must hold (x|R) = (L|x) = 0, or otherwise |fac|<1 !!!
///// Usually, L and R are block diagonal. In that case x should NOT be Block DIAGONAL (or |fac| < 1)
//template<typename KT, typename VT, typename MPStype> /// MPStype can be both a single MPS matrix or an MPSArray
//BlockMat<KT,VT>
//InvertE_fac(const MPStype& A,
//            const MPStype& B,
//            const BlockMat<KT,VT>& x,
//            dirtype dir,
//            VT fac,
//            Real tol=1e-14,
//            uint maxit=0,
//            BlockMat<KT,VT> y0 = BlockMat<KT,VT>(), /// pass by copy, as we might have to alter it if it is empty or has wrong block structure
//            bool verbose=false)
//{
//    assert(A.size() == B.size() && "A and B need to be of the same size");
//    /// determine problem size, i.e. the total size of the concatenation of the vectorizations of all blocks
//
//    dimkeypair_vec<KT> dims = x.GetSizesVector();
//    uint m_tot = x.GetNElem();
//
//    if (maxit==0) maxit = m_tot;
//
//    /// if y0.size() == 0 the second condition will not even be evaluated (short-circuiting)
//    if (y0.empty() || y0.GetSizesVector() != dims) y0 = BlockMat<KT,VT>(dims,fill::randn);
//
//    std::function<BlockMat<KT,VT> (const BlockMat<KT,VT>&)> TMfun;
//
//    if (dir ==r) TMfun = [&A,&B] (const BlockMat<KT,VT>& in) -> BlockMat<KT,VT> {return ApplyTMmixedRight(A,B,in);};
//    else if (dir == l) TMfun = [&A,&B] (const BlockMat<KT,VT>& in) -> BlockMat<KT,VT> {return ApplyTMmixedLeft(A,B,in);};
//    else throw std::logic_error("InvertE_fac(): wrong direction specified");
//
//    std::function<Col<VT> (const Col<VT>&)> Afun = [&TMfun,&dims,fac,m_tot](const Col<VT>& invec) -> Col<VT>{return InvEfun_fac(invec,TMfun,dims,fac,m_tot);};
//
//    Col<VT> y(m_tot);
//    int flag = gmres(Afun,x.Vectorize(),y,y0.Vectorize(),tol,maxit,0,verbose);
//    if (flag<0) cerr<<"InvertE_fac did not converge"<<endl;
//
//    return BlockMat<KT,VT>(y,dims);
//}

#endif // INV_E_BLOCK_FUNCTIONS

