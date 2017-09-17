#ifndef MPS_DENSE_UTIL_H
#define MPS_DENSE_UTIL_H

#ifdef _OPENMP
#include "omp.h"
#endif // _OPENMP

#include "MPSDenseMat.hpp"
#include "helpers.hpp"
#include "helpers.h"
#include "OperatorTypes.hpp"
#include "tictoc.hpp"
#include "IterativeSolvers.hpp"


static bool rng_init = false;


/// construct randMPS as a proxy struct object only storing the relevant data (mostly matrix sizes).
/// Only when assigned to some specific MPS object, construct requested random MPS via calling an implicit type conversion operator
struct randMPS
{
    const uint d_,ml_,mr_,N_;
    randMPS(uint d, uint ml, uint mr, uint Nsites=1):d_(d),ml_(ml),mr_(mr),N_(Nsites)
    {
        /// check if rng has been initialized, if not initialize, but only once!
        if (!rng_init)
        {
            arma_rng::set_seed_random();
            rng_init = true;
        }
    }
    /// type conversion operator which actually constructs the random matrix
    template<typename VT>
    operator MPSDenseMat<VT> ()
    {
        uint numel=std::pow(d_,N_);
        MPSDenseMat<VT> tmp(d_,N_);
        for (uint s=0; s<numel; ++s) tmp[s]=Mat<VT>(ml_,mr_,fill::randn);
        return tmp;
    }
};

template<typename T>
const MPSDenseMat<T>
concat(const MPSDenseMat<T>& lhs, const MPSDenseMat<T>& rhs)
{
    uint dim=lhs.GetDim();
    assert(dim==rhs.GetDim());
    assert(lhs.GetMr()==rhs.GetMl());

    /// using iterators
    MPSDenseMat<T> out(dim,lhs.GetNSites()+rhs.GetNSites());
    auto ito=out.begin();
    auto itl=lhs.begin();

    while (itl!=lhs.end())
    {
        auto itr = rhs.begin();
        while (itr!=rhs.end()) (*ito++)=(*itl)*(*itr++);
        ++itl;
    }
    out.SetDims();

//    /// using indexing ----------------------------------------------------------
//    MPSDenseMat<T> out(dim,lhs.GetNSites()+rhs.GetNSites());
//    uint ct=0;
//    for (uint i=0;i<lhs.size();++i) for (uint j=0;j<rhs.size();++j) out[ct++]=lhs[i]*rhs[j];
//    /// -----------------------------------------------------------------------------
    return out;
}

template<typename T1, typename T2>
MPSDenseMat<typename promote_type<T1,T2>::result>
ApplyOperator(const MPSDenseMat<T1>& in, const SparseOperator<T2>& op)
{
    assert(in.GetDim() == op.GetLocalDim() && "ApplyOperator: ingoing MPS and operator have different physical dimension.");
    assert(in.GetNSites() == op.GetNSites() && "ApplyOperator: ingoing MPS and operator have different support.");

    MPSDenseMat<typename promote_type<T1,T2>::result> out(in.GetDim(),in.GetMl(),in.GetMr(),in.GetNSites()); /// create out MPS filled with zeros of equal size and dimensions
    uint ii,jj;
    for (typename SparseOperator<T2>::const_iterator opit = op.begin(); opit != op.end(); ++opit)
    {
        ii = opit.row();
        jj = opit.col();
        out[ii] += (*opit)*in[jj];
    }
    return out;
}

template<typename T>
inline
MPSDenseMat<T>
NullSpace(const MPSDenseMat<T>& M, dirtype dir)
{
    Mat<T> N;
    if (dir == l) N = arma::null(M.GetDenseMat(l).t());
    else if (dir == r) N = arma::null(M.GetDenseMat(r)).t();

    return MPSDenseMat<T>(N,M.GetDim(),M.GetNSites(),dir);
}

template<typename T>
inline
MPSDenseMat<T>
qr(const MPSDenseMat<T>& M, Mat<T>& R, dirtype dir)
{
    Mat<T> Q;
    MPSDenseMat<T> out(M.GetDim(),M.GetNSites());
    bool conv;

    if (dir == l) conv = qr_econ(Q,R,M.GetDenseMat(l));
    else if (dir == r) conv = qr_econ(Q,R,M.GetDenseMat(r).t());
    else {cerr<<"qr: wrong direction specified"<<endl;abort();}

    if (!conv)
    {
            cerr<<"qr did not converge in MPSDenseMat<T>::qr";
            abort();
    }
    Col<T> rdgl = R.diag();

/// TODO (valentin#1#): this should be doable with Mat::each_row(lambda) and Mat::each_col(lambda)
        for (uint i=0; i<rdgl.n_elem; ++i)
        {
            if (rdgl(i) < 0)
            {
                R.row(i)*=-1; /// fix sign on diagonal
                Q.col(i)*=-1; /// fix sign on diagonal
            }
        }

    if (dir == r)
    {
        R = R.t();
        Q = Q.t();
    }
    return MPSDenseMat<T>(Q,M.GetDim(),M.GetNSites(),dir);
}

template<typename T>
inline
Real
norm(const MPSDenseMat<T>& M)
{
    return norm(M.GetDenseMat(l),"fro");
}

//template<typename T>
//inline
//MPSDenseMat<T>
//qr(const MPSDenseMat<T>& M, dirtype dir)
//{
//    Mat<T> Q,R;
//    RMPSDenseMat<T> out(M.GetDim(),M.GetNSites());
//    bool conv;
//
//    if (dir == l) conv = qr_econ(Q,R,M.GetDenseMat(l)));
//    else if (dir == r) conv = qr_econ(Q,R,M.GetDenseMat(r).t());
//    else {cerr<<"qr: wrong direction specified"<<endl;abort();}
//
//    if (!conv)
//    {
//            cerr<<"qr did not converge in MPSDenseMat<T>::qr";
//            abort();
//    }
//    Col<T> rdgl = R.diag();
//
///// TODO (valentin#1#): this should be doable with Mat::each_row(lambda) and Mat::each_col(lambda)
//        for (uint i=0; i<rdgl.n_elem; ++i)
//        {
//            if (rdgl(i) < 0)
//            {
//                R.row(i)*=-1; /// fix sign on diagonal
//                Q.col(i)*=-1; /// fix sign on diagonal
//            }
//        }
//
//    if (dir == r)
//    {
//        Q = Q.t();
//    }
//    else {cerr<<"qr: wrong direction specified"<<endl;abort();}
//    return MPSDenseMat<T>(Q,M.GetDim(),M.GetNSites(),dir);
//}


template<typename T>
Mat<T>
InvertE_fac(const MPSDenseMat<T>& MPS, const Mat<T>& x, T fac, dirtype dir, uint m, Real tol=1e-14, const Mat<T>& y0=Mat<T>(), uint maxit=0)
{
    assert(abs(1-fac)>1e-10);

    std::function<Col<T> (const Col<T>&)> InvEfun;
    if (maxit==0)maxit = m*m;

    if (dir == l)
    {
        InvEfun = [&MPS,m,fac](const Col<T>& invec) -> Col<T>
        {
            Col<T> outvec(invec);

            const RMatType in(const_cast<T*>(invec.memptr()),m,m,false,true);
            RMatType out(outvec.memptr(),m,m,false,true);
            out -= fac*ApplyTMLeft(MPS,in);
            return outvec;
        };
    }
    else if(dir == r)
    {
        InvEfun = [&MPS,m,fac](const Col<T>& invec) -> Col<T>
        {
            Col<T> outvec(invec);

            const RMatType in(const_cast<T*>(invec.memptr()),m,m,false,true);
            RMatType out(outvec.memptr(),m,m,false,true);
            out -= ApplyTMRight(MPS,in);

            return outvec;
        };

    }
    else {cerr<<"no"<<endl;abort();}


    Mat<T> out(m,m);/// construct matrix to be returned first
    Col<T> y(out.memptr(),m*m,false,true); /// bind memory of return matrix to vector, which will be filled in gmres with the solution

    gmres(InvEfun,Col<T>(x.memptr(),m*m),y,Col<T>(m*m,fill::randn),tol,maxit,0,true);
    return out;
}

template<typename T>
Mat<T>
InvertE_proj(const MPSDenseMat<T>& MPS, const Mat<T>& x, const Mat<T>& L, const Mat<T>& R, dirtype dir, uint m, Real tol=1e-14, const Mat<T>& y0=Mat<T>(), uint maxit=0, bool verbose=false)
{
    std::function<Col<T> (const Col<T>&)> InvEfun;
    if (maxit==0)maxit = m*m;
    Mat<T> xproj(m,m);

    if (dir == l)
    {
        xproj = x - trace(x*R)*L;
        InvEfun = [&MPS,&L,&R,m](const Col<T>& invec) -> Col<T>
        {
            Col<T> outvec(invec);

            const RMatType in(const_cast<T*>(invec.memptr()),m,m,false,true);
            RMatType out(outvec.memptr(),m,m,false,true);
            out += trace(in*R)*L - ApplyTMLeft(MPS,in);
            return outvec;
        };
    }
    else if(dir == r)
    {
        xproj = x - trace(L*x)*R;
        InvEfun = [&MPS,&L,&R,m](const Col<T>& invec) -> Col<T>
        {
            Col<T> outvec(invec);

            const RMatType in(const_cast<T*>(invec.memptr()),m,m,false,true);
            RMatType out(outvec.memptr(),m,m,false,true);
            out += trace(L*in)*R - ApplyTMRight(MPS,in);

            return outvec;
        };

    }
    else {cerr<<"InvertE_proj: wrong direction specified"<<endl;abort();}

    Mat<T> out(m,m);
    Col<T> y(out.memptr(),m*m,false,true);
    Col<T> y0v;

    if (y0.size()==0) y0v = Col<T>(m*m,fill::randn);
    else
    {
        assert(y0.n_elem==m*m && "InvertE_proj: y0 has wrong dimensions");
        y0v = Col<T>(y0.memptr(),m*m);
    }
    gmres(InvEfun,Col<T>(xproj.memptr(),m*m),y,y0v,tol,maxit,0,verbose);
    return out;
}

template<typename T, typename TH>
uint
ExpandFromH(const SparseOperator<TH>& H, MPSDenseMat<T>& AL, MPSDenseMat<T>& AR, const Mat<T>& C, uint dmmax)
{
    uint dm = dmmax;
    MPSDenseMat<T> NL = NullSpace(AL,l);
    MPSDenseMat<T> NR = NullSpace(AR,r);

/// TODO (valentin#1#2016-04-25): Here we can only use real Operators and real MPS, implement mixed type
    auto phi = ApplyOperator(AL*C*AR,H);

    Mat<T> M(NL.GetMr(),NR.GetMl(),fill::zeros);

    for (uint i=0;i<NL.size();++i)
    {
        for (uint j=0;j<NR.size();++j)
        {
//            cout<<i<<", "<<j<<endl;
            M += NL[i].t()*phi[i*NR.GetDim()+j]*NR[j].t();
        }
    }

    Mat<T> U,V;
    Lambda lam;

    svd_econ(U,lam,V,M);

    if (lam.n_elem > dmmax)
    {
        U.resize(U.n_rows,dm);
        V.resize(V.n_rows,dm);
    }
    else dm = lam.n_elem;

//    C.resize(C.n_rows + dm,C.n_cols + dm);
    for (uint s=0;s<AL.size();++s) AL[s] = join_rows(AL[s],NL[s]*U);
    for (uint s=0;s<AR.size();++s) AR[s] = join_cols(AR[s],V.t()*NR[s]);

    return dm;
}

template<typename T>
void split2(const MPSDenseMat<T>& psi, MPSDenseMat<T>& A, MPSDenseMat<T>& B, Lambda& lam, uint dim, uint mmax, double lamthresh)
{
    uint ml=psi.GetMl(),mr=psi.GetMr();
    assert(psi.size()==dim*dim);
    Mat<T> psimat = zeros(dim*ml,dim*mr);

    for (uint i=0;i<dim;++i) for (uint j=0;j<dim;++j) psimat.submat(i*ml,j*mr,(i+1)*ml-1,(j+1)*mr-1)=psi[i*dim+j];

    Mat<T> U,V;
    if(!svd(U,lam,V,psimat,"dc"))cerr<<"split2: SVD(psimat) failed"<<endl;
    lam/=norm(lam,2);

    uint truncdim=lam.n_elem;
    while ((lam(truncdim-1)<lamthresh || truncdim>mmax)) --truncdim;

    /// truncate
    lam.resize(truncdim);
    lam/=norm(lam,2);

//    A.Set(U(span::all,span(0,truncdim-1))); /// correpsonds to matlab's U(:,1:truncdim)
//    B.Set(V(span::all,span(0,truncdim-1)).t(),MPSDenseMat<T>::row); /// correpsonds to matlab's V(:,1:truncdim)'
    A.Set(U.cols(0,truncdim-1)); /// corresponds to matlab's U(:,1:truncdim)
    B.Set(Mat<T>(V.cols(0,truncdim-1)).t(),MPSDenseMat<T>::row); /// corresponds to matlab's V(:,1:truncdim)'
}
/**< Orthogonalization ********************************************************************************************/

template<typename T>
void
OrthoQR(MPSDenseMat<T>& AL, MPSDenseMat<T>& AR, const MPSDenseMat<T>& A0, Lambda& lam, double tol=1e-14, uint maxit=500)
{
//    uint dim = A.GetNumel();
    uint m = A0.GetMl();
    assert(A0.GetMr() == m);
    MPSDenseMat<T> B0 = A0.t();
//    B.SetDims();
//    cout<<"ölkök"<<endl;
//    MPSDenseMat<T> B(A.GetDenseMat(MPSDenseMat<T>::row).t(),A.GetDim(),A.GetNSites());

    uint ct;
    double dR;
    Mat<T> Ql,Rl,Qr,Rr,Rold;
    Col<T> rdgl,rdgr;
    Col<T> lsgn,rsgn;
    MPSDenseMat<T> Atmp(A0);
    MPSDenseMat<T> Btmp(B0);
//    tictoc tt;
//    tt.tic();
    /// left iteration
    ct=0;
    dR=10;
    Rold = eye<Mat<T> >(m,m)/sqrt(m);

    while (ct++<maxit && dR>tol)
    {
        if(!qr_econ(Ql,Rl,Atmp.GetDenseMat(l))) cerr<<"OrthoQR: QR failed at left step "<<ct<<endl;
        rdgl = Rl.diag();
//        #ifdef _OPENMP
//        #pragma omp parallel for
//        #endif // _OPENMP
        for (uint i=0;i<m;++i) if (rdgl(i) < 0) Rl.row(i)*=-1; /// fix sign on diagonal
        Rl/=norm(Rl,"fro");
        dR = norm(Rl.diag() - Rold.diag(),2);
//        cout<<ct<<": "<<dR<<endl;
        Atmp = Rl*A0;
//        for (lit=Atmp.begin(),rit=A.begin();lit!=Atmp.end();++lit,++rit) *lit = Rl*(*rit);
        Rold = Rl;
    }

    /// right iteration
    ct=0;
    dR=10;
    Rold = eye<Mat<T> >(m,m)/sqrt(m);
    while (ct++<maxit && dR>tol)
    {
        if(!qr_econ(Qr,Rr,Btmp.GetDenseMat(l))) cerr<<"OrthoQR: QR failed at right step "<<ct<<endl;
        rdgr = Rr.diag();
//        #ifdef _OPENMP
//        #pragma omp parallel for
//        #endif // _OPENMP
        for (uint i=0;i<m;++i) if (rdgr(i) < 0) Rr.row(i)*=-1; /// fix sign on diagonal
        Rr/=norm(Rr,"fro");
        dR = norm(Rr.diag() - Rold.diag(),2);
//        cout<<ct<<": "<<dR<<endl;
        Btmp = Rr*B0;
//        for (lit=Btmp.begin(),rit=B.begin();lit!=Btmp.end();++lit,++rit) *lit = Rr*(*rit);
        Rold = Rr;
    }
//    double t=tt.toc();
//    cout<<t<<" s."<<endl;
    Mat<T> U,V;

    if(!svd(U,lam,V,Mat<T>(Rl*Mat<T>(Rr.t())),"dc")) cerr<<"OrthoQR: SVD(Rl*Rr') failed"<<endl;

    lam/=norm(lam);
    for (uint i=0;i<m;++i)
    {
        if (rdgl(i)<0) Ql.col(i)*=-1;
        if (rdgr(i)<0) Qr.col(i)*=-1;
    }

    AL = MPSDenseMat<T>(Ql,A0.GetDim(),A0.GetNSites(),l);
    for (auto& vit : AL) vit = Mat<T>(U.t()) * Mat<T>(vit * U);

    AR = MPSDenseMat<T>(Qr.t(),A0.GetDim(),A0.GetNSites(),r);
    for (auto& vit : AR) vit = Mat<T>(V.t()) * Mat<T>(vit * V);

}


template<typename T>
RedOp<Mat<T> >
ReducedOp(const SparseOperator<T>& O, const MPSDenseMat<T>& A, dirtype dir)
{
/// TODO (valentin#1#): implement general functionality for multi-site operators
    uint d = A.GetDim();
    assert(O.GetLocalDim() == d && "H and A need to be of same physical dimension");
    assert(O.GetNSites() == 2 && "H needs to be two-site");
    RedOp<Mat<T> > Ored(d,1);

    uint m;
    if (dir == l) m = A.GetMr();
    else if(dir == r) m = A.GetMl();
    else { cerr<<"wrong direction specified"<<endl;abort();}

    for (auto& it1 : Ored)
    {
//        it1.resize(d);
        for (auto& it2 : it1) it2 = Mat<T>(m,m,fill::zeros);
    }

    typename SparseOperator<T>::const_iterator opit;
    if (dir == l)
    {
        for (opit = O.begin(); opit!=O.end(); ++opit)
        {
            std::vector<uint> ii = num2ditvec(opit.row(),d,2);
            std::vector<uint> jj = num2ditvec(opit.col(),d,2);
            Ored[ii[1]][jj[1]] += (*opit)*A[ii[0]].t()*A[jj[0]];
        }
    }
    else if (dir == r)
    {
        for (opit = O.begin(); opit!=O.end(); ++opit)
        {
            std::vector<uint> ii = num2ditvec(opit.row(),d,2);
            std::vector<uint> jj = num2ditvec(opit.col(),d,2);
            Ored[ii[0]][jj[0]] += (*opit)*A[jj[1]]*A[ii[1]].t();
        }
    }
    else { cerr<<"wrong direction specified"<<endl;abort();}

    return Ored;
}

//template<typename T>
//void
//OrthoQR(MPSDenseMat<T>& A, Lambda& lam, dirtype dir, double tol=1e-14, uint maxit=500)
//{
////    uint dim = A.GetNumel();
//    uint m = A.GetMl();
//    assert(A.GetMr() == m);
//    MPSDenseMat<T> B(A.GetDim(),A.GetNSites());
//    typename MPSDenseMat<T>::iterator lit,rit;
//    for (lit=B.begin(),rit=A.begin();rit!=A.end();++lit,++rit) *lit = rit->t();
//    B.SetDims();
////    cout<<"ölkök"<<endl;
////    MPSDenseMat<T> B(A.GetDenseMat(MPSDenseMat<T>::row).t(),A.GetDim(),A.GetNSites());
//
//    uint ct;
//    double dR;
//    Mat<T> Ql,Rl,Qr,Rr,Rold;
//    Col<T> rdgl,rdgr;
//    Col<T> lsgn,rsgn;
//    MPSDenseMat<T> Atmp(A);
//    MPSDenseMat<T> Btmp(B);
////    tictoc tt;
////    tt.tic();
//    /// left iteration
//    ct=0;
//    dR=10;
//    Rold = eye<Mat<T> >(m,m)/sqrt(m);
//    while (ct++<maxit && dR>tol)
//    {
//        if(!qr_econ(Ql,Rl,Atmp.GetDenseMat())) cerr<<"OrthoQR: QR failed at left step "<<ct<<endl;
//        rdgl = Rl.diag();
//        #ifdef _OPENMP
//        #pragma omp parallel for
//        #endif // _OPENMP
//        for (uint i=0;i<m;++i) if (rdgl(i) < 0) Rl.row(i)*=-1; /// fix sign on diagonal
//        Rl/=norm(Rl,"fro");
//        dR = norm(Rl.diag() - Rold.diag(),2);
////        cout<<ct<<": "<<dR<<endl;
//        for (lit=Atmp.begin(),rit=A.begin();lit!=Atmp.end();++lit,++rit) *lit = Rl*(*rit);
//        Rold = Rl;
//    }
//
//    /// right iteration
//    ct=0;
//    dR=10;
//    Rold = eye<Mat<T> >(m,m)/sqrt(m);
//    while (ct++<maxit && dR>tol)
//    {
//        if(!qr_econ(Qr,Rr,Btmp.GetDenseMat())) cerr<<"OrthoQR: QR failed at right step "<<ct<<endl;
//        rdgr = Rr.diag();
//        #ifdef _OPENMP
//        #pragma omp parallel for
//        #endif // _OPENMP
//        for (uint i=0;i<m;++i) if (rdgr(i) < 0) Rr.row(i)*=-1; /// fix sign on diagonal
//        Rr/=norm(Rr,"fro");
//        dR = norm(Rr.diag() - Rold.diag(),2);
////        cout<<ct<<": "<<dR<<endl;
//        for (lit=Btmp.begin(),rit=B.begin();lit!=Btmp.end();++lit,++rit) *lit = Rr*(*rit);
//        Rold = Rr;
//    }
////    double t=tt.toc();
////    cout<<t<<" s."<<endl;
//    Mat<T> U,V;
//
//    if(!svd(U,lam,V,Mat<T>(Rl*Mat<T>(Rr.t())),"dc")) cerr<<"OrthoQR: SVD(Rl*Rr') failed"<<endl;
//    lam/=norm(lam,2);
//    if (dir==l)
//    {
//        for (uint i=0;i<m;++i) if (rdgl(i)<0) Ql.col(i)*=-1;
//        A = MPSDenseMat<T>(Ql,A.GetDim(),A.GetNSites());
//        for (auto& vit : A) vit = Mat<T>(U.t()) * Mat<T>(vit * U);
//    }
//    else if(dir==r)
//    {
//        for (uint i=0;i<m;++i) if (rdgr(i)<0) Qr.col(i)*=-1;
//        A = MPSDenseMat<T>(Qr.t(),A.GetDim(),A.GetNSites(),MPSDenseMat<T>::row);
//        for (auto& vit : A) vit = Mat<T>(V.t()) * Mat<T>(vit * V);
//    }
//    else cerr<<"OrthoQR: wrong direction specified"<<endl;
//}

/**< Orthogonalization Checks *************************************************************************************/
template<typename T>
void CheckOrtho(const MPSDenseMat<T>& A, const Mat<T>& CL, const Mat<T>& CR, dirtype dir)
{
    Real chk1=0,chk2=0;
    std::string str;
    uint m;
    switch (dir)
    {
    case l:
        str = "left";
        m = A.GetMr();
        chk1 = norm(ApplyTMLeft(A) - eye<Mat<T> >(m,m));
        chk2 = norm(ApplyTMRight(A,Mat<T>(CR*CR.t())) - CL*CL.t());
        break;
    case r:
        str = "right";
        m = A.GetMl();
        chk1 = norm(ApplyTMLeft(A,Mat<T>(CL.t()*CL))-CR.t()*CR);
        chk2 = norm(ApplyTMRight(A) - eye<Mat<T> >(m,m));
        break;
//    case s:
//        str="symmetric";
//        lammat = lam;
//        L = ApplyTMmixedLeft(lam>>A,A) - lammat;
//        R = ApplyTMmixedRight(A<<lam,A) - lammat;
//        for (const auto& it : L) chk1 += pow(norm(it.second,"fro"),2)/it.second.n_elem;
//        for (const auto& it : R) chk2 += pow(norm(it.second,"fro"),2)/it.second.n_elem;
//        for (const auto& it : L) chk1 += dot(it.second)/it.second.n_elem;
//        for (const auto& it : R) chk2 += dot(it.second)/it.second.n_elem;
//        break;
    case c:
        str="canonical";
        chk1 = norm(ApplyTMLeft(A,Mat<T>(CL.t()*CL)) - CR.t()*CR);
        chk2 = norm(ApplyTMRight(A,Mat<T>(CR*CR.t())) - CL*CL.t());
        break;
    default:
        cerr<<"wrong direction specified"<<endl;
    }
    cout<<"check "<<str<<" gauge:"<<endl;
    cout<<"left: "<<chk1<<", right: "<<chk2<<endl;
}

template<typename T>
void CheckOrthoCanon(const MPSDenseMat<T>& gam, const Lambda& lam, dirtype dir)
{
    Mat<T> tmp;
    uint m;
    if(dir==l)
    {
        cout<<"left: ";
        m=gam.GetMr();
        tmp=ApplyTMLeft(lam>>gam);
    }
    else if(dir==r)
    {
        cout<<"right: ";
        m=gam.GetMl();
        tmp=ApplyTMRight(gam<<lam);
    }
    else cerr<<"CheckOrthoCanon: wrong direction specified"<<endl;
    cout<<"diag: "<<norm(tmp.diag() - ones<Col<T> >(m),2)<<", offdiag: "<<norm(tmp - diagmat(diagvec(tmp)),"fro")<<", gen.: "<<norm(tmp - eye<Mat<T> >(m,m),"fro")<<endl;
}

template<typename T>
void CheckOrthoMPS(const MPSDenseMat<T>& A, const Lambda& lam, dirtype dir)
{
    Mat<T> tmp1,tmp2,diaglam;
    uint m;
    if(dir==l)
    {
        cout<<"left: ";
        m=A.GetMr();
        tmp1=ApplyTMLeft(A);
        tmp2=ApplyTMRight(A<<lam);
        diaglam = diagmat(pow(lam,2));
    }
    else if(dir==r)
    {
        cout<<"right: ";
        m=A.GetMl();
        tmp1=ApplyTMRight(A);
        tmp2=ApplyTMLeft(lam>>A);
        diaglam = diagmat(pow(lam,2));
    }
    else cerr<<"CheckOrthoMPS: wrong direction specified"<<endl;
    cout<<"first: "<<norm(tmp1 - eye<Mat<T> >(m,m),"fro")<<", second: "<<norm(tmp2 - diaglam,"fro")<<endl;
//    cout<<"diag: "<<norm(tmp.diag() - ones<Col<T> >(m),2)<<", offdiag: "<<norm(tmp - diagmat(diagvec(tmp)),"fro")<<", gen.: "<<norm(tmp - eye<Mat<T> >(m,m),"fro")<<endl;
}

template<typename T>
void CheckOrthoSymm(const MPSDenseMat<T>& A, const Lambda& lam, dirtype dir)
{
    Mat<T> tmp;
    if(dir==l)
    {
        cout<<"left: ";
        tmp=ApplyMixedTransLeft(lam>>A,A);
    }
    else if(dir==r)
    {
        cout<<"right: ";
        tmp=ApplyMixedTransRight(A<<lam,A);
    }
    else cerr<<"CheckOrthoCanon: wrong direction specified"<<endl;

    cout<<norm(tmp - diagmat(lam),"fro")<<endl;
}


/**< Transfer Operator Applications *******************************************************************************/

/// from left --------------------------------------------------------------------------------------------------
template<typename T>
Mat<T>
ApplyTMLeft(const MPSDenseMat<T>& A)
{
    uint outdim=A.GetMr();

    Mat<T> out = zeros(outdim,outdim);
    for (auto it=A.begin();it!=A.end();++it) out+= Mat<T>(it->t())*(*it);
    return out;
}

template<typename T1,typename T2>
Mat<typename promote_type<T1,T2>::result>
ApplyTMLeft(const MPSDenseMat<T1>& A, const Mat<T2>& in)
{
    uint indim=A.GetMl(),outdim=A.GetMr();
    assert(in.n_rows==indim && in.n_cols==indim);
    Mat<typename promote_type<T1,T2>::result> out(outdim,outdim,fill::zeros);

    for (const auto& Ait : A) out += Ait.t() * in * Ait;

    return out;
}

template<typename T>
Mat<T>
ApplyMixedTMLeft(const MPSDenseMat<T>& A,const MPSDenseMat<T>& B)
{
    uint mrout=A.GetMr(),mlout=B.GetMr();
    assert(A.size()==B.size());
    assert(B.GetMl()==A.GetMl());

/// TODO (valentin#1#2016-04-27): Switch to iterators
    Mat<T> out(mlout,mrout,fill::zeros);
    for (uint i=0;i<A.size();++i) out+= (B[i].t())*A[i];
    return out;
}

template<typename T1,typename T2>
Mat<typename promote_type<T1,T2>::result>
ApplyMixedTMLeft(const MPSDenseMat<T1>& A,const MPSDenseMat<T1>& B, const Mat<T2>& in)
{
    uint mrout=A.GetMr(),mlout=B.GetMr();
    assert(A.GetDim()==B.GetDim() && "A and B need to be of same physical dimension");
    assert(A.GetNSites()==B.GetNSites() && "A and B need to act on the same number of sites");
    assert(B.GetMl()==in.n_rows && "B and in have inconsistent dimensions");
    assert(A.GetMl()==in.n_cols && "A and in have inconsistent dimensions");
//    assert(B.GetMr()==outdim);

/// TODO (valentin#1#2016-04-27): Switch to iterators
    Mat<typename promote_type<T1,T2>::result> out(mlout,mrout,fill::zeros);
    for (uint i=0;i<A.size();++i) out+= (B[i].t())*in*A[i];
    return out;
}

template<typename T1, typename T2>
inline
Mat<T1>
ApplyOpTMLeft(const SparseOperator<T2>& O, const MPSDenseMat<T1>& A)
{
    return ApplyMixedTMLeft(ApplyOperator(A,O),A);
}


/// from right --------------------------------------------------------------------------------------------------

template<typename T>
Mat<T> ApplyTMRight(const MPSDenseMat<T>& A)
{
    uint outdim=A.GetMl();

    Mat<T> out = zeros(outdim,outdim);
    for (auto it=A.begin();it!=A.end();++it) out+= (*it)*Mat<T>(it->t());
    return out;
}

template<typename T1,typename T2>
Mat<typename promote_type<T1,T2>::result>
ApplyTMRight(const MPSDenseMat<T1>& A, const Mat<T2>& in)
{
    uint indim=A.GetMr(),outdim=A.GetMl();
    assert(in.n_rows==indim && in.n_cols==indim);
    Mat<typename promote_type<T1,T2>::result> out(outdim,outdim,fill::zeros);

    for (const auto& Ait : A) out += Ait * in * Ait.t();

    return out;
}

template<typename T>
Mat<T>
ApplyMixedTMRight(const MPSDenseMat<T>& A,const MPSDenseMat<T>& B)
{
    uint mlout=A.GetMl(),mrout=B.GetMl();
    assert(A.size()==B.size());
    assert(B.GetMr()==A.GetMr());
//    assert(B.GetMr()==outdim);

/// TODO (valentin#1#2016-04-27): Switch to iterators
    Mat<T> out(mlout,mrout,fill::zeros);
    for (uint i=0;i<A.size();++i) out+= A[i]*(B[i].t());
    return out;
}

template<typename T1,typename T2>
Mat<typename promote_type<T1,T2>::result>
ApplyMixedTMRight(const MPSDenseMat<T1>& A,const MPSDenseMat<T1>& B, const Mat<T2>& in)
{
    uint mlout=A.GetMl(),mrout=B.GetMl();
    assert(A.GetDim()==B.GetDim() && "A and B need to be of same physical dimension");
    assert(A.GetNSites()==B.GetNSites() && "A and B need to act on the same number of sites");
    assert(B.GetMr()==in.n_cols && "B and in have inconsistent dimensions");
    assert(A.GetMr()==in.n_rows && "A and in have inconsistent dimensions");
//    assert(B.GetMr()==outdim);

/// TODO (valentin#1#2016-04-27): Switch to iterators
    Mat<typename promote_type<T1,T2>::result> out(mlout,mrout,fill::zeros);
    for (uint i=0;i<A.size();++i) out+= A[i]*in*(B[i].t());
    return out;
}

template<typename T1, typename T2>
inline
Mat<T1>
ApplyOpTMRight(const SparseOperator<T2>& O, const MPSDenseMat<T1>& A)
{
    return ApplyMixedTMRight(ApplyOperator(A,O),A);
}

#endif // MPS_UTIL_H
