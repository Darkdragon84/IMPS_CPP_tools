#ifndef EIGS_DENSE_FUN_H_
#define EIGS_DENSE_FUN_H_

#include "arma_typedefs.h"
#include "MPSDenseUtilities.hpp"
#include "eigs.h"

using namespace std;


template<typename VT>
void
TMMult(VT* invec, VT* outvec, const std::function<void (const Mat<VT>&, Mat<VT>&)>& TMfun, uint dim)
{
    Mat<VT> in(invec,dim,dim,false,true);
    Mat<VT> out(outvec,dim,dim,false,true);
    TMfun(in,out);
//    cout<<in<<endl;
//    cout<<out<<endl;
}


template<typename VT>
Real
TMDominantEig(const MPSDenseMat<VT>& MPS, Mat<VT>& V, dirtype dir, Real tol=1e-14, const Mat<VT>& x0=Mat<VT>(), int maxit=500, string mode="LR");

template<>
Real
TMDominantEig(const MPSDenseMat<Real>& MPS, Mat<Real>& V, dirtype dir, Real tol, const Mat<Real>& x0, int maxit, string mode)
{
    uint dim = MPS.GetMl();
    uint D_tot=dim*dim;
    assert(MPS.GetMr()==dim);
    /// function handle for the actual routine for applying the TM from the left/right onto some Matrix
    std::function<void (const Mat<Real>&, Mat<Real>&)> TMfun;
    if (dir==l) TMfun = [&MPS](const Mat<Real>& in, Mat<Real>& out) -> void {out = ApplyTMLeft(MPS,in);};
    else if (dir==r) TMfun = [&MPS](const Mat<Real>& in, Mat<Real>& out) -> void {out = ApplyTMRight(MPS,in);};
    else
    {
        cerr<<"wrong direction specified"<<endl;
        abort();
    }

    /// actual calculation of the dominant eigenpair of the TM
    CVecType valtmp;
    CMatType Vtmp;
    RVecType x0v;
    if (x0.n_rows==dim && x0.n_cols==dim) x0v = RVecType(x0.memptr(),D_tot);
    eigs_rn([dim,&TMfun](Real* invec, Real* outvec)->void {TMMult(invec,outvec,TMfun,dim);},D_tot,valtmp,Vtmp,1,mode,tol,x0v,maxit);

    /// analyze and postedit the dominant eigenpair
    V.clear();
    if (imag(valtmp(0))>2*tol) cerr<<"Warning: dominant eigenvalue is complex: "<<valtmp(0)<<endl;
    if (norm(imag(Vtmp.col(0)))>2*tol*D_tot) cerr<<"Warning: dominant eigenvector is complex"<<endl;
    RVecType Vvec = real(Vtmp.col(0));
    Real val = real(valtmp(0));
    V = reshape(Vvec,dim,dim);
    V = (V + V.t());
    V /= trace(V);

    if (dir==l) DOUT("left: "<<abs(val)<<", "<<val<<": "<<norm(ApplyTMLeft(MPS,V) - val*V,"fro"));
    else if (dir==r) DOUT("right: "<<abs(val)<<", "<<val<<": "<<norm(ApplyTMRight(MPS,V) - val*V,"fro"));

    return val;
}


template<typename VT>
CVecType
TMEigs(const MPSDenseMat<VT>& MPS, std::vector<Mat<Complex> >& V, dirtype dir, uint nev, Real tol=1e-14, const Mat<VT>& x0=Mat<VT>(), int maxit=500, string mode="LR");


template<>
CVecType
TMEigs(const MPSDenseMat<Real>& MPS, std::vector<Mat<Complex> >& V, dirtype dir, uint nev, Real tol, const Mat<Real>& x0, int maxit, string mode)
{
    if (mode!="LR" && mode!="LM")
    {
        cerr<<"wrong mode "<<mode<<endl;
        abort();
    }

    uint dim = MPS.GetMl();
    uint D_tot=dim*dim;
    assert(MPS.GetMr()==dim);

    /// function handle for the actual routine for applying the TM from the left/right onto some Matrix
    std::function<void (const Mat<Real>&, Mat<Real>&)> TMfun;
    if (dir==l) TMfun = [&MPS](const Mat<Real>& in, Mat<Real>& out) -> void {out = ApplyTMLeft(MPS,in);};
    else if (dir==r) TMfun = [&MPS](const Mat<Real>& in, Mat<Real>& out) -> void {out = ApplyTMRight(MPS,in);};
    else
    {
        cerr<<"wrong direction specified"<<endl;
        abort();
    }

    /// actual calculation of the dominant eigenpair of the TM
    CVecType vals;
    CMatType Vtmp;
    RVecType x0v;
    if (x0.n_rows==dim && x0.n_cols==dim) x0v = RVecType(x0.memptr(),D_tot);
    eigs_rn([dim,&TMfun](Real* invec, Real* outvec)->void {TMMult(invec,outvec,TMfun,dim);},D_tot,vals,Vtmp,nev,mode,tol,x0v,maxit);

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
        V[n] = reshape(Vtmp.col(n),dim,dim);
        if (dir==l) DOUT("left: "<<abs(vals(n))<<", "<<vals(n)<<": "<<norm(ApplyTMLeft(MPS,V[n]) - vals(n)*V[n]));
        else if (dir==r) DOUT("right: "<<abs(vals(n))<<", "<<vals(n)<<": "<<norm(ApplyTMRight(MPS,V[n]) - vals(n)*V[n]));
    }
    return vals;
}

template<typename VT>
MPSDenseMat<VT>
Ortho(const MPSDenseMat<VT>& A0, Lambda& lam, dirtype dir, Real tol=1e-14, const Mat<VT>& L0=Mat<VT>(), const Mat<VT>& R0=Mat<VT>(),  uint maxit=500, string mode="LM")
{
    Mat<VT> L,R,UL,UR,U,V;
    RVecType DL,DR;

    Real vall = TMDominantEig(A0,L,l,tol,L0,maxit,mode);
    Real valr = TMDominantEig(A0,R,r,tol,R0,maxit,mode);
    Real nrm = 0.5*(vall+valr);
    if (abs(vall-valr)>tol*nrm) cerr<<"warning: vall and valr differ by "<<valr-vall<<endl;

    eig_sym(DL,UL,L);
    eig_sym(DR,UR,R);

    Mat<VT> X = UR*diagmat(sqrt(DR));
    Mat<VT> Y = diagmat(sqrt(DL))*UL.t();

    svd_econ(U,lam,V,Y*X);
    Real lamnrm = norm(lam);
    lam/=lamnrm;

    Mat<VT> matl,matr;

//    MPSDenseMat<VT> A(A0.GetDim(),A0.GetNSites());
    switch (dir)
    {
    case l:
        matl = (lamnrm/sqrt(nrm))*diagmat(lam)*V.t()*diagmat(1./sqrt(DR))*UR.t();
        matr = UL*diagmat(1./sqrt(DL))*U;
        break;
    case r:
        matl = V.t()*diagmat(1./sqrt(DR))*UR.t();
        matr = (lamnrm/sqrt(nrm))*UL*diagmat(1./sqrt(DL))*U*diagmat(lam);
        break;
    case c:
        break;
    case s:
        break;
    }

    return matl*(A0*matr);
}

//template<typename T>
//void OrthoCanon(MPSDenseMat<T>& gam, Lambda& lam, double tol=1e-15, int maxit=500)
//{
///** \brief Orthogonalization of a canonical unit cell gamma, lambda.
// *  Be sure that all lambdas have been removed from gamma, i.e. it is in the right canonical form.
// */
//    const string mode = "LR";
//    int m=gam.GetMl();
//    assert(m==(int)gam.GetMr());
//
//    Mat<T> Vl,Vr;
//    MPSDenseMat<T> A = lam>>gam;
//    MPSDenseMat<T> B = gam<<lam;
//    double El=DiagTM(A,Vl,l,tol,maxit,mode);
//    double Er=DiagTM(B,Vr,r,tol,maxit,mode);
//
//    if(std::abs(El-Er)>1e-10) cerr<<"OrthoCanon: left and right dominant eigenvalue differ by "<<std::abs(El-Er)<<endl;
//
//    double nrm=0.5*(El + Er);
//
//    DOUT("check left ("<<El<<"): "<<norm(ApplyTMLeft(A,Vl) - El*Vl,"fro")<<", hermitian: "<<norm(Vl - Vl.t(),"fro"));
//    DOUT("check right ("<<Er<<"): "<<norm(ApplyTMRight(B,Vr) - Er*Vr,"fro")<<", hermitian: "<<norm(Vr - Vr.t(),"fro"));
//
//    Mat<T> Ur,Ul,dum;
//    Lambda Dr,Dl;
////    eig_sym(Dl,Ul,Vl,"dc");
////    eig_sym(Dr,Ur,Vr,"dc");
//    svd(Ul,Dl,dum,Vl,"dc");
//    svd(Ur,Dr,dum,Vr,"dc");
//
//    Lambda Drsq = sqrt(Dr);
//    Lambda Dlsq = sqrt(Dl);
//
//    Mat<T> X = Ur<<Drsq;
//    Mat<T> Y = Dlsq>>Mat<T>(Ul.t());
//    DOUT("decomposition of Vr: "<<norm(Vr - X*X.t(),"fro"));
//    DOUT("decomposition of Vl: "<<norm(Vl - Y.t()*Y,"fro"));
//
//    Mat<T> U,V;
////    svd(U,lam,V,Mat<T>(Y*Mat<T>(lam>>X)),"dc");
////    if(!svd(U,lam,V,Mat<T>(Y*Mat<T>(lam>>X)))) cerr<<"OrthoCanon: SVD(Y*lam*X) failed"<<endl;
//    if(!svd(U,lam,V,Mat<T>(Y*Mat<T>(lam>>X)),"dc")) cerr<<"OrthoCanon: SVD(Y*lam*X) failed"<<endl;
//
//    double nrmfac=norm(lam,2);
//    lam/=nrmfac;
//
//    Mat<T> matl = nrmfac * Mat<T>(V.t()) * Mat<T>(Drsq<<Mat<T>(Ur.t()));
//    Mat<T> matr = Ul * Mat<T>(Dlsq<<U) /std::sqrt(nrm);
//
//    for (auto& it : gam) it = Mat<T>(matl * it) * matr;
//}
//
//template<typename T>
//void OrthoMPS(MPSDenseMat<T>& A, Lambda& lam, dirtype dir, double tol=1e-15, int maxit=500)
//{
///** \brief Orthogonalization of a canonical unit cell gamma, lambda.
// *  Be sure that all lambdas have been removed from gamma, i.e. it is in the right canonical form.
// */
//    const string mode = "LR";
////    int m=;
//    assert(A.GetMl()==A.GetMr());
//
//    Mat<T> Vl,Vr;
//    double El=DiagTM(A,Vl,l,tol,maxit,mode);
//    double Er=DiagTM(A,Vr,r,tol,maxit,mode);
//
//    if(std::abs(El-Er)>1e-10) cerr<<"OrthoMPS: left and right dominant eigenvalue differ by "<<std::abs(El-Er)<<endl;
//
//    double nrm=0.5*(El + Er);
//
//    DOUT("check left ("<<El<<"): "<<norm(ApplyTMLeft(A,Vl) - El*Vl,"fro")<<", hermitian: "<<norm(Vl - Vl.t(),"fro"));
//    DOUT("check right ("<<Er<<"): "<<norm(ApplyTMRight(A,Vr) - Er*Vr,"fro")<<", hermitian: "<<norm(Vr - Vr.t(),"fro"));
//
//    Mat<T> Ur,Ul;
//    Lambda Dr,Dl;
//    eig_sym(Dl,Ul,Vl,"dc");
//    eig_sym(Dr,Ur,Vr,"dc");
//    Lambda Drsq = sqrt(Dr);
//    Lambda Dlsq = sqrt(Dl);
//
//    Mat<T> X = Ur<<Drsq;
//    Mat<T> Y = Dlsq>>Mat<T>(Ul.t());
//    DOUT("decomposition of Vr: "<<norm(Vr - X*X.t(),"fro"));
//    DOUT("decomposition of Vl: "<<norm(Vl - Y.t()*Y,"fro"));
//
//    Mat<T> U,V;
////    svd(U,lam,V,Mat<T>(Y*Mat<T>(lam>>X)),"dc");
////    if(!svd(U,lam,V,Mat<T>(Y*Mat<T>(lam>>X)))) cerr<<"OrthoCanon: SVD(Y*lam*X) failed"<<endl;
//    if(!svd(U,lam,V,Mat<T>(Y*X),"dc")) cerr<<"OrthoMPS: SVD(Y*X) failed"<<endl;
//
//    Lambda lamsq;
//    Mat<T> matl,matr;
////    double nrmfac=norm(lam,2);
////    lam/=nrmfac;
////
////    nrmfac/=std::sqrt(nrm);
//
//    if (dir==l)
//    {
//        matl = Mat<T>(lam >> Mat<T>(V.t())) * Mat<T>(Drsq<<Mat<T>(Ur.t()));
//        matr = Mat<T>(Ul >> Dlsq) * U / std::sqrt(nrm);
//    }
//    else if (dir==r)
//    {
//        matl = Mat<T>(Mat<T>(V.t())) * Mat<T>(Drsq<<Mat<T>(Ur.t()));
//        matr = Mat<T>(Ul >> Dlsq) * Mat<T>(U << lam) / std::sqrt(nrm);
//    }
//    else if (dir==s)
//    {
//        lamsq = sqrt(lam);
//        matl = Mat<T>(lamsq >> Mat<T>(V.t())) * Mat<T>(Drsq<<Mat<T>(Ur.t()));
//        matr = Mat<T>(Ul >> Dlsq) * Mat<T>(U << lamsq) / std::sqrt(nrm);
//    }
//    else cerr<<"OrthoMPS: wrong direction specified"<<endl;
//
//    for (auto& it : A) it = Mat<T>(matl * it) * matr;
//
//    lam /= norm(lam,2);
//
//}

//template<typename T>
//void OrthoSymm(MPSDenseMat<T>& A, Lambda& lam, double tol=1e-15, int maxit=500)
//{
//
//}

#endif // EIGS_DENSE_FUN_H_
