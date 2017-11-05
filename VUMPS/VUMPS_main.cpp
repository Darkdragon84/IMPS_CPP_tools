#include <iostream>
#include <iomanip>

#include <MPSIncludesAdvanced.h>
#include <parser.h>
#include <tictoc.hpp>
#include <Models.h>

#include "VUMPS_helpers.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::min;
using std::max;

using MPSMat = MPSBlockMat<IKey,Scalar>;
using BMat = BlockMat<IKey,Scalar>;
using BDMat = BlockDiagMat<IKey,Scalar>;
using Lam = BlockLam<IKey>;
using MPSArr = MPSBlockMatArray<IKey,Scalar>;
using BDMatArr = BlockDiagMatArray<IKey,Scalar>;
using LamArr = BlockLamArray<IKey>;

//template<typename KT, typename VT>
//void
//ApplyHA(const MPSBlockMat<KT,VT>& in, MPSBlockMat<KT,VT>& out, const BlockDiagMat<KT,VT>& EHL, const BlockDiagMat<KT,VT>& EHR, const RedOp<BlockMat<KT,VT> >& HL, const RedOp<BlockMat<KT,VT> >& HR);
//
//template<typename KT, typename VT>
//void
//ApplyHAvec(VT* in, VT* out, const std::vector<dimkeypair_vec<KT> >& Adims, uint d, uint Am_tot, const BlockDiagMat<KT,VT>& EHL, const BlockDiagMat<KT,VT>& EHR, const RedOp<BlockMat<KT,VT> >& HL, const RedOp<BlockMat<KT,VT> >& HR);
//
//template<typename KT, typename VT>
//void
//ApplyHC(const BlockDiagMat<KT,VT>& in, BlockDiagMat<KT,VT>& out, const BlockDiagMat<KT,VT>& EHL, const BlockDiagMat<KT,VT>& EHR, const SparseOperator<VT>& H, const MPSBlockMat<KT,VT>& AL, const MPSBlockMat<KT,VT>& AR);
//
//template<typename KT, typename VT>
//void
//ApplyHCvec(VT* in, VT* out, const dim_vec<KT>& Cdims, uint Cm_tot, const BlockDiagMat<KT,VT>& EHL, const BlockDiagMat<KT,VT>& EHR, const SparseOperator<VT>& H, const MPSBlockMat<KT,VT>& AL, const MPSBlockMat<KT,VT>& AR);
//
//template<typename KT, typename VT>
//BlockDiagMat<KT,VT>
//GetHL(const MPSBlockMatArray<KT,VT>& AL, const RSpOp& H);
//
//template<typename KT, typename VT>
//BlockDiagMat<KT,VT>
//GetHR(const MPSBlockMatArray<KT,VT>& AR, const RSpOp& H);
//
//template<typename KT, typename VT, typename FT>
//inline
//Real
//GradNormLeft(const MPSBlockMat<KT,VT>& AL, const BlockDiagMat<KT,VT>& C, FT&& HAfun);
//
//template<typename KT, typename VT, typename FT>
//inline
//Real
//GradNormRight(const MPSBlockMat<KT,VT>& AR, const BlockDiagMat<KT,VT>& C, FT&& HAfun);
//
//template<typename KT, typename VT, typename FT>
//inline
//Real
//GradNorm(const MPSBlockMat<KT,VT>& A, const BlockDiagMat<KT,VT>& C, FT&& HAfun, dirtype dir=r);

int main(int argc, char** argv)
{
    arma_rng::set_seed_random();
    const uint output_precision = 12;

    uint d=2,N=2,dmmax=20, expand_pause = 5;
    Real thresh = 1e-14;//, svdthresh = 5e-8, thresh=1e-10;
    Real expand_thresh = 1e-5;
    Real lamthresh = 1e-12;
    Real svdthresh = 5e-7;
    Real InvETol = 5e-15;
    Real tolmin = 5e-16;
    Real tolmax = 1e-6;
    bool verbose = true;
    bool plotnorm = false;

    bool savestate=false;
    std::string savefolder="data";

    cout.precision(output_precision);
    std::vector<uint> dims, mmax;
    std::vector<std::vector<int> > keysv, i2kv;
    std::vector<std::string> opstr;
    std::vector<int> QN;
//    std::vector<int> mv;

    parser pp(argc,argv);
    pp.GetValue(N,"N",true);
    if (N<2) throw std::domain_error("unit cell size must be at least 2");

    pp.GetValue(thresh,"thresh");
    pp.GetValue(dmmax,"dmmax");
//    pp.GetValue(mv,"mv");
    pp.GetValue(dims,"m0",true);
    pp.GetValue(mmax,"mmax",true);
    pp.GetValue(QN,"QN",true);
    pp.GetValue(opstr,"obs");
    pp.GetValue(expand_thresh,"exthresh");
    pp.GetValue(lamthresh,"lamthresh");
    pp.GetValue(InvETol,"InvETol");
    pp.GetValue(expand_pause = 5,"expand_pause");
    pp.GetValue(savestate,"save");
    pp.GetValue(verbose,"verbose");
    pp.GetValue(plotnorm,"plotnorm");
    pp.GetValue(savefolder,"folder");

    /// implement "periodic boundary conditions" within UC, i.e. N+1->1 and 0->N
    auto PBC = [N](int x) -> int {return (x + N)%N;};

    if (mmax.size() == 1)
    {
        cout<<"setting all m_max to "<<mmax.front()<<endl;
        mmax = std::vector<uint>(N,mmax.front());
    }
    else if (mmax.size() != N) throw std::domain_error("mtotmax needs to be of length N");


    auto pmod = CreateModel(pp);
    auto I2K = pmod->GetI2K(N,QN);
    auto obsvec = pmod->GetObservables(opstr);

//    std::vector<VecType> expval;
//    for (uint io = 0;io < obsvec.size();++io) expval.emplace_back(VecType(N));

    d = I2K.GetLocalDim();
    auto dimvec = pmod->MakeDims(dims,N,QN);
    auto H = pmod->GetLocalHam();

    if (dimvec.size()!=N) throw std::domain_error("dimvec is not of length N");

    MPSArr ALvec,ARvec;
    BDMatArr Cvec,Lvec,Rvec;
    LamArr Lamvec;

    randMPS_LR(ALvec,ARvec,Cvec,Lamvec,N,I2K,dimvec);

    for (const auto& it : Cvec)
    {
        Lvec.emplace_back(it.t()*it);
        Rvec.emplace_back(it*it.t());
    }

    /// initial relative shift, as the layout during simulation is
    /// [AL(n),AL(n+1),...,AL(n-1)]
    /// [AR(n+1),AR(n+2),...,AR(n)]
    shift(ARvec,-1);
    shift(Lvec,-1);

    RVecType                            E = MeasureObservables(H,ALvec,BDMatArr(),Rvec,true);
    std::map<std::string,RVecType> expval = MeasureObservables(obsvec,ALvec,BDMatArr(),Rvec,true);

    BDMat HAL, HAR, EHL, EHR, ID, EHL1, EHR1, EHLtmp;
    std::vector<BDMat> EHL0(N),EHR0(N);
    MPSMat AAL(d,2);

    /// initial effective Hamiltonians
    RedOp<BMat> HL = ReducedOp(H,ALvec.back(),l);
    RedOp<BMat> HR = ReducedOp(H,ARvec.front(),r);

//    BDMat R = Cvec.back()*Cvec.back().t();
    BDMat R = Rvec.back();
    HAL = GetHL(ALvec,H);
    EHL = InvertE_proj(ALvec,HAL,eye<Scalar>(Cvec.back().GetMr()),R,l,InvETol);
    AAL = ALvec.back()*ALvec.front();

//    BDMat L = Cvec.front().t()*Cvec.front();
    BDMat L = Lvec.back();
    HAR = GetHR(ARvec,H);
    EHR = InvertE_proj(ARvec,HAR,L,eye<Scalar>(Cvec.front().GetMl()),r,InvETol);
//    AAR = ARvec.back()*ARvec.front();

    EHL1 = ApplyTMLeft(ALvec.front(),EHL) + ApplyOpTMLeftDiag(H,ALvec.back()*ALvec.front());
    EHR1 = ApplyTMRight(ARvec.back(),EHR) + ApplyOpTMRightDiag(H,ARvec.back()*ARvec.front());

    std::function<void (Real*, Real*)> Afun, CRfun, CLfun;

    RVecType EA,ECL,ECR;
    RMatType ACv,CLv,CRv;
    Real err_l=0,err_r=0,tol=InvETol,exp_prec;
    Lam Stmp,lamL,lamR;
    MPSMat QL(d,1),QR(d,1),ALnew(d,1),ARnew(d,1),AC(d,1);
    BDMat PL,PR,CL,CR,Rtmp;
    bool dosvd = true;
    bool run_IDMRG = true;
    bool expand_now = false;
    uvec expand_ct(N,fill::zeros);
    std::vector<bool> lam_too_large(N,false), m_too_small(N,false);

    uint ct=0;
    uint mr_tot=0,CLm_tot=0,CRm_tot=0,Am_tot=0;


    RMatType precm(N,2,fill::ones);
    RVecType prec(N,fill::ones);
    RVecType F(N,fill::ones);
    std::vector<Real> Fv;

    double tstep = 0;
    tictoc tt,ttstep;
    tt.tic();

    while(run_IDMRG)
    {
        ++ct;
        exp_prec = max(prec);
        ttstep.tic();

        for (uint nn=0; nn<N; ++nn)
        {
            EHL0[nn] = EHL;
            EHR0[nn] = EHR;

            F(nn) = GradNormRight(ARvec.back(),Cvec.back(),[&](const MPSMat& in) -> MPSMat {MPSMat out(d);ApplyHA(in,out,EHL,EHR,HL,HR);return out;});
            tol = min(max(prec(nn)/100.,tolmin),tolmax);

            AC = Cvec.back()*ARvec.back();
            auto Adims = AC.GetSizesVector();
            Am_tot = AC.GetNElem();


            Afun = [&](Real* in, Real* out) -> void {ApplyHAvec(in,out,Adims,d,Am_tot,EHL,EHR,HL,HR);};

            eigs_rs(Afun,Am_tot,EA,ACv,1,"SA",tol,AC.Vectorize()); /// here AC0 helps a lot!
            AC = MPSMat(ACv.col(0),Adims,d);

            auto CLdims = Cvec.back().GetSizesVector();
            CLm_tot = Cvec.back().GetNElem();
//
            auto CRdims = Cvec.front().GetSizesVector();
            CRm_tot = Cvec.front().GetNElem();

            CLfun = [&](Real* in, Real* out) -> void {ApplyHCvec(in,out,CLdims,CLm_tot,EHL,EHR1,H,ALvec.back(),ARvec.back());};
            CRfun = [&](Real* in, Real* out) -> void {ApplyHCvec(in,out,CRdims,CRm_tot,EHL1,EHR,H,ALvec.front(),ARvec.front());};


            eigs_rs(CLfun,CLm_tot,ECL,CLv,1,"SA",tol,Cvec.back().Vectorize()); /// here CL0 helps!
            eigs_rs(CRfun,CRm_tot,ECR,CRv,1,"SA",tol,Cvec.front().Vectorize()); /// here CR0 helps!
            CLv/=sign(CLv.col(0)(0));
            CRv/=sign(CRv.col(0)(0));

            CL = BDMat(CLv.col(0),CLdims); /// with sign convention
            CR = BDMat(CRv.col(0),CRdims); /// with sign convention

            lamL = svd(CL);
            lamR = svd(CR);

//            if (dosvd) dosvd = min(precm.row(0)) > svdthresh || (min(min(lamL)) > 1e-8 && min(min(lamR)) > 1e-8);
            if (dosvd && prec(nn) < svdthresh) dosvd = false;

            if (dosvd)
            {
                QL = svd(Stmp,PL,AC*CR.t(),l);
                ALnew = QL*PL;

                QR = svd(Stmp,PR,CL.t()*AC,r);
                ARnew = PR*QR;
            }
            else
            {
                /// polar decomposition scheme

//                cout<<"doing QR"<<endl;
                /// old QR scheme
                QL = qr(Rtmp,AC,l);
                PL = qr(Rtmp,CR,l);
                ALnew = QL*PL.t();

                QR = qr(Rtmp,AC,r);
                PR = qr(Rtmp,CL,r);
                ARnew = PR.t()*QR;
            }
            err_l = norm_inf(AC - ALnew*CR);
            err_r = norm_inf(AC - CL*ARnew);
            precm(nn,0) = err_l;
            precm(nn,1) = err_r;
            prec(nn) = max(err_l,err_r);

            ALvec.front() = ALnew;
            ARvec.back() = ARnew;
            Cvec.front() = CR;
            Cvec.back() = CL;
            Lamvec.front() = lamR;
            Lamvec.back() = lamL;

            R = CL*CL.t();
//            TMDominantEig(ALvec,R,r,1e-15,CL*CL.t());
            HAL = GetHL(ALvec,H);
            AAL = ALvec.back()*ALvec.front();
            EHL = InvertE_proj(ALvec,HAL,eye<Scalar>(Cvec.back().GetMr()),R,l,max(tol/100.,InvETol),0,EHL); /// here EHL0 helps a lot!!

            /// shift unit cell one site over, all remaining instructions are effectively for nn+1
            shift(ALvec,-1);
            shift(ARvec,-1);
            shift(Cvec,-1);
            shift(Lamvec,-1);

            EHLtmp = ApplyOpTMLeftDiag(H,AAL) + ApplyTMLeft(ALvec.back(),EHL);
            AAL = ALvec.back()*ALvec.front();
            EHL1 = ApplyOpTMLeftDiag(H,AAL) + ApplyTMLeft(ALvec.front(),EHLtmp);

            L = Cvec.front().t()*Cvec.front();
            HAR = GetHR(ARvec,H);
            EHR = InvertE_proj(ARvec,HAR,L,eye<Scalar>(Cvec.front().GetMl()),r,max(tol/100.,InvETol),0,EHR0[PBC(nn+1)]); /// interestingly, here EHR0 also helps, contrary to the implementation without QN!!



            mr_tot = Cvec.back().GetTotalMr(); /// should be CR
            auto minlam = min(Lamvec.back()); /// should be lamR

//            expand_now = expand_ct[nn] >= expand_pause && prec(PBC(nn+1)) < expand_thresh && mr_tot < mmax[nn] && max(minlam) > lamthresh;
            expand_now = expand_ct[nn] >= expand_pause && exp_prec < expand_thresh && mr_tot < mmax[nn] && max(minlam) > lamthresh;
//            expand_now = expand_ct[nn] >= expand_pause && prec(nn) < expand_thresh && mr_tot < mmax[nn] && max(minlam) > lamthresh;

            if (expand_now)
            {
                expand_ct[nn] = 0;

                auto mrnew = ExpandFromH(H,I2K,ALvec.back(),ARvec.back(),Cvec.back(),min({dmmax,mmax[nn]-mr_tot}),lamthresh);


                /// artificially enlarge AR[N-2] with zeros (this is the right canonical equivalent of the just expanded AL)
                /// AR[N-2] must have exactly the same number of symmetry sectors and dimensions as AL (so even introduce new zero blocks),
                /// as otherwise dimensions for a later InvertE don't match.
                for (uint s=0; s<d; ++s)
                {
                    for (const auto& ALit : ALvec.back()[s])
                    {
                        auto ARit = ARvec[N-2][s].lower_bound(Qin(ALit));
                        if (ARit != ARvec[N-2][s].end() && ARit->first == Qin(ALit))
                            QMat(*ARit) = join_rows(QMat(*ARit),MatType(QMat(*ARit).n_rows,QMat(ALit).n_cols - QMat(*ARit).n_cols,fill::zeros));
                        else ARvec[N-2][s].emplace_hint(ARit,Qin(ALit),std::make_pair(Qout(ALit),MatType(QMat(ALit).n_rows,QMat(ALit).n_cols,fill::zeros)));
                    }
                }

                EHL = ApplyOpTMLeftDiag(H,ALvec[N-2]*ALvec.back()) + ApplyTMLeft(ALvec.back(),EHL);
            }
            else
            {
                ++expand_ct[nn];
                EHL = EHLtmp;

            }
            EHR1 = ApplyTMRight(ARvec.back(),EHR) + ApplyOpTMRightDiag(H,ARvec.back()*ARvec.front());

            HL = ReducedOp(H,ALvec.back(),l); /// this one is possibly with increased bond dimension
            HR = ReducedOp(H,ARvec.front(),r); /// this one has old bond dimension

        }
        tstep = ttstep.toc();

//        E(0) = trace(ApplyOpTMLeftDiag(H,ALvec.back()*(ALvec.front()*Cvec.front())));
//        for (uint n=1; n<N; ++n) E(n) = trace(ApplyOpTMLeftDiag(H,ALvec[n-1]*(ALvec[n]*Cvec[n])));

        cout<<ct<<":"<<endl<<endl;
        for (uint n=0; n<N; ++n)
        {
            Lvec[PBC(n-1)] = Cvec[n].t()*Cvec[n];

            if (verbose)
            {
                cout<<"Lam["<<n<<"]"<<endl;
                Lamvec[n].ShowDimsMins();
                cout<<"mtot: "<<Lamvec[n].GetNElem()<<endl<<endl;
//                uint nparams = ALvec[n].GetNElem();
//                cout<<nparams<<" var. params in AL("<<n<<"), equiv. to bond dim. "<<round(sqrt(nparams/d))<<endl;
            }
            else cout<<"mtot["<<n<<"]: "<<Lamvec[n].GetNElem()<<endl<<endl;

            lam_too_large[n] = any(Lamvec[n] > lamthresh); /// checks if in any of the sectors the minimal lam is larger than lamthresh
            m_too_small[n] = Lamvec[n].GetNElem() < mmax[n];
        }
        if (plotnorm) Fv.push_back(max(F));

        run_IDMRG = F.max() > thresh || (any(lam_too_large) && any(m_too_small));

        /// Measure energy and other observables (use ARvec)
        E = MeasureObservables(H,ARvec,Lvec,BDMatArr(),true);
        expval = MeasureObservables(obsvec,ARvec,Lvec,BDMatArr(),true);

        cout.precision(4);
        precm.print("errors");
        cout<<"max err.: "<<max(max(precm))<<endl;
        F.print("gradient norm");
        cout<<"max. grad.norm: "<<max(F)<<endl;
        cout<<"tstep = "<<tstep<<" s."<<endl;
        cout.precision(output_precision);
        cout<<"=========================================================================================================================="<<endl;
    }
    double tel = tt.toc();

    /// show the amount of actual variational parameters in Block MPS
    for (uint n=0;n<N;++n)
    {
        uint nparams = ALvec[n].GetNElem();
        cout<<nparams<<" var. params in AL("<<n<<"), equiv. to bond dim. "<<round(sqrt(nparams/d))<<endl;
    }
    if (tel>60.) cout<<"elapsed time: "<<tel/60.<<" min."<<endl;
    else cout<<"elapsed time: "<<tel<<" s."<<endl;


    /// shift AR back by one, as the layout during simulation is
    /// [AL(n),AL(n+1),...,AL(n-1)]
    /// [AR(n+1),AR(n+2),...,AR(n)]
    shift(ARvec,1);

    /// regauge state in the middle of the UC.
    /// The gauges are fine around the edges, but not inside. Actually find out, why
//    qr(Rtmp,Cvec.back(),l);
//    Cvec.back() = Rtmp;
//    for (uint n=0; n<N; ++n)
//    {
//        ALvec[n] = qr(Rtmp,Cvec[PBC(n-1)]*ARvec[n],l);
//        if (n<N-1) Cvec[n] = Rtmp;
//    }
    cout<<"check gauge:"<<endl;
    for (uint n=0;n<N;++n) cout<<n<<": "<<norm_inf(ALvec[n]*Cvec[n] - Cvec[PBC(n-1)]*ARvec[n])<<endl;


    if (savestate || plotnorm)
    {
        uint mfinal = 0;
//        for (const auto& it : Lamvec) mfinal = std::max({mfinal,it.GetNElem()});
        for (const auto& it : Lamvec) mfinal = max(mfinal,it.GetNElem());

        std::stringstream savestrstream;
        savestrstream<<pmod->GetModelString();
        if (QN.size()>0)
        {
            savestrstream<<"_QN"<<QN[0];
            for (uint n=1;n<QN.size();++n) savestrstream<<"_"<<QN[n];
        }
        savestrstream<<"_N"<<N<<"_chi"<<mfinal;
        std::string filename(savestrstream.str());

        if (savestate)
        {
            if (!saveIMPS(Lamvec,Cvec,ALvec,ARvec,savefolder+"/UMPS_"+filename)) cerr<<"failed to save UMPS"<<endl;
        }

        if (plotnorm)
        {
            RVecType Fout(Fv);
            auto Fname = GetUniqueFileName("Fevo_"+filename,"bin",savefolder);
            if (Fout.save(Fname)) cout<<"saved Fevo to "<<Fname<<endl;
            else cerr<<"failed to save Fevo"<<endl;
        }

    }
    return 0;
}
