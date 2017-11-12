#include <iostream>

#include <MPSIncludesAdvanced.h>

#include <parser.h>
#include <Models.h>
#include <tictoc.hpp>

#include "Excitations_helpers.hpp"

using std::cout;
using std::cin;
using std::endl;
using std::string;

using MPSMat = MPSBlockMat<IKey,Scalar>;
using DiagMat = BlockDiagMat<IKey,Scalar>;
using Lambda = BlockLam<IKey>;
using IBMat = BlockMat<IKey,Scalar>;
using IBMatArray = BlockMatArray<IKey,Scalar>;
using IMPSArray = MPSBlockMatArray<IKey,Scalar>;
using IDiagArray = BlockDiagMatArray<IKey,Scalar>;
using ILamArray = BlockLamArray<IKey>;

template<typename VT, typename KT, typename HVT>
void
CalculateExcitations(RMatType& dE,
                     std::vector<std::vector<BlockMatArray<KT,Complex> > >& X,
                     const MPSBlockMatArray<KT,VT>& ALvec,
                     const MPSBlockMatArray<KT,VT>& ARvec,
                     const BlockDiagMatArray<KT,VT>& Lvec,
                     const BlockDiagMatArray<KT,VT>& Rvec,
                     SparseOperator<HVT>& H,
                     const RVecType& pvec,
                     const KT& K,
                     uint nev,
                     Real tol,
                     Real InvETol,
                     bool verbose,
                     bool test=false,
                     const std::vector<RSpOp>& obs=std::vector<RSpOp>());

int main(int argc, char** argv)
{
    arma_rng::set_seed_random();
    std::string filename,mode="SR",datafolder=".",trans_op="";
    std::vector<int> Kvec, QNvec;
//    std::vector<Real> pvec;
    std::vector<std::string> obsnames;
    Real tol = 1e-10,InvETol=1e-14;
    uint N=2, nev=4, np=1, rel_shift=0, glob_shift=0;
    Real pmin, pmax;
    bool verbose=false, saveE=false, saveX=false, test=false, avg=false;

    uint coutprec = 12;
    cout.precision(coutprec);

    parser pp(argc,argv);

    pp.GetValue(tol,"tol");
    pp.GetValue(InvETol,"InvETol");
    pp.GetValue(verbose,"verbose");
    pp.GetValue(nev,"nbands");
    pp.GetValue(saveE,"saveE");
    pp.GetValue(saveX,"saveX");
    pp.GetValue(avg,"avg");

    pp.GetValue(filename,"state",true);
    pp.GetValue(datafolder,"datafolder");
    pp.GetValue(QNvec,"QN",true);
    pp.GetValue(Kvec,"K",true);
    pp.GetValue(rel_shift,"shift");
    pp.GetValue(glob_shift,"glob_shift");
    pp.GetValue(trans_op,"op");
    pp.GetValue(test,"test");
    pp.GetValue(obsnames,"obs");

    /// momentum values
    pp.GetValue(pmin,"pmin",true);
    pp.GetValue(pmax,"pmax",true);
    pp.GetValue(np,"np",true);

    int frmt = std::ceil(-std::log10(tol))+1;
    InvETol = std::max(tol/10.,InvETol);

    cout.precision(frmt);

    /// create model
    modptr pmod = CreateModel(pp,verbose);

    auto H = pmod->GetLocalHam();

    if (Kvec.size()!=pmod->GetNSym()) throw std::domain_error("K has wrong number of quantum numbers");
    IKey K(pmod->GetGroupObj(),Kvec);
    IKey QN(pmod->GetGroupObj(),QNvec);

    /// create model and simulation parameter strings and output simulation info
    std::stringstream sstr;
    string savefolder;

    sstr<<fileparts(filename).name<<"_p"<<pmin<<"_"<<pmax<<"_"<<np<<"_nb"<<nev<<"_shift"<<rel_shift;
    if (!trans_op.empty()) sstr<<"_op"<<trans_op;
    sstr<<"_K";
    sstr<<K;
    cout<<"simulation summary: "<<sstr.str()<<"_tol"<<tol<<"_InvETol"<<InvETol<<endl;

    /// load IMPS state
    ILamArray Lamvec;
    IDiagArray Cvec,Lvec,Rvec;
    IMPSArray ALvec,ARvec,NLvec;
    MPSBlockMatArray<IKey,Complex> Bin,Bout;

    if (!loadIMPS(Lamvec,Cvec,ALvec,ARvec,pmod->GetGroupObj(),filename)) throw std::runtime_error(filename+" could not be loaded");

    /// unit cell size
    N = ALvec.size();

    /// momentum values
    RVecType pvec = linspace(pmin, pmax, np);


    /// create I2K object and check its compliance with the loaded state
    auto I2K = pmod->GetI2K(N,QNvec);
    auto obs = pmod->GetObservables(obsnames);

    if (ALvec != I2K) throw std::runtime_error("AL not compatible with I2K from model");
    if (ARvec != I2K) throw std::runtime_error("AR not compatible with I2K from model");


    CheckOrthoLRSqrt(ALvec,ARvec,Cvec);


    for (const auto& it : Cvec)
    {
        Lvec.emplace_back(it.t()*it);
        Rvec.emplace_back(it*it.t());
    }


    /// apply transformations to AR if present
    if (!trans_op.empty()) ApplyTransformation(ARvec,Lvec,trans_op,I2K,pmod);


    /// shift AR unit cell
    if (rel_shift > N)
    {
        rel_shift = rel_shift%N;
        cerr<<endl<<"0 < rel_shift < N, setting to "<<rel_shift<<endl<<endl;
    }
    if (rel_shift > 0)
    {
        shift(ARvec,rel_shift);
        shift(Lvec,rel_shift);
    }


    /// apply global shift to state
    if (glob_shift > N)
    {
        glob_shift = glob_shift%N;
        cerr<<endl<<"0 < glob_shift < N, setting to "<<glob_shift<<endl<<endl;
    }

    if (verbose)
    {
        cout<<"I2K:"<<endl;
        cout<<I2K<<endl;
    }

//    /// measure observables
//    auto obs = pmod->GetObservables(obsnames);
//    if (!obs.empty())
//    {
//        Real p=1.;
////        pp.GetValue(p,"p");
//
//        LMpass.clear();
//        RMpass.clear();
//        if (abs(p)<10*InvETol && K==K0)
//        {
//            LMpass = LM;
//            RMpass = RM;
//        }
//
//        MeasureExcitations(ALvec,ARvec,NLvec,LMpass,RMpass,Lvec,Rvec,obs,xdims,mtot,p,InvETol,verbose);
//        cout<<std::string(100,'=')<<endl;
//    }


//    /// early bail out for testing
//    if (test) return 0;

    if (!test && (saveE || saveX))
    {
        savefolder = GetUniquePath(datafolder + "/" + sstr.str());
        if (!mkdir(savefolder)) throw std::runtime_error("couldn't create "+savefolder);

        /// save vector of momentum values
        string savepname = sstr.str()+"_pvec";
        if (pvec.save(Fullpath(savepname,"bin",savefolder))) cout<<"saved pvec under "<<savepname<<endl;
        else cerr<<"failed to save pvec under "<<savepname<<endl;

        /// save ground state, from which excitations are created, INCLUDING shifts or transformations to AR
        string saveALRname = sstr.str()+"_ALRN";
        if (saveALRN(Cvec,Lvec,Rvec,ALvec,ARvec,NLvec,savefolder+"/"+saveALRname)) cout<<"saved ALRN under "<<saveALRname<<endl;
        else cerr<<"failed to save ALRN under "<<saveALRname<<endl;
    }


    /// actually calculate excitations
    RMatType dE(np,nev,fill::zeros);
    std::vector<std::vector<BlockMatArray<IKey,Complex> > > X(np);

    uint ngs = 1;
    if (avg)
    {
        ngs = N;
        glob_shift = 0;
    }
    std::vector<RMatType> dEv(ngs);
    std::vector<std::vector<std::vector<BlockMatArray<IKey,Complex> > > > Xv(ngs);

    if (glob_shift>0)
    {
        shift(ALvec,glob_shift);
        shift(ARvec,glob_shift);
        shift(Cvec,glob_shift);
        shift(Lvec,glob_shift);
        shift(Rvec,glob_shift);
    }

    for (uint gs=0; gs<ngs;++gs)
    {
        cout<<"global shift = "<<glob_shift+gs<<endl;
        CalculateExcitations(dEv[gs],Xv[gs],ALvec,ARvec,Lvec,Rvec,H,pvec,K,nev,tol,InvETol,verbose,test,obs);

        if (gs<ngs-1)
        {
            shift(ALvec,1);
            shift(ARvec,1);
            shift(Cvec,1);
            shift(Lvec,1);
            shift(Rvec,1);
        }
    }

    if (ngs>1)
    {
        for (uint gs=0; gs<ngs; ++gs)
        {
            dE += dEv[gs];
//            for (uint l=0; l<nev;++l) for (uint n=0;n<N;++n) X[l][n] += Xv[gs][l][n];
        }
        dE /= double(ngs);
        for (uint n=0;n<np;++n)
        {
            cout<<"p = "<<pvec(n)<<" "<<N<<" pi: \t";
            for (uint k=0; k<dE.n_cols; ++k) cout<<dE(n,k)<<"\t";
            cout<<endl;
        }
    }
    else
    {
        dE = dEv[0];
        X = Xv[0];
    }



    if (saveE)
    {

        string saveEname = sstr.str()+"_dE";
        if (dE.save(Fullpath(saveEname,"bin",savefolder))) cout<<"saved dE under "<<saveEname<<endl;
        else cerr<<"failed to save dE under "<<saveEname<<endl;

    }

    if (saveX)
    {
        for (uint n=0;n<np;++n)
        {
            string tmpXname = sstr.str() + "_X" + std::to_string(n);
//        std::vector<BlockMatArray<IKey,Complex> > Xvec;
//        Xvec.reserve(nev);
//        for (uint l=0; l<nev; ++l) Xvec.emplace_back(BlockMatArray<IKey,Complex>(X[n].col(l),xdims));
            if (save(X[n],Fullpath(tmpXname,"bin",savefolder))) cout<<"X"<<n<<" saved under "<<tmpXname<<endl;
            else cerr<<"failed to save X"<<n<<" under "<<tmpXname<<endl;
        }
    }


    return 0;
}

template<typename VT, typename KT, typename HVT>
void
CalculateExcitations(RMatType& dE,
                     std::vector<std::vector<BlockMatArray<KT,Complex> > >& X,
                     const MPSBlockMatArray<KT,VT>& ALvec,
                     const MPSBlockMatArray<KT,VT>& ARvec,
                     const BlockDiagMatArray<KT,VT>& Lvec,
                     const BlockDiagMatArray<KT,VT>& Rvec,
                     SparseOperator<HVT>& H,
                     const RVecType& pvec,
                     const KT& K,
                     uint nev,
                     Real tol,
                     Real InvETol,
                     bool verbose,
                     bool test,
                     const std::vector<RSpOp>& obs)
{
    uint N = ALvec.size();
    uint np = pvec.n_elem;
    KT K0(K.GetGroupObj());

    /// show quantum number sectors
    if (verbose)
    {
        cout<<"AL[0] outgoing"<<endl;
        cout<<ALvec[0].GetMr()<<endl;

        cout<<"AR[0] ingoing"<<endl;
        cout<<ARvec[1].GetMl()<<endl;
    }

    BlockDiagMat<KT,Scalar> L(Lvec.back());
    BlockDiagMat<KT,Scalar> R(Rvec.back());

    /// check overlaps between AL and AR and get left and right eigenmatrices of T^R_L (obtain the ones of T^L_R by hermitian conjugation)
    BlockMat<KT,Scalar> LM,RM,LMpass,RMpass;
    cout<<"mixed TM AL and AR"<<endl;
    LRoverlaps(LM,RM,ALvec,ARvec,K,nev);

    /// calculate ground state energy density to subtract from Hamiltonian
    cout<<"== H "<<std::string(94,'=')<<endl;
    cout<<"-- AL "<<std::string(94,'-')<<endl;
    RVecType E0vL = MeasureObservables(H,ALvec,BlockDiagMatArray<KT,Scalar>(),Rvec,true);
    cout<<"-- AR "<<std::string(94,'-')<<endl;
    RVecType E0vR = MeasureObservables(H,ARvec,Lvec,BlockDiagMatArray<KT,Scalar>(),true);
    double E0L = mean(E0vL);
    double E0R = mean(E0vR);
    double dE0 = E0L - E0R;
    if (std::abs(dE0) > tol) throw std::domain_error("ground state energies from AL and AR differ by " + to_varstring(dE0));

    double E0 = 0.5*(E0L + E0R);
    H -= E0*SpId<Real>(H.GetLocalDim(),H.GetNSites()); /// subtract ground state energy from Hamiltonian

    /// calculate Null spaces for all AL
    MPSBlockMatArray<KT,VT> NLvec;
    for (const auto& Ait : ALvec) NLvec.emplace_back(Nullspace(Ait,l));
    /// determine block structure and possible symmetry sectors for the excitation DOF BlockMat X
    std::vector<dimkeypair_vec<KT> > xdims;
    uint mtot = XDims(xdims,NLvec,ARvec,K);
    if (!all(xdims)) throw std::domain_error("no excitations for this quantum number");


    /// measure observables
    if (!obs.empty())
    {
//        for (const auto& oit : obs) cout<<oit.GetName()<<endl;
        Real p=1.;
//        pp.GetValue(p,"p");

        LMpass.clear();
        RMpass.clear();
        if (abs(p)<10*InvETol && K==K0)
        {
            LMpass = LM;
            RMpass = RM;
        }

        MeasureExcitations(ALvec,ARvec,NLvec,LMpass,RMpass,Lvec,Rvec,obs,xdims,mtot,p,InvETol,verbose);
        cout<<std::string(100,'=')<<endl;
    }

    if (test) return;
    /// prepare output filenames/folders etc.


    /// calculate B independent constants for applying effective Hamiltonian
    IDiagArray HLtot,HRtot;
    HeffConstants(HLtot,HRtot,ALvec,ARvec,L,R,H,InvETol,0,verbose);

    BlockMat<KT,Complex> EBR, EHBL;
    dE.resize(np,nev);
    X.resize(np);

    /// We use them to control if we invert (1-T^L_R) and (1-T^R_L) fully or if we project out the dominant subspace
    /// if they're
    tictoc tt,tts;
    /// unfortunately, ARPACK is not thread safe, so we cannot parallelize the following loop :-(
    tt.tic();
    for (uint n=0; n<np; ++n)
    {
        CVecType dEtmp;
        Real p_act = pvec(n)*datum::pi*N;
        Complex kfac = Complex(cos(p_act),sin(p_act));

        LMpass.clear();
        RMpass.clear();
        EBR.clear();
        EHBL.clear();

        /// only project out the dominant subspace, if K=0 and p=0
        if (abs(p_act)<10*InvETol && K==K0)
        {
            LMpass = LM;
            RMpass = RM;
        }

        std::function<void (Complex*,Complex*)> Hfun = [&](Complex* in, Complex* out) -> void
        {
            ApplyHeff(in,out,xdims,mtot,kfac,ALvec,ARvec,NLvec,LMpass,RMpass,H,HLtot,HRtot,InvETol,verbose,&EBR,&EHBL);
        };

        tts.tic();
        CMatType Xmat;
        eigs_n(Hfun,mtot,dEtmp,Xmat,nev,"SR",tol);

        if (any(imag(dEtmp)>tol)) cerr<<"IMAGINARY ENERGIES FOR p="<<pvec(n)<<": "<<imag(dEtmp)<<endl;

//            X.emplace_back(std::vector<BlockMatArray<KT,Complex> >());
//            for (uint l=0; l<nev; ++l) X.back().emplace_back(BlockMatArray<KT,Complex>(Xmat.col(l),xdims));
        X[n].resize(nev);
        for (uint l=0; l<nev; ++l) X[n][l] = BlockMatArray<KT,Complex>(Xmat.col(l),xdims);


        dE.row(n) = real(dEtmp.t()); /// dEtmp is column vector
        cout<<"p = "<<pvec(n)<<" "<<N<<" pi: \t";
        for (uint k=0; k<dE.n_cols; ++k) cout<<dE(n,k)<<"\t";
        cout<<"("<<tts.toc()<<" s.)"<<endl;
    }

    double tel=tt.toc();
    if (tel>60) cout<<"elapsed time: "<<tel/double(60)<<" min."<<endl;
    else cout<<"elapsed time: "<<tel<<" s."<<endl;

}
