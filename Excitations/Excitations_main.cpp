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
    bool verbose=false, saveE=false, saveX=false, test=false;

    uint coutprec = 12;
    cout.precision(coutprec);

    parser pp(argc,argv);

    pp.GetValue(tol,"tol");
    pp.GetValue(InvETol,"InvETol");
    pp.GetValue(verbose,"verbose");
    pp.GetValue(nev,"nbands");
    pp.GetValue(saveE,"saveE");
    pp.GetValue(saveX,"saveX");

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

    uint d = pmod->GetLocalDim();
    auto H = pmod->GetLocalHam();
    uint nsym = pmod->GetNSym();

    if (Kvec.size()!=nsym) throw std::domain_error("K has wrong number of quantum numbers");
    IKey K0(pmod->GetGroupObj()); /// zero quantum number, use later to check if we can fully invert mixed transfer ops
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

//    /// output simulation info
//    cout<<"Excitations for "<<np<<" momenta "<<pmin<<"<p<"<<pmax<<" and "<<nev<<" bands"<<endl;
//    cout<<"global shift="<<glob_shift<<", relative shift (AR)="<<rel_shift<<endl;
//    cout<<"quantum number of ground state:"<<QN<<", excitation:"<<K<<endl;

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

    if (ALvec != I2K) throw std::runtime_error("AL not compatible with I2K from model");
    if (ARvec != I2K) throw std::runtime_error("AR not compatible with I2K from model");


    CheckOrthoLRSqrt(ALvec,ARvec,Cvec);

    /// apply global shift to state
    if (glob_shift > N)
    {
        glob_shift = glob_shift%N;
        cerr<<endl<<"0 < glob_shift < N, setting to "<<glob_shift<<endl<<endl;
    }
    if (glob_shift > 0)
    {
        shift(ALvec,glob_shift);
        shift(ARvec,glob_shift);
        shift(Cvec,glob_shift);
    }

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

    BlockDiagMat<IKey,Scalar> L(Lvec.back());
    BlockDiagMat<IKey,Scalar> R(Rvec.back());

    /// check overlaps between AL and AR and get left and right eigenmatrices of T^R_L (obtain the ones of T^L_R by hermitian conjugation)
    BlockMat<IKey,Scalar> LM,RM;
    cout<<"mixed TM AL and AR"<<endl;
    LRoverlaps(LM,RM,ALvec,ARvec,K,nev);

    /// calculate ground state energy density to subtract from Hamiltonian
    cout<<"== H "<<std::string(94,'=')<<endl;
    cout<<"-- AL "<<std::string(94,'-')<<endl;
    RVecType E0vL = MeasureObservables(H,ALvec,BlockDiagMatArray<IKey,Scalar>(),Rvec,true);
    cout<<"-- AR "<<std::string(94,'-')<<endl;
    RVecType E0vR = MeasureObservables(H,ARvec,Lvec,BlockDiagMatArray<IKey,Scalar>(),true);
    double E0L = mean(E0vL);
    double E0R = mean(E0vR);
    double dE0 = E0L - E0R;
    if (std::abs(dE0) > tol) throw std::domain_error("ground state energies from AL and AR differ by " + to_varstring(dE0));

    double E0 = 0.5*(E0L + E0R);
    H -= E0*SpId<Real>(d,2); /// subtract ground state energy from Hamiltonian

    /// calculate Null spaces for all AL
    for (const auto& Ait : ALvec) NLvec.emplace_back(Nullspace(Ait,l));
    /// determine block structure and possible symmetry sectors for the excitation DOF BlockMat X
    std::vector<dimkeypair_vec<IKey> > xdims;
    uint mtot = XDims(xdims,NLvec,ARvec,K);
    if (!all(xdims)) throw std::domain_error("no excitations for this quantum number");

    auto obs = pmod->GetObservables(obsnames);
    if (!obs.empty())
    {
        Real p=0.5;
        pp.GetValue(p,"p");
//
//        cout<<"-- AL "<<std::string(94,'-')<<endl;
//        auto eobsL = MeasureObservables(obs,ALvec,IDiagArray(),Rvec,true);
//        cout<<"-- AR "<<std::string(94,'-')<<endl;
//        auto eobsR = MeasureObservables(obs,ARvec,Lvec,IDiagArray(),true);

        MeasureExcitations(ALvec,ARvec,NLvec,LM,RM,Lvec,Rvec,obs,xdims,mtot,p,InvETol,verbose);
        cout<<std::string(100,'=')<<endl;
    }

    /// early bail out for testing
    if (test) return 0;

    /// prepare output filenames/folders etc.

    if (saveE || saveX)
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

    /// calculate B independent constants for applying effective Hamiltonian
    IDiagArray HLtot,HRtot;
    HeffConstants(HLtot,HRtot,ALvec,ARvec,L,R,H,InvETol,0,verbose);

    /// actually calculate excitations
    RMatType dE(np,nev,fill::zeros);
    std::vector<CMatType> X(np);

    BlockMat<IKey,Complex> EBR, EHBL;

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


        /// only project out the dominant subspace, if K=0 and p=0
        BlockMat<IKey,Scalar> LMpass, RMpass;
        if (abs(p_act)<10*InvETol && K==K0)
        {
            LMpass = LM;
            RMpass = RM;
        }

        std::function<void (Complex*,Complex*)> Hfun =
//        [&xdims,mtot,kfac,&ALvec,&ARvec,&Cvec,&NLvec,&LMpass,&RMpass,&H,&HLtot,&HRtot,tol,InvETol,verbose,&EBR,&EHBL](Complex* in, Complex* out) -> void
        [&](Complex* in, Complex* out) -> void
        {
            ApplyHeff(in,out,xdims,mtot,kfac,ALvec,ARvec,NLvec,LMpass,RMpass,H,HLtot,HRtot,InvETol,verbose,&EBR,&EHBL);
        };

        tts.tic();
        eigs_n(Hfun,mtot,dEtmp,X[n],nev,"SR",tol);

        if (any(imag(dEtmp)>tol)) cerr<<"IMAGINARY ENERGIES FOR p="<<pvec(n)<<": "<<imag(dEtmp)<<endl;
        dE.row(n) = real(dEtmp);

        cout<<"p = "<<pvec(n)<<" "<<N<<" pi: \t";
        for (uint k=0; k<dE.n_cols; ++k) cout<<dE(n,k)<<"\t";
        cout<<"("<<tts.toc()<<" s.)"<<endl;

        if (saveX)
        {
            string tmpXname = sstr.str() + "_X" + std::to_string(n);
            std::vector<BlockMatArray<IKey,Complex> > Xvec;
            Xvec.reserve(nev);
            for (uint l=0; l<nev; ++l) Xvec.emplace_back(BlockMatArray<IKey,Complex>(X[n].col(l),xdims));
            if (save(Xvec,Fullpath(tmpXname,"bin",savefolder))) cout<<"X"<<n<<" saved under "<<tmpXname<<endl;
            else cerr<<"failed to save X"<<n<<" under "<<tmpXname<<endl;
        }

    }
    double tel=tt.toc();

//    for (uint n=0; n<np; ++n)
//    {
//        cout<<"p = "<<pvec(n)<<" "<<N<<" pi: \t";
//        for (uint k=0; k<dE.n_cols; ++k) cout<<dE(n,k)<<"\t";
//        cout<<endl;
//    }

    if (saveE)
    {
        string saveEname = sstr.str()+"_dE";
        if (dE.save(Fullpath(saveEname,"bin",savefolder))) cout<<"saved dE under "<<saveEname<<endl;
        else cerr<<"failed to save dE under "<<saveEname<<endl;

    }

    if (tel>60) cout<<"elapsed time: "<<tel/double(60)<<" min."<<endl;
    else cout<<"elapsed time: "<<tel<<" s."<<endl;

    return 0;
}
