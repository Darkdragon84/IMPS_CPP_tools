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
//    arma_rng::set_seed_random();
    std::string filename,mode="SR",datafolder=".",trans_op="";
    std::vector<int> exc_keyvec, QN;
//    std::vector<Real> pvec;
    std::vector<std::string> obsnames;
    Real tol = 1e-10,InvETol=1e-14;
    uint N=2, nev=4, np=1, rel_shift=0, glob_shift=0;
    Real pmin, pmax;
    bool verbose=false, saveE=false, saveX=false, test=false, proj=false;

    uint coutprec = 12;
    cout.precision(coutprec);

    parser pp(argc,argv);

    pp.GetValue(tol,"tol");
    pp.GetValue(InvETol,"InvETol");
    pp.GetValue(verbose,"verbose");
    pp.GetValue(nev,"nbands");
    pp.GetValue(proj,"project",true);
    pp.GetValue(saveE,"saveE");
    pp.GetValue(saveX,"saveX");

    pp.GetValue(filename,"state",true);
    pp.GetValue(datafolder,"datafolder");
    pp.GetValue(QN,"QN");
    pp.GetValue(exc_keyvec,"K",true);
    pp.GetValue(rel_shift,"shift");
    pp.GetValue(glob_shift,"glob_shift");
    pp.GetValue(trans_op,"op");
    pp.GetValue(test,"test");
    pp.GetValue(obsnames,"obs");

    /// momentum values
    pp.GetValue(pmin,"pmin",true);
    pp.GetValue(pmax,"pmax",true);
    pp.GetValue(np,"np",true);

    int frmt = std::ceil(-std::log10(tol));
    cout.precision(frmt);

    /// create model
    modptr pmod = CreateModel(pp,verbose);

    uint d = pmod->GetLocalDim();
    auto H = pmod->GetLocalHam();
    uint nsym = pmod->GetNSym();

    if (exc_keyvec.size()!=nsym) throw std::domain_error("K has wrong number of quantum numbers");
    IKey K(pmod->GetGroupObj(),exc_keyvec);

    std::stringstream sstr;
    string savefolder;

    sstr<<fileparts(filename).name<<"_p"<<pmin<<"_"<<pmax<<"_"<<np<<"_nb"<<nev<<"_shift"<<rel_shift;
    if (!trans_op.empty()) sstr<<"_op"<<trans_op;
    sstr<<"_K";
    sstr<<K;

    /// momentum values
    RVecType pvec = linspace(pmin, pmax, np);

    if (saveE || saveX)
    {
        savefolder = GetUniquePath(datafolder + "/" + sstr.str());
        if (!mkdir(savefolder)) throw std::runtime_error("couldn't create "+savefolder);

        string savepname = sstr.str()+"_pvec";
        if (pvec.save(Fullpath(savepname,"bin",savefolder))) cout<<"saved pvec under "<<savepname<<endl;
        else cerr<<"failed to save pvec under "<<savepname<<endl;
    }

    /// load IMPS state
    ILamArray Lamvec;
    IDiagArray Cvec,Lvec,Rvec;
    IMPSArray ALvec,ARvec,NLvec;
    MPSBlockMatArray<IKey,Complex> Bin,Bout;

    if (!loadIMPS(Lamvec,Cvec,ALvec,ARvec,filename,pmod->GetGroupObj())) throw std::runtime_error(filename+" could not be loaded");

    /// unit cell size
    N = ALvec.size();

    /// create I2K object and check its compliance with the loaded state
    auto I2K = pmod->GetI2K(N,QN);

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
    LRoverlaps(LM,RM,ALvec,ARvec,K,nev);


    /// calculate ground state energy density to subtract from Hamiltonian
    RVecType E0vL = MeasureObservables(H,ALvec,BlockDiagMatArray<IKey,Scalar>(),Rvec,true);
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

        MeasureExcitations(ALvec,ARvec,NLvec,LM,RM,Lvec,Rvec,obs,xdims,mtot,p,std::max(tol/100,InvETol),verbose);
        cout<<std::string(100,'=')<<endl;
    }

    /// early bail out for testing
    if (test) return 0;
//    {
//        cout<<"I2K:"<<endl;
//        cout<<I2K<<endl;
//        Lamvec.ShowDims("lam");
//        TMmixedEigs(ALvec,ARvec,VL,l,-K,nev,"LM",OLtol,IBMat(),0,true);
//        TMmixedEigs(ALvec,ARvec,VR,r,K,nev,"LM",OLtol,IBMat(),0,true);
//        teststuff(ALvec,ARvec,Lvec,Rvec,LM,RM,OLL,OLR);
//        cout<<"folder: "<<savefolder<<endl;
//        cout<<"name: "<<fileparts(saveEname).name<<endl;
//        return 0;
//    }



    /// calculate B independent constants for applying effective Hamiltonian
    IDiagArray HLtot,HRtot;
    HeffConstants(HLtot,HRtot,ALvec,ARvec,L,R,H,std::max(tol/100,InvETol),0,verbose);

    /// actually calculate excitations
    RMatType dE(np,nev,fill::zeros);
    std::vector<CMatType> X(np);

    tictoc tt,tts;

    /// unfortunately, ARPACK is not thread safe, so we cannot parallelize the following loop :-(
    tt.tic();
    for (uint n=0; n<np; ++n)
    {
        CVecType dEtmp;
        Real p_act = pvec(n)*datum::pi*N;
        Complex kfac = Complex(cos(p_act),sin(p_act));

        std::function<void (Complex*,Complex*)> Hfun =
        [&xdims,mtot,kfac,&ALvec,&ARvec,&Cvec,&NLvec,&LM,&RM,&H,&HLtot,&HRtot,tol,InvETol,verbose](Complex* in, Complex* out) -> void
        {
            ApplyHeff(in,out,xdims,mtot,kfac,ALvec,ARvec,NLvec,LM,RM,H,HLtot,HRtot,std::max(tol/100,InvETol),verbose);
        };

        tts.tic();
        eigs_n(Hfun,mtot,dEtmp,X[n],nev,"SR",tol);
        cout<<"p = "<<pvec(n)<<" "<<N<<" pi done ("<<tts.toc()<<" s.)"<<endl;
        if (any(imag(dEtmp)>tol)) cerr<<"IMAGINARY ENERGIES FOR p="<<pvec(n)<<": "<<imag(dEtmp)<<endl;
        dE.row(n) = real(dEtmp);

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

    for (uint n=0; n<np; ++n)
    {
        cout<<"p = "<<pvec(n)<<" "<<N<<" pi: \t";
        for (uint k=0; k<dE.n_cols; ++k) cout<<dE(n,k)<<"\t";
        cout<<endl;
    }

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
