#include <iostream>

#include <MPSIncludesAdvanced.h>

#include <parser.h>
#include <Models.h>
#include <tictoc.hpp>

#include "Excitations_helpers.hpp"

//#ifdef _OPENMP
//#include <omp.h>
//#endif // _OPENMP

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
    std::vector<Real> pvec;
    std::vector<std::string> obsnames;
    Real tol = 1e-10,InvETol=1e-14;
    uint N=2, nev=4, np=1, UC_shift=0;
    Real pmin, pmax;
    bool verbose=false, saveE=false, saveX=false, test=false, proj=false;
//    Complex kfac;

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
    pp.GetValue(UC_shift,"shift");
    pp.GetValue(trans_op,"op");
    pp.GetValue(test,"test");
    pp.GetValue(obsnames,"obs");
//    pp.GetValue(topo,"topo");

    /// momentum values
    pp.GetValue(pmin,"pmin",true);
    pp.GetValue(pmax,"pmax",true);
    pp.GetValue(np,"np",true);

    pvec.emplace_back(pmin);
    Real dp = (np>1) ? (pmax - pmin)/double(np-1) : 0; /// if np=1, we don't need dp, and set it to 0
    for (uint k=1; k<np; ++k)pvec.emplace_back(pmin + k*dp); /// this is only executed if np>1

//    Real dp = 1;
//    if (np>1)
//    {
//        dp = (pmax - pmin)/double(np-1);
//        for (uint k=0; k<np; ++k)pvec.emplace_back(pmin + k*dp);
//    }
//    else pvec.emplace_back(pmin);

    /// create model
    modptr pmod = CreateModel(pp,verbose);
//    if (verbose) pmod->ShowParams();

    uint d = pmod->GetLocalDim();
    auto H = pmod->GetLocalHam();
    uint nsym = pmod->GetNSym();

    if (exc_keyvec.size()!=nsym) throw std::domain_error("K has wrong number of quantum numbers");
    IKey K(pmod->GetGroupObj(),exc_keyvec);

    std::stringstream sstr;
    string savefolder;

    sstr<<fileparts(filename).name<<"_p"<<pmin<<"_"<<pmax<<"_"<<np<<"_nb"<<nev<<"_shift"<<UC_shift;
    if (!trans_op.empty()) sstr<<"_op"<<trans_op;
    sstr<<"_K";
    sstr<<K;

    if (saveE || saveX)
    {
        savefolder = GetUniquePath(datafolder + "/" + sstr.str());
        if (!mkdir(savefolder)) throw std::runtime_error("couldn't create "+savefolder);
//        saveEname = GetUniqueFileName(sstr.str()+"_dE","bin",savefolder);
    }


    /// load IMPS state
    ILamArray Lamvec;
    IDiagArray Cvec,Lvec,Rvec;
    IMPSArray ALvec,ARvec,NLvec;
    MPSBlockMatArray<IKey,Complex> Bin,Bout;

    if (!loadIMPS(Lamvec,Cvec,ALvec,ARvec,filename,pmod->GetGroupObj())) throw std::runtime_error(filename+" could not be loaded");


    /// unit cell size and PBC function to wrap around indices
    N = ALvec.size();
//    auto PBC = [N](int x) -> int {return (x + N)%N;};

    CheckOrthoLRSqrt(ALvec,ARvec,Cvec);
    for (const auto& it : Cvec)
    {
        Lvec.emplace_back(it.t()*it);
        Rvec.emplace_back(it*it.t());
    }


    /// create I2K object and check its compliance with the loaded state
    auto I2K = pmod->GetI2K(N,QN);

    if (ALvec != I2K) throw std::runtime_error("AL not compatible with I2K from model");
    if (ARvec != I2K) throw std::runtime_error("AR not compatible with I2K from model");

    if (UC_shift >= N)
    {
        UC_shift = UC_shift%N;
        cerr<<endl<<"0 < shift < N, setting to "<<UC_shift<<endl<<endl;
    }

    if (!trans_op.empty()) ApplyTransformation(ARvec,Lvec,trans_op,I2K,pmod);


    std::vector<BlockMat<IKey,Complex> > VL,VR;
    BlockMat<IKey,Scalar> LM,RM;
    CVecType OLLv,OLRv;
//    Real OLL=1,OLR=1;
    if (UC_shift > 0)
    {
        shift(ARvec,UC_shift);
        shift(Lvec,UC_shift);
    }
//        Cit += UC_shift;
    Real OLtol = std::max(tol/100,1e-15);
    OLLv = TMmixedEigs(ALvec,ARvec,VL,l,-K,nev,"LM",OLtol,IBMat(),0,true);
    OLRv = TMmixedEigs(ALvec,ARvec,VR,r,K,nev,"LM",OLtol,IBMat(),0,true);

    if (abs(imag(OLLv(0))) > 10*OLtol) cerr<<"dominant left overlap between AL and AR is complex, OL = "<<OLLv(0)<<endl;
    if (abs(imag(OLRv(0))) > 10*OLtol) cerr<<"dominant right overlap between AL and AR is complex, OR = "<<OLRv(0)<<endl;
    Real OLL = real(OLLv(0));
    Real OLR = real(OLRv(0));
    Real dOL = OLL - OLR;
    if (std::abs(dOL) > 10*OLtol) cerr<<"dominant left and right overlap between AL and AR differs by "<<dOL<<endl;
    Real OL = 0.5*(OLR + OLL);

    /// fix overlap between AL and AR to be (real and) positive
    ARvec.front() *= sign(OL);

    LM = BlockMat<IKey,Scalar>(VL.front());
    RM = BlockMat<IKey,Scalar>(VR.front());

    BlockDiagMat<IKey,Scalar> L(Lvec.back());
    BlockDiagMat<IKey,Scalar> R(Rvec.back());

    if (test)
    {
        auto obs = pmod->GetObservables(obsnames);

        cout<<"I2K:"<<endl;
        cout<<I2K<<endl;
        Lamvec.ShowDims("lam");
        if (!obs.empty())
        {
            cout<<"== AL "<<std::string(94,'=')<<endl;
            MeasureObservables(obs,ALvec,IDiagArray(),Rvec,true);
            cout<<"== AR "<<std::string(94,'=')<<endl;
            MeasureObservables(obs,ARvec,Lvec,IDiagArray(),true);
        }
        cout<<"sign(OLL)="<<sign(OLLv(0))<<", sign(OLR)="<<sign(OLLv(0))<<endl;
        OLL = std::abs(OLL);
        OLR = std::abs(OLR);
        TMmixedEigs(ALvec,ARvec,VL,l,-K,nev,"LM",OLtol,IBMat(),0,true);
        TMmixedEigs(ALvec,ARvec,VR,r,K,nev,"LM",OLtol,IBMat(),0,true);
        cout<<std::string(100,'-')<<endl;

//        teststuff(ALvec,ARvec,Lvec,Rvec,LM,RM,OLL,OLR);

        cout<<"folder: "<<savefolder<<endl;
//        cout<<"name: "<<fileparts(saveEname).name<<endl;
    }

//    teststuff(ALvec,ARvec,Lvec,Rvec,LM,RM,real(OLLv(0)),real(OLRv(0)));

    /// calculate ground state energy density to subtract from Hamiltonian
    RVecType E0vL = MeasureObservables(H,ALvec,BlockDiagMatArray<IKey,Scalar>(),Rvec,true);
    RVecType E0vR = MeasureObservables(H,ARvec,Lvec,BlockDiagMatArray<IKey,Scalar>(),true);
    double E0L = mean(E0vL);
    double E0R = mean(E0vR);
    double dE0 = E0L - E0R;
    if (std::abs(dE0) > 1e-14) throw std::domain_error("ground state energies from AL and AR differ by "+std::to_string(dE0));

    double E0 = 0.5*(E0L + E0R);
    H -= E0*SpId<Real>(d,2); /// subtract ground state energy from Hamiltonian

    /// calculate Null spaces for all AL
    for (const auto& Ait : ALvec) NLvec.emplace_back(Nullspace(Ait,l));
    /// determine block structure and possible symmetry sectors for the excitation DOF BlockMat X
    std::vector<dimkeypair_vec<IKey> > xdims;
    uint mtot = XDims(xdims,NLvec,ARvec,K);
    if (!all(xdims)) throw std::domain_error("no excitations for this quantum number");

//    cout<<"xdims:"<<endl;
//    for (uint n=0;n<N;++n) cout<<n<<":"<<endl<<xdims[n]<<endl;

    if(!test)
    {
        /// calculate B independent constants for applying effective Hamiltonian
        IDiagArray HLtot,HRtot;

        HeffConstants(HLtot,HRtot,ALvec,ARvec,L,R,H,max(tol/100,InvETol),0,verbose);

        /// actually calculate excitations
        RMatType dE(np,nev,fill::zeros);
        std::vector<CMatType> X(np);
//        std::vector<RVecType> dE(np);
//        std::function<void (Complex*,Complex*)> Hfun;

        tictoc tt,tts;
//        Real p_act;


        tt.tic();
        for (uint n=0; n<np; ++n)
        {
            CVecType dEtmp;
            Real p_act = pvec[n]*datum::pi*N;
            Complex kfac = Complex(cos(p_act),sin(p_act));

            std::function<void (Complex*,Complex*)> Hfun =
            [&xdims,mtot,kfac,&ALvec,&ARvec,&Cvec,&NLvec,&LM,&RM,&H,&HLtot,&HRtot,tol,InvETol,verbose](Complex* in, Complex* out) -> void
            {
                ApplyHeff(in,out,xdims,mtot,kfac,ALvec,ARvec,NLvec,LM,RM,H,HLtot,HRtot,std::max(tol/10,InvETol),verbose);
            };

//            CVecType tmpin(mtot,fill::randn);
//            CVecType tmpout(mtot,fill::zeros);
//
//            Hfun(tmpin.memptr(),tmpout.memptr());
//            cout<<dot(tmpin,tmpout)<<endl;
            tts.tic();
            eigs_n(Hfun,mtot,dEtmp,X[n],nev,"SR",tol);
            cout<<"p = "<<pvec[n]<<" "<<N<<" pi done ("<<tts.toc()<<" s.)"<<endl;
            if (any(imag(dEtmp)>tol)) cerr<<"IMAGINARY ENERGIES FOR p="<<pvec[n]<<": "<<imag(dEtmp)<<endl;
            dE.row(n) = real(dEtmp);

            if (saveX)
            {
                string tmpXname = sstr.str() + "_X" + std::to_string(n);
                std::vector<BlockMatArray<IKey,Complex> > Xvec;
                Xvec.reserve(nev);
                for (uint l=0;l<nev;++l) Xvec.emplace_back(BlockMatArray<IKey,Complex>(X[n].col(l),xdims));
                if (save(Xvec,Fullpath(tmpXname,"bin",savefolder))) cout<<"X"<<n<<" saved under "<<tmpXname<<endl;
                else cerr<<"failed to save X"<<n<<" under "<<tmpXname<<endl;
            }
//            dEtmp.raw_print("p = "+std::to_string(pvec[n])+" "+std::to_string(N)+" pi");
        }
        double tel=tt.toc();


//        dE = dE.t();
        for (uint n=0; n<np; ++n) dE.row(n).raw_print("p = "+std::to_string(pvec[n])+" "+std::to_string(N)+" pi");
//        for (uint n=0; n<np; ++n) cout<<"p = "<<pvec[n]<<" "<<N<<" pi: "<<dE.row(n)<<endl;

        if (saveE)
        {
//            std::string tmpname = GetUniqueFileName(saveEname,"bin",datafolder);
//        uint ct=1;
//        while (fileExist(tmpname)) tmpname = savename+"_"+std::to_string(ct++)+".bin";

            string saveEname = sstr.str()+"_dE";
            if (dE.save(Fullpath(saveEname,"bin",savefolder))) cout<<"saved dE under "<<saveEname<<endl;
            else cerr<<"failed to save dE under "<<saveEname<<endl;
        }

        if (tel>60) cout<<"elapsed time: "<<tel/double(60)<<" min."<<endl;
        else cout<<"elapsed time: "<<tel<<" s."<<endl;
    }

    return 0;
}