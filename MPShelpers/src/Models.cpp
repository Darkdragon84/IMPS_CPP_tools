#include "../include/Models.h"

/**  PURE VIRTUAL METHODS THAT NEED TO IMPLEMENTED FOR EVERY MODEL:
 ** Init()
 ** GetObservables()
 ** ShowParams()
 ** MakeOps()
 *************************************************************************/



#ifdef _USE_SYMMETRIES_
std::vector<dim_map<IKey> >
ModelBase::MakeDims(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const
{
    auto I2K = this->GetI2K(N,QN);
    auto path = validpath(N,I2K);
    dim_map<IKey> dmap0 = this->GetDimMap(dims,N,QN);

    dim_map<IKey> dmapl(dmap0),dmapr;

    std::vector<dim_map<IKey> > dimvec;

    for (uint k=0; k<N; ++k)
    {
        dimvec.emplace_back(dmapl);
        dmapr = dmapl + I2K[path[N-k-1]];
        dmapl = dmapr;
    }

    return dimvec;
}
#endif // _USE_SYMMETRIES_

/*************************************************************************************************************************************/
/**< XYZ ABSTRACT MODEL TYPE *********************************************************************************************************/
/*************************************************************************************************************************************/
XYZAbstractModel::XYZAbstractModel(emod type, double Jx, double Jy, double Jz, double hz, uint dim):
    ModelBase(type,dim),Jx_(Jx),Jy_(Jy),Jz_(Jz),hz_(hz),sx_(dim,1,"sx"),syi_(dim,1,"syi"),sz_(dim,1,"sz")
{
    if(dim!=2 && dim!=3) throw std::range_error("local dim "+std::to_string(dim)+" not implemented");

    this->spin_[2]="spin-1/2 ";
    this->spin_[3]="spin-1 ";
}


void XYZAbstractModel::ShowParams () const
{
    cout<<"localdim: "<<this->localdim_<<endl;
    cout<<"model parameters:"<<endl;
    cout<<"Ising interaction Jx="<<Jx_<<", Jy="<<Jy_<<", Jz="<<Jz_<<endl;
    cout<<"Zeeman magnetic field hz="<<hz_<<endl;
}

/// TODO (valentin#1#2016-12-20): switch to ladder operators again, for proper U(1) representations
void XYZAbstractModel::MakeOps()
{
    double fac=1;

    switch (this->localdim_)
    {
    case 2:
        sx_(0,1)=0.5;
        sx_(1,0)=0.5;

        syi_(0,1)=-0.5;
        syi_(1,0)=0.5;

        sz_(0,0)=0.5;
        sz_(1,1)=-0.5;
        break;
    case 3:
        fac = 1./sqrt(2.);
        sx_(0,1)=fac;
        sx_(1,0)=fac;
        sx_(1,2)=fac;
        sx_(2,1)=fac;

        syi_(0,1)=-fac;
        syi_(1,0)=fac;
        syi_(1,2)=-fac;
        syi_(2,1)=fac;

        sz_(0,0)=1.;
        sz_(2,2)=-1.;
        break;
    default:
        cerr<<"MakeOps(): dimension not implemented"<<endl;
    }
    this->ham_ = -Jx_*kron(sx_,sx_) + Jy_*kron(syi_,syi_) - Jz_*kron(sz_,sz_) - 0.5*hz_*(kron(sz_,this->id_) + kron(this->id_,sz_));
    this->ham_.SetName("H");
}

std::vector<SparseOperator<double> > XYZAbstractModel::GetObservables(const std::vector<std::string>& opstring) const
{
    enum op_enum {SX,SYI,SZ,SP,SM,RX,RZ};
    std::map<std::string,op_enum> opmap;
    opmap["sx"] = SX;
    opmap["syi"] = SYI;
    opmap["sz"] = SZ;
    opmap["sp"] = SP;
    opmap["sm"] = SM;
    opmap["rx"] = RX;
//    opmap["roty"] = ROTY;
    opmap["rz"] = RZ;

    std::vector<SparseOperator<double> > opvec;
    opvec.reserve(opstring.size());
    for (const auto& strit : opstring)
    {
        auto it = opmap.find(strit);
        if (it == opmap.end())
        {
            cerr<<"operator "<<strit<<" not defined"<<endl;
            continue;
        }

        switch (it->second)
        {
        case SX:
            opvec.emplace_back(sx_);
            break;
        case SYI:
            opvec.emplace_back(syi_);
            break;
        case SZ:
            opvec.emplace_back(sz_);
            break;
        case RX:
            opvec.emplace_back((2*sx_).SetName("rx"));
            break;
        case RZ:
            opvec.emplace_back((2*sz_).SetName("rz"));
            break;
        case SP:
            opvec.emplace_back((sx_-syi_).SetName("sp"));
            break;
        case SM:
            opvec.emplace_back((sx_+syi_).SetName("sm"));
            break;
        default:
            cerr<<"operator "<<strit<<" not defined"<<endl;
        }
    }
    return opvec;

}

/*************************************************************************************************************************************/
/**< XYZ MODEL ***********************************************************************************************************************/
/*************************************************************************************************************************************/
//XYZModel::XYZModel(double Jx, double Jy, double Jz, double hz, uint dim):
//    XYZAbstractModel(XYZ,Jx,Jy,Jz,hz,dim)
//{
//    Init();
//}
//
//void
//XYZModel::Init()
//{
//    #ifdef _USE_SYMMETRIES_
//    if (this->localdim_ != 2) throw std::domain_error("XYZModel: only localdim=2 implemented with symmetries!");
//    this->GroupObj_.init({2}); /// one Z(2) symmetry
//    #endif // _USE_SYMMETRIES_
//    this->MakeOps();
//    std::stringstream modstream;
//    modstream<<"XYZ_d"<<localdim_<<"_J"<<this->Jx_<<"_"<<this->Jy_<<"_"<<this->Jz_<<"_h0_"<<this->hz_;
//    this->modstring_ = modstream.str();
//}
//
//void
//XYZModel::ShowParams () const
//{
//    cout<<this->spin_.find(this->localdim_)->second<<"XYZ type model:"<<endl;
//    this->XYZAbstractModel::ShowParams();
//}
//
//#ifdef _USE_SYMMETRIES_
//ItoKey<1,IKey>
//XYZModel::GetI2K(uint N, const std::vector<int>& QN) const
//{
//
//}
//#endif // _USE_SYMMETRIES_

/*************************************************************************************************************************************/
/**< XXZ MODEL ***********************************************************************************************************************/
/*************************************************************************************************************************************/
XXZModel::XXZModel(double J, double Delta, double hz, uint dim):
    XYZAbstractModel(XXZ,J,J,Delta,hz,dim),J_(J),Delta_(Delta)
{
    Init();
}

void
XXZModel::Init()
{
#ifdef _USE_SYMMETRIES_
    this->GroupObj_.init({0}); /// one U(1) symmetry
#endif // _USE_SYMMETRIES_
    this->MakeOps();
    std::stringstream modstream;
    modstream<<"XXZ_d"<<localdim_<<"_J"<<this->J_<<"_"<<this->J_<<"_"<<this->Delta_<<"_h0_"<<this->hz_;
    this->modstring_ = modstream.str();
}

void
XXZModel::ShowParams () const
{
    cout<<this->spin_.find(this->localdim_)->second<<"XXZ type model:"<<endl;
    this->XYZAbstractModel::ShowParams();
}

#ifdef _USE_SYMMETRIES_
std::vector<std::pair<int,int> >
XXZModel::GetdQ(uint N, const std::vector<int>& QN) const
{
    std::vector<std::pair<int,int> > dQ;
    if (QN.size() != this->GetNSym()) throw std::domain_error("QN must have "+std::to_string(this->GetNSym())+" entries");
    int m = QN[0];
    if (uint(abs(m)) >= N) throw std::domain_error("|m| < N!");

    uint div;

    switch (this->localdim_)
    {
        case 2: /// S=1/2
            div = gcd(N-m,N+m);
            dQ.emplace_back(std::make_pair(int(2*N/div),-int((N+m)/div)));
            break;

//        case 3: /// S=1
//            div = gcd(N-m,gcd(m,N+m));
//            break;
        default:
            throw std::domain_error("d="+std::to_string(this->localdim_)+" not implemented");
    }

    return dQ;
}

dim_map<IKey>
XXZModel::GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const
{
//    if (QN.size() != this->GetNSym()) throw std::domain_error("QN must have "+std::to_string(this->GetNSym())+" entries");
//    int m = QN[0];
//    if (uint(abs(m)) >= N) throw std::domain_error("|m| < N!");
//
//    uint period = 2*N/gcd(N-m,N+m);
    auto dQ = this->GetdQ(N,QN);

    uint Nsec = dims.size();
    dim_map<IKey> dmap0;

//    for (uint i=0; i<Nsec; ++i) dmap0[IKey(this->GroupObj_, {int(period*(i - Nsec/2))})]=dims[i];
    for (uint i=0; i<Nsec; ++i) dmap0[IKey(this->GroupObj_, {int(dQ[0].first*(i - Nsec/2))})]=dims[i];

    return dmap0;
}

ItoKey<1,IKey>
XXZModel::GetI2K(uint N, const std::vector<int>& QN) const
{
//    if (QN.size() != this->GetNSym()) throw std::domain_error("QN must have "+std::to_string(this->GetNSym())+" entries");
//
//    int m = QN[0];
//    if (uint(abs(m)) >= N) throw std::domain_error("|m| < N!");
//
//    uint div;
    ItoKey<1,IKey> I2K(this->localdim_);
    auto dQ = this->GetdQ(N,QN);

    for (int s=int(this->localdim_)-1; s>=0; --s)
    {
        I2K.emplace_back(IKey(this->GetGroupObj(),{dQ[0].second + s*dQ[0].first}));
    }

//    switch (this->localdim_)
//    {
//    case 2:
//        div = gcd(N-m,N+m);
//
////        this->periods_ = {2*N/div};
//        I2K = ItoKey<1,IKey> (2, {IKey(this->GroupObj_,{int((N-m)/div)}),IKey(this->GroupObj_,{-int((N+m)/div)})});
//        break;
////    case 3:
//////        div = gcd(this->N_-m_,m_) gcd(this->N_-m_,this->N_+m_),gcd(this->N_+m_,m_)
////
////        break;
//    default:
//        throw std::domain_error("d="+std::to_string(this->localdim_)+" not implemented");
//    }
    return I2K;
}
#endif // _USE_SYMMETRIES_
/*************************************************************************************************************************************/
/**< XXZ ANTIFERROMAGNETIC MODEL *****************************************************************************************************/
/*************************************************************************************************************************************/
//XXZAFType::XXZAFType(double J, double Delta, double hz, uint dim):
//    XYZAbstractModel(XXZAF,-J,J,-Delta,hz,dim),J_(J),Delta_(Delta)
//{
//    Init();
//}
//
//void
//XXZAFType::Init()
//{
//    this->MakeOps();
//    std::stringstream modstream;
//    modstream<<"XXZ_d"<<localdim_<<"_J"<<this->J_<<"_"<<this->J_<<"_"<<this->Delta_<<"_h0_"<<this->hz_;
//    this->modstring_ = modstream.str();
//}
//
//void
//XXZAFType::ShowParams () const
//{
//    cout<<this->spin_.find(this->localdim_)->second<<"XXZ type model:"<<endl;
//    this->XYZAbstractModel::ShowParams();
//}

/*************************************************************************************************************************************/
/**< QUANTUM ISING MODEL *************************************************************************************************************/
/*************************************************************************************************************************************/
QuantumIsingModel::QuantumIsingModel(double J, double h):
    XYZAbstractModel(QIM,J,0,0,h,2),J_(J),h_(h)
{
    Init();
}

void
QuantumIsingModel::Init()
{
#ifdef _USE_SYMMETRIES_
    this->GroupObj_.init({2}); /// one Z(2) symmetry
#endif // _USE_SYMMETRIES_
    this->MakeOps();
    std::stringstream modstream;
    modstream<<"QIM_d"<<localdim_<<"_J"<<this->J_<<"_0_0_h0_"<<this->h_;
    this->modstring_ = modstream.str();
}

void
QuantumIsingModel::ShowParams () const
{
    cout<<this->spin_.find(this->localdim_)->second<<"Quantum Ising type model:"<<endl;
    this->XYZAbstractModel::ShowParams();
}

#ifdef _USE_SYMMETRIES_
std::vector<std::pair<int,int> >
QuantumIsingModel::GetdQ(uint N, const std::vector<int>& QN) const
{
    return std::vector<std::pair<int,int> >{std::make_pair(1,0)};
}

dim_map<IKey>
QuantumIsingModel::GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const
{
    uint Nsec = dims.size();
    dim_map<IKey> dmap0;

    if (Nsec>2) cerr<<"there should be exactly 2 symmetry sectors, reducing to Nsec = 2"<<endl;
    else if (Nsec<2) throw std::domain_error("there should be exactly 2 symmetry sectors");

    dmap0[IKey(this->GroupObj_, {0})] = dims[0];
    dmap0[IKey(this->GroupObj_, {1})] = dims[1];

    return dmap0;
}

ItoKey<1,IKey>
QuantumIsingModel::GetI2K(uint N, const std::vector<int>& QN) const
{
    return ItoKey<1,IKey>(this->localdim_, {IKey(this->GroupObj_,{0}),IKey(this->GroupObj_,{1})});
}
#endif // _USE_SYMMETRIES_

/*************************************************************************************************************************************/
/**< QUANTUM POTTS MODEL *************************************************************************************************************/
/*************************************************************************************************************************************/
QuantumPottsModel::QuantumPottsModel(uint q, double h):
    ModelBase(QPOTTS,q),h_(h),xop_(q,1,"X"),zop_(q,1,"Z")
{
    Init();
}


void
QuantumPottsModel::Init()
{
#ifdef _USE_SYMMETRIES_
    this->GroupObj_.init({this->localdim_}); /// one U(1) symmetry
#endif // _USE_SYMMETRIES_
    this->MakeOps();

    std::stringstream modstream;
    modstream<<"QPOTTS_q"<<this->localdim_<<"_h"<<h_;
    this->modstring_ = modstream.str();
}

void
QuantumPottsModel::ShowParams() const
{
    cout<<"Quantum Potts type model:"<<endl;
    cout<<"localdim: "<<this->localdim_<<endl;
    cout<<"model parameters:"<<endl;
    cout<<"Zeeman magnetic field hz="<<h_<<endl;
}

void
QuantumPottsModel::MakeOps()
{
    uint q = this->localdim_;
    xop_(q-1,0) = 1.;
    zop_(0,0) = q - 1;

    for (uint i=1; i<q; ++i)
    {
        xop_(i-1,i) = 1.;
        zop_(i,i) = -1.;
    }

    std::vector<SparseOperator<double> > MXq;
    MXq.reserve(q-1);
    MXq.emplace_back(xop_);
    for (uint i = 1; i < q-1; ++i) MXq.emplace_back(MXq.back()*xop_);

    this->ham_ = -0.5*h_*(kron(zop_,this->id_) + kron(this->id_,zop_));

    for (uint i = 0; i < q-1; ++i) this->ham_ -= kron(MXq[i],MXq[q-2-i]);
    this->ham_.SetName("H");
}

std::vector<SparseOperator<double> >
QuantumPottsModel::GetObservables(const std::vector<std::string>& opstring) const
{
    enum op_enum {X,Z};
    std::map<std::string,op_enum> opmap;
    opmap["X"] = X;
    opmap["Z"] = Z;

    std::vector<SparseOperator<double> > opvec;
    opvec.reserve(opstring.size());

    for (const auto& strit : opstring)
    {
        auto it = opmap.find(strit);
        if (it == opmap.end())
        {
            cerr<<"operator "<<strit<<" not defined"<<endl;
            continue;
        }

        switch (it->second)
        {
        case X:
            opvec.emplace_back(xop_);
            break;
        case Z:
            opvec.emplace_back(zop_);
            break;
        default:
            cerr<<"operator "<<strit<<" not defined"<<endl;
        }
    }
    return opvec;

}

#ifdef _USE_SYMMETRIES_
std::vector<std::pair<int,int> >
QuantumPottsModel::GetdQ(uint N, const std::vector<int>& QN) const
{
    return std::vector<std::pair<int,int> >{std::make_pair(1,0)};
}

dim_map<IKey>
QuantumPottsModel::GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const
{
    uint Nsec = dims.size();
    dim_map<IKey> dmap0;
    uint q = this->localdim_;

    if (Nsec < q) throw std::domain_error("not enough symmetry sectors, there should be at least "+std::to_string(q));
    else if (Nsec > q)
    {
        cerr<<"there should be exactly "<<q<<" symmetry sectors, reducing to Nsec="<<q<<endl;
        Nsec = q;
    }

    for (int i = 0; i < int(q); ++i)
    {
        dmap0[IKey(this->GroupObj_,{i})] = dims[i];
    }

    return dmap0;
}

ItoKey<1,IKey>
QuantumPottsModel::GetI2K(uint N, const std::vector<int>& QN) const
{
    ItoKey<1,IKey> I2K(this->localdim_);
    for (int i=0; i < int(this->localdim_); ++i) I2K.emplace_back(IKey(this->GroupObj_, {i}));
    return I2K;
}
#endif // _USE_SYMMETRIES_
/*************************************************************************************************************************************/
/**< FERMI HUBBARD MODEL *************************************************************************************************************/
/*************************************************************************************************************************************/
FermiHubbardModel::FermiHubbardModel(double t, double U, double V, double mu, double hz):
    ModelBase(FHUB,4),t_(t),U_(U),V_(V),mu_(mu),hz_(hz),cup_(4,1,"cup"),cdo_(4,1,"cdo"),nupop_(4,1,"nup"),ndoop_(4,1,"ndo")
{
    Init();
};

void
FermiHubbardModel::Init()
{
#ifdef _USE_SYMMETRIES_
    this->GroupObj_.init({0,0}); /// two U(1) symmetries
#endif // _USE_SYMMETRIES_
    this->MakeOps();

    std::stringstream modstream;
    modstream<<"FHUB_t"<<t_<<"_U"<<U_<<"_V"<<V_<<"_mu"<<mu_<<"_h"<<hz_;
    this->modstring_ = modstream.str();
}

void
FermiHubbardModel::ShowParams() const
{
    cout<<"Fermi Hubbard type model:"<<endl;
    cout<<"localdim: "<<this->localdim_<<endl;
    cout<<"model parameters:"<<endl;
    cout<<"hopping t="<<t_<<endl;
    cout<<"onsite interaction U="<<U_<<endl;
    cout<<"nearest-neighbor interaction V="<<V_<<endl;
    cout<<"chem. potential mu="<<mu_<<endl;
    cout<<"Zeeman magnetic field hz="<<hz_<<endl;
}

void
FermiHubbardModel::MakeOps()
{
    cup_(0,2)=1.;
    cup_(1,3)=1.;

    cdo_(0,1)=1.;
    cdo_(2,3)=-1.;

    nupop_(2,2)=1.;
    nupop_(3,3)=1.;

    ndoop_(1,1)=1.;
    ndoop_(3,3)=1.;

    SparseOperator<double> nup2(nupop_-0.5*this->id_), ndo2(ndoop_-0.5*this->id_); /// with this reparametrization, the ground state for mu=0 is always half filled
    SparseOperator<double> nund(nup2*ndo2),nupnd(nupop_+ndoop_),nupnd2(nup2+ndo2),sz(0.5*(nupop_-ndoop_));
    SparseOperator<double> F(4,1);
    F(0,0)=1.;
    F(1,1)=-1.;
    F(2,2)=-1.;
    F(3,3)=1.;

    this->ham_ = -t_*(kron(cup_.t()*F,cup_) - kron(cup_*F,cup_.t()) + kron(cdo_.t()*F,cdo_) - kron(cdo_*F,cdo_.t()))
                 + 0.5*U_*(kron(nund,this->id_) + kron(this->id_,nund))
                 + V_*kron(nupnd2,nupnd2)
                 - 0.5*mu_*(kron(nupnd,this->id_) + kron(this->id_,nupnd))
                 - 0.5*hz_*(kron(sz,this->id_) + kron(this->id_,sz));
    this->ham_.SetName("H");
}

std::vector<SparseOperator<double> >
FermiHubbardModel::GetObservables(const std::vector<std::string>& opstring) const
{
    enum op_enum {CUP,CDO,CUPD,CDOD,NUP,NDO,SZ,N,RX,RZ};
    std::map<std::string,op_enum> opmap;
    opmap["cup"] = CUP;
    opmap["cdo"] = CDO;
    opmap["cupd"] = CUPD;
    opmap["cdod"] = CDOD;
    opmap["nup"] = NUP;
    opmap["ndo"] = NDO;
    opmap["sz"] = SZ;
    opmap["n"] = N;
    opmap["rx"] = RX;
    opmap["rz"] = RZ;

    std::vector<SparseOperator<double> > opvec;
    opvec.reserve(opstring.size());
    for (const auto& strit : opstring)
    {
        auto it = opmap.find(strit);
        if (it == opmap.end())
        {
            cerr<<"operator "<<strit<<" not defined"<<endl;
            continue;
        }

        switch (it->second)
        {
        case CUP:
            opvec.emplace_back(cup_);
            break;
        case CDO:
            opvec.emplace_back(cdo_);
            break;
        case CUPD:
            opvec.emplace_back(cup_.t().SetName("cupd"));
            break;
        case CDOD:
            opvec.emplace_back(cdo_.t().SetName("cdod"));
            break;
        case NUP:
            opvec.emplace_back(nupop_);
            break;
        case NDO:
            opvec.emplace_back(ndoop_);
            break;
        case SZ:
            opvec.emplace_back(0.5*(nupop_-ndoop_).SetName("sz"));
            break;
        case N:
            opvec.emplace_back((nupop_+ndoop_).SetName("n"));
            break;
        case RX:
            opvec.emplace_back(SparseOperator<double>(4,1,{{0,1,2,3},{0,2,1,3}},{1,1,1,-1},"rx"));
            break;
        case RZ:
            opvec.emplace_back(SparseOperator<double>(4,1,{{0,1,2,3},{0,1,2,3}},{1,-1,1,-1},"rz"));
            break;
        default:
            cerr<<"operator "<<strit<<" not defined"<<endl;
        }
    }
    return opvec;
}

#ifdef _USE_SYMMETRIES_

std::vector<std::pair<int,int> >
FermiHubbardModel::GetdQ(uint N, const std::vector<int>& QN) const
{
    std::vector<std::pair<int,int> > dQ;

    #ifdef FHUB_NM_REP
///** (N,M) REPRESENTATION *************************************************************************************************/
    if (QN.size() != this->GetNSym()) throw std::domain_error("QN must have "+std::to_string(this->GetNSym())+" entries");
    int n = QN[0];
    int m = QN[1];
    if (n+m <= 0 || n+m >= int(2*N)) throw std::domain_error(std::to_string(-m)+" < n < "+std::to_string(2*N-m)+"!");
    if (n-m <= 0 || n-m >= int(2*N)) throw std::domain_error(std::to_string(m)+" < n < "+std::to_string(2*N+m)+"!");

    uint divn = gcd(gcd(N-n,2*N-n),n);
    uint divm = gcd(gcd(N-m,N+m),m);

    dQ.emplace_back(std::make_pair(int(N/divn),-int(n/divn)));
    dQ.emplace_back(std::make_pair(int(N/divm),-int((N+m)/divm)));
    #else
///** (NUP,NDOWN) REPRESENTATION *************************************************************************************************/
    if (QN.size() != this->GetNSym()) throw std::domain_error("QN must have "+std::to_string(this->GetNSym())+" entries");
    int nup = QN[0];
    int ndo = QN[1];
    if (nup <= 0 || nup >= int(N)) throw std::domain_error("0 < nup < N!");
    if (ndo <= 0 || ndo >= int(N)) throw std::domain_error("0 < ndo < N!");

    uint divup = gcd(N-nup,nup);
    uint divdo = gcd(N-ndo,ndo);

    dQ.emplace_back(std::make_pair(int(N/divup),-int(nup/divup)));
    dQ.emplace_back(std::make_pair(int(N/divdo),-int(ndo/divdo)));
    #endif // NM_REP_

    return dQ;
}

dim_map<IKey>
FermiHubbardModel::GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const
{
    auto dQ = this->GetdQ(N,QN);

    uint Nsec = dims.size();
    dim_map<IKey> dmap0;

    for (uint i=0; i<Nsec; ++i)
    {
        for (uint j=0; j<Nsec; ++j)
        {
            #ifdef FHUB_NM_REP
            dmap0[IKey(this->GroupObj_,{int(dQ[0].first*(i+j-Nsec+1)),int(dQ[1].first*(i-j))})] = dims[i]+dims[j]; /// (N,M) - representation
            #else
            dmap0[IKey(this->GroupObj_,{int(dQ[0].first*(i - Nsec/2)),int(dQ[1].first*(j - Nsec/2))})] = dims[i]+dims[j]; /// (NUP,NDOWN) - representation
            #endif // FHUB_NM_REP
        }
    }

    return dmap0;
}

ItoKey<1,IKey>
FermiHubbardModel::GetI2K(uint N, const std::vector<int>& QN) const
{
    auto dQ = this->GetdQ(N,QN);

    #ifdef FHUB_NM_REP
    /// original (N,M) - representation:
    /// |0> = N(0, 0) - (n-m) = (  -n,  -m)
    /// |1> = N(1,-1) - (n-m) = ( N-n,-N-m)
    /// |2> = N(1, 1) - (n-m) = ( N-n, N-m)
    /// |3> = N(2, 0) - (n-m) = (2N-n,  -m)
    return ItoKey<1,IKey>(this->localdim_,
    {
        IKey(this->GroupObj_,{dQ[0].second                ,dQ[1].second +   dQ[1].first}),
        IKey(this->GroupObj_,{dQ[0].second +   dQ[0].first,dQ[1].second                }),
        IKey(this->GroupObj_,{dQ[0].second +   dQ[0].first,dQ[1].second + 2*dQ[1].first}),
        IKey(this->GroupObj_,{dQ[0].second + 2*dQ[0].first,dQ[1].second +   dQ[1].first})
    });
    #else
    /// original (Nup,Ndown) - representation:
    /// |0> = N(0,0) - (nup-ndo) = ( -nup, -ndo)
    /// |1> = N(0,1) - (nup-ndo) = ( -nup,N-ndo)
    /// |2> = N(1,0) - (nup-ndo) = (N-nup, -ndo)
    /// |3> = N(1,1) - (nup-ndo) = (N-nup,N-ndo)
    return ItoKey<1,IKey>(this->localdim_,
    {
        IKey(this->GroupObj_,{dQ[0].second              ,dQ[1].second              }),
        IKey(this->GroupObj_,{dQ[0].second              ,dQ[1].second + dQ[1].first}),
        IKey(this->GroupObj_,{dQ[0].second + dQ[0].first,dQ[1].second              }),
        IKey(this->GroupObj_,{dQ[0].second + dQ[0].first,dQ[1].second + dQ[1].first})
    });
    #endif // FHUB_NM_REP
}

#endif // _USE_SYMMETRIES_
/*************************************************************************************************************************************/
/**< BOSE HUBBARD MODEL **************************************************************************************************************/
/*************************************************************************************************************************************/
BoseHubbardModel::BoseHubbardModel(uint d, double t, double U, double mu):
    ModelBase(BHUB,d), t_(t),U_(U),mu_(mu),bop_(d,1,"b"),nop_(d,1,"n")
{
    Init();
}

void
BoseHubbardModel::Init()
{
#ifdef _USE_SYMMETRIES_
    this->GroupObj_.init({0}); /// one U(1) symmetry
#endif // _USE_SYMMETRIES_
    this->MakeOps();
    std::stringstream modstream;
    modstream<<"BHUB_t"<<t_<<"_U"<<U_<<"_mu"<<mu_;
    this->modstring_ = modstream.str();
}

void
BoseHubbardModel::ShowParams() const
{
    cout<<"Bose Hubbard type model:"<<endl;
    cout<<"localdim: "<<this->localdim_<<endl;
    cout<<"model parameters:"<<endl;
    cout<<"hopping t="<<t_<<endl;
    cout<<"onsite interaction U="<<U_<<endl;
    cout<<"chem. potential mu="<<mu_<<endl;
}

void
BoseHubbardModel::MakeOps()
{
    for (uint i=1; i<this->localdim_; ++i)
    {
        bop_(i-1,i) = sqrt(double(i));
        nop_(i,i) = i;
    }

    auto onsite = 0.5*U_*nop_*(nop_ - this->id_) - mu_*nop_;
    this->ham_ = -t_*(kron(bop_,bop_.t()) + kron(bop_.t(),bop_)) + 0.5*(kron(onsite,this->id_) + kron(this->id_,onsite));
    this->ham_.SetName("H");
}

std::vector<SparseOperator<double> >
BoseHubbardModel::GetObservables(const std::vector<std::string>& opstring) const
{
    enum op_enum {B,B2,N,NMAX};
    std::map<std::string,op_enum> opmap;
    opmap["b"] = B;
    opmap["b2"] = B2;
    opmap["n"] = N;
    opmap["nmax"] = NMAX;

    std::vector<SparseOperator<double> > opvec;
    opvec.reserve(opstring.size());
    for (const auto& strit : opstring)
    {
        auto it = opmap.find(strit);
        if (it == opmap.end())
        {
            cerr<<"operator "<<strit<<" not defined"<<endl;
            continue;
        }

        switch (it->second)
        {
        case B:
            opvec.emplace_back(bop_);
            break;
        case B2:
            opvec.emplace_back((bop_*bop_).SetName("b2"));
            break;
        case N:
            opvec.emplace_back(nop_);
            break;
        case NMAX:
        {
            SparseOperator<double> nmax(this->localdim_,1,"nmax");
            nmax(this->localdim_-1,this->localdim_-1) = 1;
            opvec.emplace_back(nmax);
            break;
        }
        default:
            cerr<<"operator "<<strit<<" not defined"<<endl;
        }
    }
    return opvec;
}

#ifdef _USE_SYMMETRIES_

std::vector<std::pair<int,int> >
BoseHubbardModel::GetdQ(uint N, const std::vector<int>& QN) const
{
    std::vector<std::pair<int,int> > dQ;

    if (QN.size() != this->GetNSym()) throw std::domain_error("QN must have "+std::to_string(this->GetNSym())+" entries");
    int n = QN[0];
    if (n<1 || n>=int(N*this->localdim_)) throw std::domain_error("0 < n < "+std::to_string(N*this->localdim_)+"!");

    uint div = gcd(N,n);

    dQ.emplace_back(int(N/div),-int(n/div));
    return dQ;
}

dim_map<IKey>
BoseHubbardModel::GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const
{
//    if (QN.size() != this->GetNSym()) throw std::domain_error("QN must have "+std::to_string(this->GetNSym())+" entries");
//    int n = QN[0];
//    if (n<1 || n>=int(this->localdim_)) throw std::domain_error("0 < n < "+std::to_string(this->localdim_)+"!");
//
//    uint period = N/gcd(N,n);
    auto dQ = this->GetdQ(N,QN);
    uint Nsec = dims.size();
    dim_map<IKey> dmap0;

    for (int i=0; i<int(Nsec); ++i) dmap0[IKey(this->GroupObj_,{int(i*dQ[0].first)})] = dims[i];

    return dmap0;
}

ItoKey<1,IKey>
BoseHubbardModel::GetI2K(uint N, const std::vector<int>& QN) const
{
//    if (QN.size() != this->GetNSym()) throw std::domain_error("QN must have "+std::to_string(this->GetNSym())+" entries");
//    int n = QN[0];
//    if (n<1 || n>=int(this->localdim_)) throw std::domain_error("0 < n < "+std::to_string(this->localdim_)+"!");
//
//    uint div = gcd(N,n);
//    this->periods_ = {N/div};
    auto dQ = this->GetdQ(N,QN);

    std::vector<IKey> keys;
//    for (uint i=0; i<this->localdim_; ++i) keys.emplace_back(IKey(this->GroupObj_, {int(i*N - n)/int(div)}));
    for (int i=0; i<int(this->localdim_); ++i) keys.emplace_back(IKey(this->GroupObj_, {dQ[0].second + i*dQ[0].first}));
    return ItoKey<1,IKey>(this->localdim_,keys);
}
#endif // _USE_SYMMETRIES_


const modptr CreateModel(const parser& P, bool verbose)
{

    std::string modstr;

    Col<uint> uvals;
    RVecType rvals;

    P.GetValue(modstr,"model",true);
//    try {P.GetValue(modstr,"model",true);}
//    catch (std::exception& e){std::terminate();}

    std::map<std::string,ModelBase::emod> modmap;
//    modmap["XYZ"]=ModelBase::XYZ;
    modmap["XXZ"]=ModelBase::XXZ;
    modmap["QIM"]=ModelBase::QIM;
    modmap["QPOTTS"]=ModelBase::QPOTTS;
    modmap["BHUB"]=ModelBase::BHUB;
    modmap["FHUB"]=ModelBase::FHUB;

    modptr pmod;

    auto it = modmap.find(modstr);
    if (it==modmap.end()) throw std::domain_error(modstr+" not found");

    ModelBase::emod modtype = it->second;

    switch (modtype)
    {
//    case ModelBase::XYZ:
//        uvals.resize(1);
//        rvals.resize(5);
//        /// default values
//        uvals(0)=2;
//        rvals(4)=0;
//        try{
//            P.GetValue(uvals(0),"d");
//            P.GetValue(rvals(0),"Jx",true);
//            P.GetValue(rvals(1),"Jy",true);
//            P.GetValue(rvals(2),"Jz",true);
//            rvals(3)=0;//P.GetValue(rvals(3),"hx");
//            P.GetValue(rvals(4),"hz");
//        }
//        catch (std::exception& e) {throw std::logic_error("parameters missing for XYZ model");}
//
//        if (verbose)
//        {
//            cout<<"XYZ model (d="<<uvals(0)<<")"<<endl;
//            cout<<"J=("<<rvals(0)<<","<<rvals(1)<<","<<rvals(2)<<"), h=("<<rvals(3)<<","<<rvals(4)<<")"<<endl;
//        }
//        #ifdef _USE_SYMMETRIES_
//        P.GetValue(N,"N",true);
//        if (verbose) cout<<"N="<<N<<endl;
//        pmod = make_shared<XYZModel>(rvals(0),rvals(1),rvals(2),rvals(4),uvals(0),N);
//        #else
//        pmod = make_shared<XYZModel>(rvals(0),rvals(1),rvals(2),rvals(4),uvals(0));
//        #endif // _USE_SYMMETRIES_
//        break;
    case ModelBase::XXZ:
    {
        uvals.resize(1);
        rvals.resize(3);
        /// default values
        uvals(0)=2;
        rvals(0)=1;
        rvals(2)=0;
        try
        {
            P.GetValue(uvals(0),"d");
            P.GetValue(rvals(0),"J");
            P.GetValue(rvals(1),"Delta",true);
            P.GetValue(rvals(2),"hz");
        }
        catch (std::exception& e)
        {
            throw std::logic_error("parameters missing for XXZ model");
        }

        if (verbose)
        {
            cout<<"XXZ model (d="<<uvals(0)<<")"<<endl;
            cout<<"J="<<rvals(0)<<", Delta="<<rvals(1)<<", hz="<<rvals(2)<<endl;
        }
        pmod = make_shared<XXZModel>(rvals(0),rvals(1),rvals(2),uvals(0));
//        #ifdef _USE_SYMMETRIES_
//        int m=0;
//        P.GetValue(N,"N",true);
//        P.GetValue(m,"m",true);
//        if (verbose) cout<<"N="<<N<<", m="<<m<<endl;
//        pmod = make_shared<XXZModel>(rvals(0),rvals(1),rvals(2),uvals(0),N,m);
//        #else
//        pmod = make_shared<XXZModel>(rvals(0),rvals(1),rvals(2),uvals(0));
//        #endif // _USE_SYMMETRIES_
        break;
    }
    case ModelBase::QIM:
        rvals.resize(2);
        /// default values
        rvals(0)=1;
        try
        {
            P.GetValue(rvals(0),"J");
            P.GetValue(rvals(1),"hz",true);
        }
        catch (std::exception& e)
        {
            throw std::logic_error("parameters missing for Quantum Ising model");
        }

        if (verbose)
        {
            cout<<"Transverse Ising model"<<endl;
            cout<<"J="<<rvals(0)<<", h="<<rvals(1)<<endl;
        }
        pmod = make_shared<QuantumIsingModel>(rvals(0),rvals(1));

//        #ifdef _USE_SYMMETRIES_
//        P.GetValue(N,"N",true);
//        if (verbose) cout<<"N="<<N<<endl;
//        pmod = make_shared<QuantumIsingModel>(rvals(0),rvals(1),N);
//        #else
//        pmod = make_shared<QuantumIsingModel>(rvals(0),rvals(1));
//        #endif // _USE_SYMMETRIES_
        break;
    case ModelBase::FHUB:
    {
        rvals.resize(5);
        /// default values
        rvals(0)=1;
        rvals(2)=0;
        rvals(3)=0;
        rvals(3)=0;
        try
        {
            P.GetValue(rvals(0),"t");
            P.GetValue(rvals(1),"U",true);
            P.GetValue(rvals(2),"V");
            P.GetValue(rvals(3),"mu");
            P.GetValue(rvals(4),"hz");
        }
        catch (std::exception& e)
        {
            throw std::logic_error("parameters missing for Fermi Hubbard model");
        }

        if (verbose)
        {
            cout<<"Fermi Hubbard model"<<endl;
            cout<<"t="<<rvals(0)<<", U="<<rvals(1)<<", V="<<rvals(2)<<", mu="<<rvals(3)<<", h="<<rvals(4)<<endl;
        }
        pmod = make_shared<FermiHubbardModel>(rvals(0),rvals(1),rvals(2),rvals(3),rvals(4));
//        #ifdef _USE_SYMMETRIES_
//        uint nup = 1, ndo = 1;
//        P.GetValue(N,"N",true);
//        P.GetValue(nup,"nup",true);
//        P.GetValue(ndo,"ndo",true);
//        if (verbose) cout<<"N="<<N<<", nup="<<nup<<", ndo="<<ndo<<endl;
//        pmod = make_shared<FermiHubbardModel>(rvals(0),rvals(1),rvals(2),rvals(3),rvals(4),N,nup,ndo);
//        #else
//        pmod = make_shared<FermiHubbardModel>(rvals(0),rvals(1),rvals(2),rvals(3),rvals(4));
//        #endif // _USE_SYMMETRIES_
        break;
    }
    case ModelBase::BHUB:
    {
        uvals.resize(1);
        rvals.resize(3);
        /// default values
        rvals(0)=1;
        rvals(2)=0;
        try
        {
            P.GetValue(uvals(0),"d",true);
            P.GetValue(rvals(0),"t");
            P.GetValue(rvals(1),"U",true);
            P.GetValue(rvals(2),"mu");
        }
        catch (std::exception& e)
        {
            throw std::logic_error("parameters missing for Bose Hubbard model");
        }

        if (verbose)
        {
            cout<<"Bose Hubbard model"<<endl;
            cout<<"t="<<rvals(0)<<", U="<<rvals(1)<<", mu="<<rvals(2)<<endl;
        }
        pmod = make_shared<BoseHubbardModel>(uvals(0),rvals(0),rvals(1),rvals(2));
//        #ifdef _USE_SYMMETRIES_
//        uint n = 1;
//        P.GetValue(N,"N",true);
//        P.GetValue(n,"n",true);
//        if (verbose) cout<<"N="<<N<<", n="<<n<<endl;
//        pmod = make_shared<BoseHubbardModel>(uvals(0),rvals(0),rvals(1),rvals(2),N,n);
//        #else
//        pmod = make_shared<BoseHubbardModel>(uvals(0),rvals(0),rvals(1),rvals(2));
//        #endif // _USE_SYMMETRIES_
        break;
    }
    case ModelBase::QPOTTS:
        uvals.resize(1);
        rvals.resize(1);

        /// default values
        rvals(0)=0;
        try
        {
            P.GetValue(uvals(0),"q",true);
            P.GetValue(rvals(0),"h");
        }
        catch (std::exception& e)
        {
            throw std::logic_error("parameters missing for Quantum Potts model");
        }
        if (verbose)
        {
            cout<<"Quantum Potts model"<<endl;
            cout<<"q="<<uvals(0)<<", h="<<rvals(0)<<endl;
        }
        pmod = make_shared<QuantumPottsModel>(uvals(0),rvals(0));

//        #ifdef _USE_SYMMETRIES_
//        P.GetValue(N,"N",true);
//        if (verbose) cout<<"N="<<N<<endl;
//        pmod = make_shared<QuantumPottsModel>(uvals(0),rvals(0),N);
//        #else
//        pmod = make_shared<QuantumPottsModel>(uvals(0),rvals(0));
//        #endif // _USE_SYMMETRIES_
        break;
    default:
        throw std::logic_error("model not recognized");
    }
    return pmod;
}

