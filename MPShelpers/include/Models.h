#ifndef MODELS_H_
#define MODELS_H_

#include "parser.h"
#include "MPSIncludesBasic.h"

//#include <vector>
//#include <string>
//
//#include "OperatorTypes.hpp"
//#include "arma_typedefs.h"
//#include "parser.h"
//#include "helpers.h"
//
//#ifdef _USE_SYMMETRIES_
//#include "ItoKey.hpp"
//#include "KeyTypes.hpp"
//#include "symobj.hpp"
//#endif // _USE_SYMMETRIES_

using std::cout;
using std::endl;
using std::make_shared;

/// TODO (valentin#1#2016-10-31): find out why I implemented an Init() method rather than just doing all this in the ctor
/// TODO (valentin#1#2017-11-05): implement general model with user defined Hamiltonian, symmetries, etc.
/// TODO (valentin#1#2017-11-05): implement user defined observables

class ModelBase
{
public:
    typedef enum emod {XYZ,XXZ,XXZAF,QIM,QPOTTS,FHUB,BHUB,CUSTOM} emod;

    ModelBase(emod modtype, uint localdim):modtype_(modtype),localdim_(localdim),id_(SpId<Real>(localdim,1)) {};
    virtual ~ModelBase() {};

    virtual void Init() = 0; /// define as pure virtual function

    /// GETTERS
    inline uint GetLocalDim() const {return localdim_;};
    inline emod GetModelType() const {return modtype_;};
    inline std::string GetModelString() const {return modstring_;};
    inline SparseOperator<double> GetLocalHam() const { return ham_;};
    virtual std::vector<SparseOperator<double> > GetObservables(const std::vector<std::string>& opstring) const = 0; /// define as pure virtual function

    virtual void ShowParams() const = 0; /// define as pure virtual function

    #ifdef _USE_SYMMETRIES_
    const symobj<int>& GetGroupObj() const {return GroupObj_;};
    inline uint GetNSym() const {return GroupObj_.Nsym_;};
    /// get the stride between the quantum numbers (first) and the initial offset (second)
    virtual std::vector<std::pair<int,int> > GetdQ(uint N, const std::vector<int>& QN) const = 0; /// define as pure virtual function
    virtual ItoKey<1,IKey> GetI2K(uint N, const std::vector<int>& QN=std::vector<int>()) const = 0; /// define as pure virtual function
    std::vector<dim_map<IKey> > MakeDims(const std::vector<uint>& dims, uint N, const std::vector<int>& QN=std::vector<int>()) const;
    #endif // _USE_SYMMETRIES_
protected:
    const emod modtype_;
    const uint localdim_;
    std::string modstring_;

    SparseOperator<double> ham_,id_;
    virtual void MakeOps() = 0; /// define as pure virtual function

    #ifdef _USE_SYMMETRIES_
    virtual dim_map<IKey> GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const = 0; /// define as pure virtual function
    symobj<int> GroupObj_; /// this doesn't need to be a reference, we will use this very instance for all other Keys appearing anywhere
    #endif // _USE_SYMMETRIES_
};


using modptr = std::shared_ptr<ModelBase>;
const modptr CreateModel(const parser& P, bool verbose=false);

/**==================================================================================================================================*/
/**== ACTUAL MODEL TYPE DECLARATIONS ================================================================================================*/
/**==================================================================================================================================*/

/*************************************************************************
 ** PURE VIRTUAL METHODS THAT NEED TO IMPLEMENTED FOR EVERY MODEL:
 ** Init()
 ** GetObservables()
 ** ShowParams()
 ** MakeOps()
 **
 ** WITH SYMMETRIES:
 ** GetI2K()
 ** GetDimMap()
 *************************************************************************/

/*************************************************************************************************************************************/
/**< XYZ ABSTRACT MODEL TYPE *********************************************************************************************************/
/*************************************************************************************************************************************/
class XYZAbstractModel : public ModelBase
{
public:
    XYZAbstractModel(emod type, double Jx, double Jy, double Jz, double hz, uint dim);

    virtual void ShowParams () const;
    std::vector<SparseOperator<double> > GetObservables(const std::vector<std::string>& opstring) const;

protected:
    void MakeOps();

    double Jx_,Jy_,Jz_,hz_;
    SparseOperator<double> sx_,syi_,sz_,sp_,sm_;
    std::map<uint,std::string> spin_;
};

/*************************************************************************************************************************************/
/**< XYZ MODEL ***********************************************************************************************************************/
/*************************************************************************************************************************************/
//class XYZModel : public XYZAbstractModel
//{
//public:
//    XYZModel(double Jx, double Jy, double Jz, double hz, uint dim);
//
//    void Init();
//    void ShowParams () const;
//    #ifdef _USE_SYMMETRIES_
//    ItoKey<1,IKey> GetI2K(uint N, const std::vector<int>& QN) const;
//    #endif // _USE_SYMMETRIES_
//};

/*************************************************************************************************************************************/
/**< XXZ MODEL ***********************************************************************************************************************/
/*************************************************************************************************************************************/
class XXZModel : public XYZAbstractModel
{
public:
    XXZModel(double J, double Delta, double hz, uint dim);

    void Init();
    void ShowParams () const;
    #ifdef _USE_SYMMETRIES_
    std::vector<std::pair<int,int> > GetdQ(uint N, const std::vector<int>& QN) const;
    ItoKey<1,IKey> GetI2K(uint N, const std::vector<int>& QN) const;
    #endif // _USE_SYMMETRIES_
protected:
    #ifdef _USE_SYMMETRIES_
    dim_map<IKey> GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const;
    #endif // _USE_SYMMETRIES_
    double J_,Delta_;
};

/*************************************************************************************************************************************/
/**< XXZ ANTIFERROMAGNETIC MODEL *****************************************************************************************************/
/*************************************************************************************************************************************/
//class XXZAFType : public XYZAbstractModel
//{
//public:
//    XXZAFType(double J, double Delta, double hz, uint dim);
//
//    void Init();
//    void ShowParams () const;
//protected:
//    double J_,Delta_;
//};

/*************************************************************************************************************************************/
/**< QUANTUM ISING MODEL *************************************************************************************************************/
/*************************************************************************************************************************************/
class QuantumIsingModel : public XYZAbstractModel
{
public:
    QuantumIsingModel(double J, double h);

    void Init();
    void ShowParams () const;
    #ifdef _USE_SYMMETRIES_
    std::vector<std::pair<int,int> > GetdQ(uint N, const std::vector<int>& QN) const;
    ItoKey<1,IKey> GetI2K(uint N, const std::vector<int>& QN) const;
    #endif // _USE_SYMMETRIES_

protected:
    #ifdef _USE_SYMMETRIES_
    dim_map<IKey> GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const;
    #endif // _USE_SYMMETRIES_
    double J_,h_;
};

/*************************************************************************************************************************************/
/**< QUANTUM POTTS MODEL *************************************************************************************************************/
/*************************************************************************************************************************************/
class QuantumPottsModel : public ModelBase
{
public:
    QuantumPottsModel(uint q, double h);

    void Init();
    void ShowParams () const;
    std::vector<SparseOperator<double> > GetObservables(const std::vector<std::string>& opstring) const;

    #ifdef _USE_SYMMETRIES_
    std::vector<std::pair<int,int> > GetdQ(uint N, const std::vector<int>& QN) const;
    ItoKey<1,IKey> GetI2K(uint N, const std::vector<int>& QN) const;
    #endif // _USE_SYMMETRIES_
protected:
    void MakeOps();
    #ifdef _USE_SYMMETRIES_
    dim_map<IKey> GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const;
    #endif // _USE_SYMMETRIES_

    double h_;
    SparseOperator<double> xop_,zop_;
};
/*************************************************************************************************************************************/
/**< FERMI HUBBARD MODEL *************************************************************************************************************/
/*************************************************************************************************************************************/
class FermiHubbardModel : public ModelBase
{
public:
    FermiHubbardModel(double t, double U, double V, double mu, double hz);

    void Init();
    void ShowParams() const;
    std::vector<SparseOperator<double> > GetObservables(const std::vector<std::string>& opstring) const;

    #ifdef _USE_SYMMETRIES_
    std::vector<std::pair<int,int> > GetdQ(uint N, const std::vector<int>& QN) const;
    ItoKey<1,IKey> GetI2K(uint N, const std::vector<int>& QN) const;
    #endif // _USE_SYMMETRIES_

protected:
    void MakeOps();
    #ifdef _USE_SYMMETRIES_
    dim_map<IKey> GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const;
    #endif // _USE_SYMMETRIES_

    double t_,U_,V_,mu_,hz_;
    SparseOperator<double> cup_,cdo_,nupop_,ndoop_;
};

/*************************************************************************************************************************************/
/**< BOSE HUBBARD MODEL **************************************************************************************************************/
/*************************************************************************************************************************************/
class BoseHubbardModel : public ModelBase
{
public:
    BoseHubbardModel(uint d, double t, double U, double mu);

    void Init();
    void ShowParams() const;
    std::vector<SparseOperator<double> > GetObservables(const std::vector<std::string>& opstring) const;

    #ifdef _USE_SYMMETRIES_
    std::vector<std::pair<int,int> > GetdQ(uint N, const std::vector<int>& QN) const;
    ItoKey<1,IKey> GetI2K(uint N, const std::vector<int>& QN) const;
    #endif // _USE_SYMMETRIES_
protected:
    void MakeOps();
    #ifdef _USE_SYMMETRIES_
    dim_map<IKey> GetDimMap(const std::vector<uint>& dims, uint N, const std::vector<int>& QN) const;
    #endif // _USE_SYMMETRIES_

    double t_,U_,mu_;
    SparseOperator<double> bop_,nop_;
};

/*************************************************************************************************************************************/
/**< CUSTOM MODEL FROM FILE **********************************************************************************************************/
/*************************************************************************************************************************************/
class CustomModel : public ModelBase
{
public:
    CustomModel() = default

    void Init() {}; /// define as pure virtual function

    /// GETTERS
    std::vector<SparseOperator<double> > GetObservables(const std::vector<std::string>& opstring) const = {}; /// define as pure virtual function

    void ShowParams() const {}; /// define as pure virtual function

    #ifdef _USE_SYMMETRIES_
    const symobj<int>& GetGroupObj() const {return GroupObj_;};
    inline uint GetNSym() const {return GroupObj_.Nsym_;};
    /// get the stride between the quantum numbers (first) and the initial offset (second)
    std::vector<std::pair<int,int> > GetdQ(uint N, const std::vector<int>& QN) const {}; /// define as pure virtual function
    ItoKey<1,IKey> GetI2K(uint N, const std::vector<int>& QN=std::vector<int>()) const {}; /// define as pure virtual function
    std::vector<dim_map<IKey> > MakeDims(const std::vector<uint>& dims, uint N, const std::vector<int>& QN=std::vector<int>()) const;
    #endif // _USE_SYMMETRIES_
}
#endif // MODELS_H_
