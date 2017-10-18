
#include "../include/eigs.h"

static const int std_ncv = 20;

int eigs_n(std::function<void (Complex*,Complex*)> MultOPx, int N, CVecType& vals, CMatType& vecs, int nev, std::string whch, double tol, const CVecType& x0, int maxit, int ncv)
{
    vals.reset();
    vecs.reset();
    /// PARAMS FOR ZNAUPD ---------------------------------------------------------------------------------------------------------//
    int mode=1; /// standard EV problem A*x = lam*x

    int MAXIT = (maxit==0) ? N : std::min(maxit,N);

    int     IDO=0;
    char    BMAT='I';
    char    WHICH[3]; strcpy(WHICH,whch.c_str());
    int     NEV = nev;
    if (NEV > N-2) /// for non-symm, nev=N-2 at most
    {
        cerr<<"eigs_cn(): too many requested eigenvalues (nev < "<<N-1<<"), setting to "<<N-2<<endl;
        NEV = N-2;
    }
    if (NEV < 1) throw std::domain_error("NEV must be at least 1");
    double  TOL=tol;
    int     NCV = ncv;
    if (NCV == 0) NCV = std::min(std::max(std_ncv,2*NEV+1),N); /// NCV = 2*NEV + 1 advised, but take no less than 20. But NCV <= N too!
    if (NCV > N) NCV = N; /// NCV <= N !
    int     LDV = N; /// even though the C++ equivalent of a COMPLEX*16 array of length N is a double array of length 2*N, LDV is still N, as fortran jumps by 2 indices through the C style double array.
    int     LWORKL=3*NCV*NCV + 5*NCV;
    int     IPARAM[11] = {1,0,MAXIT,1,0,0,mode,0,0,0,0}; /// for check on output: IPARARM[2] = iter, IPARAM[8] = numopx
    int     IPNTR[14];
    int     INFONAUP=1; /// if 1 on entry use RESID as initial vector -> always set INFONAUP=1 and use own random starting vector if none other supplied

    CVecType RESID; /// on entry contains starting vector for Arnoldi, on exit contains the final residual vector
    if (x0.size()==0) RESID = CVecType(N,fill::randn);
    else if (x0.size()==uint(N)) RESID = x0;
    else throw std::domain_error("eigs_cn: x0 has wrong dimension");

    CMatType V(N,NCV); /// will contain Krylov Basis from Arnoldi
    CVecType WORKD(3*N);
    CVecType WORKL(LWORKL);
    RVecType RWORK(NCV);

//    CVecType ritzv;

//    uint CNTR = 0;
//    uint MAXITER = 5;

    /// ZNAUPD -------------------------------------------------------------------------------------------------------------------------//
    while (IDO!=99)
    {
        /// standard EV problem: A*x = lambda*x (here OP=A and B=I)
        /// generalized EV problem: A*x = lambda*M*x (here OP = inv[M]*A and B=M)
        znaupd_(&IDO,&BMAT,&N,WHICH,&NEV,&TOL,RESID.memptr(),&NCV,V.memptr(),&LDV,IPARAM,IPNTR,WORKD.memptr(),WORKL.memptr(),&LWORKL,RWORK.memptr(),&INFONAUP);

        switch (IDO)
        {
        case -1: /// compute Y = OP * X
            /// initialization phase, which is somehow never used...
            cerr<<"-1 not implemented"<<endl;
            break;
        case 1: /// compute Z = B * X and Y = OP * Z
            /// For standard EV Problem Y = Z as B = I (see above)
            MultOPx(&WORKD[IPNTR[0]-1],&WORKD[IPNTR[1]-1]);
//            ritzv = CVecType(&WORKLmem[IPNTR[6]-1],NEV,false,true);
//            ritzv.print("Ritz Values");
            break;
        case 2: /// compute Y = M * X (only for generalized)
            cerr<<"2 not implemented"<<endl;
            break;
        case 3: /// calculate shifts, if specified to do this separately (if IPARAM(1) = 0)
            cerr<<"3 not implemented"<<endl;
            break;
        case 4: /// compute Z = OP * X, documentation doesn't even say, which memory to use, probably never used...
            cerr<<"4 not implemented"<<endl;
            break;
        case 99:/// ARPACK HAS CONVERGED
//            cout<<"convergence"<<endl;
            break;
        default:
            cerr<<"IDO has unknown value"<<endl;
        }
    }
    int nconv=0;

    if (IDO==99 && INFONAUP==0)
    {
        nconv = IPARAM[4];
    }
    else
    {
/// Error flag on output (from documentation of ZNAUPD)
/// INFONAUP = 0: Normal exit.
/// INFONAUP = 1: Maximum number of iterations taken. All possible eigenvalues of OP has been found. IPARAM(5) returns the number of wanted converged Ritz values.
/// INFONAUP = 2: No longer an informational error. Deprecated starting with release 2 of ARPACK.
/// INFONAUP = 3: No shifts could be applied during a cycle of the Implicitly restarted Arnoldi iteration. One possibility is to increase the size of NCV relative to NEV. See remark 4 below.
/// INFONAUP = -1: N must be positive.
/// INFONAUP = -2: NEV must be positive.
/// INFONAUP = -3: NCV-NEV >= 2 and less than or equal to N.
/// INFONAUP = -4: The maximum number of Arnoldi update iteration must be greater than zero.
/// INFONAUP = -5: WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'
/// INFONAUP = -6: BMAT must be one of 'I' or 'G'.
/// INFONAUP = -7: Length of private work array is not sufficient.
/// INFONAUP = -8: Error return from LAPACK eigenvalue calculation;
/// INFONAUP = -9: Starting vector is zero.
/// INFONAUP = -10: IPARAM(7) must be 1,2,3.
/// INFONAUP = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.
/// INFONAUP = -12: IPARAM(1) must be equal to 0 or 1.
/// INFONAUP = -9999: Could not build an Arnoldi factorization. User input error highly likely. Please check actual array dimensions and layout. IPARAM(5) returns the size of the current Arnoldi factorization.
        cerr<<"no convergence in ZNAUPD, on exit: "<<INFONAUP<<", "<<IPARAM[4]<<" eigenvalues converged"<<endl;
        return 0;
    }

    /// PARAMS FOR ZNEUPD -------------------------------------------------------------------------------------------------------------------------//
    int INFONEUP=0;
    int RVEC=1;
    char HOWMNY='A';
    IVecType SELECT(NCV,fill::zeros);
    vals.resize(NEV+1);
    vecs.resize(N,NEV+1);
    int LDZ=N;
    Complex SIGMA(0,0);
    CVecType WORKEV(2*NCV);

    zneupd_(&RVEC,&HOWMNY,SELECT.memptr(),vals.memptr(),vecs.memptr(),&LDZ,&SIGMA,WORKEV.memptr(),&BMAT,&N,WHICH,&NEV,&TOL,RESID.memptr(),
            &NCV,V.memptr(),&LDV,IPARAM,IPNTR,WORKD.memptr(),WORKL.memptr(),&LWORKL,RWORK.memptr(),&INFONEUP);

    if (INFONEUP==0)
    {
        vals.resize(nconv);
        vecs.resize(N,nconv);
    }
    else
    {
/// Error flag on output  (from documentation of ZNEUPD)
/// INFONEUP = 0: Normal exit.
/// INFONEUP = 1: The Schur form computed by LAPACK routine csheqr could not be reordered by LAPACK routine ztrsen. Re-enter subroutine zneupd with IPARAM(5)=NCV and increase the size of
///               the array D to have dimension at least dimension NCV and allocate at least NCV columns for Z. NOTE: Not necessary if Z and V share the same space. Please notify the authors if this error occurs.
/// INFONEUP = -1: N must be positive.
/// INFONEUP = -2: NEV must be positive.
/// INFONEUP = -3: NCV-NEV >= 1 and less than or equal to N.
/// INFONEUP = -5: WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'
/// INFONEUP = -6: BMAT must be one of 'I' or 'G'.
/// INFONEUP = -7: Length of private work WORKL array is not sufficient.
/// INFONEUP = -8: Error return from LAPACK eigenvalue calculation. This should never happened.
/// INFONEUP = -9: Error return from calculation of eigenvectors. Informational error from LAPACK routine ztrevc.
/// INFONEUP = -10: IPARAM(7) must be 1,2,3
/// INFONEUP = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.
/// INFONEUP = -12: HOWMNY = 'S' not yet implemented
/// INFONEUP = -13: HOWMNY must be one of 'A' or 'P' if RVEC = .true.
/// INFONEUP = -14: ZNAUPD did not find any eigenvalues to sufficient accuracy.
/// INFONEUP = -15: ZNEUPD got a different count of the number of converged Ritz values than ZNAUPD got. This indicates the user probably made an error in passing data from ZNAUPD to ZNEUPD or that the
///                 data was modified before entering ZNEUPD
        cerr<<"no convergence in ZNEUPD, on exit "<<INFONEUP<<endl;
        return 0;
    }
    return nconv;
}

int eigs_n(const CMatType& A, CVecType& vals, CMatType& vecs,  int nev, std::string whch, double tol, const CVecType& x0, int maxit, int ncv)//, const CVecType& x0)
{
    uint m=A.n_rows;
    assert(m==A.n_cols);
    if (maxit==0) maxit = m;

    auto MultAx=[&A,m](Complex in[], Complex out[]) -> void
    {
        CVecType invec(in,m,false), outvec(out,m,false);
        outvec = A*invec;
    };

//    cout<<"ncv:"<<ncv<<endl;
//    return eigs_cn(MultAx,m,vals,vecs,nev,whch,tol,x0,maxit,ncv);
    return eigs_n(MultAx,m,vals,vecs,nev,whch,tol,x0,maxit,ncv);
}

int eigs_n(std::function<void (double*,double*)> MultOPx, int N, CVecType& vals, CMatType& vecs, int nev, std::string whch, double tol, const RVecType& x0, int maxit, int ncv)
{
    vals.reset();
    vecs.reset();
    /// PARAMS FOR DNAUPD ---------------------------------------------------------------------------------------------------------//
    int mode=1; /// standard EV problem A*x = lam*x

    int     MAXIT = (maxit==0) ? N : std::min(maxit,N);
    int     IDO=0;
    char    BMAT='I';
    char    WHICH[3]; strcpy(WHICH,whch.c_str());
    int     NEV = nev;
    if (NEV > N-2) /// for non-symm, nev = N-2 at most
    {
        cerr<<"eigs_rn(): too many requested eigenvalues (nev < "<<N-1<<"), setting to "<<N-2<<endl;
        NEV = N-2;
    }
    if (NEV < 1) throw std::domain_error("NEV must be at least 1");
    double  TOL = tol;
    int     NCV = ncv;
    if (NCV == 0) NCV = std::min(std::max(std_ncv,2*NEV+1),N); /// NCV = 2*NEV + 1 advised, but take no less than 20. But NCV <= N too!
    if (NCV > N) NCV = N; /// NCV <= N !
    int     LDV = N;
    int     LWORKL = 3*NCV*(NCV + 2);
    int     IPARAM[11] = {1,0,MAXIT,1,0,0,mode,0,0,0,0}; /// for check on output: IPARARM[2] = iter, IPARAM[8] = numopx
    int     IPNTR[14];
    int     INFONAUP = 1;/// if 1 on entry use RESID as initial vector -> always set INFONAUP=1 and use own random starting vector if none other supplied

    RVecType RESID; /// on entry contains starting vector for Arnoldi, on exit contains the final residual vector
    if (x0.size()==0) RESID = RVecType(N,fill::randn);
    else if (x0.size()==uint(N)) RESID = x0;
    else
    {
        cerr<<"x0 has wrong dimension"<<endl;
        abort();
    }

    RMatType V(LDV,NCV); /// will contain Krylov Basis from Arnoldi
    RVecType WORKD(3*N);
    RVecType WORKL(LWORKL);

    /// DNAUPD -------------------------------------------------------------------------------------------------------------------------//
    while (IDO!=99)
    {
        dnaupd_(&IDO,&BMAT,&N,WHICH,&NEV,&TOL,RESID.memptr(),&NCV,V.memptr(),&LDV,IPARAM,IPNTR,WORKD.memptr(),WORKL.memptr(),&LWORKL,&INFONAUP);

        switch (IDO)
        {
        case -1:
            cerr<<"-1 not implemented"<<endl;
            break;
        case 1: /// compute Z = B * X and Y = OP * Z
            /// For Simple EV Problem Y = Z
            MultOPx(&WORKD[IPNTR[0]-1],&WORKD[IPNTR[1]-1]);
            break;
        case 2:
            cerr<<"2 not implemented"<<endl;
            break;
        case 3:
            cerr<<"3 not implemented"<<endl;
            break;
        case 4:
            cerr<<"4 not implemented"<<endl;
            break;
        case 99:/// ARPACK HAS CONVERGED
            break;
        default:
            cerr<<"IDO has unknown value"<<endl;
        }

    }
    int nconv=0;

    if (IDO==99 && INFONAUP==0)
    {
        nconv=IPARAM[4];
    }
    else
    {
        cerr<<"no convergence in DNAUPD, on exit: "<<INFONAUP<<endl;
        return 0;
    }


    /// PARAMS FOR DNEUPD -------------------------------------------------------------------------------------------------------------------------//
    int INFONEUP=0;
    int RVEC=1;
    char HOWMNY='A';
    IVecType SELECT(NCV,fill::zeros);
    RVecType EVR(NEV+1);
    RVecType EVI(NEV+1);
    RMatType Zmat(N,NEV+1);
    int LDZ=N;
    double SIGMAR=0., SIGMAI=0.;
    RVecType WORKEV(3*NCV);


    /// DNEUPD -------------------------------------------------------------------------------------------------------------------------------------//

    dneupd_(&RVEC,&HOWMNY,SELECT.memptr(),EVR.memptr(),EVI.memptr(),Zmat.memptr(),&LDZ,&SIGMAR,&SIGMAI,WORKEV.memptr(),
            &BMAT,&N,WHICH,&NEV,&TOL,RESID.memptr(),&NCV,V.memptr(),&LDV,IPARAM,IPNTR,WORKD.memptr(),WORKL.memptr(),&LWORKL,&INFONEUP);

    if(INFONEUP==0)
    {
        nconv=IPARAM[4];
        uint newsize=nconv;

        EVR.resize(newsize);
        EVI.resize(newsize);

        vals=CVecType(EVR,EVI); /// fill in eigenvalues

        if (RVEC==1)
        {
            vecs.set_size(N,newsize);
            uint i=0;
            while (i<newsize)
            {
                if (std::abs(EVI(i))<1e-14) /// real eigenvalue
                {
                    vecs.col(i)=CVecType(Zmat.col(i),zeros(N));
                    ++i;
                }
                else /// complex pair
                {
                    vecs.col(i)=CVecType(Zmat.col(i),Zmat.col(i+1));
                    vecs.col(i+1)=CVecType(Zmat.col(i),-Zmat.col(i+1));
                    i+=2;
                }
            }
        }
        else vals=sort(vals,"descend");
    }
    else
    {
        cerr<<"no convergence in DNEUP, on exit: "<<INFONEUP<<endl;
        return 0;
    }

    return nconv;
}


int eigs_n(const RMatType& A, CVecType& vals, CMatType& vecs,  int nev, std::string whch, double tol, const RVecType& x0, int maxit, int ncv)
{
    uint m=A.n_rows;
    assert(m==A.n_cols);
    if (maxit==0) maxit = m;


    auto MultAx=[&A,m](double in[], double out[])
    {
        RVecType invec(in,m,false,true), outvec(out,m,false,true);
        outvec = A*invec;
    };

    return eigs_n(MultAx,m,vals,vecs,nev,whch,tol,x0,maxit,ncv);
}

int eigs_rs(std::function<void (double*,double*)> MultOPx, int N, RVecType& vals, RMatType& vecs, int nev, std::string whch, double tol, const RVecType& x0, int maxit, int ncv)
{

    vals.reset();
    vecs.reset();
    /// PARAMS FOR DSAUPD ---------------------------------------------------------------------------------------------------------//
    int mode=1; /// standard EV problem A*x = lam*x

    int MAXIT = (maxit==0) ? N : std::min(maxit,N);

    int     IDO = 0;
    char    BMAT = 'I';
    char    WHICH[3]; strcpy(WHICH,whch.c_str());
    int     NEV = nev;
    if (NEV > N-1) /// for real symm, nev = N-1 at most
    {
        cerr<<"eigs_rs(): too many requested eigenvalues (nev < "<<N<<"), setting to "<<N-1<<endl;
        NEV = N-1;
    }
    if (NEV < 1) throw std::domain_error("NEV must be at least 1");
    double  TOL = tol;
    int     NCV = ncv;
    if (NCV == 0) NCV = std::min(std::max(std_ncv,2*NEV),N); /// NCV = 2*NEV advised, but take no less than 20. But NCV <= N too!
    if (NCV > N) NCV = N; /// NCV <= N !
    int     LDV = N;
    int     LWORKL = NCV*(NCV + 8);
    int     IPARAM[11] = {1,0,MAXIT,1,0,0,mode,0,0,0,0}; /// for check on output: IPARARM[2] = iter, IPARAM[8] = numopx
    int     IPNTR[11];
    int     INFONAUP = 1; /// if 1 on entry use RESID as initial vector -> always set INFONAUP=1 and use own random starting vector if none other supplied

    RVecType RESID; /// on entry contains starting vector for Arnoldi, on exit contains the final residual vector
    if (x0.size()==0) RESID = RVecType(N,fill::randn);
    else if (x0.size()==uint(N)) RESID = x0;
    else
    {
        cout<<N<<" vs. "<<x0.size()<<endl;
        throw std::domain_error("eigs_rs: x0 has wrong dimension");
    }

    RMatType V(LDV,NCV); /// will contain Krylov Basis from Arnoldi
    RVecType WORKD(3*N);
    RVecType WORKL(LWORKL);
    /// DSAUPD -------------------------------------------------------------------------------------------------------------------------//
    while (IDO!=99)
    {
        dsaupd_(&IDO,&BMAT,&N,WHICH,&NEV,&TOL,RESID.memptr(),&NCV,V.memptr(),&LDV,IPARAM,IPNTR,WORKD.memptr(),WORKL.memptr(),&LWORKL,&INFONAUP);

        switch (IDO)
        {
        case -1:
            cerr<<"-1 not implemented"<<endl;
            break;
        case 1: /// compute Z = B * X and Y = OP * Z
            /// For Simple EV Problem Y = Z
            MultOPx(&WORKD[IPNTR[0]-1],&WORKD[IPNTR[1]-1]);
            break;
        case 2:
            cerr<<"2 not implemented"<<endl;
            break;
        case 3:
            cerr<<"3 not implemented"<<endl;
            break;
        case 99:/// ARPACK HAS CONVERGED
            break;
        default:
            cerr<<"IDO has unknown value"<<endl;
        }
    }
    int nconv=0;

    if (IDO==99 && INFONAUP==0)
    {
        nconv=IPARAM[4];
    }
    else
    {
        cerr<<"no convergence in DSAUPD, on exit: "<<INFONAUP<<endl;
        return 0;
    }


    /// PARAMS FOR DSEUPD -------------------------------------------------------------------------------------------------------------------------//
    int INFONEUP=0;
    int RVEC=1;
    char HOWMNY='A';
    IVecType SELECT(NCV,fill::zeros);
    vals.resize(NEV);
    vecs.resize(N,NEV);
    int LDZ=N;
    double SIGMA=0.;


    /// DSEUPD -------------------------------------------------------------------------------------------------------------------------------------//

    dseupd_(&RVEC,&HOWMNY,SELECT.memptr(),vals.memptr(),vecs.memptr(),&LDZ,&SIGMA,&BMAT,&N,WHICH,&NEV,&TOL,RESID.memptr(),
            &NCV,V.memptr(),&LDV,IPARAM,IPNTR,WORKD.memptr(),WORKL.memptr(),&LWORKL,&INFONEUP);

    if(INFONEUP==0)
    {
        nconv=IPARAM[4];
        int newsize=nconv;

        if (newsize!=NEV)
        {
            DOUT(newsize<<" instead of "<<NEV<<" eigenpairs converged");
            vals.resize(newsize);
            if (RVEC==1) vecs.resize(N,newsize);
        }
    }
    else
    {
        cerr<<"no convergence in DNEUP, on exit: "<<INFONEUP<<endl;
        return 0;
    }

    return nconv;
}

int eigs_rs(const RMatType& A, RVecType& vals, RMatType& vecs,  int nev, std::string whch, double tol, const RVecType& x0, int maxit, int ncv)
{
    uint m=A.n_rows;
    assert(m==A.n_cols);
    if (maxit==0) maxit = m;


    auto MultAx=[&A,m](double in[], double out[])
    {
        RVecType invec(in,m,false,true), outvec(out,m,false,true);
        outvec = A*invec;
    };

    return eigs_rs(MultAx,m,vals,vecs,nev,whch,tol,x0,maxit,ncv);
}





