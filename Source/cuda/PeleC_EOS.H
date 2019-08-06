#ifndef EOS_H_
#define EOS_H_

#include <AMReX.H>
#include <AMReX_REAL.H> 
#include <AMReX_Utility.H>
#include <PeleC_index_macros.H>  
#include "mechanism.h"
#include "PeleC_Parameters.H"
#include "chemistry_file.H" 
#include <cmath>
#define naux 0

using namespace amrex; 
 
extern "C" {
AMREX_GPU_HOST_DEVICE void get_imw(double imw[]);
AMREX_GPU_HOST_DEVICE void get_mn(double mw[]);  
AMREX_GPU_HOST_DEVICE void CKPY(double*  rho,double*  T,double*  y,double *  P);
AMREX_GPU_HOST_DEVICE void CKCVMS(double*  T, double*  cvms);
AMREX_GPU_HOST_DEVICE void CKCVBS(double*  T, double* massfrac, double* cv); 
AMREX_GPU_HOST_DEVICE void CKCPMS(double*  T, double*  cvms);
AMREX_GPU_HOST_DEVICE void CKUMS(double*  T,double*  ums);
AMREX_GPU_HOST_DEVICE void CKHMS(double*  T,double*  ums);
AMREX_GPU_HOST_DEVICE void CKWYR(double* rho, double* T, double* y, double* wdot); 
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_EY(double*  e,double*  y, double*  t, int *ierr);
AMREX_GPU_HOST_DEVICE void CKYTX(double massfrac[], double molefrac[]);
 
}

namespace EOS
{

/*
AMREX_GPU_HOST_DEVICE 
inline void bottom()
{
    CKCVMS(&T,  cvi);
    CKCPMS(&T,  cpi); 
    CKHMS(&T,   hi);
    cv = 0.e0, cp = 0.e0, h = 0.e0; 
    for(int i = 0; i < NUM_SPECIES; ++i){
         cv+=massfrac[i]*cvi[i];
         cp+=massfrac[i]*cpi[i]; 
         h +=massfrac[i]* hi[i]; 
    }
    amrex::Real Cvx = wbar*cv; 
    gam1 = (Cvx + Ru)/Cvx; 
    cs = std::sqrt(gam1*p/rho); 
    dpdr_e = p/rho;
    dpde = (gam1 - 1.0)*rho; 
    s = 1.e0; 
    dpdr = 0.e0;
}
*/ 

AMREX_GPU_HOST_DEVICE
inline void get_cp(amrex::Real T, amrex::Real mass[], amrex::Real &cp)
{
    amrex::Real temp[NUM_SPECIES]; //cpi 
    CKCPMS(&T, temp); 
    cp = 0.0; 
    for(int i =0; i < NUM_SPECIES; ++i) cp += mass[i]*temp[i];         
}

AMREX_GPU_HOST_DEVICE
inline void get_wb(amrex::Real mass[], amrex::Real &wbar)
{
    amrex::Real imw[NUM_SPECIES]; 
    get_imw(imw);
    amrex::Real summ =0.0; 
    for(int i = 0; i < NUM_SPECIES; ++i) summ+= mass[i]*imw[i]; 
    wbar = 1.0/summ; 

}

AMREX_GPU_HOST_DEVICE
inline void get_cs(amrex::Real rho, amrex::Real T, amrex::Real mass[], amrex::Real &c)
{
    amrex::Real pres; 
    CKPY(&rho, &T, mass, &pres); 
    amrex::Real temp[NUM_SPECIES]; 
    CKCVMS(&T, temp); //temp = cvi 
    amrex::Real cv = 0.0; 
    for(int i =0; i < NUM_SPECIES; ++i) 
        cv += mass[i]*temp[i]; 
    amrex::Real wbar; 
    get_wb(mass, wbar); 
    amrex::Real ga = (wbar*cv + Ru)/(wbar*cv); 
    c = std::sqrt(ga*pres/rho); 
}

AMREX_GPU_HOST_DEVICE
inline void cmpT(amrex::Real e, amrex::Real massfrac[], amrex::Real &T)
{
    //For Fuego this function is really just a wrapper for GET_T_GIVEN_EY
    //In SRK this will be different probably 
    int lierr = 0; 
    GET_T_GIVEN_EY(&e, massfrac, &T, &lierr); 
}


/*Prototype for moving EOS to be a namespace instead of a class */ 
AMREX_GPU_HOST_DEVICE
inline void ctop(amrex::Real massfrac[], amrex::Real rho,
                   amrex::Real &e, amrex::Real &T, 
                   amrex::Real &p, amrex::Real &dpdr_e,
                   amrex::Real &dpde, amrex::Real &gam1, amrex::Real &cs, 
                   amrex::Real &wbar)
{
    int lierr = 0;
    amrex::Real temp[NUM_SPECIES]; 
    //here temp is the inverse molecular weights 
    get_imw(temp);
    amrex::Real summ =0.0; 
    for(int i = 0; i < NUM_SPECIES; ++i) summ+= massfrac[i]*temp[i]; 
    wbar = 1.0/summ; 
//  get Temperature
    GET_T_GIVEN_EY(&e, massfrac, &T, &lierr);    
//  get pressure 
    CKPY(&rho, &T, massfrac, &p); 

    CKCVMS(&T,  temp); // temp changes from imw to cvi
    amrex::Real cv = 0.e0; 
    for(int i = 0; i < NUM_SPECIES; ++i){
         cv+=massfrac[i]*temp[i];
    }
    amrex::Real Cvx = wbar*cv; 
    gam1 = (Cvx + Ru)/Cvx; 
    cs = std::sqrt(gam1*p/rho); 
    dpdr_e = p/rho;
    dpde = (gam1 - 1.0)*rho; 
}


AMREX_GPU_HOST_DEVICE
inline void TrhoY2p(amrex::Real dens, amrex::Real T, amrex::Real mass[], amrex::Real &pres)
{
    CKPY(&dens, &T, mass, &pres); 
}


AMREX_GPU_HOST_DEVICE
inline void mpr2wdot(amrex::Real dens, amrex::Real T, amrex::Real mass[], amrex::Real wdot[])
{
    CKWYR(&dens, &T, mass, wdot); 
    amrex::Real mw[NUM_SPECIES]; 
    get_mw(mw); 
    for(int n = 0; n < NUM_SPECIES; n++) wdot[n] *= mw[n]; 
}

AMREX_GPU_HOST_DEVICE
inline void get_ei(amrex::Real T, amrex::Real enrgi[])
{
    CKUMS(&T, enrgi); 
}

AMREX_GPU_HOST_DEVICE 
inline void get_cv(amrex::Real mass[], amrex::Real T, amrex::Real &cv)
{
    cv = 0.e0; 
    amrex::Real temp[NUM_SPECIES];
    CKCVMS(&T, temp); //here temp is cvi
    for(int i = 0; i < NUM_SPECIES; ++i) cv+= mass[i]*temp[i]; 
}



AMREX_GPU_HOST_DEVICE 
inline void ytx(amrex::Real mass[], amrex::Real mole[])
{
    CKYTX(mass, mole); 
}


AMREX_GPU_HOST_DEVICE
inline void get_hi(amrex::Real T, amrex::Real enthalpy[])
{
   CKHMS(&T, enthalpy);
}


//Hydro -> Advected -> Species -> Aux 
//If num_adv == 0 -> QFA = QFS and UFA = UFS, see PeleC_index_macros.H 
//For explicit definitions. 
AMREX_GPU_HOST_DEVICE
inline int upass_map(const int i)
{
/*UMY and UMZ are passive*/
#if (AMREX_SPACEDIM==1)
    if(i <=1); 
        return i+2; 
    else 
        return (i-2) + UFA;
/*UMZ is passive*/
#elif (AMREX_SPACEDIM==2)
    if(i == 0)
        return 3; 
    else 
        return (i-1) + UFA;
#else 
    return i + UFA;
#endif
}

AMREX_GPU_HOST_DEVICE 
inline int qpass_map(const int i)
{
/*V and W are passive*/
#if(AMREX_SPACEDIM==1)
    if(i <=1); 
        return i+2; 
    else 
        return (i-2) + QFA; 
/*W is passive*/
#elif(AMREX_SPACEDIM==2)
    if(i==0) 
        return 3; 
    else
        return (i-1) + QFA;
#else 
    return i + QFA;  
#endif
}

};
#endif
