#include "PeleC_EOS.H"
#include "mechanism.h"
#include "PeleC_index_macros.H"
#include "PeleC_Parameters.H"
#include <cmath>

#ifdef AMREX_USE_CUDA
extern __managed__
#else 
extern 
#endif
double imw[NUM_SPECIES]; 

AMREX_GPU_HOST_DEVICE void CKPY(double*  rho,double*  T,double*  y, int * iwrk,double*  rwrk, double *  P);
AMREX_GPU_HOST_DEVICE void CKCVMS(double*  T, int * iwrk,double*  rwrk,double*  cvms);
AMREX_GPU_HOST_DEVICE void CKCPMS(double*  T, int * iwrk,double*  rwrk,double*  cvms);
AMREX_GPU_HOST_DEVICE void CKUMS(double*  T, int * iwrk,double*  rwrk,double*  ums);
AMREX_GPU_HOST_DEVICE void CKHMS(double*  T, int * iwrk,double*  rwrk,double*  ums);
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_EY(double*  e,double*  y, int * iwrk,double*  rwrk,double*  t, int *ierr);

EOS::EOS()
{}

EOS::~EOS()
{}

AMREX_GPU_DEVICE 
void EOS::eos_bottom()
{
    CKCVMS(&T, &iwrk, &rwrk, cvi); 
    CKCPMS(&T, &iwrk, &rwrk, cpi); 
    CKHMS( &T, &iwrk, &rwrk,  hi);
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


AMREX_GPU_DEVICE
void EOS::eos_wb()
{
    wbar = 1.0; 
    amrex::Real summ =0.0; 
#pragma unroll 
    for(int i = 0; i < NUM_SPECIES; ++i) summ+= massfrac[i]*imw[i]; 
    wbar /= summ; 
}


AMREX_GPU_DEVICE
void EOS::eos_re()
{
    int lierr=0; 
    eos_wb(); 
    
    GET_T_GIVEN_EY(&e, massfrac, &iwrk, &rwrk, &T, &lierr); 
    T = amrex::max(T, smallT); 
    CKUMS(&T, &iwrk, &rwrk, ei); 
    CKPY(&rho, &T, massfrac, &iwrk, &rwrk, &p); 
    eos_bottom(); 
}


/* THESE ASSUME THE ONLY PASSIVE VARS ARE THE SPECIES AND NON-USED VELOCITIES. TODO add additional passive vars*/
AMREX_GPU_DEVICE
int EOS::upass_map(const int i)
{
#if (AMREX_SPACEDIM==1)
    if(i == 0) 
        return 2;
    else if(i == 1)
        return 3; 
    else 
        return (i-2) + UFS;
#elif (AMREX_SPACEDIM==2)
    if(i == 0)
        return 3; 
    else 
        return (i-1) + UFS;
#else 
    return i + UFS;
#endif
}

AMREX_GPU_DEVICE 
int EOS::qpass_map(const int i)
{
#if(AMREX_SPACEDIM==1)
    if(i==0)
        return 2; 
    else if(i ==1); 
        return 3; 
    else 
        return (i-2) + QFS; 
#elif(AMREX_SPACEDIM==2)
    if(i==0) 
        return 3; 
    else
        return (i-1) + QFS;
#else 
    return i + QFS;  
#endif
}


