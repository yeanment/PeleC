#include "PeleC_EOS.H"
#include "mechanism.h"
#include "PeleC_index_macros.H"
#include "PeleC_Parameters.H"
#include <cmath>


extern "C" {
AMREX_GPU_HOST_DEVICE void get_imw(double* neww); 
AMREX_GPU_HOST_DEVICE void ckpy_(double*  rho,double*  T,double*  y,double *  P);
AMREX_GPU_HOST_DEVICE void ckcvms_(double*  T, double*  cvms);
AMREX_GPU_HOST_DEVICE void ckcpms_(double*  T, double*  cvms);
AMREX_GPU_HOST_DEVICE void ckums_(double*  T,double*  ums);
AMREX_GPU_HOST_DEVICE void ckhms_(double*  T,double*  ums);
AMREX_GPU_HOST_DEVICE void get_t_given_ey_(double*  e,double*  y, double*  t, int *ierr);
AMREX_GPU_HOST_DEVICE void ckytx_(double massfrac[], double molefrac[]); 
}

AMREX_GPU_HOST_DEVICE EOS::EOS()
{}

AMREX_GPU_HOST_DEVICE EOS::~EOS()
{}

AMREX_GPU_HOST_DEVICE 
void EOS::eos_bottom()
{
    ckcvms_(&T,  cvi);
    ckcpms_(&T,  cpi); 
    ckhms_( &T,   hi);
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


AMREX_GPU_HOST_DEVICE
void EOS::eos_wb()
{
    amrex::Real imw[NUM_SPECIES]; 
    get_imw(imw);
    amrex::Real summ =0.0; 
#pragma unroll 
    for(int i = 0; i < NUM_SPECIES; ++i) summ+= massfrac[i]*imw[i]; 
    wbar = 1.0/summ; 
}


AMREX_GPU_HOST_DEVICE
void EOS::eos_re()
{
    int lierr=0; 
    eos_wb();
    get_t_given_ey_(&e, massfrac, &T, &lierr);
//    T = amrex::max(T, smallT); //*/
    ckums_(&T,  ei); 
    ckpy_(&rho, &T, massfrac, &p);

    eos_bottom(); 
}

AMREX_GPU_HOST_DEVICE
void EOS::eos_rp()
{
    eos_wb(); 
    T = p*wbar/(rho*Ru);
    ckums_(&T,  ei);
    e = 0.0;  
#pragma unroll
    for(int i = 0; i < NUM_SPECIES; ++i) e += massfrac[i]*ei[i]; 
    eos_bottom();     
}

AMREX_GPU_HOST_DEVICE
void EOS::eos_ytx()
{
    for(int i = 0; i < NUM_SPECIES; ++i) std::cout<< massfrac[i] << std::endl; 
    std::cin.get(); 
    ckytx_(massfrac, molefrac); 
}

AMREX_GPU_HOST_DEVICE
void EOS::eos_hi()
{
   ckhms_( &T,   hi);
}

/* THESE ASSUME THE ONLY PASSIVE VARS ARE THE SPECIES AND NON-USED VELOCITIES. TODO add additional passive vars*/
AMREX_GPU_HOST_DEVICE
int EOS::upass_map(const int i)
{
/*UMY and UMZ are passive*/
#if (AMREX_SPACEDIM==1)
    if(i <=1); 
        return i+2; 
    else 
        return (i-2) + UFS;
/*UMZ is passive*/
#elif (AMREX_SPACEDIM==2)
    if(i == 0)
        return 3; 
    else 
        return (i-1) + UFS;
#else 
    return i + UFS;
#endif
}

AMREX_GPU_HOST_DEVICE 
int EOS::qpass_map(const int i)
{
/*V and W are passive*/
#if(AMREX_SPACEDIM==1)
    if(i <=1); 
        return i+2; 
    else 
        return (i-2) + QFS; 
/*W is passive*/
#elif(AMREX_SPACEDIM==2)
    if(i==0) 
        return 3; 
    else
        return (i-1) + QFS;
#else 
    return i + QFS;  
#endif
}


