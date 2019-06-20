#include "PeleC_EOS.H"
#include "mechanism.h"
#include "PeleC_Parameters.H"
#include "chemistry_file.H" 
#include <cmath>


extern "C" {
AMREX_GPU_HOST_DEVICE void get_imw(double neww[]);
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

AMREX_GPU_HOST_DEVICE EOS::EOS()
{}

AMREX_GPU_HOST_DEVICE EOS::~EOS()
{}

AMREX_GPU_HOST_DEVICE 
void EOS::eos_bottom()
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


AMREX_GPU_HOST_DEVICE
void EOS::eos_wb()
{
    amrex::Real imw[NUM_SPECIES]; 
    get_imw(imw);
    amrex::Real summ =0.0; 
    for(int i = 0; i < NUM_SPECIES; ++i) summ+= massfrac[i]*imw[i]; 
    wbar = 1.0/summ; 
}


AMREX_GPU_HOST_DEVICE
void EOS::eos_re()
{
    int lierr=0; 
    eos_wb();
    GET_T_GIVEN_EY(&e, massfrac, &T, &lierr);
//    T = amrex::max(T, smallT); //*/
    CKUMS(&T,  ei); 
    CKPY(&rho, &T, massfrac, &p);

    eos_bottom(); 
}

AMREX_GPU_HOST_DEVICE
void EOS::eos_rt()
{
    eos_wb(); 
    CKPY(&rho, &T, massfrac, &p); 
    CKUMS(&T, ei);
    for(int i = 0; i < NUM_SPECIES; ++i) e+= massfrac[i]*ei[i]; 
    
    eos_bottom(); 
}


AMREX_GPU_HOST_DEVICE
void EOS::eos_mpr2wdot(amrex::Real wdot[])
{
    BL_PROFILE_VAR("PeleC::EOS::eos_mpr2wdot", EOS_mpr2wdot); 
    CKWYR(&rho, &T, massfrac, wdot);
    eos_rt(); 
    amrex::Real mw[NUM_SPECIES]; 
    get_mw(mw); 
    for(int n = 0; n < NUM_SPECIES; n++){
        wdot[n] *= mw[n];
    }  
    BL_PROFILE_VAR_STOP(EOS_mpr2wdot); 
}


AMREX_GPU_HOST_DEVICE
void EOS::eos_rp()
{
    eos_wb(); 
    T = p*wbar/(rho*Ru);
    CKUMS(&T,  ei);
    e = 0.0;  
    for(int i = 0; i < NUM_SPECIES; ++i) e += massfrac[i]*ei[i]; 
    eos_bottom();     
}

AMREX_GPU_HOST_DEVICE
void EOS::eos_ytx()
{
    CKYTX(massfrac, molefrac); 
}

AMREX_GPU_HOST_DEVICE
void EOS::eos_hi()
{
   CKHMS( &T,   hi);
}
AMREX_GPU_HOST_DEVICE
void EOS::eos_cv()
{
    CKCVBS(&T, massfrac, &cv); 
}

//Hydro -> Advected -> Species -> Aux 
//If num_adv == 0 -> QFA = QFS and UFA = UFS, see PeleC_index_macros.H 
//For explicit definitions. 
AMREX_GPU_HOST_DEVICE
int EOS::upass_map(const int i)
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
int EOS::qpass_map(const int i)
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


