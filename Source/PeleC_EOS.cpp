#include "PeleC_EOS.H"
#include "mechanism.h"

#if 0
using namespace amrex;
EOS::EOS()
{}

EOS::~EOS()
{}

AMREX_GPU_DEVICE 
void eos_bottom()
{
    CKCVMS(T,iwrk,rwrk, cvi); 
    CKCPMS(T,iwrk,rwrk, cpi); 
    CKHMS( T,iwrk,rwrk,  hi);
    cv = 0.e0, cp = 0.e0, h = 0.e0; 
    for(int i = 0; i < nspecies; ++i){
         cv+=massfrac[i]*cvi[i];
         cp+=massfrac[i]*cpi[i]; 
         h +=massfrac[i]* hi[i]; 
    }
    Real Cvx = wbar*cv; 
//Do we know what Ru is here? 
    gam1 = (Cvx + Ru)/Cvx; 
    cs = std::sqrt(gam1*p/rho); 
    dpdr_e = p/rho;
    dpde = (gam1 - 1.0)*rho; 
    s = 1.e0; 
    dPdr = 0.e0; 
}


AMREX_GPU_DEVICE
void EOS::eos_re()
{
    eos_wb(); 
//TODO make this work? 
    get_T_given_eY(e, massfrac, iwrk, rwrk, T, lierr); 
//Do we know what smallT is here? 
    T = amrex::max(T, smallT); 
    CKUMS(T, iwrk, rwrk, ei); 
    CKPY(rho, T, massfrac, iwrk, rwrk, p); 
    eos_bottom(); 
}


#endif
