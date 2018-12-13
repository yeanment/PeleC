#ifndef EOS_H_
#define EOS_H_

#include "mechanism.h" 
#if 0
using namespace amrex;
 
class EOS
{
public:
    EOS(); 
    ~EOS(); 

//TODO Stupid Vars we don't actually need but are arguements to fnctns 
    int iwrk = 0; 
    amrex::Real rwrk = 1.e0; 

// variables 
    amrex::Real rho, T, p, e, h, s, f;
    amrex::Real massfrac[nspecies], molefrac[nspecies], aux[naux]; 
    amrex::Real dpdT, dpdr, dedT, dedr, dhdT, dhdr, dsdT, dsdr, dpde, dpdr_e;
    amrex::Real cv,  cp, cpi[nspecies], cvi[npecies], hi[npecies], ei[nspecies], si[nspecies]; 
    amrex::Real wbar, mui[nspecies], Acti[npecies], dedY[nspecies], dhdY[nspecies], gam1, cs; 
// Quantaties for non-ideal eos
    amrex::Real am, bm, damdYk[nspecies], d2amdYkdT[nspecies], dPdYk[nspecies], damdT, d2amdT2;
    amrex::Real dpdtau, Z, taui[nspecies], diP[nspecies], dijY[nspecies][nspecies]; 

//functions 
AMREX_GPU_DEVICE    void eos_re(); 
AMREX_GPU_DEVICE    void eos_bottom(); 
//TODO add more later!  
}
#endif
#endif
