#ifndef EOS_H_
#define EOS_H_

#include <AMReX.H>
#include <AMReX_REAL.H> 
#include <AMReX_Utility.H> 
#include "mechanism.h"
#define naux 0
 
using namespace amrex;
 
class EOS
{
public:
    EOS(); 
    ~EOS(); 

//TODO Stupid Vars we don't actually need but are arguments to functions 
    int iwrk = 0; 
    amrex::Real rwrk = 1.e0; 
// variables 
    amrex::Real rho, T, p, e, h, s, f;
    amrex::Real massfrac[NUM_SPECIES], molefrac[NUM_SPECIES], aux[naux]; 
    amrex::Real dpdT, dpdr, dedT, dedr, dhdT, dhdr, dsdT, dsdr, dpde, dpdr_e;
    amrex::Real cv,  cp, cpi[NUM_SPECIES], cvi[NUM_SPECIES], hi[NUM_SPECIES], ei[NUM_SPECIES], si[NUM_SPECIES]; 
    amrex::Real wbar, mui[NUM_SPECIES], Acti[NUM_SPECIES], dedY[NUM_SPECIES], dhdY[NUM_SPECIES], gam1, cs; 
// Quantaties for non-ideal eos
    amrex::Real am, bm, damdYk[NUM_SPECIES], d2amdYkdT[NUM_SPECIES], dPdYk[NUM_SPECIES], damdT, d2amdT2;
    amrex::Real dpdtau, Z, taui[NUM_SPECIES], diP[NUM_SPECIES], dijY[NUM_SPECIES][NUM_SPECIES]; 

//functions
AMREX_GPU_DEVICE    int  upass_map(const int i); 
AMREX_GPU_DEVICE    int  qpass_map(const int i); 
AMREX_GPU_DEVICE    void eos_wb();  
AMREX_GPU_DEVICE    void eos_re(); 
AMREX_GPU_DEVICE    void eos_bottom(); 
//TODO add more later!  
};
#endif
