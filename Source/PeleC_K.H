#ifndef PeleC_K_H_
#define PeleC_K_H_

#include <AMReX_FArrayBox.H> 
AMREX_GPU_DEVICE 
void PeleC_ctoprim(amrex::Box const& bx, amrex::FArrayBox& state, amrex::FArrayBox& Q,
                   amrex::FArrayBox& Qaux); 
//TODO Add more fuctions! 

#endif 
