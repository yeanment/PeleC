#ifndef PeleC_misc_H_
#define PeleC_misc_H_
#include <AMReX_FArrayBox.H>


AMREX_GPU_DEVICE
void PeleC_cmpTemp(const int i, const int j, const int k, amrex::Array4<amrex::Real> const& S); 

AMREX_GPU_DEVICE
void PeleC_rst_int_e(const int i, const int j, const int k, amrex::Array4<amrex::Real> const& S); 

#endif
