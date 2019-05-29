#ifndef actual_transport_H_
#define actual_transport_H_

#include <mechanism.h> 
#include <chemistry_file.H> 

#include <AMReX_REAL.H>
#include <AMReX_Array.H>

#include <AMReX_GpuQualifiers.H>

AMREX_GPU_HOST_DEVICE
void PeleC_actual_transport(bool wtr_get_xi, bool wtr_get_mu, bool wtr_get_lam, bool wtr_get_Ddiag,
                            amrex::Real& Tloc,
                            amrex::Real& rholoc, amrex::Real* Yloc,
                            amrex::Real* Ddiag, amrex::Real& mu, amrex::Real& xi, amrex::Real& lam);

AMREX_GPU_HOST_DEVICE
 void PeleC_comp_pure_bulk(amrex::Real Tloc,  amrex::Real* muloc,
                           amrex::Real* xiloc );


#endif
