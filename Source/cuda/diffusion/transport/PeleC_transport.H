#ifndef PeleC_transport_H_
#define PeleC_transport_H_

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include "PeleC_trans_params.H"
#include <AMReX_FArrayBox.H>


void PeleC_transport_init();
void PeleC_transport_close();

AMREX_GPU_DEVICE
void PeleC_get_transport_coeffs(amrex::Box const& bx, amrex::Array4<const amrex::Real> const& q,
       amrex::Array4<amrex::Real> const& D); 


#endif
