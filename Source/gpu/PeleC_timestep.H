#ifndef PELEC_TIMESTEP_H
#define PELEC_TIMESTEP_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H> 

AMREX_GPU_HOST_DEVICE
amrex::Real 
PeleC_estdt_hydro ( amrex::Box const& bx, amrex::FArrayBox const& statefab, 
             D_DECL(const amrex::Real& dx, const amrex::Real& dy, const amrex::Real& dz)); 

AMREX_GPU_DEVICE 
amrex::Real 
PeleC_estdt_veldif(amrex::Box const box, amrex::FArrayBox const& statefab,
             D_DECL(const amrex::Real& dx, const amrex::Real& dy, const amrex::Real& dz));

AMREX_GPU_DEVICE 
amrex::Real 
PeleC_estdt_tempdif(amrex::Box const box, amrex::FArrayBox const& statefab,
             D_DECL(const amrex::Real& dx, const amrex::Real& dy, const amrex::Real& dz));

AMREX_GPU_DEVICE 
amrex::Real 
PeleC_estdt_enthdif(amrex::Box const box, amrex::FArrayBox const& statefab,
             D_DECL(const amrex::Real& dx, const amrex::Real& dy, const amrex::Real& dz));

#endif 
