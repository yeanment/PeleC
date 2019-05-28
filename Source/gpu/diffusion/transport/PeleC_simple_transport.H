#ifndef actual_transport_H_
#define actual_transport_H_

#include "mechanism.h"

#include <AMReX_REAL.H>
#include <AMReX_Array.H>

#include <AMReX_GpuQualifiers.H>


void actual_transport(bool wtr_get_xi, bool wtr_get_mu, bool wtr_get_lam, bool wtr_get_Ddiag, amrex::Real& Tloc,
              amrex::Real& rholoc, amrex::Real* Yloc,
              amrex::Real* Ddiag, amrex::Real& mu, amrex::Real& xi, amrex::Real& lam);

 void comp_pure_bulk(amrex::Real Tloc,  amrex::Real* muloc,
          amrex::Real* xiloc );


#endif
