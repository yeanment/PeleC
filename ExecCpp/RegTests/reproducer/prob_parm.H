#ifndef _PROB_PARM_H_
#define _PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuMemory.H>

struct ProbParmDevice : amrex::Gpu::Managed
{
  amrex::Real p0 = 1.013e6; // [erg cm^-3]
  amrex::Real T0 = 300.0;
};

struct ProbParmHost : amrex::Gpu::Managed
{
};

#endif
