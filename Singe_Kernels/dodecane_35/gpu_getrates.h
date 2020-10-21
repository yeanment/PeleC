#ifndef __BASELINE_GPU_GET_RATES__
#define __BASELINE_GPU_GET_RATES__
#include "chemistry_file.H"

AMREX_GPU_GLOBAL
void gpu_getrates(const double *temperature_array, const double *pressure_array, 
  const double *avmolwt_array, const double *mass_frac_array, const double 
  *diffusion_array, const double dt, const int 
  spec_stride/*NX*NY*NZ in number of doubles*/, double *wdot_array); 
#endif // __BASELINE_GPU_GET_RATES__
