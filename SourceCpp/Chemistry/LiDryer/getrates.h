#ifndef __BASELINE_CPU_GET_RATES__
#define __BASELINE_CPU_GET_RATES__

#include "chemistry_file.H"

#ifdef __cplusplus
extern "C" {
#endif
AMREX_GPU_GLOBAL
//AMREX_GPU_HOST_DEVICE
void getrates(const double pressure, const double temperature, const double 
  avmolwt, double *mass_frac, double *wdot); 
#ifdef __cplusplus
}
#endif
#endif // __BASELINE_CPU_GET_RATES__
