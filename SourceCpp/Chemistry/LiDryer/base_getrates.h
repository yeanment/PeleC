#ifndef __BASE_CPU_GET_RATES__
#define __BASE_CPU_GET_RATES__

#include "chemistry_file.H"

#ifdef __cplusplus
extern "C" {
#endif
AMREX_GPU_HOST_DEVICE
void base_getrates(const double pressure, const double temperature, const double 
  avmolwt, const double *mass_frac, double *wdot); 
#ifdef __cplusplus
}
#endif
#endif // __BASE_CPU_GET_RATES__
