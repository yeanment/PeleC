#ifndef __BASELINE_CPU_GET_RATES__
#define __BASELINE_CPU_GET_RATES__
#include "chemistry_file.H"

#define GETRATES_NEEDS_DIFFUSION

#define GETRATES_STIFF_SPECIES 22

const unsigned int stif_species_indexes[22] = {0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 
  15, 16, 17, 18, 19, 20, 22, 23, 29, 31, 32, 33}; 

#ifdef __cplusplus
extern "C" {
#endif
AMREX_GPU_HOST_DEVICE
void base_getrates(const double pressure, const double temperature, const double avmolwt, 
const double *mass_frac, const double *diffusion, const double dt, double *wdot); 
#ifdef __cplusplus
}
#endif
#endif // __BASELINE_CPU_GET_RATES__
