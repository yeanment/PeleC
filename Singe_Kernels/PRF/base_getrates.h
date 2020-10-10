#ifndef __BASELINE_CPU_GET_RATES__
#define __BASELINE_CPU_GET_RATES__
#include "chemistry_file.H"

#define GETRATES_NEEDS_DIFFUSION

#define GETRATES_STIFF_SPECIES 93

static AMREX_GPU_DEVICE_MANAGED unsigned int stif_species_indexes[93] = {0, 2, 4, 6, 7, 10, 11, 12, 13, 
  14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
  35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 
  56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 74, 75, 76, 77, 
  78, 79, 80, 81, 82, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 98, 100, 103, 104, 
  107, 108, 109, 110, 111, 112, 113, 114}; 

#ifdef __cplusplus
extern "C" {
#endif
AMREX_GPU_HOST_DEVICE
void base_getrates(const double pressure, const double temperature, const double 
  avmolwt, const double *mass_frac, const double *diffusion, const double dt, 
  double *wdot); 
#ifdef __cplusplus
}
#endif
#endif // __BASELINE_CPU_GET_RATES__
