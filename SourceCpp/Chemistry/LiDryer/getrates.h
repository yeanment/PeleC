#ifndef __BASELINE_CPU_GET_RATES__
#define __BASELINE_CPU_GET_RATES__

/*#ifndef __SINGE_MOLE_MASSES__
#define __SINGE_MOLE_MASSES__
const double molecular_masses[9] = {2.01594, 31.9988, 18.01534, 1.00797, 
  15.9994, 17.00737, 33.00677, 34.01474, 28.0134}; 
#endif


#ifndef __SINGE_RECIP_MOLE_MASSES__
#define __SINGE_RECIP_MOLE_MASSES__
const double recip_molecular_masses[9] = {0.4960465093207139, 
  0.03125117191894696, 0.05550825019122593, 0.9920930186414277, 
  0.06250234383789392, 0.05879803873262004, 0.03029681486555637, 
  0.02939901936631002, 0.03569720205330306}; 
#endif*/
#include "chemistry_file.H"

#ifdef __cplusplus
extern "C" {
#endif
AMREX_GPU_HOST_DEVICE
void getrates(const double pressure, const double temperature, const double 
  avmolwt, const double *mass_frac, double *wdot); 
#ifdef __cplusplus
}
#endif
#endif // __BASELINE_CPU_GET_RATES__
