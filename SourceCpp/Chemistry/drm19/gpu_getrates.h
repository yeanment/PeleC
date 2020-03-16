#ifndef __BASELINE_GPU_GET_RATES__
#define __BASELINE_GPU_GET_RATES__

__global__ void
gpu_getrates(const double *temperature_array, const double *pressure_array, 
  const double *avmolwt_array, const double *mass_frac_array, const int 
  slice_stride/*NX*NY in number of doubles*/, const int row_stride/*NX in number 
  of doubles*/, const int total_steps/*NZ in number of doubles*/, const int 
  spec_stride/*NX*NY*NZ in number of doubles*/, const int step_stride/*always 
  zero*/, double *wdot_array); 
#endif // __BASELINE_GPU_GET_RATES__
