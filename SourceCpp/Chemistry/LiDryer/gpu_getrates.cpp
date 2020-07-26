#include "chemistry_file.H"
#include "gpu_getrates.h"

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

static AMREX_GPU_DEVICE_MANAGED double molecular_masses[9] = {2.01594, 31.9988, 18.01534, 1.00797, 
  15.9994, 17.00737, 33.00677, 34.01474, 28.0134}; 

static AMREX_GPU_DEVICE_MANAGED double recip_molecular_masses[9] = {0.4960465093207139, 
  0.03125117191894696, 0.05550825019122593, 0.9920930186414277, 
  0.06250234383789392, 0.05879803873262004, 0.03029681486555637, 
  0.02939901936631002, 0.03569720205330306}; 


AMREX_GPU_GLOBAL
void
gpu_getrates(const double * temperature_array, const double * pressure_array, 
  const double * avmolwt_array, const double *mass_frac_array, const int 
  spec_stride/*NX*NY*NZ in number of doubles*/, double *wdot_array) 
{
  
  const double PA = 1.013250e+06;
  const double R0 = 8.314510e+07;
  const double R0c = 1.9872155832;
  const double DLn10 = 2.3025850929940459e0;
  
  {
    const int offset = (blockIdx.x*blockDim.x + threadIdx.x);
    temperature_array += offset;
    pressure_array += offset;
    avmolwt_array += offset;
    mass_frac_array += offset;
    wdot_array += offset;
  }
  double temperature;
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(temperature) : 
    "l"(temperature_array) : "memory"); 
  const double otc     = 1.0 / temperature;
  const double ortc    = 1.0 / (temperature * R0c);
  const double vlntemp = log(temperature);
  const double prt     = PA / (R0 * temperature);
  const double oprt    = 1.0 / prt;
  
  double mass_frac[9];
  double avmolwt;
  double pressure;
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[0]) : 
    "l"(mass_frac_array+0*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[1]) : 
    "l"(mass_frac_array+1*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[2]) : 
    "l"(mass_frac_array+2*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[3]) : 
    "l"(mass_frac_array+3*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[4]) : 
    "l"(mass_frac_array+4*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[5]) : 
    "l"(mass_frac_array+5*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[6]) : 
    "l"(mass_frac_array+6*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[7]) : 
    "l"(mass_frac_array+7*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(mass_frac[8]) : 
    "l"(mass_frac_array+8*spec_stride) : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(avmolwt) : "l"(avmolwt_array) 
    : "memory"); 
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(pressure) : 
    "l"(pressure_array) : "memory"); 
  double cgspl[9];
  // Gibbs computation
  {
    const double &tk1 = temperature;
    double tklog = log(tk1);
    double tk2 = tk1 * tk1;
    double tk3 = tk1 * tk2;
    double tk4 = tk1 * tk3;
    double tk5 = tk1 * tk4;
    
    // Species H2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[0] = 2.991423*tk1*(1-tklog) + -3.500322e-04*tk2 + 9.389715e-09*tk3 
          + 7.692981666666665e-13*tk4 + -7.913759999999998e-17*tk5 + (-835.034 - 
          tk1*-1.35511); 
      }
      else
      {
        cgspl[0] = 3.298124*tk1*(1-tklog) + -4.124721e-04*tk2 + 
          1.357169166666667e-07*tk3 + 7.896194999999997e-12*tk4 + 
          -2.067436e-14*tk5 + (-1.012521e+03 - tk1*-3.294094); 
      }
    }
    // Species O2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[1] = 3.697578*tk1*(1-tklog) + -3.0675985e-04*tk2 + 2.09807e-08*tk3 
          + -1.479400833333333e-12*tk4 + 5.682175e-17*tk5 + (-1.23393e+03 - 
          tk1*3.189166); 
      }
      else
      {
        cgspl[1] = 3.212936*tk1*(1-tklog) + -5.63743e-04*tk2 + 
          9.593583333333335e-08*tk3 + -1.0948975e-10*tk4 + 
          4.384276999999999e-14*tk5 + (-1.005249e+03 - tk1*6.034738); 
      }
    }
    // Species H2O
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[2] = 2.672146*tk1*(1-tklog) + -1.5281465e-03*tk2 + 
          1.455043333333333e-07*tk3 + -1.00083e-11*tk4 + 
          3.195808999999999e-16*tk5 + (-2.989921e+04 - tk1*6.862817); 
      }
      else
      {
        cgspl[2] = 3.386842*tk1*(1-tklog) + -1.737491e-03*tk2 + 1.059116e-06*tk3 
          + -5.807150833333332e-10*tk4 + 1.253294e-13*tk5 + (-3.020811e+04 - 
          tk1*2.590233); 
      }
    }
    // Species H
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[3] = 2.5*tk1*(1-tklog) + -0.0*tk2 + -0.0*tk3 + -0.0*tk4 + -0.0*tk5 
          + (2.547163e+04 - tk1*-0.4601176); 
      }
      else
      {
        cgspl[3] = 2.5*tk1*(1-tklog) + -0.0*tk2 + -0.0*tk3 + -0.0*tk4 + -0.0*tk5 
          + (2.547163e+04 - tk1*-0.4601176); 
      }
    }
    // Species O
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[4] = 2.54206*tk1*(1-tklog) + 1.377531e-05*tk2 + 
          5.171338333333334e-10*tk3 + -3.792555833333332e-13*tk4 + 
          2.184026e-17*tk5 + (2.92308e+04 - tk1*4.920308); 
      }
      else
      {
        cgspl[4] = 2.946429*tk1*(1-tklog) + 8.19083e-04*tk2 + 
          -4.035053333333334e-07*tk3 + 1.3357025e-10*tk4 + -1.945348e-14*tk5 + 
          (2.914764e+04 - tk1*2.963995); 
      }
    }
    // Species OH
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[5] = 2.86472886*tk1*(1-tklog) + -5.2825224e-04*tk2 + 
          4.318045966666668e-08*tk3 + -2.543488949999999e-12*tk4 + 
          6.659793799999999e-17*tk5 + (3.68362875e+03 - tk1*5.70164073); 
      }
      else
      {
        cgspl[5] = 4.12530561*tk1*(1-tklog) + 1.612724695e-03*tk2 + 
          -1.087941151666667e-06*tk3 + 4.832113691666665e-10*tk4 + 
          -1.031186895e-13*tk5 + (3.34630913e+03 - tk1*-0.69043296); 
      }
    }
    // Species HO2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[6] = 4.0172109*tk1*(1-tklog) + -1.119910065e-03*tk2 + 
          1.056096916666667e-07*tk3 + -9.520530833333331e-12*tk4 + 
          5.395426749999999e-16*tk5 + (111.856713 - tk1*3.78510215); 
      }
      else
      {
        cgspl[6] = 4.30179801*tk1*(1-tklog) + 2.374560255e-03*tk2 + 
          -3.526381516666667e-06*tk3 + 2.02303245e-09*tk4 + -4.64612562e-13*tk5 
          + (294.80804 - tk1*3.71666245); 
      }
    }
    // Species H2O2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[7] = 4.573167*tk1*(1-tklog) + -2.168068e-03*tk2 + 2.457815e-07*tk3 
          + -1.957419999999999e-11*tk4 + 7.158269999999999e-16*tk5 + 
          (-1.800696e+04 - tk1*0.5011370000000001); 
      }
      else
      {
        cgspl[7] = 3.388754*tk1*(1-tklog) + -3.284613e-03*tk2 + 
          2.475021666666667e-08*tk3 + 3.854838333333332e-10*tk4 + 
          -1.2357575e-13*tk5 + (-1.766315e+04 - tk1*6.785363); 
      }
    }
    // Species N2
    {
      if (tk1 > 1.0e+03)
      {
        cgspl[8] = 2.92664*tk1*(1-tklog) + -7.439885e-04*tk2 + 
          9.474601666666666e-08*tk3 + -8.414199999999998e-12*tk4 + 
          3.376675499999999e-16*tk5 + (-922.7977 - tk1*5.980528); 
      }
      else
      {
        cgspl[8] = 3.298677*tk1*(1-tklog) + -7.0412e-04*tk2 + 6.60537e-07*tk3 + 
          -4.7012625e-10*tk4 + 1.2224275e-13*tk5 + (-1.0209e+03 - tk1*3.950372); 
      }
    }
  }
  
  double mole_frac[9];
  // Compute mole fractions
  {
    double sumyow = temperature * avmolwt * R0;
    sumyow = pressure/sumyow;
    mole_frac[0] = mass_frac[0] * recip_molecular_masses[0];
    mole_frac[0] = (mole_frac[0] > 1e-200) ? mole_frac[0] : 1e-200;
    mole_frac[0] *= sumyow;
    mole_frac[1] = mass_frac[1] * recip_molecular_masses[1];
    mole_frac[1] = (mole_frac[1] > 1e-200) ? mole_frac[1] : 1e-200;
    mole_frac[1] *= sumyow;
    mole_frac[2] = mass_frac[2] * recip_molecular_masses[2];
    mole_frac[2] = (mole_frac[2] > 1e-200) ? mole_frac[2] : 1e-200;
    mole_frac[2] *= sumyow;
    mole_frac[3] = mass_frac[3] * recip_molecular_masses[3];
    mole_frac[3] = (mole_frac[3] > 1e-200) ? mole_frac[3] : 1e-200;
    mole_frac[3] *= sumyow;
    mole_frac[4] = mass_frac[4] * recip_molecular_masses[4];
    mole_frac[4] = (mole_frac[4] > 1e-200) ? mole_frac[4] : 1e-200;
    mole_frac[4] *= sumyow;
    mole_frac[5] = mass_frac[5] * recip_molecular_masses[5];
    mole_frac[5] = (mole_frac[5] > 1e-200) ? mole_frac[5] : 1e-200;
    mole_frac[5] *= sumyow;
    mole_frac[6] = mass_frac[6] * recip_molecular_masses[6];
    mole_frac[6] = (mole_frac[6] > 1e-200) ? mole_frac[6] : 1e-200;
    mole_frac[6] *= sumyow;
    mole_frac[7] = mass_frac[7] * recip_molecular_masses[7];
    mole_frac[7] = (mole_frac[7] > 1e-200) ? mole_frac[7] : 1e-200;
    mole_frac[7] *= sumyow;
    mole_frac[8] = mass_frac[8] * recip_molecular_masses[8];
    mole_frac[8] = (mole_frac[8] > 1e-200) ? mole_frac[8] : 1e-200;
    mole_frac[8] *= sumyow;
  }
  
  double thbctemp[2];
  // Computing third body values
  {
    double ctot = 0.0;
    ctot += mole_frac[0];
    ctot += mole_frac[1];
    ctot += mole_frac[2];
    ctot += mole_frac[3];
    ctot += mole_frac[4];
    ctot += mole_frac[5];
    ctot += mole_frac[6];
    ctot += mole_frac[7];
    ctot += mole_frac[8];
    thbctemp[0] = ctot + 1.5*mole_frac[0] + 11.0*mole_frac[2];
    thbctemp[1] = ctot + mole_frac[0] - 0.22*mole_frac[1] + 10.0*mole_frac[2];
  }
  
  double rr_f[19];
  double rr_r[19];
  //   0)  O2 + H <=> O + OH
  {
    double forward = 3.547e+15 * exp(-0.406*vlntemp - 1.6599e+04*ortc);
    double xik = -cgspl[1] - cgspl[3] + cgspl[4] + cgspl[5];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[0] = forward * mole_frac[1] * mole_frac[3];
    rr_r[0] = reverse * mole_frac[4] * mole_frac[5];
  }
  //   1)  H2 + O <=> H + OH
  {
    double forward = 5.08e+04 * exp(2.67*vlntemp - 6.29e+03*ortc);
    double xik = -cgspl[0] + cgspl[3] - cgspl[4] + cgspl[5];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[1] = forward * mole_frac[0] * mole_frac[4];
    rr_r[1] = reverse * mole_frac[3] * mole_frac[5];
  }
  //   2)  H2 + OH <=> H2O + H
  {
    double forward = 2.16e+08 * exp(1.51*vlntemp - 3.43e+03*ortc);
    double xik = -cgspl[0] + cgspl[2] + cgspl[3] - cgspl[5];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[2] = forward * mole_frac[0] * mole_frac[5];
    rr_r[2] = reverse * mole_frac[2] * mole_frac[3];
  }
  //   3)  H2O + O <=> 2 OH
  {
    double forward = 2.97e+06 * exp(2.02*vlntemp - 1.34e+04*ortc);
    double xik = -cgspl[2] - cgspl[4] + 2.0 * cgspl[5];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[3] = forward * mole_frac[2] * mole_frac[4];
    rr_r[3] = reverse * mole_frac[5] * mole_frac[5];
  }
  //   4)  H2 + M <=> 2 H + M
  {
    double forward = 4.577e+19 * exp(-1.4*vlntemp - 1.0438e+05*ortc);
    double xik = -cgspl[0] + 2.0 * cgspl[3];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[4] = forward * mole_frac[0];
    rr_r[4] = reverse * mole_frac[3] * mole_frac[3];
    rr_f[4] *= thbctemp[0];
    rr_r[4] *= thbctemp[0];
  }
  //   5)  2 O + M <=> O2 + M
  {
    double forward = 6.165e+15 * exp(-0.5 * vlntemp);
    double xik = cgspl[1] - 2.0 * cgspl[4];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[5] = forward * mole_frac[4] * mole_frac[4];
    rr_r[5] = reverse * mole_frac[1];
    rr_f[5] *= thbctemp[0];
    rr_r[5] *= thbctemp[0];
  }
  //   6)  H + O + M <=> OH + M
  {
    double forward = 4.714e+18 * otc;
    double xik = -cgspl[3] - cgspl[4] + cgspl[5];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[6] = forward * mole_frac[3] * mole_frac[4];
    rr_r[6] = reverse * mole_frac[5];
    rr_f[6] *= thbctemp[0];
    rr_r[6] *= thbctemp[0];
  }
  //   7)  H + OH + M <=> H2O + M
  {
    double forward = 3.8e+22 * otc * otc;
    double xik = cgspl[2] - cgspl[3] - cgspl[5];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[7] = forward * mole_frac[3] * mole_frac[5];
    rr_r[7] = reverse * mole_frac[2];
    rr_f[7] *= thbctemp[0];
    rr_r[7] *= thbctemp[0];
  }
  //   8)  O2 + H (+M) <=> HO2 (+M)
  {
    double rr_k0 = 6.366e+20 * exp(-1.72*vlntemp - 524.8*ortc);
    double rr_kinf = 1.475e+12 * exp(0.6 * vlntemp);
    double pr = rr_k0 / rr_kinf * thbctemp[1];
    double fcent = log10(MAX(0.2 * exp(-9.999999999999999e+29 * temperature) + 
      0.8 * exp(-9.999999999999999e-31 * temperature),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = -cgspl[1] - cgspl[3] + cgspl[6];
    double reverse = forward * MIN(exp(xik*otc) * prt,1e200);
    rr_f[8] = forward * mole_frac[1] * mole_frac[3];
    rr_r[8] = reverse * mole_frac[6];
  }
  //   9)  H + HO2 <=> H2 + O2
  {
    double forward = 1.66e+13 * exp(-823.0*ortc);
    double xik = cgspl[0] + cgspl[1] - cgspl[3] - cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[9] = forward * mole_frac[3] * mole_frac[6];
    rr_r[9] = reverse * mole_frac[0] * mole_frac[1];
  }
  //  10)  H + HO2 <=> 2 OH
  {
    double forward = 7.079e+13 * exp(-295.0*ortc);
    double xik = -cgspl[3] + 2.0 * cgspl[5] - cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[10] = forward * mole_frac[3] * mole_frac[6];
    rr_r[10] = reverse * mole_frac[5] * mole_frac[5];
  }
  //  11)  O + HO2 <=> O2 + OH
  {
    double forward = 3.25e+13;
    double xik = cgspl[1] - cgspl[4] + cgspl[5] - cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[11] = forward * mole_frac[4] * mole_frac[6];
    rr_r[11] = reverse * mole_frac[1] * mole_frac[5];
  }
  //  12)  OH + HO2 <=> O2 + H2O
  {
    double forward = 2.89e+13 * exp(497.0*ortc);
    double xik = cgspl[1] + cgspl[2] - cgspl[5] - cgspl[6];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[12] = forward * mole_frac[5] * mole_frac[6];
    rr_r[12] = reverse * mole_frac[1] * mole_frac[2];
  }
  //  13, 14)  2 HO2 <=> O2 + H2O2
  {
    double forward = 4.2e+14 * exp(-1.1982e+04*ortc);
    forward = forward + 1.3e+11 * exp(1.6293e+03*ortc);
    double xik = cgspl[1] - 2.0 * cgspl[6] + cgspl[7];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[13] = forward * mole_frac[6] * mole_frac[6];
    rr_r[13] = reverse * mole_frac[1] * mole_frac[7];
  }
  //  14)  H2O2 (+M) <=> 2 OH (+M)
  {
    double rr_k0 = 1.202e+17 * exp(-4.55e+04*ortc);
    double rr_kinf = 2.951e+14 * exp(-4.843e+04*ortc);
    double pr = rr_k0 / rr_kinf * thbctemp[0];
    double fcent = log10(MAX(0.5 * exp(-9.999999999999999e+29 * temperature) + 
      0.5 * exp(-9.999999999999999e-31 * temperature),1e-200)); 
    double flogpr = log10(MAX(pr,1e-200)) - 0.4 - 0.67 * fcent;
    double fdenom = 0.75 - 1.27 * fcent - 0.14 * flogpr;
    double fquan = flogpr / fdenom;
    fquan = fcent / (1.0 + fquan * fquan);
    double forward = rr_kinf * pr/(1.0 + pr) * exp(fquan*DLn10);
    double xik = 2.0 * cgspl[5] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc) * oprt,1e200);
    rr_f[14] = forward * mole_frac[7];
    rr_r[14] = reverse * mole_frac[5] * mole_frac[5];
  }
  //  15)  H + H2O2 <=> H2O + OH
  {
    double forward = 2.41e+13 * exp(-3.97e+03*ortc);
    double xik = cgspl[2] - cgspl[3] + cgspl[5] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[15] = forward * mole_frac[3] * mole_frac[7];
    rr_r[15] = reverse * mole_frac[2] * mole_frac[5];
  }
  //  16)  H + H2O2 <=> H2 + HO2
  {
    double forward = 4.82e+13 * exp(-7.95e+03*ortc);
    double xik = cgspl[0] - cgspl[3] + cgspl[6] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[16] = forward * mole_frac[3] * mole_frac[7];
    rr_r[16] = reverse * mole_frac[0] * mole_frac[6];
  }
  //  17)  O + H2O2 <=> OH + HO2
  {
    double forward = 9.55e+06 * temperature * temperature * exp(-3.97e+03*ortc);
    double xik = -cgspl[4] + cgspl[5] + cgspl[6] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[17] = forward * mole_frac[4] * mole_frac[7];
    rr_r[17] = reverse * mole_frac[5] * mole_frac[6];
  }
  //  18, 20)  OH + H2O2 <=> H2O + HO2
  {
    double forward = 1.0e+12;
    forward = forward + 5.8e+14 * exp(-9.557e+03*ortc);
    double xik = cgspl[2] - cgspl[5] + cgspl[6] - cgspl[7];
    double reverse = forward * MIN(exp(xik*otc),1e200);
    rr_f[18] = forward * mole_frac[5] * mole_frac[7];
    rr_r[18] = reverse * mole_frac[2] * mole_frac[6];
  }
  double wdot[9];
  double ropl[19];
  for (int i = 0; i < 19; i++)
  {
    ropl[i] = rr_f[i] - rr_r[i];
  }
  // 0. H2
  wdot[0] = -ropl[1] - ropl[2] - ropl[4] + ropl[9] + ropl[16];
  // 1. O2
  wdot[1] = -ropl[0] + ropl[5] - ropl[8] + ropl[9] + ropl[11] + ropl[12] + 
    ropl[13]; 
  // 2. H2O
  wdot[2] = ropl[2] - ropl[3] + ropl[7] + ropl[12] + ropl[15] + ropl[18];
  // 3. H
  wdot[3] = -ropl[0] + ropl[1] + ropl[2] + 2.0*ropl[4] - ropl[6] - ropl[7] - 
    ropl[8] - ropl[9] - ropl[10] - ropl[15] - ropl[16]; 
  // 4. O
  wdot[4] = ropl[0] - ropl[1] - ropl[3] - 2.0*ropl[5] - ropl[6] - ropl[11] - 
    ropl[17]; 
  // 5. OH
  wdot[5] = ropl[0] + ropl[1] - ropl[2] + 2.0*ropl[3] + ropl[6] - ropl[7] + 
    2.0*ropl[10] + ropl[11] - ropl[12] + 2.0*ropl[14] + ropl[15] + ropl[17] - 
    ropl[18]; 
  // 6. HO2
  wdot[6] = ropl[8] - ropl[9] - ropl[10] - ropl[11] - ropl[12] - 2.0*ropl[13] + 
    ropl[16] + ropl[17] + ropl[18]; 
  // 7. H2O2
  wdot[7] = ropl[13] - ropl[14] - ropl[15] - ropl[16] - ropl[17] - ropl[18];
  // 8. N2
  wdot[8] = 0.0;
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+0*spec_stride) , 
    "d"(wdot[0]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+1*spec_stride) , 
    "d"(wdot[1]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+2*spec_stride) , 
    "d"(wdot[2]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+3*spec_stride) , 
    "d"(wdot[3]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+4*spec_stride) , 
    "d"(wdot[4]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+5*spec_stride) , 
    "d"(wdot[5]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+6*spec_stride) , 
    "d"(wdot[6]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+7*spec_stride) , 
    "d"(wdot[7]) : "memory"); 
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(wdot_array+8*spec_stride) , 
    "d"(wdot[8]) : "memory"); 
}

