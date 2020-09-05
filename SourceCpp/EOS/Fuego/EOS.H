#ifndef _EOS_H_
#define _EOS_H_

//#include <cmath>

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include "mechanism.h"
#include "chemistry_file.H"
#include "gpu_getrates.h"
#include "getrates.h"
#include "base_getrates.h"
#include <iostream>
#include "IndexDefines.H"
#include "Constants.H"

extern "C" {
AMREX_GPU_HOST_DEVICE void get_imw(amrex::Real imw[]);
AMREX_GPU_HOST_DEVICE void get_mw(amrex::Real mw[]);
AMREX_GPU_HOST_DEVICE void CKPY(amrex::Real* rho, amrex::Real* T, amrex::Real* y, amrex::Real* P);
AMREX_GPU_HOST_DEVICE void CKCVMS(amrex::Real* T, amrex::Real* cvms);
AMREX_GPU_HOST_DEVICE void CKCVBS(amrex::Real* T, amrex::Real* massfrac, amrex::Real* cv);
AMREX_GPU_HOST_DEVICE void CKCPMS(amrex::Real* T, amrex::Real* cvms);
AMREX_GPU_HOST_DEVICE void CKUMS(amrex::Real* T, amrex::Real* ums);
AMREX_GPU_HOST_DEVICE void CKHMS(amrex::Real* T, amrex::Real* ums);
AMREX_GPU_HOST_DEVICE void CKWYR(amrex::Real* rho, amrex::Real* T, amrex::Real* y, amrex::Real* wdot);
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_EY(amrex::Real* e, amrex::Real* y, amrex::Real* t, int* ierr);
AMREX_GPU_HOST_DEVICE void CKYTX(amrex::Real massfrac[], amrex::Real molefrac[]);
AMREX_GPU_HOST_DEVICE void CKXTY(amrex::Real molefrac[], amrex::Real massfrac[]);
AMREX_GPU_HOST_DEVICE void CKRHOY(amrex::Real* pres, amrex::Real* T, amrex::Real massfrac[], amrex::Real* rho);
}

namespace EOS {

void init();
/*
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void bottom()
{
    CKCVMS(&T,  cvi);
    CKCPMS(&T,  cpi);
    CKHMS(&T,   hi);
    cv = 0.0, cp = 0.0, h = 0.0;
    for(int i = 0; i < NUM_SPECIES; ++i){
         cv+=massfrac[i]*cvi[i];
         cp+=massfrac[i]*cpi[i];
         hi +=massfrac[i]* hi[i];
    }
    amrex::Real Cvx = wbar*cv;
    gam1 = (Cvx + RU)/Cvx;
    cs = std::sqrt(gam1*p/rho);
    dpdr_e = p/rho;
    dpde = (gam1 - 1.0)*rho;
    s = 1.0;
    dpdr = 0.0;
}
*/

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_cp(amrex::Real T, amrex::Real mass[], amrex::Real& cp)
{
  amrex::Real temp[NUM_SPECIES]; // cpi
  CKCPMS(&T, temp);
  cp = 0.0;
  for (int i = 0; i < NUM_SPECIES; ++i)
    cp += mass[i] * temp[i];
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_wb(amrex::Real mass[], amrex::Real& wbar)
{
  amrex::Real imw[NUM_SPECIES];
  get_imw(imw);
  amrex::Real summ = 0.0;
  for (int i = 0; i < NUM_SPECIES; ++i)
    summ += mass[i] * imw[i];
  wbar = 1.0 / summ ;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_cs_no_T(amrex::Real rho, amrex::Real p, amrex::Real mass[], amrex::Real& cs)
{
  amrex::Real temp[NUM_SPECIES];
  amrex::Real wbar;
  get_wb(mass, wbar);
  amrex::Real T = p * wbar / (rho * RU);
  CKCVMS(&T, temp); // temp = cvi
  amrex::Real cv = 0.0;
  for (int i = 0; i < NUM_SPECIES; ++i)
    cv += mass[i] * temp[i];
  amrex::Real ga = (wbar * cv + RU) / (wbar * cv);
  cs = std::sqrt(ga * p / rho);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_cs(amrex::Real rho, amrex::Real T, amrex::Real mass[], amrex::Real& cs)
{
  amrex::Real pres;
  CKPY(&rho, &T, mass, &pres);
  amrex::Real temp[NUM_SPECIES];
  CKCVMS(&T, temp); // temp = cvi
  amrex::Real cv = 0.0;
  for (int i = 0; i < NUM_SPECIES; ++i)
    cv += mass[i] * temp[i];
  amrex::Real wbar;
  get_wb(mass, wbar);
  amrex::Real ga = (wbar * cv + RU) / (wbar * cv);
  cs = std::sqrt(ga * pres / rho);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
cmpT(amrex::Real e, amrex::Real massfrac[], amrex::Real& T)
{
  // For Fuego this function is really just a wrapper for GET_T_GIVEN_EY
  // In SRK this will be different probably
  int lierr = 0;
  GET_T_GIVEN_EY(&e, massfrac, &T, &lierr);
}

/*Prototype for moving EOS to be a namespace instead of a class */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
ctop(
  amrex::Real massfrac[],
  amrex::Real rho,
  amrex::Real& e,
  amrex::Real& T,
  amrex::Real& p,
  amrex::Real& dpdr_e,
  amrex::Real& dpde,
  amrex::Real& gam1,
  amrex::Real& cs,
  amrex::Real& wbar)
{
  int lierr = 0;
  amrex::Real temp[NUM_SPECIES];
  // here temp is the inverse molecular weights
  get_imw(temp);
  amrex::Real summ = 0.0;
  for (int i = 0; i < NUM_SPECIES; ++i)
    summ += massfrac[i] * temp[i];
  wbar = 1.0 / summ;
  //  get Temperature
  GET_T_GIVEN_EY(&e, massfrac, &T, &lierr);
  //  get pressure
  CKPY(&rho, &T, massfrac, &p);

  CKCVMS(&T, temp); // temp changes from imw to cvi
  amrex::Real cv = 0.0;
  for (int i = 0; i < NUM_SPECIES; ++i) {
    cv += massfrac[i] * temp[i];
  }
  amrex::Real Cvx = wbar * cv;
  gam1 = (Cvx + RU) / Cvx;
  cs = std::sqrt(gam1 * p / rho);
  dpdr_e = p / rho;
  dpde = (gam1 - 1.0) * rho;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_p(
  amrex::Real rho,
  amrex::Real massfrac[],
  amrex::Real& e,
  amrex::Real& T,
  amrex::Real& p)
{
  int lierr = 0;
  GET_T_GIVEN_EY(&e, massfrac, &T, &lierr);
  CKPY(&rho, &T, massfrac, &p);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
TrhoY2p(amrex::Real dens, amrex::Real T, amrex::Real mass[], amrex::Real& pres)
{
  CKPY(&dens, &T, mass, &pres);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
mpr2wdot(
  amrex::Real dens, amrex::Real T, amrex::Real mass[], amrex::Real wdot[], int NX)
{
  //CKWYR(&dens, &T, mass, wdot);
  
  amrex::Real P; 
  CKPY(&dens, &T, mass, &P);
  amrex::Real wbar;
  get_wb(mass, wbar);
  wbar = 1.0/wbar; 
  amrex::Real mw[NUM_SPECIES];
  get_mw(mw);
  base_getrates (P, T, wbar, mass, wdot);
  for (int n = 0; n < NUM_SPECIES; n++)  wdot[n] *= mw[n];
  
  /*amrex::Real *Yd, *Yh, *wdotd, *Pd, *Ph, *Td, *Th, *wd, *wh;

  Yh = (amrex::Real*)malloc(NUM_SPECIES*sizeof(amrex::Real));
  Ph = (amrex::Real*)malloc(sizeof(amrex::Real));
  Th = (amrex::Real*)malloc(sizeof(amrex::Real));
  wh = (amrex::Real*)malloc(sizeof(amrex::Real));
  
  for (int i=0; i<NUM_SPECIES; i++) {
    Yh[i] = mass[i]; 
  }
    
  *Ph = P; *Th = T; *wh = wbar;
  cudaMalloc((void **) &Yd, NUM_SPECIES*sizeof(amrex::Real));
  cudaMalloc((void **) &wdotd, NUM_SPECIES*sizeof(amrex::Real));
  cudaMalloc((void **) &Pd, sizeof(amrex::Real));
  cudaMalloc((void **) &Td, sizeof(amrex::Real));
  cudaMalloc((void **) &wd, sizeof(amrex::Real));
  

  cudaMemcpyAsync(Yd, Yh, NUM_SPECIES*sizeof(amrex::Real), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
    
  cudaMemcpyAsync(Pd, Ph, sizeof(amrex::Real), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  
  cudaMemcpyAsync(Td, Th, sizeof(amrex::Real), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  
  cudaMemcpyAsync(wd, wh, sizeof(amrex::Real), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  
  //cudaError_t err = cudaGetLastError();
  //if(err!=cudaSuccess)  {
  //  printf("Cuda Error: %s\n", cudaGetErrorString(err));
  //}
  
  //dim3 grid(NUM_SPECIES,1);
  //getrates <<<1, 1>>> (*Pd, *Td, *wd, Yd, wdotd);
  //gpu_getrates <<<1, 1>>> (Td, Pd, wd, Yd, 1, wdotd);
  cudaDeviceSynchronize();
  cudaFree(Yd); cudaFree(Pd); cudaFree(Td); cudaFree(wd); free(Yh); free(Ph); free(Th); free(wh);
  
  for (int n = 0; n < NUM_SPECIES; n++)  wdot[n] = wdotd[n]*mw[n];
  
  cudaFree(wdotd);*/  

}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_ei(amrex::Real T, amrex::Real enrgi[])
{
  CKUMS(&T, enrgi);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_cv(amrex::Real mass[], amrex::Real T, amrex::Real& cv)
{
  cv = 0.0;
  amrex::Real temp[NUM_SPECIES];
  CKCVMS(&T, temp); // here temp is cvi
  for (int i = 0; i < NUM_SPECIES; ++i)
    cv += mass[i] * temp[i];
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
ytx(amrex::Real mass[], amrex::Real mole[])
{
  CKYTX(mass, mole);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
xty(amrex::Real mole[], amrex::Real mass[])
{
  CKXTY(mole, mass);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
tp(
  amrex::Real pres,
  amrex::Real massfrac[],
  amrex::Real T,
  amrex::Real& rho,
  amrex::Real& energy)
{
  CKRHOY(&pres, &T, massfrac, &rho);
  amrex::Real ei[NUM_SPECIES];
  CKUMS(&T, ei);
  energy = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++)
    energy += massfrac[n] * ei[n];
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
rhopY2e(
  amrex::Real rho, amrex::Real massfrac[], amrex::Real p, amrex::Real& energy)
{
  amrex::Real wbar;
  get_wb(massfrac, wbar);
  amrex::Real T = p * wbar / (rho * RU);
  amrex::Real ei[NUM_SPECIES];
  CKUMS(&T, ei);
  energy = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++)
    energy += massfrac[n] * ei[n];
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_hi(amrex::Real mass[], amrex::Real T, amrex::Real enthalpy[])
{
  CKHMS(&T, enthalpy);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_s(amrex::Real& s)
{
  s = 1.0;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
get_gamma(amrex::Real e, amrex::Real& gamma)
{
  amrex::Real wbar, cv, T, Cvx;
  amrex::Real massfrac[NUM_SPECIES];
  cmpT(e, massfrac, T);
  get_cv(massfrac, T, cv);
  get_wb(massfrac, wbar);
  Cvx = wbar * cv;
  gamma = (Cvx + RU) / Cvx;
}

// Hydro -> Advected -> Species -> Aux
// If num_adv == 0 -> QFA = QFS and UFA = UFS, see IndexDefines.H
// For explicit definitions.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
int
upass_map(const int i)
{
  return i + UFA;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
int
qpass_map(const int i)
{
  /*V and W are passive*/
  return i + QFA;
}

}; // namespace EOS

void pc_eos_init();

#endif
