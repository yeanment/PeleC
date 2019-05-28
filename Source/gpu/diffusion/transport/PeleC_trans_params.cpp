#include "mechanism.h"
#include "trans_params.H"
#include <AMReX_Arena.H>
#include <cstdio>

#if defined(BL_FORT_USE_UPPERCASE)
#define egtransetWT EGTRANSETWT
#elif defined(BL_FORT_USE_LOWERCASE)
#define egtransetWT egtransetwt
#elif defined(BL_FORT_USE_UNDERSCORE)
#define egtransetWT egtransetwt_
#endif
#if defined(BL_FORT_USE_UPPERCASE)
#define egtransetEPS EGTRANSETEPS
#elif defined(BL_FORT_USE_LOWERCASE)
#define egtransetEPS egtranseteps
#elif defined(BL_FORT_USE_UNDERSCORE)
#define egtransetEPS egtranseteps_
#endif
#if defined(BL_FORT_USE_UPPERCASE)
#define egtransetSIG EGTRANSETSIG
#elif defined(BL_FORT_USE_LOWERCASE)
#define egtransetSIG egtransetsig
#elif defined(BL_FORT_USE_UNDERSCORE)
#define egtransetSIG egtransetsig_
#endif
#if defined(BL_FORT_USE_UPPERCASE)
#define egtransetDIP EGTRANSETDIP
#elif defined(BL_FORT_USE_LOWERCASE)
#define egtransetDIP egtransetdip
#elif defined(BL_FORT_USE_UNDERSCORE)
#define egtransetDIP egtransetdip_
#endif
#if defined(BL_FORT_USE_UPPERCASE)
#define egtransetPOL EGTRANSETPOL
#elif defined(BL_FORT_USE_LOWERCASE)
#define egtransetPOL egtransetpol
#elif defined(BL_FORT_USE_UNDERSCORE)
#define egtransetPOL egtransetpol_
#endif
#if defined(BL_FORT_USE_UPPERCASE)
#define egtransetZROT EGTRANSETZROT
#elif defined(BL_FORT_USE_LOWERCASE)
#define egtransetZROT egtransetzrot
#elif defined(BL_FORT_USE_UNDERSCORE)
#define egtransetZROT egtransetzrot_
#endif
#if defined(BL_FORT_USE_UPPERCASE)
#define egtransetNLIN EGTRANSETNLIN
#elif defined(BL_FORT_USE_LOWERCASE)
#define egtransetNLIN egtransetnlin
#elif defined(BL_FORT_USE_UNDERSCORE)
#define egtransetNLIN egtransetnlin_
#endif
#if defined(BL_FORT_USE_UPPERCASE)
#define egtransetCOFETA EGTRANSETCOFETA
#elif defined(BL_FORT_USE_LOWERCASE)
#define egtransetCOFETA egtransetcofeta
#elif defined(BL_FORT_USE_UNDERSCORE)
#define egtransetCOFETA egtransetcofeta_
#endif
#if defined(BL_FORT_USE_UPPERCASE)
#define egtransetCOFLAM EGTRANSETCOFLAM
#elif defined(BL_FORT_USE_LOWERCASE)
#define egtransetCOFLAM egtransetcoflam
#elif defined(BL_FORT_USE_UNDERSCORE)
#define egtransetCOFLAM egtransetcoflam_
#endif
#if defined(BL_FORT_USE_UPPERCASE)
#define egtransetCOFD EGTRANSETCOFD
#elif defined(BL_FORT_USE_LOWERCASE)
#define egtransetCOFD egtransetcofd
#elif defined(BL_FORT_USE_UNDERSCORE)
#define egtransetCOFD egtransetcofd_
#endif


extern "C" {
void egtransetWT(amrex::Real* wt);
void egtransetEPS(amrex::Real* eps);
void egtransetSIG(amrex::Real* sig);
void egtransetDIP(amrex::Real* dip);
void egtransetPOL(amrex::Real* pol);
void egtransetZROT(amrex::Real* zrot);
void egtransetNLIN(int* nlin);
void egtransetCOFETA(amrex::Real* fitmu);
void egtransetCOFLAM(amrex::Real* fitlam);
void egtransetCOFD(amrex::Real* fitdbin);
}

using namespace amrex;

namespace trans_params {

AMREX_GPU_DEVICE_MANAGED amrex::Real* wt;
AMREX_GPU_DEVICE_MANAGED amrex::Real* iwt;
AMREX_GPU_DEVICE_MANAGED amrex::Real* eps;
AMREX_GPU_DEVICE_MANAGED amrex::Real* sig;
AMREX_GPU_DEVICE_MANAGED amrex::Real* dip;
AMREX_GPU_DEVICE_MANAGED amrex::Real* pol;
AMREX_GPU_DEVICE_MANAGED amrex::Real* zrot;
AMREX_GPU_DEVICE_MANAGED amrex::Real* fitmu;
AMREX_GPU_DEVICE_MANAGED amrex::Real* fitlam;
AMREX_GPU_DEVICE_MANAGED amrex::Real* fitdbin;
AMREX_GPU_DEVICE_MANAGED int* nlin;
AMREX_GPU_DEVICE_MANAGED int array_size;
AMREX_GPU_DEVICE_MANAGED int fit_length;

void init ()
{
    array_size = NUM_SPECIES;
    fit_length = NUM_FIT;
//    std::cout << " array_size " << array_size << std::endl;
//    std::cout << " fit_length " << fit_length << std::endl;
    wt   = static_cast<Real*>(The_Managed_Arena()->alloc(sizeof(Real)*array_size));
    iwt  = static_cast<Real*>(The_Managed_Arena()->alloc(sizeof(Real)*array_size));
    eps  = static_cast<Real*>(The_Managed_Arena()->alloc(sizeof(Real)*array_size));
    sig  = static_cast<Real*>(The_Managed_Arena()->alloc(sizeof(Real)*array_size));
    dip  = static_cast<Real*>(The_Managed_Arena()->alloc(sizeof(Real)*array_size));
    pol  = static_cast<Real*>(The_Managed_Arena()->alloc(sizeof(Real)*array_size));
    zrot = static_cast<Real*>(The_Managed_Arena()->alloc(sizeof(Real)*array_size));

    fitmu   = static_cast<Real*>(The_Managed_Arena()->alloc(sizeof(Real)*array_size*fit_length));
    fitlam  = static_cast<Real*>(The_Managed_Arena()->alloc(sizeof(Real)*array_size*fit_length));
    fitdbin = static_cast<Real*>(The_Managed_Arena()->alloc(sizeof(Real)*array_size*array_size*fit_length));

    nlin = static_cast<int*>(The_Managed_Arena()->alloc(sizeof(int)*array_size));

    egtransetWT(wt);
    egtransetEPS(eps);
    egtransetSIG(sig);
    egtransetDIP(dip);
    egtransetPOL(pol);
    egtransetZROT(zrot);
    egtransetNLIN(nlin);
    egtransetCOFETA(fitmu);
    egtransetCOFLAM(fitlam);
    egtransetCOFD(fitdbin);

    for (int i=0; i < array_size; ++i){
        iwt[i] = 1. / wt[i];
    }
}

void finalize ()
{
    The_Managed_Arena()->free(wt);
    The_Managed_Arena()->free(iwt);
    The_Managed_Arena()->free(eps);
    The_Managed_Arena()->free(sig);
    The_Managed_Arena()->free(dip);
    The_Managed_Arena()->free(pol);
    The_Managed_Arena()->free(zrot);
    The_Managed_Arena()->free(fitmu);
    The_Managed_Arena()->free(fitlam);
    The_Managed_Arena()->free(fitdbin);
    The_Managed_Arena()->free(nlin);
}


}
