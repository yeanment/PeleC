#ifndef transport_H_
#define transport_H_

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include "PeleC_trans_params.H"
#include <AMReX_FArrayBox.H>
#include <limits>
#include <cmath>
#include "mechanism.h"
#include "PeleC_simple_transport.H"


void transport_init() // bind(C, name="transport_init")
{

     trans_params::init();

}

void transport_close()
{

     trans_params::finalize();

}

AMREX_GPU_HOST_DEVICE
inline
void get_transport_coeffs(amrex::Box const& bx, amrex::Array4<const amrex::Real> const& massfrac,
       amrex::Array4<const amrex::Real> const& temperature, amrex::Array4<const amrex::Real> const& density,
       amrex::Array4<amrex::Real> const& D) noexcept
{

        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);

    bool wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag;

    wtr_get_xi    = true;
    wtr_get_mu    = true;
    wtr_get_lam   = true;
    wtr_get_Ddiag = true;

    amrex::Real T;
    amrex::Real rho;
    amrex::GpuArray<amrex::Real,NUM_SPECIES>  massloc;

    amrex::Real muloc,xiloc,lamloc;
    amrex::GpuArray<amrex::Real,NUM_SPECIES>  Ddiag;




    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {

                T = temperature(i,j,k);
                rho = density(i,j,k);

                for (int n = 0; n < NUM_SPECIES; ++n){
                  massloc[n] = massfrac(i,j,k,n);
                }

                actual_transport(wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, T, rho, massloc.data(),
                    Ddiag.data(),muloc,xiloc,lamloc);

                //   mu, xi and lambda are stored after D in the diffusion multifab

                for (int n = 0; n < NUM_SPECIES; ++n){
                  D(i,j,k,n) = Ddiag[n];
                }
                  D(i,j,k,NUM_SPECIES) = muloc;
                  D(i,j,k,NUM_SPECIES+1) = xiloc;
                  D(i,j,k,NUM_SPECIES+2) = lamloc;

              
            }
        }
    }

}

#endif
