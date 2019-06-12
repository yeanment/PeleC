#include "PeleC_diffterm_2D.H" 
#include "PeleC_gradutil_2D.H"
#include "PeleC_diffusion.H"  
/*
    This file contains the driver for generating the diffusion fluxes, which are then used to generate the diffusion flux divergence.
    PeleC_compute_diffusion_flux utilizes functions from the PeleC_diffterm_2D header: 
    PeleC_move_transcoefs_to_ec -> Moves Cell Centered Transport Coefficients to Edge Centers 
    PeleC_compute_tangential_vel_derivs -> Computes the Tangential Velocity Derivatives 
    PeleC_diffusion_flux -> Computes the diffusion flux per direction with the coefficients and velocity derivatives. 
*/

void 
PeleC_compute_diffusion_flux(const Box& box, const amrex::Array4<const amrex::Real> &q, const amrex::Array4<const amrex::Real> &coef, 
                             const amrex::Array4<amrex::Real> &flx1, const amrex::Array4<amrex::Real> &flx2, 
                             const amrex::Array4<const amrex::Real> &a1, const amrex::Array4<const amrex::Real> &a2, 
                             const amrex::Real del[], const int do_harmonic, const int diffuse_vel) 
{        
        Box exbox = amrex::surroundingNodes(box,0);
        Box eybox = amrex::surroundingNodes(box,1); 
        Gpu::AsyncFab cx_ec(exbox, dComp_lambda+1); 
        Gpu::AsyncFab cy_ec(eybox, dComp_lambda+1); 
        auto const &cx = cx_ec.array();
        auto const &cy = cy_ec.array(); 
        const amrex::Real dx = del[0]; 
        const amrex::Real dy = del[1]; 

        // Get Edge-centered transport coefficients
        BL_PROFILE("PeleC::pc_move_transport_coeffs_to_ec call");
        AMREX_PARALLEL_FOR_4D (box, dComp_lambda+1, i, j, k, n, { 
               PeleC_move_transcoefs_to_ec(i,j,k,n, coef, cx, 0, do_harmonic); 
               PeleC_move_transcoefs_to_ec(i,j,k,n, coef, cy, 1, do_harmonic); 
        });       

        int nCompTan = 2;
        Gpu::AsyncFab tx_der(exbox, nCompTan);
        Gpu::AsyncFab ty_der(eybox, nCompTan); 
        auto const &tx = tx_der.array(); 
        auto const &ty = ty_der.array(); 
        // Tangential derivatives on faces only needed for velocity diffusion
        if (diffuse_vel == 0) {
          (tx_der.fab()).setVal(0);
          (ty_der.fab()).setVal(0);
        } 
        else
        {
            BL_PROFILE("PeleC::pc_compute_tangential_vel_derivs call");
        // Tangential derivs 
        // X 
            AMREX_PARALLEL_FOR_3D(exbox, i, j, k, {
                PeleC_compute_tangential_vel_derivs(i,j,k,tx, q, 0, dy); 
            });
        // Y
            AMREX_PARALLEL_FOR_3D(eybox, i, j, k, {
                PeleC_compute_tangential_vel_derivs(i,j,k,ty, q, 1, dx); 
            });
        }  // diffuse_vel

        //Compute Extensive diffusion fluxes
        BL_PROFILE("PeleC::diffusion_flux()"); 
        AMREX_PARALLEL_FOR_3D(exbox, i, j, k,  {
            PeleC_diffusion_flux(i,j,k, q, cx, tx, a1, flx1, dx, 0); 
        });
        AMREX_PARALLEL_FOR_3D(eybox, i, j, k,  {
            PeleC_diffusion_flux(i,j,k, q, cy, ty, a2, flx2, dy, 1);
        });
}
