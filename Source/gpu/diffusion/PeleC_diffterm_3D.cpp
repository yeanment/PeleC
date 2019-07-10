#include "PeleC_diffterm_3D.H" 
#include "PeleC_gradutil_3D.H"
#include "PeleC_diffusion.H"  
/*
    This file contains the driver for generating the diffusion fluxes, which are then used to generate the diffusion flux divergence.
    PeleC_compute_diffusion_flux utilizes functions from the PeleC_diffterm_3D header: 
    PeleC_move_transcoefs_to_ec -> Moves Cell Centered Transport Coefficients to Edge Centers 
    PeleC_compute_tangential_vel_derivs -> Computes the Tangential Velocity Derivatives 
    PeleC_diffusion_flux -> Computes the diffusion flux per direction with the coefficients and velocity derivatives. 
*/

void 
PeleC_compute_diffusion_flux(const Box& box, const amrex::Array4<const amrex::Real> &q,
                             const amrex::Array4<const amrex::Real> &coef, const amrex::Array4<amrex::Real> &flx1, 
                             const amrex::Array4<amrex::Real> &flx2, const amrex::Array4<amrex::Real> &flx3, 
                             const amrex::Array4<const amrex::Real> &a1, const amrex::Array4<const amrex::Real> &a2, 
                             const amrex::Array4<const amrex::Real> &a3, const amrex::Real del[], 
                             const int do_harmonic, const int diffuse_vel) 
{        
        Box exbox = amrex::surroundingNodes(box,0);
        Box eybox = amrex::surroundingNodes(box,1); 
        Box ezbox = amrex::surroundingNodes(box,2); 
        const amrex::Real dx = del[0]; 
        const amrex::Real dy = del[1]; 
        const amrex::Real dz = del[2]; 

        //Compute Extensive diffusion fluxes
        {
            //X
            BL_PROFILE("PeleC::diffusion_flux()"); 
            AMREX_PARALLEL_FOR_3D(exbox, i, j, k,  {
                amrex::Real tx[4]={}; 
                amrex::Real cx[dComp_lambda+1]={};
                for(int n = 0; n < dComp_lambda+1; n++) 
                    PeleC_move_transcoefs_to_ec(i,j,k,n, coef, cx, 0, do_harmonic); 
                PeleC_compute_tangential_vel_derivs(i,j,k,tx,q,0,dy,dz); 
                PeleC_diffusion_flux(i,j,k, q, cx, tx, a1, flx1, dx, 0); 
            });
            //Y
            AMREX_PARALLEL_FOR_3D(eybox, i, j, k,  {
                amrex::Real ty[4]={}; 
                amrex::Real cy[dComp_lambda+1]={};
                for(int n = 0; n < dComp_lambda+1; n++) 
                    PeleC_move_transcoefs_to_ec(i,j,k,n, coef, cy, 1, do_harmonic);
                PeleC_compute_tangential_vel_derivs(i,j,k,ty, q, 1, dx, dz); 
                PeleC_diffusion_flux(i,j,k, q, cy, ty, a2, flx2, dy, 1);
            });
            //Z
            AMREX_PARALLEL_FOR_3D(ezbox, i, j, k,  {
                amrex::Real tz[4]={}; 
                amrex::Real cz[dComp_lambda+1]={};
                for(int n = 0; n < dComp_lambda+1; n++) 
                    PeleC_move_transcoefs_to_ec(i,j,k,n, coef, cz, 2, do_harmonic);  
                PeleC_compute_tangential_vel_derivs(i,j,k,tz, q, 2, dx, dy); 
                PeleC_diffusion_flux(i,j,k, q, cz, tz, a3, flx3, dz, 2);
            });
        }
}
