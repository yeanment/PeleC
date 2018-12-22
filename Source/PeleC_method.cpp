#include "PeleC_K.H" 


void PeleC_umeth2d(amrex::Box const& bx, amrex::FArrayBox &flatn,
                   amrex::FArrayBox const& quax, amrex::FArrayBox const& srcQ, 
                   amrex::FArrayBox const& bcMask, amrex::FArrayBox &flx1, 
                   amrex::FArrayBox &flx2, amrex::FArrayBox &q1, 
                   amrex::FArrayBox &q2, amrex::FArrayBox &a1, 
                   amrex::FArrayBox &a2, amrex::FArrayBox &pdiv, 
                   amrex::FArrayBox &vol, amrex::Real *dx, amrex::Real dt)
{
    amrex::Real const dxdt  = dt/dx[0]; 
    amrex::Real const hdtdx = 0.5*dtdx; 
    amrex::Real const hdtdy = 0.5*dt/dx[1]; 
    amrex::Real const hdt   = 0.5*dt; 

    if(hybrid_riemann == 1)
    {
        shock(q, shk, dx, dy); 
    }
    else shk.setVal(0.0); 

    if(ppm_type .eq. 0){
        
    }
}

