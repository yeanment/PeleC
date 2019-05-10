#ifndef PeleC_gradutil_2D_H 
#define PeleC_gradutil_2D_H 
#include <AMReX_FArrayBox.H>
#include <PeleC_index_macros.H> 

/*This header file contains functions and declarations for Gradient Utilities in 2D for PeleC GPU.  
  As per the convention of AMReX, inlined functions are defined here. Where as non-inline functions 
  are declared here. */ 

AMREX_GPU_DEVICE 
inline
void PeleC_compute_tangential_vel_derivs(const int i, const int j, const int k, 
                                         const amrex::Array4<amrex::Real> &td, 
                                         const amrex::Array4<const amrex::Real> &q, 
                                         const int dir, const amrex::Real dx) 
{
    amrex::Real dxinv = 1.e0/dx; //Here dx will be dy if dir = 0 and dx if dir = 1 
    //This is because it is a tangential velocity derivative.  
    if(dir == 0)
    {
        td(i,j,k,0) = 0.25e0*dxinv*(q(i,j+1,k,QU) + q(i-1,j+1,QU) - q(i,j-1,k,QU) -q(i-1,j-1,k,QU));
        td(i,j,k,1) = 0.25e0*dxinv*(q(i,j+1,k,QV) + q(i-1,j+1,QV) - q(i,j-1,k,QV) -q(i-1,j-1,k,QV)); 
    }
    if(dir == 1)
    {
        td(i,j,k,0) = 0.25e0*dxinv*(q(i+1,j,k,QU) + q(i+1,j-1,QU) - q(i-1,j,k,QU) -q(i-1,j-1,k,QU));
        td(i,j,k,1) = 0.25e0*dxinv*(q(i+1,j,k,QV) + q(i+1,j-1,QV) - q(i-1,j,k,QV) -q(i-1,j-1,k,QV)); 
    }
}

#ifdef PELEC_USE_EB
//TODO add eb stuffs  
#endif


#endif
