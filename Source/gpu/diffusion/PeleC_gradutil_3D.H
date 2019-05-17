#ifndef PeleC_gradutil_3D_H 
#define PeleC_gradutil_3D_H 
#include <AMReX_FArrayBox.H>
/*This header file contains functions and declarations for Gradient Utilities in 3D for PeleC GPU.  
  As per the convention of AMReX, inlined functions are defined here. Where as non-inline functions 
  are declared here. */
 
AMREX_GPU_DEVICE 
inline
void PeleC_compute_tangential_vel_derivs(const int i, const int j, const int k, 
                                         const amrex::Array4<amrex::Real> &td, 
                                         const amrex::Array4<const amrex::Real> &q, 
                                         const int dir, const amrex::Real dx1, 
                                         const amrex::Real dx2) 
{
    amrex::Real dx1inv = 1.e0/dx1; //dx1 and dx2 will be the trangential grid spacing 
    amrex::Real dx2inv = 1.e0/dx2; 
    //This is because it is a tangential velocity derivative.  
    if(dir == 0)
    {
        //dx1 = dy, dx2 = dz 
        td(i,j,k,0) = 0.25e0*dx1inv*(q(i,j+1,k,QU) + q(i-1,j+1,k,QU) - q(i,j-1,k,QU) -q(i-1,j-1,k,QU)); //dudy
        td(i,j,k,1) = 0.25e0*dx1inv*(q(i,j+1,k,QV) + q(i-1,j+1,k,QV) - q(i,j-1,k,QV) -q(i-1,j-1,k,QV)); //dvdy
        td(i,j,k,2) = 0.25e0*dx2inv*(q(i,j,k+1,QU) + q(i-1,j,k+1,QU) - q(i,j,k-1,QU) -q(i-1,j,k-1,QU)); //dudz 
        td(i,j,k,3) = 0.25e0*dx2inv*(q(i,j,k+1,QW) + q(i-1,j,k+1,QW) - q(i,j,k-1,QW) -q(i-1,j,k-1,QW)); //dwdz
    }
    else if(dir == 1)
    {
        //dx1 = dx, dx2 = dz 
        td(i,j,k,0) = 0.25e0*dx1inv*(q(i+1,j,k,QU) + q(i+1,j-1,k,QU) - q(i-1,j,k,QU) -q(i-1,j-1,k,QU)); //dudx
        td(i,j,k,1) = 0.25e0*dx1inv*(q(i+1,j,k,QV) + q(i+1,j-1,k,QV) - q(i-1,j,k,QV) -q(i-1,j-1,k,QV)); //dvdx
        td(i,j,k,2) = 0.25e0*dx2inv*(q(i,j,k+1,QV) + q(i,j-1,k+1,QV) - q(i,j,k-1,QV) -q(i,j-1,k-1,QV)); //dvdz
        td(i,j,k,3) = 0.25e0*dx2inv*(q(i,j,k+1,QW) + q(i,j-1,k+1,QW) - q(i,j,k-1,QW) -q(i,j-1,k-1,QW)); //dwdz
    }
    else{
        //dx1 = dx, dx2 = dy 
        td(i,j,k,0) = 0.25e0*dx1inv*(q(i+1,j,k,QU) + q(i+1,j,k-1,QU) - q(i-1,j,k,QU) -q(i-1,j,k-1,QU)); //dudx
        td(i,j,k,1) = 0.25e0*dx1inv*(q(i+1,j,k,QW) + q(i+1,j,k-1,QW) - q(i-1,j,k,QW) -q(i-1,j,k-1,QW)); //dwdx
        td(i,j,k,2) = 0.25e0*dx2inv*(q(i,j+1,k,QV) + q(i,j+1,k-1,QV) - q(i,j-1,k,QV) -q(i,j-1,k-1,QV)); //dvdy
        td(i,j,k,3) = 0.25e0*dx2inv*(q(i,j+1,k,QW) + q(i,j+1,k-1,QW) - q(i,j-1,k,QW) -q(i,j-1,k-1,QW)); //dwdy
    }
}

#endif
