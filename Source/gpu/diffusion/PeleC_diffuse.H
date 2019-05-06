#ifndef PeleC_diffuse_H 
#define PeleC_diffuse_H
#include <AMReX_FArrayBox.H>


AMREX_GPU_DEVICE 
inline 
void PeleC_move_transcoefs_to_ec(const int i,const int j, const int k, const int n, 
                                 const amrex::Array4<const amrex::Real> &carr, 
                                 const amrex::Array4<amrex::Real> &earr, 
                                 const int dir)

{
//TODO add harmonic case
    if(dir==0) earr(i,j,k,n) = 0.5e0*(carr(i,j,k,n) - carr(i-1,j,k,n)); 
    else if(dir==1) earr(i,j,k,n) = 0.5e0*(carr(i,j,k,n) - carr(i,j-1,k,n)); 
    else earr(i,j,k,n) = 0.5e0*(carr(i,j,k,n) - carr(i,j,k-1,n));         
}                                    


AMREX_GPU_DEVICE
inline
void PeleC_diffextrap(const int i, const int j, const int k, const int n, 
                      const amrex::Array4<amrex::Real> dif, 
                      const int lo,  const int hi, const int dir) 
{
        //TODO: Need to think about how to fill ghost cells on GPU. 

}











#endif
