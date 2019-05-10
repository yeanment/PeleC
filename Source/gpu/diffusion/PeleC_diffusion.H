#ifndef PeleC_diffusion_H 
#define PeleC_diffusion_H
#include <AMReX_FArrayBox.H>
/*This header file contains functions and declarations for the diffusion portion of PeleC GPU. 
  As per the convention of AMReX, inlined functions are defined here. Where as non-inline functions 
  are declared here. */ 



AMREX_GPU_DEVICE 
inline 
void PeleC_move_transcoefs_to_ec(const int i,const int j, const int k, const int n, 
                                 const amrex::Array4<const amrex::Real> &carr, 
                                 const amrex::Array4<amrex::Real> &earr, 
                                 const int dir, const int do_harmonic)

{
    if(do_harmonic==0){
        if(dir==0) earr(i,j,k,n) = 0.5e0*(carr(i,j,k,n) - carr(i-1,j,k,n)); 
        else if(dir==1) earr(i,j,k,n) = 0.5e0*(carr(i,j,k,n) - carr(i,j-1,k,n)); 
        else earr(i,j,k,n) = 0.5e0*(carr(i,j,k,n) - carr(i,j,k-1,n));         
    }
    else{
        if(dir==0){
            if(carr(i,j,k,n)*carr(i-1,j,k,n) > 0) 
                earr(i,j,k,n) = 2.e0*(carr(i,j,k,n)*carr(i-1,j,k,n))/(carr(i,j,k,n) + carr(i-1,j,k,n)); 
            else earr(i,j,k,n) = 0; 
        }
        else if(dir==1){
            if(carr(i,j,k,n)*carr(i,j-1,k,n) > 0) 
                earr(i,j,k,n) = 2.e0*(carr(i,j,k,n)*carr(i,j-1,k,n))/(carr(i,j,k,n) + carr(i,j-1,k,n)); 
            else earr(i,j,k,n) = 0; 
        }
        else{
            if(carr(i,j,k,n)*carr(i,j,k-1,n) > 0) 
                earr(i,j,k,n) = 2.e0*(carr(i,j,k,n)*carr(i,j,k-1,n))/(carr(i,j,k,n) + carr(i,j,k-1,n)); 
            else earr(i,j,k,n) = 0; 
        }
    }
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
