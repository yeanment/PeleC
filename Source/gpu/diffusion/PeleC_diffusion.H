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
        if(dir==0) earr(i,j,k,n) = 0.5e0*(carr(i,j,k,n) + carr(i-1,j,k,n)); 
        else if(dir==1) earr(i,j,k,n) = 0.5e0*(carr(i,j,k,n) + carr(i,j-1,k,n)); 
        else earr(i,j,k,n) = 0.5e0*(carr(i,j,k,n) + carr(i,j,k-1,n));         
    }
    else{
         amrex::Real a = carr(i,j,k,n); 
        if(dir==0){
            amrex::Real b = carr(i-1,j,k,n); 
            if(a*b > 0.e0) 
                earr(i,j,k,n) = 2.e0*(a*b)/(a+b); 
            else earr(i,j,k,n) = 0.e0; 
        }
        else if(dir==1){
            amrex::Real b = carr(i,j-1,k,n); 
            if(a*b > 0.e0) 
                earr(i,j,k,n) = 2.e0*(a*b)/(a+b); 
            else earr(i,j,k,n) = 0.e0; 
        }
        else{
            amrex::Real b = carr(i,j,k-1,n); 
            if(a*b > 0.e0) 
                earr(i,j,k,n) = 2.e0*(a*b)/(a+b); 
            else earr(i,j,k,n) = 0.e0; 
        }
    }
}                                    

inline
void setV(const Box box, const int Ncomp, const amrex::Array4<amrex::Real> array, const amrex::Real val)
{
    AMREX_PARALLEL_FOR_4D(box, Ncomp, i, j, k, n, { 
        array(i,j,k,n) = val; 
    }); 
}

inline
void setC(const Box box,const int Nstart, const int Nend, const amrex::Array4<amrex::Real> array, const amrex::Real val)
{
    AMREX_PARALLEL_FOR_3D(box, i, j, k, 
      { 
        for(int n = Nstart; n < Nend; n++) array(i,j,k,n) = val;
    }); 
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
