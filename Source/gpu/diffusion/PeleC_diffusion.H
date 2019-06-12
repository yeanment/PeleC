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
                                 amrex::Real *earr, 
                                 const int dir, const int do_harmonic)

{
    if(do_harmonic==0){
        if(dir==0) earr[n] = 0.5e0*(carr(i,j,k,n) + carr(i-1,j,k,n)); 
        else if(dir==1) earr[n] = 0.5e0*(carr(i,j,k,n) + carr(i,j-1,k,n)); 
        else earr[n] = 0.5e0*(carr(i,j,k,n) + carr(i,j,k-1,n));         
    }
    else{
         amrex::Real a = carr(i,j,k,n); 
        if(dir==0){
            amrex::Real b = carr(i-1,j,k,n); 
            if(a*b > 0.e0) 
                earr[n] = 2.e0*(a*b)/(a+b); 
            else earr[n] = 0.e0; 
        }
        else if(dir==1){
            amrex::Real b = carr(i,j-1,k,n); 
            if(a*b > 0.e0) 
                earr[n] = 2.e0*(a*b)/(a+b); 
            else earr[n] = 0.e0; 
        }
        else{
            amrex::Real b = carr(i,j,k-1,n); 
            if(a*b > 0.e0) 
                earr[n] = 2.e0*(a*b)/(a+b); 
            else earr[n] = 0.e0; 
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
void PeleC_diffextrap(const int i, const int j, const int k, 
                      const amrex::Array4<amrex::Real> dif,
                      const int ng, const int first, const int last, 
                      D_DECL(const int lx, const int ly, const int lz), 
                      D_DECL(const int hx, const int hy, const int hz),
                      const amrex::Dim3 dlo, const amrex::Dim3 dhi) 
{
            for(int n = first; n < last; n++)
            {
//============================= Sides =============================================
             //low x  
                if(dlo.x < lx){
                    if(i == lx) dif(i-1,j,k,n) = dif(i,j,k,n); 
                }
             //high x 
                if(dhi.x-1 > hx){
                    if(i == hx)  dif(i+1,j,k,n) = dif(i,j,k,n); 
                }
             //low y
                if(dlo.y < ly){
                    if(j == ly) dif(i,j-1,k,n) = dif(i,j,k,n); 
                }
             //high y 
                if(dhi.y-1 > hy){
                    if(j == hy) dif(i,j+1,k,n) = dif(i,j,k,n); 
                }
#if AMREX_SPACEDIM==3 
             //low z 
                if(dlo.z < lz){
                    if(k == lz) dif(i,j,k-1,n) = dif(i,j,k,n); 
                }
             //high z 
                if(dhi.z-1 > hz){
                    if(k == hz) dif(i,j,k+1,n) = dif(i,j,k,n); 
                }
#endif            
//============================= Edges ==============================================            
             //low x low y 
             if(dlo.x < lx && dlo.y < ly){
                if(i == lx && j == ly) dif(i-1,j-1,k,n) = dif(i,j,k,n); 
             }
             //low x high y 
             if(dlo.x < lx && dhi.y-1 > hy){
                if(i == lx && j == hy) dif(i-1,j+1,k,n) = dif(i,j,k,n); 
             }
             //high x low y 
             if(dhi.x-1 > hx && dlo.y < ly){
                if(i == hx && j == ly) dif(i+1,j-1,k,n) = dif(i,j,k,n); 
             }
             //high x high y 
             if(dhi.x-1 > hx && dhi.y-1 > hy){
                if(i == hx && j == hy) dif(i+1,j+1,k,n) = dif(i,j,k,n); 
             }
#if AMREX_SPACEDIM==3 
             //low x low z 
             if(dlo.x < lx && dlo.z < lz){
                if(i == lx && k == lz) dif(i-1,j,k-1,n) = dif(i,j,k,n); 
             }
             //low x high z 
             if(dlo.x < lx && dhi.z-1 > hz){
                if(i == lx && k == hz) dif(i-1,j,k+1,n) = dif(i,j,k,n); 
             }
             //high x low z 
             if(dhi.x-1 > hx && dlo.z < lz){
                if(i == hx && k == lz) dif(i+1,j,k-1,n) = dif(i,j,k,n); 
             }
             //high x high y 
             if(dhi.x-1 > hx && dhi.z-1 > hz){
                if(i == hx && k == hz) dif(i+1,j,k+1,n) = dif(i,j,k,n); 
             }

             //low y low z 
             if(dlo.y < ly && dlo.z < lz){
                if(j == ly && k == lz) dif(i,j-1,k-1,n) = dif(i,j,k,n); 
             }
             //low y hi z 
             if(dlo.y < ly && dhi.z-1 > hz){
                if(j == ly && k == hz) dif(i,j-1,k+1,n) = dif(i,j,k,n); 
             }
             //hi y low z 
             if(dhi.y-1 > hy && dlo.z < lz){
                if(j == hy && k == lz) dif(i,j+1,k-1,n) = dif(i,j,k,n); 
             }
             //low y low z 
             if(dhi.y-1 > hy && dhi.z-1 > hz){
                if(j == hy && k == hz) dif(i,j+1,k+1,n) = dif(i,j,k,n); 
             }
//============================= Corners ======================================
            //low x low y low z 
            if(dlo.x < lx && dlo.y < ly && dlo.z < lz)
                if(i == lx && j == ly && k == lz) dif(i-1,j-1,k-1,n) = dif(i,j,k,n); 
             //hi x low y low z 
            if(dhi.x -1 > hx && dlo.y < ly && dlo.z < lz)
                if(i == hx && j == ly && k == lz) dif(i+1,j-1,k-1,n) = dif(i,j,k,n); 
           
            //low x hi y low z 
            if(dlo.x < lx && dhi.y-1 > ly && dlo.z < lz)
                if(i == lx && j == hy && k == lz) dif(i-1,j+1,k-1,n) = dif(i,j,k,n); 

            //low x low y hi z 
            if(dlo.x < lx && dlo.y < ly && dhi.z-1 > hz)
                if(i == lx && j == ly && k == hz) dif(i-1,j-1,k+1,n) = dif(i,j,k,n); 

            //hi x hi y low z 
            if(dhi.x -1 > hx && dhi.y-1 > hy && dlo.z < lz)
                if(i == hx && j == hy && k == lz) dif(i+1,j+1,k-1,n) = dif(i,j,k,n); 

            //hi x low y hi z 
            if(dhi.x -1 > hx && dlo.y < ly && dhi.z -1 > lz)
                if(i == hx && j == ly && k == hz) dif(i+1,j-1,k+1,n) = dif(i,j,k,n); 

            //low x hi y hi z 
            if(dlo.x < lx && dhi.y-1 > hy && dlo.z-1 > hz)
                if(i == lx && j == hy && k == hz) dif(i-1,j+1,k+1,n) = dif(i,j,k,n); 

            //hi x hi y hi z 
            if(dhi.x-1 > hx && dhi.y-1 > hy && dlo.z-1 > hz)
                if(i == hx && j == hy && k == hz) dif(i+1,j+1,k+1,n) = dif(i,j,k,n); 

#endif      
         }

}

#endif
