#include "PeleC_NSCBC_2D.H"
#include <PeleC.H> 

void impose_NSCBC(Box box, const amrex::Array4<amrex::Real> &u, 
                  const amrex::Array4<amrex::Real> &q, 
                  const amrex::Array4<amrex::Real> &qaux, 
                  const amrex::Array4<const int> x_bcMask, 
                  const amrex::Array4<const int> y_bcMask, 
                  const int nscbc_isAnyPerio, const int nscbc_perio, 
                  const amrex::Real time, const amrex::Real *delta, 
                  const amrex::Real dt) 
{
        const amrex::Real *problo = geom.ProbLo(); 
        const amrex::Real *probhi = geom.ProbHi(); 
        Box dom = geom.Domain(); 
        const int* domlo = dom.loVect(); 
        const int* domhi = dom.hiVect(); 
        const int* qlo = box.loVect(); 
        const int* qhi = box.hiVect(); 
        if(domlo[0] < qlo[0] && domlo[1] < qlo[1] && domhi[0] > qhi[0] && domhi[1] > qhi[1]) 
            return; 

        
        if(nscbc_isAnyPerio==0){ 
                

        }
}
