#ifndef PeleC_diffterm_3D_H
#define PeleC_diffterm_3D_H
#include <AMReX_FArrayBox.H> 
#include <PeleC_index_macros.H> 
#include <PeleC_EOS.H> 

/*This header file contains functions and declarations for diffterm in 2D for PeleC GPU.  
  As per the convention of AMReX, inlined functions are defined here. Where as non-inline functions 
  are declared here. */ 

/*The original Diffterm function passes too many arguments, which contribute to kernel 
  launch overhead. In order to combat this, we split some of the computation up into separate 
  Device functions which will then be launched by a CPU function called PeleC_diffterm. 
  PeleC_diffusion_flux calculates the diffusion flux per diction. */
AMREX_GPU_DEVICE
inline
void PeleC_diffusion_flux(const int i,const int j,const int k,
                          const amrex::Array4<const amrex::Real> &q, 
                          const amrex::Array4<const amrex::Real> &coef, 
                          const amrex::Array4<const amrex::Real> &t, 
                          const amrex::Array4<const amrex::Real> &a,                 
                          const amrex::Array4<amrex::Real> &flx,
                          const amrex::Real delta, const int dir) 
{
    amrex::Real dxinv = 1./delta;
    //Here "x" is the normal direction and "y" is the transverse 
    if(dir ==0){
        amrex::Real dTdx = dxinv*(q(i,j,k,QTEMP) - q(i-1,j,k,QTEMP)); 
        amrex::Real dudx = dxinv*(q(i,j,k,QU) - q(i-1,j,k,QU));
        amrex::Real dvdx = dxinv*(q(i,j,k,QV) - q(i-1,j,k,QV)); 
        amrex::Real dudy = t(i,j,k,0); 
        amrex::Real dvdy = t(i,j,k,1); 
        amrex::Real divu = dudx + dvdy; 
        amrex::Real taux = coef(i,j,k,dComp_mu)*(2.e0*dudx - 2.0/3.0*divu)
                      + coef(i,j,k,dComp_xi)*divu; 
        amrex::Real tauy = coef(i,j,k,dComp_mu)*(dudy + dvdx); 
        flx(i,j,k,UMX) = -taux;
        flx(i,j,k,UMY) = -tauy; 
        flx(i,j,k,UMZ) = 0; 
        flx(i,j,k,UEDEN) = - taux*(0.5e0*(q(i,j,k,QU) + q(i-1,j,k,QU))) 
                           - tauy*(0.5e0*(q(i,j,k,QV) + q(i-1,j,k,QV))) 
                           - coef(i,j,k,dComp_lambda)*dTdx;

        amrex::Real dlnp = dxinv*(q(i,j,k,QPRES) - q(i-1,j,k,QPRES))/
                                  (0.5e0*(q(i,j,k,QPRES) + q(i-1,j,k,QPRES))); 

// ------------------- Construct EOS objects for i and i -1 ---------------------------------------- // 
        EOS eos; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns) {
            eos.massfrac[ns] = q(i,j,k,ns+QFS);
        }
        eos.eos_ytx();
        eos.T = q(i,j,k,QTEMP); 
        eos.rho = q(i,j,k,QRHO); 
        eos.p = q(i,j,k,QPRES);
        eos.eos_hi();
        EOS eos_prev; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns) {
            eos_prev.massfrac[ns] = q(i-1,j,k,ns+QFS);
        }
        eos_prev.T = q(i-1,j,k,QTEMP); 
        eos_prev.rho = q(i-1,j,k,QRHO); 
        eos_prev.p = q(i-1,j,k,QPRES); 
        eos_prev.eos_ytx();
        eos_prev.eos_hi(); 
// ------------------ Get species/enthalpy diffusion, compute correction vel
        amrex::Real Vc = 0.e0; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns){
            amrex::Real Xface = 0.5e0*(eos.molefrac[ns] + eos_prev.molefrac[ns]);
            amrex::Real Yface = 0.5e0*(eos.massfrac[ns] + eos_prev.massfrac[ns]); 
            amrex::Real hface = 0.5e0*(eos.hi[ns] + eos_prev.hi[ns]); 
            amrex::Real dXdx = dxinv*(eos.molefrac[ns] -eos_prev.molefrac[ns]); 
            amrex::Real Vd = -coef(i,j,k,dComp_rhoD+ns)*(dXdx + (Xface - Yface)*dlnp); 
            flx(i,j,k,UFS+ns) = Vd; 
            Vc += Vd; 
            flx(i,j,k,UEDEN) += Vd*hface; 
        }
// ----------------- Add correction velocity to fluxes 
         for(int ns = 0; ns < NUM_SPECIES; ++ns){
            amrex::Real Yface = 0.5e0*(eos.massfrac[ns] + eos_prev.massfrac[ns]); 
            amrex::Real hface = 0.5e0*(eos.hi[ns] + eos_prev.hi[ns]); 
            flx(i,j,k,UFS+ns) -= Yface*Vc; 
            flx(i,j,k,UEDEN) -= Yface*hface*Vc; 
        }
// ---------------- Scale by area 
        amrex::Real a1 = a(i,j,k); 
        flx(i,j,k,UMX) *= a1; 
        flx(i,j,k,UMY) *= a1; 
        flx(i,j,k,UEDEN) *= a1; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns) flx(i,j,k,UFS+ns) *= a1;            
    }

    else// if(dir == 1)  recall that here dxinv really is dyinv. 
    {
        amrex::Real dTdy = dxinv*(q(i,j,k,QTEMP) - q(i,j-1,k,QTEMP)); 
        amrex::Real dudy = dxinv*(q(i,j,k,QU) - q(i,j-1,k,QU)); 
        amrex::Real dvdy = dxinv*(q(i,j,k,QV) - q(i,j-1,k,QV)); 
        amrex::Real dudx = t(i,j,k,0); 
        amrex::Real dvdx = t(i,j,k,1); 
        amrex::Real divu = dudx + dvdy; 
        amrex::Real tauy = coef(i,j,k,dComp_mu)*(2.e0*dvdy - 2.0/3.0*divu)
                          + coef(i,j,k,dComp_xi)*divu; 
        amrex::Real taux = coef(i,j,k,dComp_mu)*(dudy + dvdx); 
        flx(i,j,k,UMX) = -taux; 
        flx(i,j,k,UMY) = -tauy; 
        flx(i,j,k,UMZ) = 0; 
        flx(i,j,k,UEDEN) = - taux*(0.5e0*(q(i,j,k,QU) + q(i,j-1,k,QU))) 
                           - tauy*(0.5e0*(q(i,j,k,QV) + q(i,j-1,k,QV))) 
                           - coef(i,j,k,dComp_lambda)*dTdy; 
        amrex::Real dlnp = dxinv*(q(i,j,k,QPRES) - q(i,j-1,k,QPRES))/
                                  (0.5e0*(q(i,j,k,QPRES) + q(i,j-1,k,QPRES))); 
    
// ------------------- Construct EOS objects for j and j -1 ---------------------------------------- // 
        EOS eos, eos_prev; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns) {
            eos.massfrac[ns] = q(i,j,k,QFS+ns);
        }   
        eos.T = q(i,j,k,QTEMP); 
        eos.rho = q(i,j,k,QRHO); 
        eos.p = q(i,j,k,QPRES); 
        eos.eos_ytx(); 
        eos.eos_hi();       
        for(int ns = 0; ns < NUM_SPECIES; ++ns) {
            eos_prev.massfrac[ns] = q(i,j-1,k,QFS+ns);
        }   
        eos_prev.T = q(i,j-1,k,QTEMP); 
        eos_prev.rho = q(i,j-1,k,QRHO); 
        eos_prev.p = q(i,j-1,k,QPRES); 
        eos_prev.eos_ytx(); 
        eos_prev.eos_hi(); 
// ------------------ Get species/enthalpy diffusion, compute correction vel
        amrex::Real Vc = 0.e0; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns){
            amrex::Real Xface = 0.5e0*(eos.molefrac[ns] + eos_prev.molefrac[ns]); 
            amrex::Real Yface = 0.5e0*(eos.massfrac[ns] + eos_prev.massfrac[ns]); 
            amrex::Real hface = 0.5e0*(eos.hi[ns] + eos_prev.hi[ns]); 
            amrex::Real dXdx = dxinv*(eos.molefrac[ns] -eos_prev.molefrac[ns]); 
            amrex::Real Vd = -coef(i,j,k,dComp_rhoD+ns)*(dXdx + (Xface - Yface)*dlnp); 
            flx(i,j,k,UFS+ns) = Vd; 
            Vc += Vd; 
            flx(i,j,k,UEDEN) += Vd*hface; 
        }
// ----------------- Add correction velocity to species fluxes and 
        for(int ns = 0; ns < NUM_SPECIES; ++ns){
            amrex::Real Yface = 0.5e0*(eos.massfrac[ns] + eos_prev.massfrac[ns]); 
            amrex::Real hface = 0.5e0*(eos.hi[ns] + eos_prev.hi[ns]); 
            flx(i,j,k,UFS+ns) -= Yface*Vc; 
            flx(i,j,k,UEDEN) -= Yface*hface*Vc; 
        }
// ---------------- Scale by area 
        amrex::Real a1 = a(i,j,k); 
        flx(i,j,k,UMX) *= a1; 
        flx(i,j,k,UMY) *= a1; 
        flx(i,j,k,UEDEN) *= a1; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns) flx(i,j,k,UFS+ns) *= a1;            
    }
}


/* This function computes the diffusion flux divergence.  */ 
AMREX_GPU_HOST_DEVICE 
inline void 
PeleC_diffup(const int i, const int j, const int k, const int n, 
             const amrex::Array4<const amrex::Real> &fx, 
             const amrex::Array4<const amrex::Real> &fy, 
             const amrex::Array4<const amrex::Real> &fz, 
             const amrex::Array4<const amrex::Real> &V, 
             const amrex::Array4<amrex::Real> &D)
{
        D(i,j,k,n) = -(fx(i+1,j,k,n) - fx(i,j,k,n)  
                     + fy(i,j+1,k,n) - fy(i,j,k,n) 
                     + fy(i,j,k+1,n) - fz(i,j,k,n))/V(i,j,k); 
}

//Host Driver Function for generating diffusive fluxes. 

void 
PeleC_compute_diffusion_flux(const Box& box, const amrex::Array4<const amrex::Real> &q, const amrex::Array4<const amrex::Real> &coef, 
                             const amrex::Array4<amrex::Real> &flx1, const amrex::Array4<amrex::Real> &flx2, 
                             const amrex::Array4<amrex::Real> &flx3, const amrex::Array4<const amrex::Real> &a1,
                             const amrex::Array4<const amrex::Real> &a2, const amrex::Array4<const amrex::Real> &a3, 
                             const int nCompTr, const amrex::Real del[], const int do_harmonic, const int diffuse_vel); 

#endif 
