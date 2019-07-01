#ifndef PeleC_diffterm_3D_H
#define PeleC_diffterm_3D_H
#include <AMReX_FArrayBox.H> 
#include <PeleC_index_macros.H> 
#include <PeleC_EOS.H> 

/*This header file contains functions and declarations for diffterm in 3D for PeleC GPU.  
  As per the convention of AMReX, inlined device functions are defined here.
  Where as non-inline host functions are declared here. */ 

/*The original Diffterm function passes too many arguments, which contribute to kernel 
  launch overhead. In order to combat this, we split some of the computation up into separate 
  Device functions which will then be launched by a CPU function called PeleC_diffterm. 
  PeleC_diffusion_flux calculates the diffusion flux per diction. */

AMREX_GPU_DEVICE
inline
void PeleC_diffusion_flux(const int i,const int j,const int k,
                          const amrex::Array4<const amrex::Real> &q, 
                          const amrex::Real coef[], 
                          const amrex::Real t[],
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
        amrex::Real dwdx = dxinv*(q(i,j,k,QW) - q(i-1,j,k,QW)); 
        amrex::Real dudy = t[0]; 
        amrex::Real dvdy = t[1]; 
        amrex::Real dudz = t[2]; 
        amrex::Real dwdz = t[3]; 
        amrex::Real divu = dudx + dvdy + dwdz; 
        amrex::Real taux = coef[dComp_mu]*(2.e0*dudx - 2.0/3.0*divu)
                      + coef[dComp_xi]*divu; 
        amrex::Real tauy = coef[dComp_mu]*(dudy + dvdx); 
        amrex::Real tauz = coef[dComp_mu]*(dudz + dwdx); 
        flx(i,j,k,UMX) = -taux;
        flx(i,j,k,UMY) = -tauy; 
        flx(i,j,k,UMZ) = -tauz; 
        flx(i,j,k,UEDEN) = - taux*(0.5e0*(q(i,j,k,QU) + q(i-1,j,k,QU))) 
                           - tauy*(0.5e0*(q(i,j,k,QV) + q(i-1,j,k,QV))) 
                           - tauz*(0.5e0*(q(i,j,k,QW) + q(i-1,j,k,QW)))
                           - coef[dComp_lambda]*dTdx;

        amrex::Real dlnp = dxinv*(q(i,j,k,QPRES) - q(i-1,j,k,QPRES))/
                                 (0.5e0*(q(i,j,k,QPRES) + q(i-1,j,k,QPRES))); 

// ------------------- Get massfrac, molefrac, enthalpy for i and i -1 ---------------------------------------- // 
        amrex::Real mass1[NUM_SPECIES], mass2[NUM_SPECIES]; 
        amrex::Real mole1[NUM_SPECIES], mole2[NUM_SPECIES];  
        amrex::Real hi1[NUM_SPECIES], hi2[NUM_SPECIES]; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns) {
            mass1[ns] = q(i,j,k,ns+QFS);
        }
        EOS::ytx(mass1, mole1);
        amrex::Real T = q(i,j,k,QTEMP); 
        EOS::get_hi(T, hi1);
        for(int ns = 0; ns < NUM_SPECIES; ++ns) {
            mass2[ns] = q(i-1,j,k,ns+QFS);
        }
        T = q(i-1,j,k,QTEMP); 
        EOS::ytx(mass2, mole2);
        EOS::get_hi(T, hi2); 
// ------------------ Get species/enthalpy diffusion, compute correction vel
        amrex::Real Vc = 0.e0; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns){
            amrex::Real Xface = 0.5e0*(mole1[ns] + mole2[ns]);
            amrex::Real Yface = 0.5e0*(mass1[ns] + mass2[ns]); 
            amrex::Real hface = 0.5e0*(hi1[ns] + hi2[ns]); 
            amrex::Real dXdx = dxinv*(mole1[ns] -mole2[ns]); 
            amrex::Real Vd = -coef[dComp_rhoD+ns]*(dXdx + (Xface - Yface)*dlnp); 
            flx(i,j,k,UFS+ns) = Vd; 
            Vc += Vd; 
            flx(i,j,k,UEDEN) += Vd*hface; 
        }
// ----------------- Add correction velocity to fluxes 
         for(int ns = 0; ns < NUM_SPECIES; ++ns){
            amrex::Real Yface = 0.5e0*(mass1[ns] + mass2[ns]); 
            amrex::Real hface = 0.5e0*(hi1[ns] + hi2[ns]); 
            flx(i,j,k,UFS+ns) -= Yface*Vc; 
            flx(i,j,k,UEDEN) -= Yface*hface*Vc; 
        }
// ---------------- Scale by area 
        amrex::Real a1 = a(i,j,k); 
        flx(i,j,k,UMX) *= a1; 
        flx(i,j,k,UMY) *= a1; 
        flx(i,j,k,UMZ) *= a1; 
        flx(i,j,k,UEDEN) *= a1; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns) flx(i,j,k,UFS+ns) *= a1;            
    }

    else if(dir == 1) // recall that here dxinv really is dyinv. 
    {
        amrex::Real dTdy = dxinv*(q(i,j,k,QTEMP) - q(i,j-1,k,QTEMP)); 
        amrex::Real dudy = dxinv*(q(i,j,k,QU) - q(i,j-1,k,QU)); 
        amrex::Real dvdy = dxinv*(q(i,j,k,QV) - q(i,j-1,k,QV)); 
        amrex::Real dwdy = dxinv*(q(i,j,k,QW) - q(i,j-1,k,QW)); 
        amrex::Real dudx = t[0]; 
        amrex::Real dvdx = t[1]; 
        amrex::Real dvdz = t[2]; 
        amrex::Real dwdz = t[3]; 
        amrex::Real divu = dudx + dvdy + dwdz; 
        amrex::Real tauy = coef[dComp_mu]*(2.e0*dvdy - 2.0/3.0*divu)
                          + coef[dComp_xi]*divu; 
        amrex::Real taux = coef[dComp_mu]*(dudy + dvdx); 
        amrex::Real tauz = coef[dComp_mu]*(dwdy + dvdz); 
        flx(i,j,k,UMX) = -taux; 
        flx(i,j,k,UMY) = -tauy; 
        flx(i,j,k,UMZ) = -tauz; 
        flx(i,j,k,UEDEN) = - taux*(0.5e0*(q(i,j,k,QU) + q(i,j-1,k,QU))) 
                           - tauy*(0.5e0*(q(i,j,k,QV) + q(i,j-1,k,QV))) 
                           - tauz*(0.5e0*(q(i,j,k,QW) + q(i,j-1,k,QW)))
                           - coef[dComp_lambda]*dTdy; 
        amrex::Real dlnp = dxinv*(q(i,j,k,QPRES) - q(i,j-1,k,QPRES))/
                                  (0.5e0*(q(i,j,k,QPRES) + q(i,j-1,k,QPRES))); 
    
// ------------------- Construct massfrac, molefrac, enthalpy, for j and j -1 ---------------------------------------- // 
        amrex::Real mass1[NUM_SPECIES], mass2[NUM_SPECIES]; 
        amrex::Real mole1[NUM_SPECIES], mole2[NUM_SPECIES];  
        amrex::Real hi1[NUM_SPECIES], hi2[NUM_SPECIES]; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns) {
            mass1[ns] = q(i,j,k,ns+QFS);
        }
        EOS::ytx(mass1, mole1);
        amrex::Real T = q(i,j,k,QTEMP); 
        EOS::get_hi(T, hi1);
        for(int ns = 0; ns < NUM_SPECIES; ++ns) {
            mass2[ns] = q(i,j-1,k,ns+QFS);
        }
        T = q(i,j-1,k,QTEMP); 
        EOS::ytx(mass2, mole2);
        EOS::get_hi(T, hi2); 
// ------------------ Get species/enthalpy diffusion, compute correction vel
        amrex::Real Vc = 0.e0; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns){
            amrex::Real Xface = 0.5e0*(mole1[ns] + mole2[ns]);
            amrex::Real Yface = 0.5e0*(mass1[ns] + mass2[ns]); 
            amrex::Real hface = 0.5e0*(hi1[ns] + hi2[ns]); 
            amrex::Real dXdx = dxinv*(mole1[ns] -mole2[ns]); 
            amrex::Real Vd = -coef[dComp_rhoD+ns]*(dXdx + (Xface - Yface)*dlnp); 
            flx(i,j,k,UFS+ns) = Vd; 
            Vc += Vd; 
            flx(i,j,k,UEDEN) += Vd*hface; 
        }
// ----------------- Add correction velocity to species fluxes and 
        for(int ns = 0; ns < NUM_SPECIES; ++ns){
            amrex::Real Yface = 0.5e0*(mass1[ns] + mass2[ns]); 
            amrex::Real hface = 0.5e0*(hi1[ns] + hi2[ns]); 
            flx(i,j,k,UFS+ns) -= Yface*Vc; 
            flx(i,j,k,UEDEN) -= Yface*hface*Vc; 
        }
// ---------------- Scale by area 
        amrex::Real a1 = a(i,j,k); 
        flx(i,j,k,UMX) *= a1; 
        flx(i,j,k,UMY) *= a1;
        flx(i,j,k,UMZ) *= a1;  
        flx(i,j,k,UEDEN) *= a1; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns) flx(i,j,k,UFS+ns) *= a1;            
    }
    else// if(dir == 2)  recall that here dxinv really is dzinv. 
    {
        amrex::Real dTdz = dxinv*(q(i,j,k,QTEMP) - q(i,j,k-1,QTEMP)); 
        amrex::Real dudz = dxinv*(q(i,j,k,QU) - q(i,j,k-1,QU)); 
        amrex::Real dvdz = dxinv*(q(i,j,k,QV) - q(i,j,k-1,QV)); 
        amrex::Real dwdz = dxinv*(q(i,j,k,QW) - q(i,j,k-1,QW)); 
        amrex::Real dudx = t[0]; 
        amrex::Real dwdx = t[1]; 
        amrex::Real dvdy = t[2]; 
        amrex::Real dwdy = t[3]; 
        amrex::Real divu = dudx + dvdy + dwdz; 
        amrex::Real tauz = coef[dComp_mu]*(2.e0*dwdz - 2.0/3.0*divu)
                          + coef[dComp_xi]*divu; 
        amrex::Real taux = coef[dComp_mu]*(dudz + dwdx); 
        amrex::Real tauy = coef[dComp_mu]*(dvdz + dwdy); 
        flx(i,j,k,UMX) = -taux; 
        flx(i,j,k,UMY) = -tauy; 
        flx(i,j,k,UMZ) = -tauz; 
        flx(i,j,k,UEDEN) = - taux*(0.5e0*(q(i,j,k,QU) + q(i,j,k-1,QU))) 
                           - tauy*(0.5e0*(q(i,j,k,QV) + q(i,j,k-1,QV))) 
                           - tauz*(0.5e0*(q(i,j,k,QW) + q(i,j,k-1,QW)))
                           - coef[dComp_lambda]*dTdz; 
        amrex::Real dlnp = dxinv*(q(i,j,k,QPRES) - q(i,j,k-1,QPRES))/
                                 (0.5e0*(q(i,j,k,QPRES) + q(i,j,k-1,QPRES))); 
    
// ------------------- Construct massfrac, molefrac, enthalpy, for k and k -1 ---------------------------------------- // 
        amrex::Real mass1[NUM_SPECIES], mass2[NUM_SPECIES]; 
        amrex::Real mole1[NUM_SPECIES], mole2[NUM_SPECIES];  
        amrex::Real hi1[NUM_SPECIES], hi2[NUM_SPECIES]; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns) {
            mass1[ns] = q(i,j,k,ns+QFS);
        }
        EOS::ytx(mass1, mole1);
        amrex::Real T = q(i,j,k,QTEMP); 
        EOS::get_hi(T, hi1);
        for(int ns = 0; ns < NUM_SPECIES; ++ns) {
            mass2[ns] = q(i,j,k-1,ns+QFS);
        }
        T = q(i,j,k-1,QTEMP); 
        EOS::ytx(mass2, mole2);
        EOS::get_hi(T, hi2); 
// ------------------ Get species/enthalpy diffusion, compute correction vel
        amrex::Real Vc = 0.e0; 
        for(int ns = 0; ns < NUM_SPECIES; ++ns){
            amrex::Real Xface = 0.5e0*(mole1[ns] + mole2[ns]);
            amrex::Real Yface = 0.5e0*(mass1[ns] + mass2[ns]); 
            amrex::Real hface = 0.5e0*(hi1[ns] + hi2[ns]); 
            amrex::Real dXdx = dxinv*(mole1[ns] -mole2[ns]); 
            amrex::Real Vd = -coef[dComp_rhoD+ns]*(dXdx + (Xface - Yface)*dlnp); 
            flx(i,j,k,UFS+ns) = Vd; 
            Vc += Vd; 
            flx(i,j,k,UEDEN) += Vd*hface; 
        }
// ----------------- Add correction velocity to species fluxes and 
        for(int ns = 0; ns < NUM_SPECIES; ++ns){
            amrex::Real Yface = 0.5e0*(mass1[ns] + mass2[ns]); 
            amrex::Real hface = 0.5e0*(hi1[ns] + hi2[ns]); 
            flx(i,j,k,UFS+ns) -= Yface*Vc; 
            flx(i,j,k,UEDEN) -= Yface*hface*Vc; 
        }
// ---------------- Scale by area 
        amrex::Real a1 = a(i,j,k); 
        flx(i,j,k,UMX) *= a1; 
        flx(i,j,k,UMY) *= a1;
        flx(i,j,k,UMZ) *= a1;  
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
                     + fz(i,j,k+1,n) - fz(i,j,k,n))/V(i,j,k); 
}

//Host Driver Function for generating diffusive fluxes. 

void 
PeleC_compute_diffusion_flux(const Box& box, const amrex::Array4<const amrex::Real> &q, const amrex::Array4<const amrex::Real> &coef, 
                             const amrex::Array4<amrex::Real> &flx1, const amrex::Array4<amrex::Real> &flx2, 
                             const amrex::Array4<amrex::Real> &flx3, const amrex::Array4<const amrex::Real> &a1,
                             const amrex::Array4<const amrex::Real> &a2, const amrex::Array4<const amrex::Real> &a3, 
                             const amrex::Real del[], const int do_harmonic, const int diffuse_vel); 

#endif 
