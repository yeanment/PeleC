#ifndef PeleC_K_H_
#define PeleC_K_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include "PeleC_index_macros.H" 
#include "PeleC_EOS.H"
#if AMREX_SPACEDIM == 1
#include "PeleC_method_1D.H" 
#elif AMREX_SPACEDIM == 2
#include "PeleC_method_2D.H"
#else
#include "PeleC_method_3D.H"
#endif 


/*This header file contains functions and declarations for miscellaneous utilities and hydro portion 
  of PeleC gpu. 
  As per the convention of AMReX, inlined functions are defined here. Where as non-inline functions 
  are declared here. */ 

//TODO Add more fuctions! 
struct Real3{
    amrex::Real x;
    amrex::Real y; 
    amrex::Real z; 
};



AMREX_GPU_DEVICE
inline
void PeleC_ctoprim(const int i, const int j, const int k, 
                   amrex::Array4<const amrex::Real> const& u,
                   amrex::Array4<amrex::Real> const& q,
                   amrex::Array4<amrex::Real> const& qa)
{
    const amrex::Real smallr = 1.e-19; 
    const amrex::Real smallp = 1.e-10; 
    amrex::Real R = k_B*n_A; 

    Real3 v; 
    int n, nq;
    amrex::Real rho = u(i,j,k,URHO);  
    amrex::Real rhoinv = 1.0/rho; 
    v.x = u(i,j,k,UMX)*rhoinv;
    v.y = u(i,j,k,UMY)*rhoinv;
    v.z = u(i,j,k,UMZ)*rhoinv; 
    amrex::Real kineng = 0.5*rho*(v.x*v.x + v.y*v.y + v.z*v.z); 
    q(i,j,k,QRHO) = rho; 
    q(i,j,k,QU) = v.x; 
    q(i,j,k,QV) = v.y;
    q(i,j,k,QW) = v.z; 
    for(int ipassive = 0; ipassive < npassive; ++ipassive){
        n = EOS::upass_map(ipassive); 
        nq = EOS::qpass_map(ipassive); 
        q(i,j,k,nq) = u(i,j,k,n)/rho; 
    }

    amrex::Real e = (u(i,j,k,UEDEN) - kineng)*rhoinv;
    amrex::Real T = u(i,j,k,UTEMP); 
    amrex::Real massfrac[NUM_SPECIES]; 
//    amrex::Real aux[naux]; 
    for(int sp = 0; sp < NUM_SPECIES; ++sp) massfrac[sp] = q(i,j,k,sp+QFS);
//    for(int ax = 0; ax < naux; ++ax) aux[ax] = q(i,j,k,ax+QFX);
    amrex::Real dpdr_e, dpde, gam1, cs, wbar, p; 
//  Use eos conversion;  
    EOS::ctop(massfrac, rho, e, T, p, dpdr_e, dpde, gam1, cs, wbar); 
    q(i,j,k,QTEMP) = T;
    q(i,j,k,QREINT) = e * rho;
    q(i,j,k,QPRES) = p;
    q(i,j,k,QGAME) = p/(e*rho) + 1.0; 

//Auxilary Fab
    qa(i,j,k,QDPDR)  = dpdr_e; 
    qa(i,j,k,QDPDE)  = dpde;
    qa(i,j,k,QGAMC)  = gam1; 
    qa(i,j,k,QC)     = cs; 
    qa(i,j,k,QCSML)  = amrex::max(small, small*cs); 
    qa(i,j,k,QRSPEC) = R/wbar; 
}

AMREX_GPU_DEVICE
inline
void PeleC_srctoprim(const int i, const int j, const int k,
                     amrex::Array4<const amrex::Real> const& q,
                     amrex::Array4<const amrex::Real> const& qa,
                     amrex::Array4<const amrex::Real> const& src, 
                     amrex::Array4<amrex::Real> const& srcq)
{
    int n, nq; 
    amrex::Real rhoinv; 
    amrex::Real srcrho;
    Real3 v; 
    Real3 um; 
//=============== Local loading Vars ==> Load once from global memory use multiple times ===========
    srcrho = src(i,j,k,URHO); 
    rhoinv = 1.e0/q(i,j,k,QRHO); 
    v.x = q(i,j,k,QU); 
    v.y = q(i,j,k,QV); 
    v.z = q(i,j,k,QW); 
    um.x = src(i,j,k,UMX); 
    um.y = src(i,j,k,UMY); 
    um.z = src(i,j,k,UMZ);
//============== SrcQ Calc =======================================================================
    srcq(i,j,k,QRHO  ) = srcrho; 
    srcq(i,j,k,QU    ) = (um.x - v.x*srcrho)*rhoinv; 
    srcq(i,j,k,QV    ) = (um.y - v.y*srcrho)*rhoinv; 
    srcq(i,j,k,QW    ) = (um.z - v.z*srcrho)*rhoinv; 
    srcq(i,j,k,QREINT) = src(i,j,k,UEDEN) - v.x*um.x - v.y*um.y - v.z*um.z 
                   + srcrho*(v.x*v.x + v.y*v.y + v.z*v.z);
    srcq(i,j,k,QPRES)  = qa(i,j,k,QDPDE)*(srcq(i,j,k,QREINT) - q(i,j,k,QREINT)*srcrho*rhoinv)*rhoinv
                   + qa(i,j,k,QDPDR)*srcrho; 

    for(int ip = 0; ip < npassive; ++ip)
    {
        n  = EOS::upass_map(ip); 
        nq = EOS::qpass_map(ip);
        srcq(i,j,k,nq) = ( src(i,j,k,n) - q(i,j,k,nq)*srcrho)*rhoinv;
    }                        
}

AMREX_GPU_DEVICE
inline 
void 
PeleC_divu(const int i, const int j, const int k, amrex::Array4<const amrex::Real> const &q, 
           D_DECL(const amrex::Real dx, const amrex::Real dy, const amrex::Real dz),
           amrex::Array4<amrex::Real> const& divu)
{
#if(AMREX_SPACEDIM==2)
     amrex::Real ux = 0.5e0*(q(i,j,k,QU) - q(i-1,j,k,QU) + 
                    q(i,j-1,k,QU) - q(i-1,j-1,k,QU))/dx; 
     amrex::Real vy = 0.5e0*(q(i,j,k,QV) - q(i,j-1,k,QV) + 
                    q(i-1,j,k,QV) - q(i-1,j-1,k,QV))/dy; 
     divu(i,j,k) = ux + vy; 
#elif(AMREX_SPACEDIM==3)
    amrex::Real ux = 0.25e0*(q(i  ,j  ,k  ,QU) - q(i-1,j  ,k  ,QU) 
                           + q(i  ,j  ,k-1,QU) - q(i-1,j  ,k-1,QU) 
                           + q(i  ,j-1,k  ,QU) - q(i-1,j-1,k  ,QU) 
                           + q(i  ,j-1,k-1,QU) - q(i-1,j-1,k-1,QU) )/dx; 

    amrex::Real vy = 0.25e0*(q(i  ,j  ,k  ,QV) - q(i  ,j-1,k  ,QV) 
                           + q(i  ,j  ,k-1,QV) - q(i  ,j-1,k-1,QV) 
                           + q(i-1,j  ,k  ,QV) - q(i-1,j-1,k  ,QV) 
                           + q(i-1,j  ,k-1,QV) - q(i-1,j-1,k-1,QV) )/dy; 

    amrex::Real wz = 0.25e0*(q(i  ,j  ,k  ,QW) - q(i  ,j  ,k-1,QW) 
                           + q(i-1,j  ,k  ,QW) - q(i-1,j  ,k-1,QW) 
                           + q(i  ,j-1,k  ,QW) - q(i  ,j-1,k-1,QW) 
                           + q(i-1,j-1,k  ,QW) - q(i-1,j-1,k-1,QW) )/dz; 
    divu(i,j,k) = ux + vy + wz; 
#endif
}

AMREX_GPU_DEVICE 
inline 
amrex::Real  
PeleC_pres(const int i , const int j, const int k, amrex::Array4<const amrex::Real> const &u)
{
    amrex::Real rho = u(i,j,k,URHO); 
    amrex::Real T   = u(i,j,k,UTEMP); 
    amrex::Real massfrac[NUM_SPECIES]; 
    amrex::Real rhoInv = 1.e0/rho; 
    amrex::Real p; 
    for(int n = 0; n < NUM_SPECIES; ++n) massfrac[n] = u(i,j,k,UFS+n)*rhoInv; 
    EOS::TrhoY2p(rho, T, massfrac, p); 
    return p;  
}

//======================================= Host functions ================================

void 
PeleC_umdrv(const int is_finest_level, const amrex::Real time, amrex::Box const &bx,
            const int* dlo, const int* dhi, const int* bclo, const int* bchi,  
            amrex::Array4<const amrex::Real> const &uin, 
            amrex::Array4<amrex::Real> const& uout, 
            amrex::Array4<const amrex::Real> const& q,
            amrex::Array4<const amrex::Real> const& qaux,
            amrex::Array4<const amrex::Real> const& src_q, //amrex::IArrayBox const& bcMask,
            const amrex::Real *dx, const amrex::Real dt, 
            D_DECL(amrex::Array4<amrex::Real> const& flux1,
                   amrex::Array4<amrex::Real> const& flux2, 
                   amrex::Array4<amrex::Real> const& flux3), 
            D_DECL(amrex::Array4<const amrex::Real> const& a1,
                   amrex::Array4<const amrex::Real> const& a2, 
                   amrex::Array4<const amrex::Real> const& a3), 
#if (AMREX_SPACEDIM < 3)
            amrex::Array4<amrex::Real> const &dloga, 
#endif
            amrex::Array4<amrex::Real> const &vol, amrex::Real cflLoc);

void PeleC_consup(amrex::Box const &bx, amrex::Array4<const amrex::Real> const& u, 
                  amrex::Array4<amrex::Real> const& update, 
           D_DECL(amrex::Array4<amrex::Real> const& flx1,
                  amrex::Array4<amrex::Real> const& flx2,
                  amrex::Array4<amrex::Real> const& flx3),
           D_DECL(amrex::Array4<const amrex::Real> const &a1,
                  amrex::Array4<const amrex::Real> const &a2, 
                  amrex::Array4<const amrex::Real> const &a3), 
                  amrex::Array4<const amrex::Real> const &vol,
                  amrex::Array4<const amrex::Real> const &div, 
                  amrex::Array4<const amrex::Real> const &pdivu,
                  amrex::Real const *del, amrex::Real const difmag);

AMREX_GPU_DEVICE
inline
void PeleC_norm_spec_flx(const int i, const int j, const int k,
                         amrex::Array4<amrex::Real> const& flx) 
{
    int num = NUM_SPECIES + UFS; 
    amrex::Real summ, fac;
    summ = 0.e0;  
    for(int n = UFS; n < num; ++n) summ += flx(i,j,k,n);

    fac = (summ != 0.e0) ? flx(i,j,k,URHO)/summ : 1.e0; 
    for(int n = UFS; n < num; ++n) flx(i,j,k,n) *= fac; 

}

AMREX_GPU_DEVICE
inline
void PeleC_ext_flx(const int i, const int j, const int k,
                   amrex::Array4<amrex::Real> const& flx,
                   amrex::Array4<const amrex::Real> const& a) 
{
    amrex::Real area = a(i,j,k); //avoid reading from global memory NVAR times
    for(int n = 0; n < NVAR; ++n) flx(i,j,k,n) *= area; 
}

AMREX_GPU_DEVICE
inline 
void PeleC_update(const int i, const int j, const int k,
                  amrex::Array4<amrex::Real> const& update,
                  D_DECL( 
                  amrex::Array4<const amrex::Real> const& flx1,
                  amrex::Array4<const amrex::Real> const& flx2, 
                  amrex::Array4<const amrex::Real> const& flx3), 
                  amrex::Array4<const amrex::Real> const& vol,
                  amrex::Array4<const amrex::Real> const& pdivu)
{
    amrex::Real v = vol(i,j,k); 
    for(int n = 0; n < NVAR; ++n)
    { 
#if(AMREX_SPACEDIM==2)
       update(i,j,k,n) += (flx1(i,j,k,n) - flx1(i+1,j,k,n)
                       +   flx2(i,j,k,n) - flx2(i,j+1,k,n))/v; 
#elif(AMREX_SPACEDIM==3)
       update(i,j,k,n) += (flx1(i,j,k,n) - flx1(i+1,j,k,n)
                       +   flx2(i,j,k,n) - flx2(i,j+1,k,n)
                       +   flx3(i,j,k,n) - flx3(i,j,k+1,n))/v;
#endif
    }

    update(i,j,k,UEINT) -= pdivu(i,j,k);     
}


//Derive routines 
void PeleC_derpres(const Box& bx, FArrayBox& pfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& ufab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/);

#endif 
