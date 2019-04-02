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

//TODO Add more fuctions! 
struct Real3{
    amrex::Real x;
    amrex::Real y; 
    amrex::Real z; 
};

AMREX_GPU_HOST_DEVICE
inline 
amrex::Real 
PeleC_estdt ( amrex::Box const& bx, amrex::FArrayBox const& statefab, 
             D_DECL(const amrex::Real& dx, const amrex::Real& dy, const amrex::Real& dz)) noexcept 
{
    const auto lo = amrex::lbound(bx); 
    const auto hi = amrex::ubound(bx); 
    const auto u = statefab.array(); 
#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ !=9) || (__CUDACC_VER_MINOR__ != 2) 
    amrex::Real dt = std::numeric_limits<amrex::Real>::max(); 
#else 
    amrex::Real dt = 1.e37;
#endif
    EOS state;  
    amrex::Real rhoInv; 
    amrex::Real ux, dt1, c; 
#if AMREX_SPACEDIM > 1
    amrex::Real uy, dt2; 
#if AMREX_SPACEDIM > 2
    amrex::Real uz, dt3; 
#endif
#endif 

    for         (int k = lo.z; k <= hi.z; ++k){
        for     (int j = lo.y; j <= hi.y; ++j){
            for (int i = lo.x; i <= hi.x; ++i){
                rhoInv = 1.e0/ u(i,j,k,URHO); 
                state.rho = u(i,j,k,URHO);
                state.T   = u(i,j,k,UTEMP); 
                state.e   = u(i,j,k,UEINT)*rhoInv; 
                for(int n = 0; n < NUM_SPECIES; ++n) state.massfrac[n] = u(i,j,k,UFS+n)*rhoInv; 
                for(int n = 0; n < NUM_AUXILIARY; ++n) state.aux[n] = u(i,j,k,UFX+n)*rhoInv; 
                state.eos_re();

                c = state.cs;
                ux = u(i,j,k,UMX)*rhoInv; 
               dt1 = dx/(c+std::abs(ux)); 
                dt = amrex::min(dt, dt1); 
#if AMREX_SPACEDIM > 1
                uy = u(i,j,k,UMY)*rhoInv; 
               dt2 = dy/(c+std::abs(uy));
                dt = amrex::min(dt, dt2);               
#if AMREX_SPACEDIM > 2
                uz = u(i,j,k,UMZ)*rhoInv; 
               dt3 = dz/(c+std::abs(uz)); 
                dt = amrex::min(dt, dt3); 
#endif
#endif
           }
        }
    }
            
    return dt; 

}


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
    EOS eos_state; 

    Real3 v; 
    int n, nq; 
    eos_state.rho = u(i,j,k,URHO); 
    amrex::Real rhoinv = 1.0/eos_state.rho; 
    v.x = u(i,j,k,UMX)*rhoinv;
    v.y = u(i,j,k,UMY)*rhoinv;
    v.z = u(i,j,k,UMZ)*rhoinv; 
    amrex::Real kineng = 0.5*eos_state.rho*(v.x*v.x + v.y*v.y + v.z*v.z); 
    q(i,j,k,QRHO) = eos_state.rho; 
    q(i,j,k,QU) = v.x; 
    q(i,j,k,QV) = v.y;
    q(i,j,k,QW) = v.z; 
// Maybe #pragma unroll                 
    for(int ipassive = 0; ipassive < npassive; ++ipassive){
        n = eos_state.upass_map(ipassive); 
        nq = eos_state.qpass_map(ipassive); 
        q(i,j,k,nq) = u(i,j,k,n)/eos_state.rho; 
    }

    eos_state.e = (u(i,j,k,UEDEN) - kineng)*rhoinv;
    eos_state.T = u(i,j,k,UTEMP); 
#pragma unroll 
    for(int sp = 0; sp < NUM_SPECIES; ++sp) eos_state.massfrac[sp] = q(i,j,k,sp+QFS);
#pragma unroll
    for(int ax = 0; ax < naux; ++ax) eos_state.aux[ax] = q(i,j,k,ax+QFX); 
// Call eos_re
    eos_state.eos_re(); 
    q(i,j,k,QTEMP) = eos_state.T;
    q(i,j,k,QREINT) = eos_state.e * eos_state.rho;
    q(i,j,k,QPRES) = eos_state.p;
    q(i,j,k,QGAME) = eos_state.p/(eos_state.e*eos_state.rho) + 1.0; 

//Auxilary Fab
    qa(i,j,k,QDPDR)  = eos_state.dpdr_e; 
    qa(i,j,k,QDPDE)  = eos_state.dpde;
    qa(i,j,k,QGAMC)  = eos_state.gam1; 
    qa(i,j,k,QC)     = eos_state.cs; 
    qa(i,j,k,QCSML)  = amrex::max(small, small*qa(i,j,k,QC)); 
    qa(i,j,k,QRSPEC) = R/eos_state.wbar; 
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

#pragma unroll 
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
    amrex::Real ux = 0.25e0*(q(i,j,k,QU) - q(i-1,j,k,QU) 
                           + q(i,j,k-1,QU) - q(i-1,j,k-1,QU) 
                           + q(i,j-1,k  ,QU) - q(i-1,j-1,k,QU) 
                           + q(i,j-1,k-1,QU) - q(i-1,j-1,k-1,QU) )/dx; 

    amrex::Real vy = 0.25e0*(q(i,j,k,QV) - q(i,j-1,k,QV) 
                           + q(i,j,k-1,QV) - q(i,j-1,k-1,QV) 
                           + q(i-1,j,k  ,QV) - q(i-1,j-1,k,QV) 
                           + q(i-1,j,k-1,QV) - q(i-1,j-1,k-1,QV) )/dy; 

    amrex::Real wz = 0.25e0*(q(i,j,k,QW) - q(i,j,k-1,QW) 
                           + q(i-1,j,k,QW) - q(i-1,j,k-1,QW) 
                           + q(i,j-1,k  ,QW) - q(i,j-1,k-1,QW) 
                           + q(i-1,j-1,k,QW) - q(i-1,j-1,k-1,QW) )/dz; 
    divu(i,j,k) = ux + vy + wz; 
#endif
}

AMREX_GPU_DEVICE 
inline 
amrex::Real  
PeleC_pres(const int i , const int j, const int k, amrex::Array4<const amrex::Real> const &u)
{
    EOS state; 
    amrex::Real rhoInv = 1.e0/ u(i,j,k,URHO); 
    state.rho = u(i,j,k,URHO); 
    state.T   = u(i,j,k,UTEMP); 
    state.e   = u(i,j,k,UEINT)*rhoInv; 
    for(int n = 0; n < NUM_SPECIES; ++n) state.massfrac[n] = u(i,j,k,UFS+n)*rhoInv; 
    for(int n = 0; n < NUM_AUXILIARY; ++n) state.aux[n] = u(i,j,k,UFX+n)*rhoInv;
    state.eos_re(); 
    return state.p;  
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

// ====================================== Consup Support Functions =======================================
// Designed for CPU or GPU 
AMREX_GPU_DEVICE
inline 
void PeleC_artif_visc(const int i, const int j, const int k,
                      amrex::Array4<amrex::Real> const& flx,
                      amrex::Array4<const amrex::Real> const& div, 
                      amrex::Array4<const amrex::Real> const &u,
                      amrex::Real const dx, amrex::Real const difmag,  const int cdir)
{
    amrex::Real div1;
    if(cdir==0){
        div1 = 0.5e0*(div(i,j,k) + div(i,j+1,k)); 
        div1 = difmag*amrex::min(0.0, div1); 

#pragma unroll
        for(int n=0; n < NVAR; ++n){
            if(n != UTEMP){
// Passing in difmag TODO not pass in. 
                flx(i,j,k,n) += dx*div1*(u(i,j,k,n) - u(i-1,j,k,n));
            }
        }
    }
    else if(cdir==1){
        div1 = 0.5e0*(div(i,j,k) + div(i+1,j,k)); 
        div1 = difmag*amrex::min(0.0, div1); 
#pragma unroll 
        for(int n=0; n < NVAR; ++n){
            if(n  != UTEMP){
                flx(i,j,k,n) += dx*div1*(u(i,j,k,n) - u(i,j-1,k,n)); 
            }
        }
    }
    flx(i,j,k,UTEMP) = 0.e0;  
}

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
#pragma unroll 
    for(int n = UFS; n < num; ++n) flx(i,j,k,n) *= fac; 

}

AMREX_GPU_DEVICE
inline
void PeleC_ext_flx(const int i, const int j, const int k,
                   amrex::Array4<amrex::Real> const& flx,
                   amrex::Array4<const amrex::Real> const& a) 
{
    amrex::Real area = a(i,j,k); //avoid reading from global memory NVAR times
#pragma unroll 
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
#pragma unroll 
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
