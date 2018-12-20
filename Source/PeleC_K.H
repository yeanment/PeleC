#ifndef PeleC_K_H_
#define PeleC_K_H_

#include <AMReX_FArrayBox.H>
#include "PeleC_Parameters.H" 
#include "PeleC_index_macros.H" 
#include "PeleC_EOS.H"

/*AMREX_GPU_DEVICE
void PeleC_ctoprim(amrex::Box const& bx, amrex::FArrayBox const& state, amrex::FArrayBox& Q,
                   amrex::FArrayBox& Qaux); 
AMREX_GPU_DEVICE
void PeleC_srctoprim(amrex::Box const& bx, amrex::FArrayBox const& qfab, amrex::FArrayBox const& qauxfab,
                   amrex::FArrayBox const& srcfab, amrex::FArrayBox &srcQfab); */
//TODO Add more fuctions! 
struct Real3{
    amrex::Real x;
    amrex::Real y; 
    amrex::Real z; 
};

AMREX_GPU_DEVICE
inline
void PeleC_ctoprim(amrex::Box const& bx, amrex::FArrayBox const& ufab, amrex::FArrayBox & qfab,
                   amrex::FArrayBox & qauxfab)
{

    const auto len = length(bx); 
    const auto lo  = lbound(bx); 
    const auto u   = ufab.view(lo); 
    const auto q   = qfab.view(lo); 
    const auto qa  = qauxfab.view(lo); 
    const amrex::Real smallr = 1.e-19; 
    const amrex::Real smallp = 1.e-10; 
    amrex::Real R = k_B*n_A; 
    EOS eos_state; 

    for         (int k = 0; k < len.z; ++k){
        for     (int j = 0; j < len.y; ++j){
            AMREX_PRAGMA_SIMD
            for (int i = 0; i < len.x; ++i){
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
//TODO figure out massfrac and auxilary vars
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
        }
    }

}

AMREX_GPU_DEVICE
inline
void PeleC_srctoprim(amrex::Box const& bx, amrex::FArrayBox const& qfab, amrex::FArrayBox const& qauxfab,
                   amrex::FArrayBox const& srcfab, amrex::FArrayBox &srcQfab)
{
    const auto len = length(bx); 
    const auto lo  = lbound(bx); 
    const auto q   = qfab.view(lo); 
    const auto qa  = qauxfab.view(lo); 
    const auto src  = srcfab.view(lo); 
    const auto srcq = srcQfab.view(lo); 
    int n, nq; 
    amrex::Real rhoinv; 
    amrex::Real srcrho;
    Real3 v; 
    Real3 um; 
    for         (int k = 0; k < len.z; ++k){
        for     (int j = 0; j < len.y; ++j){
            AMREX_PRAGMA_SIMD
            for (int i = 0; i < len.x; ++i){
//--------------- Local loading Vars ==> Load once from global memory use multiple times -------------------
                srcrho = src(i,j,k,URHO); 
                rhoinv = 1.e0/q(i,j,k,QRHO); 
                v.x = q(i,j,k,QU); 
                v.y = q(i,j,k,QV); 
                v.z = q(i,j,k,QW); 
                um.x = src(i,j,k,UMX); 
                um.y = src(i,j,k,UMY); 
                um.z = src(i,j,k,UMZ);
//--------------- SrcQ Calc ---------------------------------------------------------------------------------
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
        }
    }
}


/*
AMREX_GPU_DEVICE
inline 
void PeleC_consup(amrex::Box const &bx, amrex::FArrayBox const& uifab, amrex::FArrayBox const& qfab, 
                  amrex::FArrayBox &uofab, amrex:FArrayBox const& upfab, 
                  amrex::FArrayBox const &q1fab, amrex::FArrayBox const &q2fab, 
                  amrex::FArrayBox const &flx1fab , amrex::FArrayBox const &flx2fab,
                  amrex::FArrayBox const &a1fab   , amrex::FArrayBox const &a2fab, 
                  amrex::FArrayBox const &volfab  , amrex::FArrayBox const &divfab, 
                  amrex::FArrayBox const &pdivufab, amrex:Real const *dx, amrex::Real *added_flux, 
                  amrex::Real *lost)
{
    const auto len   = length(bx); 
    const auto lo    = lbound(bx); 
    const auto u     = ufab.view(lo); 
    const auto q     = qfab.view(lo); 
    const auto uo    = uofab.view(lo); 
    const auto up    = upfab.view(lo); 
    const auto q1    = q1fab.view(lo); 
    const auto q2    = q2fab.view(lo); 
    const auto flx1  = flx1fab.view(lo); 
    const auto flx2  = flx2fab.view(lo); 
    const auto a1    = a1fab.view(lo); 
    const auto a2    = a2fab.view(lo); 
    const auto vol   = volfab.view(lo); 
    const auto div   = divfab.view(lo); 
    const auto pdivu = pdivufab.view(lo); 
   
    amrex::Real div1; 

    for         (int k = 0; k < len.z; ++k){
        for     (int j = 0; j < len.y; ++j){
            AMREX_PRAGMA_SIMD
            for (int i = 0; i < len.x; ++i){
//-------------- Add some artificial viscosity ------------------------
                for(int n=0; n < NVAR; ++n){
                    if(n == UTEMP){
                        flx1(i,j,k,n) = 0; 
                        flx2(i,j,k,n) = 0; 
                    }
                    else{
                        div1 = 0.5e0*(div(i,j,k) + div(i,j+1,k)); 
                        //  Do we know what difmag is?                     
                        div1 = difmag*std::min(0, div1); 
                        flx1(i,j,k,n) = flx1(i,j,k,n) + dx*div1*(uin(i,j,k,n) - uin(i-1,j,k,n));
                        div1 = 0.5e0*(div(i,j,k) + div(i+1,j,k)); 
                        div1 = difmag*std::min(0, div1); 
                        flx2(i,j,k,n) = flx2(i,j,k,n) + dy*div1*(uin(i,j,k,n) - uin(i,j-1,k,n)); 
                    }
                }
// -------------  Normalize Species Fluxes -----------------------------
                amrex::Real summ = 0.0; 
                for(int n = UFS; n < NVAR; ++n){
                    summ+= flx2(i,j,k,n) 
                }
                                 
            }
        }
    }
}
*/
#endif 
