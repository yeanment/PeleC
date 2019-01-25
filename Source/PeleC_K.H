#ifndef PeleC_K_H_
#define PeleC_K_H_

#include <AMReX_FArrayBox.H>
#include "PeleC_Parameters.H" 
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
//=============== Local loading Vars ==> Load once from global memory use multiple times =============
                srcrho = src(i,j,k,URHO); 
                rhoinv = 1.e0/q(i,j,k,QRHO); 
                v.x = q(i,j,k,QU); 
                v.y = q(i,j,k,QV); 
                v.z = q(i,j,k,QW); 
                um.x = src(i,j,k,UMX); 
                um.y = src(i,j,k,UMY); 
                um.z = src(i,j,k,UMZ);
//=============== SrcQ Calc ==========================================================================
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


//TODO make consup a host function and split the interior up into device functions. 
AMREX_GPU_DEVICE
inline 
void PeleC_consup(amrex::Box const &bx, amrex::Array4<amrex::Real> const& u, 
                  amrex::Array4<amrex::Real> const& q, amrex::Array4<amrex::Real> &update, 
                  amrex::Array4<amrex::Real> const &q1, amrex::Array4<amrex::Real> const &q2, 
                  amrex::Array4<amrex::Real> &flx1 , amrex::Array4<amrex::Real> &flx2,
                  amrex::Array4<amrex::Real> const &a1   , amrex::Array4<amrex::Real> const &a2, 
                  amrex::Array4<amrex::Real> const &vol  , amrex::Array4<amrex::Real> const &div, 
                  amrex::Array4<amrex::Real> const &pdivu, amrex::Real const *dx,
                  amrex::Real const difmag)
{
    const auto len    = length(bx); 
    const auto lo     = lbound(bx); 
   
    amrex::Real div1; 
    amrex::Real summ; 
    amrex::Real fac; 
    
//============== Add some artificial viscosity ========================
//-------------------------- x-flux ----------------------------------- 
    for         (int k = 0; k < len.z; ++k){
        for     (int j = 0; j < len.y; ++j){
            AMREX_PRAGMA_SIMD
            for (int i = 0; i < len.x+1; ++i){
#pragma unroll
                for(int n=0; n < NVAR; ++n){
                    if(n == UTEMP){
                        flx1(i,j,k,n) = 0; 
                    }
                    else{
                        div1 = 0.5e0*(div(i,j,k) + div(i,j+1,k)); 
// Passing in difmag TODO not pass in. 
                        div1 = difmag*std::min(0.0, div1); 
                        flx1(i,j,k,n) = flx1(i,j,k,n) + dx[0]*div1*(u(i,j,k,n) - u(i-1,j,k,n));
                    }
                }
            }
        }
 //-------------------------- y-flux ----------------------------------- 
        for     (int j = 0; j < len.y+1; ++j){
            AMREX_PRAGMA_SIMD
            for (int i = 0; i < len.x; ++i){
#pragma unroll 
                for(int n=0; n < NVAR; ++n){
                    if(n == UTEMP){
                        flx2(i,j,k,n) = 0; 
                    }
                    else{
                        div1 = 0.5e0*(div(i,j,k) + div(i+1,j,k)); 
                        div1 = difmag*std::min(0.0, div1); 
                        flx2(i,j,k,n) = flx2(i,j,k,n) + dx[1]*div1*(u(i,j,k,n) - u(i,j-1,k,n)); 
                    }
                }
            }
        }
    }

// ===========  Normalize Species Fluxes =============================
// -------------------------- x-flux ---------------------------------
    for         (int k = 0; k < len.z; ++k){
        for     (int j = 0; j < len.y; ++j){
            AMREX_PRAGMA_SIMD
            for (int i = 0; i < len.x+1; ++i){
                summ = 0.0; 
                for(int n = UFS; n < NVAR; ++n) summ+= flx1(i,j,k,n);
                if(summ != 0.e0){
                    fac = flx1(i,j,k,URHO) / summ; 
                }
                else{
                    fac = 1.e0; 
                }
#pragma unroll
                for(int n = UFS; n < NVAR; ++n) flx1(i,j,k,n) *= fac; 
            }
        }

// ------------------------ y-flux -----------------------------------
               
        for     (int j = 0; j < len.y+1; ++j){
            AMREX_PRAGMA_SIMD
            for (int i = 0; i < len.x; ++i){
                summ = 0.0; 
                for(int n = UFS; n < NVAR; ++n) summ+= flx2(i,j,k,n);
                if(summ != 0.e0){
                    fac = flx2(i,j,k,URHO) / summ; 
                }
                else{
                    fac = 1.e0; 
                }
#pragma unroll
                for(int n = UFS; n < NVAR; ++n) flx2(i,j,k,n) *= fac; 
            }
        }
    }                                                 

// ============================ Extensive Fluxes =====================
// -------------------------- x-flux ---------------------------------
    for         (int k = 0; k < len.z; ++k){
        for     (int j = 0; j < len.y; ++j){
            AMREX_PRAGMA_SIMD
            for (int i = 0; i < len.x+1; ++i){
#pragma unroll 
                for(int n = 0; n < NVAR; ++n) flx1(i,j,k,n) *= a1(i,j,k);
            }
        }

// ------------------------ y-flux -----------------------------------
               
        for     (int j = 0; j < len.y+1; ++j){
            AMREX_PRAGMA_SIMD
            for (int i = 0; i < len.x; ++i){
                for(int n = 0; n < NVAR; ++n) flx2(i,j,k,n) *= a2(i,j,k); 
            }
        }
    }                                                 

// === Fill the update array (this is essenitally the sources fab) ===

    for         (int k = 0; k < len.z; ++k){
        for     (int j = 0; j < len.y; ++j){
            AMREX_PRAGMA_SIMD
            for (int i = 0; i < len.x; ++i){
#pragma unroll
                for(int n = 0; i < NVAR; ++n)
                    update(i,j,k,n) += (flx1(i,j,k,n) - flx1(i+1,j,k,n)
                                    +   flx2(i,j,k,n) - flx2(i,j+1,k,n))
                                    /   vol(i,j,k);
                update(i,j,k,UEINT) -= pdivu(i,j,k);  
            }
        }
    }   

}

AMREX_GPU_DEVICE
inline 
void 
PeleC_divu(const int i, const int j, const int k, amrex::Array4<amrex::Real> const &q, 
           const amrex::Real *dx, amrex::Array4<amrex::Real> &divu)
{
     amrex::Real ux = 0.5e0*(q(i,j,k,QU) - q(i-1,j,k,QU) + 
                    q(i,j-1,k,QU) - q(i-1,j-1,k,QU))/dx[0]; 
     amrex::Real vy = 0.5e0*(q(i,j,k,QV) - q(i,j-1,k,QV) + 
                    q(i-1,j,k,QV) - q(i-1,j-1,k,QV))/dx[1]; 
     divu(i,j,k) = ux + vy; 
}


void 
PeleC_umdrv(const int is_finest_llevel, const amrex::Real time, amrex::Box const &bx, 
//            amrex::Real const *dom_lo, amrex::Real const *dom_hi, 
            amrex::FArrayBox const &uin, 
            amrex::FArrayBox &uout, amrex::FArrayBox const &q, amrex::FArrayBox const &qaux,
            amrex::FArrayBox const &src_q, amrex::FArrayBox const &bcMask,
            const amrex::Real *dx, const amrex::Real dt, amrex::FArrayBox flux[], 
#if (AMREX_SPACEDIM < 3) 
            amrex::FArrayBox pradial, 
#endif
            D_DECL(amrex::FArrayBox const &a1, amrex::FArrayBox const &a2, 
                   amrex::FArrayBox const &a3), 
#if (AMREX_SPACEDIM < 3)
            amrex::FArrayBox const &dloga, 
#endif
            amrex::FArrayBox const &vol, amrex::Real cflLoc)
//TODO assess if we want verbose           
{

//    TODO Check for CFL violation, this is a reduce operation. 
//    amrex::Real cour = PeleC_cfl(bx, q, qaux, dt, dx, courno); 
//    if(cour > 1.e0) 
        
    //TODO Do Flattening when we go to PPM. 
    
    //Call the method!
    const auto q1bx = surroundingNodes(bx,0); 
    amrex::FArrayBox q1(q1bx,QVAR); 
#if AMREX_SPACEDIM > 1
    const auto q2bx = surroundingNodes(bx,1); 
    amrex::FArrayBox q2(q2bx,QVAR); 
#endif
#if AMREX_SPACEDIM > 2 
    const auto q3bx = surroundingNodes(bx,2); 
    amrex::FArrayBox q3(q3bx,QVAR); 
#endif

    amrex::FArrayBox divu(bx, 1); 
    amrex::FArrayBox pdivu(bx, 1); 

#if AMREX_SPACEDIM == 1
    PeleC_umeth1D(bx, q,  qaux, src_q, bcMask, flux[0], q1, pdivu, dx, dt);  
#elif AMREX_SPACEDIM==2 
    PeleC_umeth2D(bx, q,  qaux, src_q, bcMask, flux[0], flux[1], dloga, q1, q2, a1, a2, 
                  pdivu, vol, dx, dt); 
#else
    PeleC_umeth3D(bx, q,  qaux, src_q, bcMask, flux[0], flux[1], flux[2], 
                  dloga, q1, q2, q3, a1, a2, a3, pdivu, vol, dx, dt);   
#endif

    //divu 
    amrex::Array4<amrex::Real> qfab = q.array(), divfab = divu.array(); 
    AMREX_PARALLEL_FOR_3D (bx, i,j,k, {
        PeleC_divu(i,j,k, qfab, dx, divfab); 
    });

    //consup 
    amrex::Array4<amrex::Real> D_DECL(flxx = flux[0].array(),
                                      flxy = flux[1].array(), 
                                      flxz = flux[2].array()); 
    amrex::Array4<amrex::Real> D_DECL(a1fab = a1.array(), 
                                      a2fab = a2.array(), 
                                      a3fab = a3.array()); 
    amrex::Array4<amrex::Real> D_DECL(q1fab = q1.array(), 
                                      q2fab = q2.array(), 
                                      q3fab = q3.array()); 
    amrex::Array4<amrex::Real> uinfab = uin.array(), uoutfab = uout.array(); 
    amrex::Array4<amrex::Real> volfab = vol.array(), pdivufab = pdivu.array();

    //TODO have difmag be parm parsed
    amrex::Real difmag = 0.0;  
    AMREX_LAUNCH_DEVICE_LAMBDA (bx, tbx , {
    PeleC_consup(tbx, uinfab, qfab, uoutfab,
                D_DECL(q1fab, q2fab, q3fab), 
                D_DECL(flxx, flxy, flxz),
                D_DECL(a1fab, a2fab, a3fab), 
                volfab, divfab, pdivufab, dx, difmag); 
    }); 

}


#endif 
