#include "PeleC_Parameters.H" 
#include "PeleC_index_macros.H" 
#include "PeleC_EOS.H"
#include "PeleC_K.H" 

struct Real3{
    amrex::Real x;
    amrex::Real y; 
    amrex::Real z; 
};

AMREX_GPU_DEVICE
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
