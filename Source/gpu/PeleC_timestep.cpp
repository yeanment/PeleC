#include <PeleC_transport.H>
#include <PeleC_simple_transport.H> 
#include "PeleC_EOS.H" 
#include "PeleC_timestep.H" 
#include <PeleC_index_macros.H>

/*================================= EstDt routines! =================================*/ 

AMREX_GPU_HOST_DEVICE
amrex::Real 
PeleC_estdt_hydro ( amrex::Box const& bx, amrex::FArrayBox const& statefab, 
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
                state.rho = u(i,j,k,URHO); 
                rhoInv = 1.e0/state.rho; 
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

AMREX_GPU_HOST_DEVICE
inline
void PeleC_trans4dt(const int which_trans,EOS eos, amrex::Real &D)
{
    bool get_xi = false, get_mu = false, get_lam = false, get_Ddiag = false; 
    amrex::Real dum1 = 0., dum2 = 0.;
 
    if(which_trans==0){
        get_mu = true; 
        PeleC_actual_transport(get_xi, get_mu, get_lam, get_Ddiag, eos.T, eos.rho, eos.massfrac, 
                               nullptr, D, dum1, dum2); 
    }
    else if(which_trans==1){
        get_lam = true; 
        PeleC_actual_transport(get_xi, get_mu, get_lam, get_Ddiag, eos.T, eos.rho, eos.massfrac, 
                               nullptr, dum1, dum2, D);        
    }
}

/*Diffusion Velocity */ 
AMREX_GPU_HOST_DEVICE 
amrex::Real PeleC_estdt_veldif(amrex::Box const box, amrex::FArrayBox const& statefab,
             D_DECL(const amrex::Real& dx, const amrex::Real& dy, const amrex::Real& dz)) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    const auto u = statefab.array(); 
#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ !=9) || (__CUDACC_VER_MINOR__ != 2) 
    amrex::Real dt = std::numeric_limits<amrex::Real>::max();
#else 
    amrex::Real dt = 1.e37;
#endif
    amrex::Real rhoInv;  
    amrex::Real D, dt1;
#if AMREX_SPACEDIM > 1
    amrex::Real dt2;
#if AMREX_SPACEDIM > 2
    amrex::Real dt3;
#endif
#endif
    EOS eos; 
    int which_trans= 0; 
    for         (int k = lo.z; k <= hi.z; ++k){
        for     (int j = lo.y; j <= hi.y; ++j){
            for (int i = lo.x; i <= hi.x; ++i){
                eos.rho = u(i,j,k,URHO); 
                rhoInv = 1.e0/eos.rho;         
                #pragma unroll 
                for(int n = 0; n < NUM_SPECIES; ++n){
                     eos.massfrac[n] = u(i,j,k,n+UFS) * rhoInv;    
                } 
                eos.T = u(i,j,k,UTEMP);
                PeleC_trans4dt(which_trans, eos, D);
                D *= rhoInv; 
                dt1 = 0.5e0*dx*dx/(AMREX_SPACEDIM*D);
                dt  = amrex::min(dt, dt1);  
#if AMREX_SPACEDIM > 1 
                dt2 = 0.5e0*dy*dy/(AMREX_SPACEDIM*D); 
                dt  = amrex::min(dt, dt2); 
#if AMREX_SPACEDIM > 2 
                dt3 = 0.5e0*dz*dz/(AMREX_SPACEDIM*D); 
                dt  = amrex::min(dt,dt3); 
#endif 
#endif
           }
        }
    }
            
    return dt;
}

/*Diffusion Temperature */ 
AMREX_GPU_HOST_DEVICE 
amrex::Real PeleC_estdt_tempdif(amrex::Box const bx, amrex::FArrayBox const& statefab,
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
    amrex::Real rhoInv;  
    amrex::Real D, dt1;
#if AMREX_SPACEDIM > 1
    amrex::Real dt2;
#if AMREX_SPACEDIM > 2
    amrex::Real dt3;
#endif
#endif
    EOS eos; 
    int which_trans = 1; 
    for         (int k = lo.z; k <= hi.z; ++k){
        for     (int j = lo.y; j <= hi.y; ++j){
            for (int i = lo.x; i <= hi.x; ++i){
                eos.rho = u(i,j,k,URHO); 
                rhoInv = 1.e0/eos.rho;         
                #pragma unroll 
                for(int n = 0; n < NUM_SPECIES; ++n) eos.massfrac[n] = u(i,j,k,n+UFS) * rhoInv;    
                eos.T = u(i,j,k,UTEMP); 
                PeleC_trans4dt(which_trans, eos, D);
                eos.eos_cv(); 
                D *= rhoInv/eos.cv; 
                dt1 = 0.5e0*dx*dx/(AMREX_SPACEDIM*D);
                dt  = amrex::min(dt, dt1);  
#if AMREX_SPACEDIM > 1 
                dt2 = 0.5e0*dy*dy/(AMREX_SPACEDIM*D); 
                dt  = amrex::min(dt, dt2); 
#if AMREX_SPACEDIM > 2 
                dt3 = 0.5e0*dz*dz/(AMREX_SPACEDIM*D); 
                dt  = amrex::min(dt,dt3); 
#endif 
#endif
           }
        }
    }
            
    return dt;
}

/* Diffusion Enthalpy */ 
AMREX_GPU_HOST_DEVICE 
amrex::Real PeleC_estdt_enthdif(amrex::Box const bx, amrex::FArrayBox const& statefab,
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
    amrex::Real rhoInv;  
    amrex::Real D, dt1;
#if AMREX_SPACEDIM > 1
    amrex::Real dt2;
#if AMREX_SPACEDIM > 2
    amrex::Real dt3;
#endif
#endif
    EOS eos; 
    int which_trans = 1; 
    for         (int k = lo.z; k <= hi.z; ++k){
        for     (int j = lo.y; j <= hi.y; ++j){
            for (int i = lo.x; i <= hi.x; ++i){
                eos.rho = u(i,j,k,URHO); 
                rhoInv = 1.e0/eos.rho;         
                #pragma unroll 
                for(int n = 0; n < NUM_SPECIES; ++n) eos.massfrac[n] = u(i,j,k,n+UFS) * rhoInv;    
                eos.e = u(i,j,k,UEINT)*rhoInv; 
                eos.T = u(i,j,k,UTEMP); 
                eos.eos_re(); 
                PeleC_trans4dt(which_trans, eos, D); 
                D *= rhoInv/eos.cp; 
                dt1 = 0.5e0*dx*dx/(AMREX_SPACEDIM*D);
                dt  = amrex::min(dt, dt1);  
#if AMREX_SPACEDIM > 1 
                dt2 = 0.5e0*dy*dy/(AMREX_SPACEDIM*D); 
                dt  = amrex::min(dt, dt2); 
#if AMREX_SPACEDIM > 2 
                dt3 = 0.5e0*dz*dz/(AMREX_SPACEDIM*D); 
                dt  = amrex::min(dt,dt3); 
#endif 
#endif
           }
        }
    }
            
    return dt;
}

