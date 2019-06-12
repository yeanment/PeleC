#include "PeleC_K.H"
#include "PeleC_misc.H"
#if AMREX_SPACEDIM==2 
#include "PeleC_method_2D.H"
#elif AMREX_SPACEDIM==3
#include "PeleC_method_3D.H" 
#endif 

AMREX_GPU_DEVICE
void PeleC_cmpTemp(const int i, const int j, const int k, amrex::Array4<amrex::Real> const& State)
{
    EOS eos_state; 
    amrex::Real rho = State(i,j,k,URHO); 
    amrex::Real rhoInv = 1.0/ rho; 
    eos_state.rho = rho; 
    eos_state.T = State(i,j,k,UTEMP); 
    eos_state.e = State(i,j,k,UEINT)*rhoInv;
#pragma unroll  
    for(int n = 0; n < NUM_SPECIES; ++n){
         eos_state.massfrac[n] = State(i,j,k,UFS+n)*rhoInv;
    }
#pragma unroll      
    for(int n = 0; n < NUM_AUXILIARY; ++n) eos_state.aux[n] = State(i,j,k,UFX+n)*rhoInv; 
    
    eos_state.eos_re(); 
    State(i,j,k,UTEMP) = eos_state.T; 
}

AMREX_GPU_DEVICE
void PeleC_rst_int_e(const int i, const int j, const int k, amrex::Array4<amrex::Real> const& S)
{
    amrex::Real rho = S(i,j,k,URHO); 
    amrex::Real u = S(i,j,k,UMX)/rho; 
    amrex::Real v = S(i,j,k,UMY)/rho; 
    amrex::Real w = S(i,j,k,UMZ)/rho; 
    amrex::Real ke = 0.5e0*(u*u + v*v + w*w); 
    S(i,j,k,UEINT) = S(i,j,k,UEDEN) - rho*ke; 
}


void 
PeleC_umdrv(const int is_finest_level, const amrex::Real time, amrex::Box const &bx,
            const int* domlo, const int* domhi,
            const int* bclo,   const int* bchi, 
            amrex::Array4<const amrex::Real> const &uin, 
            amrex::Array4<amrex::Real> const& uout, 
            amrex::Array4<const amrex::Real> const& q,
            amrex::Array4<const amrex::Real> const& qaux,
            amrex::Array4<const amrex::Real> const& src_q,// amrex::IArrayBox const& bcMask,
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
            amrex::Array4<amrex::Real> const &vol, amrex::Real cflLoc)
{

//  Set Up for Hydro Flux Calculations 
    auto const& bxg2 = grow(bx, 2); 
    auto const& q1bx = surroundingNodes(bxg2,0); 
    Gpu::AsyncFab q1(q1bx, NGDNV); 
    amrex::Real const delx    = dx[0]; 

#if AMREX_SPACEDIM > 1
    amrex::Real const dely    = dx[1]; 
    auto const& q2bx = surroundingNodes(bxg2,1); 
    Gpu::AsyncFab q2(q2bx, NGDNV); 

#if AMREX_SPACEDIM > 2
    amrex::Real const delz    = dx[2]; 
    auto const& q3bx = surroundingNodes(bxg2,2); 
    Gpu::AsyncFab q3(q3bx, NGDNV); 
#endif
#endif


//  Temporary FArrayBoxes 
    Gpu::AsyncFab divu(bxg2, 1); 
    Gpu::AsyncFab pdivu(bx, 1); 
    auto const&  divarr = divu.array(); 
    auto const& pdivuarr = pdivu.array();

    BL_PROFILE_VAR("PeleC::umeth()", umeth); 
#if AMREX_SPACEDIM == 1
    Abort("No 1D implementation!"); 
/*    PeleC_umeth_1D(bx, bclo, bchi, domlo, domhi,  q,  qaux, src_q, 
                   bcMask, flux1, q1, pdivu, dx, dt);  */ 
#elif AMREX_SPACEDIM==2 
    PeleC_umeth_2D(bx, bclo, bchi, domlo, domhi,q,  qaux, src_q,// bcMask, 
                   flux1, flux2, dloga,
                   q1.array(), q2.array(), a1, a2, pdivuarr, vol, dx, dt); 
#else
    auto const& q1arr = q1.array(); 
    auto const& q2arr = q2.array(); 
    auto const& q3arr = q3.array(); 

    PeleC_umeth_3D(bx, bclo, bchi, domlo, domhi, q,  qaux, src_q, //bcMask,
                   flux1, flux2, flux3,  q1arr, q2arr, q3arr, a1, a2, a3, pdivuarr, vol, dx, dt);   
#endif
    BL_PROFILE_VAR_STOP(umeth); 

    //divu 
    AMREX_PARALLEL_FOR_3D (bxg2, i,j,k, {
        PeleC_divu(i,j,k, q, D_DECL(delx, dely, delz), divarr); 
    });

    //consup 
    amrex::Real difmag = 0.1e0; 

    PeleC_consup(bx, uin, uout,
                 D_DECL(flux1, flux2, flux3),
                 D_DECL(a1, a2, a3), 
                 vol, divarr, pdivuarr, dx, difmag); 
}


void PeleC_derpres(const Box& bx, FArrayBox& pfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& ufab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const u = ufab.array();
    auto       p    = pfab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        p(i,j,k,dcomp) = PeleC_pres(i, j, k, u);
    });
}



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
                  amrex::Real const *del, amrex::Real const difmag)
{
    
//============== Flux alterations ========================
//-------------------------- x-flux -----------------------------------
    amrex::Box const &xfbx = surroundingNodes(bx, 0); 
    const amrex::Real dx  = del[0]; 

    AMREX_PARALLEL_FOR_3D(xfbx, i, j, k, {
        PeleC_artif_visc(i,j,k,flx1, div, u, dx, difmag, 0);
        //Normalize Species Flux
        PeleC_norm_spec_flx(i,j,k, flx1); 
        //Make flux extensive
        PeleC_ext_flx(i,j,k,flx1, a1);                            
    });
#if(AMREX_SPACEDIM>1)
//------------------------- y-flux ------------------------------------
    const amrex::Real dy  = del[1];  
    amrex::Box const &yfbx = surroundingNodes(bx, 1);
    AMREX_PARALLEL_FOR_3D(yfbx, i, j, k, {
        //Artificial Viscosity! 
        PeleC_artif_visc(i,j,k,flx2, div, u, dy, difmag, 1); 
        //Normalize Species Flux 
        PeleC_norm_spec_flx(i,j,k,flx2); 
        //Make flux extensive
        PeleC_ext_flx(i,j,k,flx2,a2); 
    };);
#if(AMREX_SPACEDIM>2)
//------------------------- y-flux ------------------------------------
    const amrex::Real dz  = del[2];  
    amrex::Box const &zfbx = surroundingNodes(bx, 2);
    AMREX_PARALLEL_FOR_3D(zfbx, i, j, k, {
        //Artificial Viscosity! 
        PeleC_artif_visc(i,j,k,flx3, div, u, dz, difmag, 2); 
        //Normalize Species Flux 
        PeleC_norm_spec_flx(i,j,k,flx3); 
        //Make flux extensive
        PeleC_ext_flx(i,j,k,flx3,a3); 
    };);
#endif
#endif
//================ Combine for Hydro Sources ==========================
      
    AMREX_PARALLEL_FOR_3D(bx, i, j, k, {
        PeleC_update(i,j,k, update, D_DECL(flx1, flx2, flx3), vol, pdivu); 
    };);
}
