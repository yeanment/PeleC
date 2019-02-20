#include "PeleC_K.H" 

void 
PeleC_umdrv(const int is_finest_level, const amrex::Real time, amrex::Box const &bx, 
            amrex::Array4<const amrex::Real> const &uin, 
            amrex::Array4<amrex::Real> const& uout, 
            amrex::Array4<const amrex::Real> const& q,
            amrex::Array4<const amrex::Real> const& qaux,
            amrex::Array4<const amrex::Real> const& src_q, amrex::IArrayBox const& bcMask,
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

//    TODO Check for CFL violation, this is a reduce operation. 
//    amrex::Real cour = PeleC_cfl(bx, q, qaux, dt, dx, courno); 
//    if(cour > 1.e0) 
        
    //TODO Do Flattening when we go to PPM. 
    
    //Call the method!
    const auto& bxg2 = grow(bx, 2); 
    const auto q1bx = surroundingNodes(bxg2,0); 
    Gpu::AsyncFab q1(q1bx, NGDNV); 
#if AMREX_SPACEDIM > 1
    const auto q2bx = surroundingNodes(bxg2,1); 
    Gpu::AsyncFab q2(q2bx, NGDNV); 
#endif
#if AMREX_SPACEDIM > 2 
    const auto q3bx = surroundingNodes(bxg2,2); 
    Gpu::AsyncFab q3(q3bx, NGDNV); 
#endif

    Gpu::AsyncFab divu(bxg2, 1); 
    Gpu::AsyncFab pdivu(bx, 1); 
    auto const&  divarr = divu.array(); 
    auto const& pdivuarr = pdivu.array();

#if AMREX_SPACEDIM == 1
    PeleC_umeth_1D(bx, q,  qaux, src_q, bcMask, flux1, q1, pdivu, dx, dt);  
#elif AMREX_SPACEDIM==2 
    PeleC_umeth_2D(bx, q,  qaux, src_q, bcMask, flux1, flux2, dloga, q1.array(), q2.array(), a1, a2, 
                   pdivuarr, vol, dx, dt); 
#else
    PeleC_umeth_3D(bx, q,  qaux, src_q, bcMask, flux1, flux2, flux3, 
                   q1, q2, q3, a1, a2, a3, pdivu, vol, dx, dt);   
#endif

    //divu 
    const amrex::Real delx = dx[0]; 
    const amrex::Real dely = dx[1];  
    AMREX_PARALLEL_FOR_3D (bxg2, i,j,k, {
        PeleC_divu(i,j,k, q, delx, dely, divarr); 
    });
    //consup 

/* Only used for cylindrical coords "at this point" 
    auto const& D_DECL(q1fab = q1.array(), 
                       q2fab = q2.array(), 
                       q3fab = q3.array()); */ 


    //TODO have difmag be parm parsed
    amrex::Real difmag = 0.1e0; //0.005e0; 
    PeleC_consup(bx, uin, uout,
                 D_DECL(flux1, flux2, flux3),
                 D_DECL(a1, a2, a3), 
                 vol, divarr, pdivuarr, dx, difmag); 
}


//NOTE THIS IS ONLY FOR 2D! 
void PeleC_consup(amrex::Box const &bx, amrex::Array4<const amrex::Real> const& u, 
                  amrex::Array4<amrex::Real> const& update, 
                  amrex::Array4<amrex::Real> const& flx1,
                  amrex::Array4<amrex::Real> const& flx2,
                  amrex::Array4<const amrex::Real> const &a1,
                  amrex::Array4<const amrex::Real> const &a2, 
                  amrex::Array4<const amrex::Real> const &vol,
                  amrex::Array4<const amrex::Real> const &div, 
                  amrex::Array4<const amrex::Real> const &pdivu,
                  amrex::Real const *del, amrex::Real const difmag)
{
    
//============== Flux alterations ========================
//-------------------------- x-flux -----------------------------------
    amrex::Box const &xfbx = surroundingNodes(bx, 0); 
    const amrex::Real dx  = del[0]; 
    const amrex::Real dy  = del[1];  

    AMREX_PARALLEL_FOR_3D(xfbx, i, j, k, {
        PeleC_artif_visc(i,j,k,flx1, div, u, dx, difmag, 0);
        //Normalize Species Flux
        PeleC_norm_spec_flx(i,j,k, flx1); 
        //Make flux extensive
        PeleC_ext_flx(i,j,k,flx1, a1);                            
    });
//------------------------- y-flux ------------------------------------

    amrex::Box const &yfbx = surroundingNodes(bx, 1); 
    AMREX_PARALLEL_FOR_3D(yfbx, i, j, k, {
        //Artificial Viscosity! 
        PeleC_artif_visc(i,j,k,flx2, div, u, dy, difmag, 1); 
        //Normalize Species Flux 
        PeleC_norm_spec_flx(i,j,k,flx2); 
        //Make flux extensive
        PeleC_ext_flx(i,j,k,flx2,a2); 
    };);
//================ Combine for Hydro Sources ==========================

    AMREX_PARALLEL_FOR_3D(bx, i, j, k, {
        PeleC_update(i,j,k, update, flx1, flx2, vol, pdivu); 
    };);
}
