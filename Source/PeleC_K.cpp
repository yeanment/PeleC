#include "PeleC_K.H" 

void 
PeleC_umdrv(const int is_finest_level, const amrex::Real time, amrex::Box const &bx, 
            amrex::FArrayBox const &uin, 
            amrex::FArrayBox &uout, amrex::FArrayBox const &q, amrex::FArrayBox const &qaux,
            amrex::FArrayBox const &src_q, amrex::IArrayBox const &bcMask,
            const amrex::Real *dx, const amrex::Real dt, 
            D_DECL(amrex::FArrayBox &flux1, amrex::FArrayBox &flux2, 
                   amrex::FArrayBox &flux3), 
            D_DECL(amrex::FArrayBox const &a1, amrex::FArrayBox const &a2, 
                   amrex::FArrayBox const &a3), 
#if (AMREX_SPACEDIM < 3)
            amrex::FArrayBox const &dloga, 
#endif
            amrex::FArrayBox const &vol, amrex::Real cflLoc)
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

    Gpu::AsyncFab divu(bx, 1); 
    Gpu::AsyncFab pdivu(bx, 1); 

    amrex::Print() << " Calling umeth! " << std::endl; 
#if AMREX_SPACEDIM == 1
    PeleC_umeth_1D(bx, q,  qaux, src_q, bcMask, flux1, q1, pdivu, dx, dt);  
#elif AMREX_SPACEDIM==2 
    PeleC_umeth_2D(bx, q,  qaux, src_q, bcMask, flux1, flux2, dloga, q1.fab(), q2.fab(), a1, a2, 
                  pdivu.fab(), vol, dx, dt); 
#else
    PeleC_umeth_3D(bx, q,  qaux, src_q, bcMask, flux1, flux2, flux3, 
                   q1, q2, q3, a1, a2, a3, pdivu, vol, dx, dt);   
#endif

    //divu 
    auto const& qfab = q.array();
    auto const&  divfab = divu.array(); 
    amrex::Print() << "Calling divu! " << std::endl; 

    AMREX_PARALLEL_FOR_3D (bx, i,j,k, {
        PeleC_divu(i,j,k, qfab, dx, divfab); 
    });

    //consup 
    auto const& D_DECL(flxx = flux1.array(),
                       flxy = flux2.array(), 
                       flxz = flux3.array()); 

    auto const& D_DECL(a1fab = a1.array(), 
                       a2fab = a2.array(), 
                       a3fab = a3.array()); 

/* Only used for cylindrical coords "at this point" 
    auto const& D_DECL(q1fab = q1.array(), 
                       q2fab = q2.array(), 
                       q3fab = q3.array()); */ 

    auto const& uinfab = uin.array();
    auto const& uoutfab = uout.array(); 
    auto const& volfab = vol.array();
    auto const& pdivufab = pdivu.array();

    //TODO have difmag be parm parsed
    amrex::Real difmag = 0.e0; //0.005e0; 


    amrex::Print() << "calling consup!" << std::endl; 
    PeleC_consup(bx, uinfab, uoutfab,
                 D_DECL(flxx, flxy, flxz),
                 D_DECL(a1fab, a2fab, a3fab), 
                 volfab, divfab, pdivufab, dx, difmag); 

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
                  amrex::Real const *dx, amrex::Real const difmag)
{
    
//============== Flux alterations ========================
//-------------------------- x-flux -----------------------------------
    amrex::Box const &xfbx = surroundingNodes(bx, 0); 
    AMREX_PARALLEL_FOR_3D(xfbx, i, j, k, {
        //Artificial Viscosity! 
        PeleC_artif_visc(i,j,k,flx1, div, u, dx[0], difmag, 0); 
        //Normalize Species Flux
        PeleC_norm_spec_flx(i,j,k, flx1); 
        //Make flux extensive
        PeleC_ext_flx(i,j,k,flx1, a1);                              
    };);

//------------------------- y-flux ------------------------------------
    amrex::Box const &yfbx = surroundingNodes(bx, 1); 
    AMREX_PARALLEL_FOR_3D(yfbx, i, j, k, {
        //Artificial Viscosity! 
        PeleC_artif_visc(i,j,k,flx2, div, u, dx[1], difmag, 1); 
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
