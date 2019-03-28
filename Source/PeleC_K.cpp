#include "PeleC_K.H" 
#include "PeleC_method_2D.H"

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
    PeleC_umeth_1D(bx, bclo, bchi, domlo, domhi,  q,  qaux, src_q, 
                   bcMask, flux1, q1, pdivu, dx, dt);  
#elif AMREX_SPACEDIM==2 
/*    PeleC_umeth_2D(bx, bclo, bchi, domlo, domhi,q,  qaux, src_q,// bcMask, 
                   flux1, flux2, dloga,
                   q1.array(), q2.array(), a1, a2, pdivuarr, vol, dx, dt); */ 

// ================================= UMETH 2D ====================================  
    amrex::Real const dtdx  = dt/delx; 
    amrex::Real const hdtdx = 0.5*dtdx; 
    amrex::Real const hdtdy = 0.5*dt/dely; 
    amrex::Real const hdt   = 0.5*dt; 

    const int bclx = bclo[0]; 
    const int bcly = bclo[1]; 
    const int bchx = bchi[0]; 
    const int bchy = bchi[1]; 
    const int dlx  = domlo[0]; 
    const int dly  = domlo[1];     
    const int dhx  = domhi[0]; 
    const int dhy  = domhi[1]; 
    auto const& q1arr = q1.array(); 
    auto const& q2arr = q2.array(); 
//    auto const& bcMaskarr = bcMask.array();
    const Box& bxg1 = grow(bx, 1); 
//    const Box& bxg2 = grow(bx, 2);
    AsyncFab slope(bxg2, QVAR);
    auto const& slarr = slope.array();

//===================== X slopes ===================================
    int cdir = 0; 
    const Box& xslpbx = grow(bxg1, cdir, 1);

    const Box& xmbx = growHi(xslpbx,0, 1); 
    const Box& xflxbx = surroundingNodes(bxg1,cdir);
    AMREX_PARALLEL_FOR_4D (xslpbx, QVAR, i,j,k,n, { 
        PeleC_slope_x(i,j,k,n, slarr, q);
    }); 
//==================== X interp ====================================
    AsyncFab qxm(xmbx, QVAR); 
    AsyncFab qxp(xslpbx, QVAR);
    auto const& qxmarr = qxm.array(); 
    auto const& qxparr = qxp.array(); 
   
    AMREX_PARALLEL_FOR_3D (xslpbx,i,j,k, {
      PeleC_plm_x(i, j, k, qxmarr, qxparr, slarr, q, qaux(i,j,k,QC), 
                   dloga, delx, dt);
    }); 

//===================== X initial fluxes ===========================
    AsyncFab fx(xflxbx, NVAR);
    auto const& fxarr = fx.array(); 
    AsyncFab qgdx(bxg2, NGDNV); 
    auto const& gdtemp = qgdx.array(); 

    AMREX_PARALLEL_FOR_3D (xflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k,bclx, bchx, dlx, dhx, qxmarr, qxparr, fxarr, gdtemp, qaux, 0);
    });

//==================== Y slopes ====================================
    cdir = 1; 
    const Box& yflxbx = surroundingNodes(bxg1,cdir); 
    const Box& yslpbx = grow(bxg1, cdir, 1);
    const Box& ymbx = growHi(yslpbx, 1, 1); 
    AsyncFab qym(ymbx, QVAR);
    AsyncFab qyp(yslpbx, QVAR);
    auto const& qymarr = qym.array(); 
    auto const& qyparr = qyp.array();  
    AMREX_PARALLEL_FOR_4D (yslpbx, QVAR, i, j, k, n,{
        PeleC_slope_y(i, j, k, n, slarr, q);
    });

//==================== Y interp ====================================
    AMREX_PARALLEL_FOR_3D (yslpbx, i,j,k, {
        PeleC_plm_y(i,j,k, qymarr, qyparr, slarr, q, qaux(i,j,k,QC), 
                    dely, dt);
    });
 
//===================== Y initial fluxes ===========================
    AsyncFab fy(yflxbx, NVAR); 
    auto const& fyarr = fy.array();
    AMREX_PARALLEL_FOR_3D (yflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, bcly, bchy, dly, dhy, qymarr, qyparr, fyarr, q2arr, qaux, 1); 
    }); 

//===================== X interface corrections ====================
    cdir = 0;
    AsyncFab qm(bxg2, QVAR); 
    AsyncFab qp(bxg1, QVAR);
    const Box& tybx = grow(bx, cdir, 1);
    auto const& qmarr = qm.array();
    auto const& qparr = qp.array();  
    AMREX_PARALLEL_FOR_3D (tybx, i,j,k, {
        PeleC_transy(i,j,k, qmarr, qparr, qxmarr, qxparr, fyarr,
                     src_q, qaux, q2arr, a2, vol, hdt, hdtdy);
   });
 
//===================== Final Riemann problem X ====================
    const Box& xfxbx = surroundingNodes(bx, cdir);
    AMREX_PARALLEL_FOR_3D (xfxbx, i,j,k, {      
      PeleC_cmpflx(i,j,k, bclx, bchx, dlx, dhx, qmarr, qparr, flux1, q1arr, qaux,0); 
    }); 

//===================== Y interface corrections ====================
    cdir = 1; 
    const Box& txbx = grow(bx, cdir, 1);
    AMREX_PARALLEL_FOR_3D (txbx, i, j , k, {
        PeleC_transx(i,j,k, qmarr, qparr, qymarr, qyparr, fxarr,
                     src_q, qaux, gdtemp, a1, vol, hdt, hdtdx);                
    }); 
//===================== Final Riemann problem Y ====================
    
    const Box& yfxbx = surroundingNodes(bx, cdir);
    AMREX_PARALLEL_FOR_3D (yfxbx, i, j, k, {
      PeleC_cmpflx(i,j,k, bcly, bchy, dly, dhy, qmarr, qparr, flux2, q2arr, qaux, 1);
    });

//===================== Construct p div{U} =========================
    AMREX_PARALLEL_FOR_3D (bx, i, j, k, {
        PeleC_pdivu(i,j,k, pdivuarr, q1arr, q2arr, a1, a2, vol); 
    });
/* ======================== END UMETH2D =========================== */ 
#else
    PeleC_umeth_3D(bx, bclo, bchi, domlo, domhi,q,  qaux, src_q, bcMask, flux1, flux2,
                   flux3,  q1, q2, q3, a1, a2, a3, pdivu, vol, dx, dt);   
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
                  D_DECL(
                  amrex::Array4<const amrex::Real> const &a1,
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
    AMREX_PARALLEL_FOR_3D(yfbx, i, j, k, {
        //Artificial Viscosity! 
        PeleC_artif_visc(i,j,k,flx3, div, u, dz, difmag, 1); 
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
