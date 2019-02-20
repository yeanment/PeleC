#include "PeleC_method_2D.H" 

//Host function to call gpu hydro functions
void PeleC_umeth_2D(amrex::Box const& bx, amrex::Array4<const amrex::Real> const &q, 
           amrex::Array4<const amrex::Real> const& qaux,
           amrex::Array4<const amrex::Real> const& srcQ, amrex::IArrayBox const& bcMask,
           amrex::Array4<amrex::Real> const& flx1, amrex::Array4<amrex::Real> const& flx2, 
           amrex::Array4<const amrex::Real> const& dloga, amrex::Array4<amrex::Real> const& q1,
           amrex::Array4<amrex::Real> const& q2, amrex::Array4<const amrex::Real> const& a1, 
           amrex::Array4<const amrex::Real> const& a2, amrex::Array4<amrex::Real> const& pdivu, 
           amrex::Array4<const amrex::Real> const& vol, const amrex::Real *del, const amrex::Real dt)
{
    amrex::Real const dx    = del[0]; 
    amrex::Real const dy    = del[1]; 
    amrex::Real const dtdx  = dt/dx; 
    amrex::Real const hdtdx = 0.5*dtdx; 
    amrex::Real const hdtdy = 0.5*dt/dy; 
    amrex::Real const hdt   = 0.5*dt; 

    auto const& bcMaskarr = bcMask.array();
    const Box& bxg1 = grow(bx, 1); 
    const Box& bxg2 = grow(bx, 2);
    AsyncFab slope(bxg2, QVAR);
    auto const& slarr = slope.array();

//===================== X slopes ===================================
    int cdir = 0; 
    const Box& xslpbx = grow(bxg1, cdir, 1);

    IntVect lox(AMREX_D_DECL(xslpbx.loVect()[0], xslpbx.loVect()[1], xslpbx.loVect()[2])); 
    IntVect hix(AMREX_D_DECL(xslpbx.hiVect()[0]+1, xslpbx.hiVect()[1], xslpbx.hiVect()[2])); 
    const Box xmbx(lox, hix);
    const Box& xflxbx = surroundingNodes(bxg1,cdir);
    AMREX_PARALLEL_FOR_3D (xslpbx,i,j,k, { 
        PeleC_slope_x(i,j,k, slarr, q);
    }); // */
//==================== X interp ====================================
    AsyncFab qxm(xmbx, QVAR); 
    AsyncFab qxp(xslpbx, QVAR);
    auto const& qxmarr = qxm.array(); 
    auto const& qxparr = qxp.array(); 
   
    AMREX_PARALLEL_FOR_3D (xslpbx, i,j,k, {
      PeleC_plm_x(i, j, k, qxmarr, qxparr, slarr, q, qaux(i,j,k,QC), 
                   dx, dt); // dloga, dx[0], dt);
    });

//===================== X initial fluxes ===========================
    AsyncFab fx(xflxbx, NVAR);
    auto const& fxarr = fx.array(); 

    //bcMaskarr at this point does nothing. 
    AMREX_PARALLEL_FOR_3D (xflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qxmarr, qxparr, fxarr, q1, qaux,0);
                // bcMaskarr, 0);
    });

//==================== Y slopes ====================================
    cdir = 1; 
    const Box& yflxbx = surroundingNodes(bxg1,cdir); 
    const Box& yslpbx = grow(bxg1, cdir, 1);
    IntVect loy(AMREX_D_DECL(yslpbx.loVect()[0], yslpbx.loVect()[1], yslpbx.loVect()[2])); 
    IntVect hiy(AMREX_D_DECL(yslpbx.hiVect()[0], yslpbx.hiVect()[1]+1, yslpbx.hiVect()[2])); 

    const Box ymbx(loy,hiy); 
    AsyncFab qym(ymbx, QVAR);
    AsyncFab qyp(yslpbx, QVAR);
    auto const& qymarr = qym.array(); 
    auto const& qyparr = qyp.array();  
    AMREX_PARALLEL_FOR_3D (yslpbx, i,j,k,{
        PeleC_slope_y(i,j,k, slarr, q); 
    }); // */

//==================== Y interp ====================================
    AMREX_PARALLEL_FOR_3D (yslpbx, i,j,k, {
        PeleC_plm_y(i,j,k, qymarr, qyparr, slarr, q, qaux(i,j,k,QC), 
                    dy, dt); // dloga, dx[1], dt);
    });

//===================== Y initial fluxes ===========================
    AsyncFab fy(yflxbx, NVAR); 
    auto const& fyarr = fy.array();
    AMREX_PARALLEL_FOR_3D (yflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qymarr, qyparr, fyarr, q2, qaux, 1); 
        // bcMaskarr, 1);
    }); 

//===================== X interface corrections ====================
    cdir = 0; 
    AsyncFab qm(bxg2, QVAR); 
    AsyncFab qp(bxg2, QVAR);
    const Box& tybx = grow(bx, cdir, 1); 
    auto const& qmarr = qm.array();
    auto const& qparr = qp.array();  
    AMREX_PARALLEL_FOR_3D (tybx, i,j,k, {
        PeleC_transy(i,j,k, qmarr, qparr, qxmarr, qxparr, fyarr,
                     srcQ, qaux, q2, a2, vol, hdt, hdtdy);
   });

//===================== Final Riemann problem X ====================
    const Box& xfxbx = surroundingNodes(bx, cdir);
    AMREX_PARALLEL_FOR_3D (xfxbx, i,j,k, {      
      PeleC_cmpflx(i,j,k, qmarr, qparr, flx1, q1, qaux,0); // bcMaskarr, 0);
    }); 

//===================== Y interface corrections ====================
    cdir = 1; 
    const Box& txbx = grow(bx, cdir, 1);
    AMREX_PARALLEL_FOR_3D (txbx, i, j , k, {
        PeleC_transx(i,j,k, qmarr, qparr, qymarr, qyparr, fxarr,
                     srcQ, qaux, q1, a1, vol, hdt, hdtdx);                
    });

//===================== Final Riemann problem Y ====================
    
    const Box& yfxbx = surroundingNodes(bx, cdir);
    AMREX_PARALLEL_FOR_3D (yfxbx, i, j, k, {
      PeleC_cmpflx(i,j,k, qmarr, qparr, flx2, q2, qaux, 1); // bcMaskarr, 1); 
    });

//===================== Construct p div{U} =========================
    AMREX_PARALLEL_FOR_3D (bx, i, j, k, {
        PeleC_pdivu(i,j,k, pdivu, q1, q2, a1, a2, vol); 
    });
}

