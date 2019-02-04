#include "PeleC_method_2D.H" 

//Host function to call gpu hydro functions
void PeleC_umeth_2D(amrex::Box const& bx, amrex::FArrayBox const &q, 
           amrex::FArrayBox const& qaux,
           amrex::FArrayBox const& srcQ, amrex::IArrayBox const& bcMask,
           amrex::FArrayBox &flx1, amrex::FArrayBox &flx2, 
           amrex::FArrayBox const& dloga, amrex::FArrayBox &q1,
           amrex::FArrayBox &q2, amrex::FArrayBox const &a1, 
           amrex::FArrayBox const &a2, amrex::FArrayBox &pdivu, 
           amrex::FArrayBox const &vol, const amrex::Real *dx, const amrex::Real dt)
{
    amrex::Real const dtdx  = dt/dx[0]; 
    amrex::Real const hdtdx = 0.5*dtdx; 
    amrex::Real const hdtdy = 0.5*dt/dx[1]; 
    amrex::Real const hdt   = 0.5*dt; 

    const Box& bxg1 = grow(bx, 1); 
    const Box& bxg2 = grow(bx, 2);
    const Box& bxg3 = grow(bx, 3);  
    AsyncFab slope(bxg3, QVAR);
    auto const& slfab = slope.array();
    auto const& qfab  = q.array(); 
    auto const& qauxfab = qaux.array();
    auto const& srcQfab = srcQ.array(); 
    auto const& dlogafab = dloga.array();
    auto const& area1 = a1.array();
    auto const& area2 = a2.array(); 
    auto const& bcMaskfab = bcMask.array();
    auto const& volfab = vol.array();  

//===================== X slopes ===================================
    int cdir = 0; 
    const Box& xslpbx = grow(bxg2, cdir, 1);
     
    AMREX_PARALLEL_FOR_3D (xslpbx,i,j,k, { 
        PeleC_slope_x(i,j,k, slfab, qfab);
    }); 


//==================== X interp ====================================
    const Box& xflxbx = surroundingNodes(bxg1,cdir); 
    AsyncFab qxm(xslpbx, QVAR); 
    AsyncFab qxp(xslpbx, QVAR);
    auto const& qxmfab = qxm.array(); 
    auto const& qxpfab = qxp.array(); 

    AMREX_PARALLEL_FOR_3D (bxg2, i,j,k, {
        PeleC_plm_x(i, j, k, qxmfab, qxpfab, slfab, qfab, qauxfab, 
                    srcQfab, dlogafab, dx[0], dt); 
    });

//===================== X initial fluxes ===========================
    AsyncFab fx(xflxbx, NVAR);
    auto const& fxfab = fx.array(); 
    auto const& q1fab = q1.array();  

    //bcMask at this point does nothing.  
    AMREX_PARALLEL_FOR_3D (xflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qxmfab, qxpfab, fxfab, q1fab, qauxfab,
                 bcMaskfab, 0);
    });

//==================== Y slopes ====================================
    cdir = 1; 
    const Box& yflxbx = surroundingNodes(bxg1,cdir); 
    const Box& yslpbx = grow(bxg2, cdir, 1);
    AsyncFab qym(yslpbx, QVAR);
    AsyncFab qyp(yslpbx, QVAR);
    auto const& qymfab = qym.array(); 
    auto const& qypfab = qyp.array();  
 
    AMREX_PARALLEL_FOR_3D (yslpbx, i,j,k,{
        PeleC_slope_y(i,j,k, slfab, qfab); 
    });

//==================== Y interp ====================================
    AMREX_PARALLEL_FOR_3D (bxg2, i,j,k, {
        PeleC_plm_y(i,j,k, qymfab, qypfab, slfab, qfab, qauxfab, 
                srcQfab, dlogafab, dx[1], dt); 
    });

//===================== Y initial fluxes ===========================
    AsyncFab fy(yflxbx, NVAR); 
    auto const& fyfab = fy.array(), q2fab = q2.array(); 

    AMREX_PARALLEL_FOR_3D (yflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qymfab, qypfab, fyfab, q2fab, qauxfab, bcMaskfab, 1); 
    }); 

//===================== X interface corrections ====================
    cdir = 0; 
    AsyncFab qm(bxg2, QVAR); 
    AsyncFab qp(bxg2, QVAR);
    auto const& qmfab = qm.array();
    auto const& qpfab = qp.array();  

    AMREX_PARALLEL_FOR_3D (bxg1, i,j,k, {
        PeleC_transy(i,j,k, qmfab, qpfab, qxmfab, qxpfab, fyfab,
                     srcQfab, qauxfab, q2fab, area2, volfab, hdt, hdtdy); 
    }); 

//===================== Final Riemann problem X ====================
    const Box& xfxbx = surroundingNodes(bx, cdir); 
    auto const& flx1fab = flx1.array(); 
    AMREX_PARALLEL_FOR_3D (xfxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qmfab, qpfab, flx1fab,q1fab, qauxfab, bcMaskfab, 0);
    }); 

//===================== Y interface corrections ====================
    cdir = 1; 
    AMREX_PARALLEL_FOR_3D (bxg1, i, j , k, {
    PeleC_transx(i,j,k, qmfab, qpfab, qymfab, qypfab, fxfab,
                 srcQfab, qauxfab, q1fab, area1, volfab, hdt, hdtdx); 
    });

//===================== Final Riemann problem Y ====================
    const Box& yfxbx = surroundingNodes(bx, cdir);
    auto const& flx2fab = flx2.array();  
    AMREX_PARALLEL_FOR_3D (yfxbx, i, j, k, {
        PeleC_cmpflx(i,j,k, qmfab, qpfab, flx2fab, q2fab, qauxfab, bcMaskfab, 1); 
    });

//===================== Construct p div{U} =========================
    auto const& pdivufab = pdivu.array(); 
    AMREX_PARALLEL_FOR_3D (bx, i, j, k, {
        PeleC_pdivu(i,j,k, pdivufab, q1fab, q2fab, area1, area2, volfab); 
    });

}

