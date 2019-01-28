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
    Gpu::AsyncFab slope(bxg2, QVAR);
    auto slfab = slope.array(), qfab  = q.array(); 
    auto qauxfab = qaux.array(), srcQfab = srcQ.array(); 
    auto dlogafab = dloga.array(), area1 = a1.array(), area2 = a2.array(); 
    auto bcMaskfab = bcMask.array();
    auto volfab = vol.array();  

//===================== X slopes ===================================
    int cdir = 0; 
    const Box& xslpbx = grow(bxg1, cdir, 1);
     amrex::Print() << "================== Before Slope  X ================== " << std::endl;

    AMREX_PARALLEL_FOR_3D (xslpbx,i,j,k, { 
        PeleC_slope_x(i,j,k, slfab, qfab);
    }); 


//==================== X interp ====================================
//Need to include dlogA.
    Gpu::AsyncFab qxm(xslpbx, QVAR); 
    Gpu::AsyncFab qxp(xslpbx, QVAR);
    amrex::Array4<amrex::Real> qxmfab = qxm.array(); 
    amrex::Array4<amrex::Real> qxpfab = qxp.array(); 

    amrex::Print() << "================== Before PLM X ================== " << std::endl;

    AMREX_PARALLEL_FOR_3D (xslpbx, i,j,k, {
        PeleC_plm_x(i, j, k, qxmfab, qxpfab, slfab, qfab, qauxfab, 
                    srcQfab, dlogafab, dx[0], dt); 
    });

//===================== X initial fluxes ===========================
    const Box& xflxbx = surroundingNodes(bxg1,cdir); 
    Gpu::AsyncFab fx(xflxbx, NVAR);
    amrex::Array4<amrex::Real> fxfab = fx.array(); 
    amrex::Array4<amrex::Real> q1fab = q1.array();  
    amrex::Print() << "================== Before flux X ================== " << std::endl;

    //bcMask at this point does nothing.  
    AMREX_PARALLEL_FOR_3D (xflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qxmfab, qxpfab, fxfab, q1fab, qauxfab,
                 bcMaskfab, 0);
    });

//==================== Y slopes ====================================
    cdir = 1; 
    const Box& yslpbx = grow(bxg1, cdir, 1); 
    Gpu::AsyncFab qym(yslpbx, QVAR);
    Gpu::AsyncFab qyp(yslpbx, QVAR);
    amrex::Array4<amrex::Real> qymfab = qym.array(); 
    amrex::Array4<amrex::Real> qypfab = qyp.array();  
    amrex::Print() << "================== Before Slope Y ================== " << std::endl;

 
    AMREX_PARALLEL_FOR_3D (yslpbx, i,j,k,{
        PeleC_slope_y(i,j,k, slfab, qfab); 
    });

//==================== Y interp ====================================
    amrex::Print() << "================== Before PLM Y ================== " << std::endl;

    AMREX_PARALLEL_FOR_3D (yslpbx, i,j,k, {
        PeleC_plm_y(i,j,k, qymfab, qypfab, slfab, qfab, qauxfab, 
                srcQfab, dlogafab, dx[1], dt); 
    });

//===================== Y initial fluxes ===========================
/*    const Box& yflxbx = surroundingNodes(bxg1,cdir); 
    Gpu::AsyncFab fy(yflxbx, NVAR); 
    amrex::Array4<amrex::Real> fyfab = fy.array(), q2fab = q2.array(); 
    amrex::Print() << "================== Before flux Y ================== " << std::endl;

    AMREX_PARALLEL_FOR_3D (yflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qymfab, qypfab, fyfab, q2fab, qauxfab, bcMaskfab, 1); 
    }); 

//===================== X interface corrections ====================
    cdir = 0; 
    Gpu::AsyncFab qm(bxg1, QVAR); 
    Gpu::AsyncFab qp(bxg1, QVAR);
    amrex::Array4<amrex::Real> qmfab = qm.array(), qpfab = qp.array();  
    //TODO think about box size for these 
    const Box& xbx = grow(bx, cdir, 1);
    AMREX_PARALLEL_FOR_3D (xbx, i,j,k, {
        PeleC_transy(i,j,k, qmfab, qpfab, qxmfab, qxpfab, fyfab,
                     srcQfab, qauxfab, q2fab, area2, volfab, hdt, hdtdy); 
    }); 

//===================== Final Riemann problem X ====================
    const Box& xfxbx = surroundingNodes(bx, cdir); 
    amrex::Array4<amrex::Real> flx1fab = flx1.array(); 
    AMREX_PARALLEL_FOR_3D (xfxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qmfab, qpfab, flx1fab,q1fab, qauxfab, bcMaskfab, 0);
    }); 

//===================== Y interface corrections ====================
    cdir = 1; 
    const Box& ybx = grow(bx, cdir, 1);
    AMREX_PARALLEL_FOR_3D (ybx, i, j , k, {
    PeleC_transx(i,j,k, qmfab, qpfab, qymfab, qypfab, fxfab,
                 srcQfab, qauxfab, q1fab, area1, volfab, hdt, hdtdx); 
    });

//===================== Final Riemann problem Y ====================
    const Box& yfxbx = surroundingNodes(bx, cdir);
    amrex::Array4<amrex::Real> flx2fab = flx2.array();  
    AMREX_PARALLEL_FOR_3D (yfxbx, i, j, k, {
        PeleC_cmpflx(i,j,k, qmfab, qpfab, flx2fab, q2fab, qauxfab, bcMaskfab, 1); 
    });

//===================== Construct p div{U} =========================
    amrex::Array4<amrex::Real> pdivufab = pdivu.array(); 
    AMREX_PARALLEL_FOR_3D (bx, i, j, k, {
        PeleC_pdivu(i,j,k, pdivufab, q1fab, q2fab, area1, area2, volfab); 
    });

    amrex::Print() << "======================= End of UMETH! ============================" << std::endl; */
}

