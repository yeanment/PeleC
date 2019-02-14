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
    AsyncFab slope(bxg2, QVAR);
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
    const Box& xslpbx = grow(bxg1, cdir, 1);

    IntVect lox(AMREX_D_DECL(xslpbx.loVect()[0], xslpbx.loVect()[1], xslpbx.loVect()[2])); 
    IntVect hix(AMREX_D_DECL(xslpbx.hiVect()[0]+1, xslpbx.hiVect()[1], xslpbx.hiVect()[2])); 
    const Box xmbx(lox, hix);
    const Box& xflxbx = surroundingNodes(bxg1,cdir);
    amrex::Print()<< "Slope x! " << std::endl; 
    AMREX_PARALLEL_FOR_3D (xslpbx,i,j,k, { 
        PeleC_slope_x(i,j,k, slfab, qfab);
    }); // */
    Gpu::Device::synchronize(); 
//==================== X interp ====================================
    AsyncFab qxm(xmbx, QVAR); 
    AsyncFab qxp(xslpbx, QVAR);
    auto const& qxmfab = qxm.array(); 
    auto const& qxpfab = qxp.array(); 

    amrex::Print() << "PLM X" << std::endl;
    amrex::Print() << "C = " << qauxfab(2,3,0,QC) << std::endl;  
    AMREX_PARALLEL_FOR_3D (xslpbx, i,j,k, {
       PeleC_plm_x(i, j, k, qxmfab, qxpfab, slfab, qfab, qauxfab(i,j,k,QC) , 
                    srcQfab,dx[0], dt); // dlogafab, dx[0], dt);
/*                 if(i==25 && j==204){
                    amrex::Print()<<"plm_x" << std::endl;
                    for(int n = 0; n < QVAR; ++n){
                        amrex::Print() << n << std::endl; 
                        amrex::Print() << "qm " << qxmfab(i+1,j,k,n) << '\t' << 
                                      "qp " << qxpfab(i,j,k,n) << '\t' <<
                                      "slope " << slfab(i,j,k,n) << std::endl;  
                    }
                    std::cin.get();
                 } // */
   });
   Gpu::Device::synchronize(); 

//===================== X initial fluxes ===========================
    AsyncFab fx(xflxbx, NVAR);
    auto const& fxfab = fx.array(); 
    auto const& q1fab = q1.array();  

    //bcMask at this point does nothing.  
    amrex::Print() << "FLUX X " << std::endl; 
    AMREX_PARALLEL_FOR_3D (xflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qxmfab, qxpfab, fxfab, q1fab, qauxfab,
                // bcMaskfab,
                 0);  
    });
    Gpu::Device::synchronize(); 

//==================== Y slopes ====================================
    cdir = 1; 
    const Box& yflxbx = surroundingNodes(bxg1,cdir); 
    const Box& yslpbx = grow(bxg1, cdir, 1);
    IntVect loy(AMREX_D_DECL(yslpbx.loVect()[0], yslpbx.loVect()[1], yslpbx.loVect()[2])); 
    IntVect hiy(AMREX_D_DECL(yslpbx.hiVect()[0], yslpbx.hiVect()[1]+1, yslpbx.hiVect()[2])); 

    const Box ymbx(loy,hiy); 
    AsyncFab qym(ymbx, QVAR);
    AsyncFab qyp(yslpbx, QVAR);
    auto const& qymfab = qym.array(); 
    auto const& qypfab = qyp.array();  
    
    amrex::Print() << "Slope y !" << std::endl;  
    AMREX_PARALLEL_FOR_3D (yslpbx, i,j,k,{
        PeleC_slope_y(i,j,k, slfab, qfab); 
    }); // */
   Gpu::Device::synchronize(); 

//==================== Y interp ====================================
    amrex::Print() << "PLM Y " << std::endl; 
    AMREX_PARALLEL_FOR_3D (yslpbx, i,j,k, {
          qymfab(i,j,k,QRHO) = qauxfab(i,j,k,QC); 
//        PeleC_plm_y(i,j,k, qymfab, qypfab, slfab, qfab, qauxfab(i,j,k,QC), 
//                srcQfab, dx[1], dt); // dlogafab, dx[1], dt);
/*                 if(i==25 && j==204){
                    amrex::Print()<<"plm_y" << std::endl;
                    for(int n = 0; n < QVAR; ++n){
                        amrex::Print() << n << std::endl; 
                        amrex::Print() << "qm " << qymfab(i,j+1,k,n) << '\t' << 
                                      "qp " << qypfab(i,j,k,n) << '\t' <<
                                      "slope " << slfab(i,j,k,n) << std::endl;  
                    }
                    std::cin.get();
                 } // */
  });
   Gpu::Device::synchronize(); 

//===================== Y initial fluxes ===========================
    amrex::Print()<< " FLX Y " << std::endl; 
    AsyncFab fy(yflxbx, NVAR); 
    auto const& fyfab = fy.array();
    auto const& q2fab = q2.array();
    AMREX_PARALLEL_FOR_3D (yflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qymfab, qypfab, fyfab, q2fab, qauxfab, 1); 
        // bcMaskfab, 1);
    }); 
   Gpu::Device::synchronize(); 

//===================== X interface corrections ====================
    cdir = 0; 
    AsyncFab qm(bxg2, QVAR); 
    AsyncFab qp(bxg2, QVAR);
    const Box& tybx = grow(bx, cdir, 1); 
    auto const& qmfab = qm.array();
    auto const& qpfab = qp.array();  
    amrex::Print() << " Transy " << std::endl; 
    AMREX_PARALLEL_FOR_3D (tybx, i,j,k, {
        PeleC_transy(i,j,k, qmfab, qpfab, qxmfab, qxpfab, fyfab,
                     srcQfab, qauxfab, q2fab, area2, volfab, hdt, hdtdy);
/*                if(i==25 && j==204){
                    amrex::Print()<<"transy" << std::endl;
                    for(int n = 0; n < QVAR; ++n){
                        amrex::Print() << n << std::endl; 
                        amrex::Print() << "qm " << qmfab(i+1,j,k,n) << '\t' << 
                                      "qp " << qpfab(i,j,k,n) << '\t' << 
                                       "fx " << fyfab(i, j+1, k, n) << '\t' 
                                       << fyfab(i,j,k,n) << '\n' 
                                       << "qym " << qxmfab(i+1,j,k,n) << '\t' << 
                                          "qyp " << qxpfab(i,j,k,n) << std::endl;
                    }
                    std::cin.get();
                 } // */

/*/
         for(int n = 0; n < QVAR; ++n){
            qmfab(i+1,j,k,n) = qxmfab(i+1,j,k,n); 
            qpfab(i,j,k,n) = qxpfab(i,j,k,n); 
        } // */ 
   });
   Gpu::Device::synchronize(); 

//===================== Final Riemann problem X ====================
    const Box& xfxbx = surroundingNodes(bx, cdir); 
    auto const& flx1fab = flx1.array();
    amrex::Print() << "FLX x 2" << std::endl; 
    AMREX_PARALLEL_FOR_3D (xfxbx, i,j,k, {
      PeleC_cmpflx(i,j,k, qmfab, qpfab, flx1fab,q1fab, qauxfab,0); // bcMaskfab, 0);
    }); 
    Gpu::Device::synchronize(); 

//===================== Y interface corrections ====================
    cdir = 1; 
    const Box& txbx = grow(bx, cdir, 1);
    amrex::Print() << "Transx" << std::endl; 
    AMREX_PARALLEL_FOR_3D (txbx, i, j , k, {
        PeleC_transx(i,j,k, qmfab, qpfab, qymfab, qypfab, fxfab,
                 srcQfab, qauxfab, q1fab, area1, volfab, hdt, hdtdx);                
/*                if(i==25 && j==204){
                    amrex::Print() << "transx" << std::endl; 
                    for(int n = 0; n < QVAR; ++n){
                        amrex::Print() << n << std::endl; 
                        amrex::Print() << "qm " << qmfab(i,j+1,k,n) << '\t' << 
                                      "qp " << qpfab(i,j,k,n) << '\t' << 
                                       "fx " << fxfab(i+1, j, k, n) << '\t' 
                                       << fxfab(i,j,k,n) << '\n' 
                                       << "qym " << qymfab(i,j+1,k,n) << '\t' << 
                                          "qyp " << qypfab(i,j,k,n) << std::endl;
                    }
                    std::cin.get();
                 } // */
/*        for(int n = 0; n < QVAR; ++n){
            qmfab(i,j+1,k,n) = qymfab(i,j+1,k,n); 
            qpfab(i,j,k,n) = qypfab(i,j,k,n); 
        } // */
  });
  Gpu::Device::synchronize(); 

//===================== Final Riemann problem Y ====================
    
    const Box& yfxbx = surroundingNodes(bx, cdir);
    auto const& flx2fab = flx2.array();  
    amrex::Print()<< " FLX Y 2 " << std::endl; 
    AMREX_PARALLEL_FOR_3D (yfxbx, i, j, k, {
      PeleC_cmpflx(i,j,k, qmfab, qpfab, flx2fab, q2fab, qauxfab, 1); // bcMaskfab, 1); 
    });
    Gpu::Device::synchronize(); 

//===================== Construct p div{U} =========================
    auto const& pdivufab = pdivu.array(); 
    amrex::Print() << "P divu " << std::endl; 
    AMREX_PARALLEL_FOR_3D (bx, i, j, k, {
        PeleC_pdivu(i,j,k, pdivufab, q1fab, q2fab, area1, area2, volfab); 
    });
    Gpu::Device::synchronize(); 

}

