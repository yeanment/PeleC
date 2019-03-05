#include "PeleC_method_2D.H" 

//Host function to call gpu hydro functions
void PeleC_umeth_2D(amrex::Box const& bx, const int* bclo, const int* bchi, 
           const int* domlo, const int* domhi, 
           amrex::Array4<amrex::Real> const &q, 
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

    const int bclx = bclo[0]; 
    const int bcly = bclo[1]; 
    const int bchx = bchi[0]; 
    const int bchy = bchi[1]; 
    const int dlx  = domlo[0]; 
    const int dly  = domlo[1];     
    const int dhx  = domhi[0]; 
    const int dhy  = domhi[1]; 
    
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
    AMREX_PARALLEL_FOR_4D (xslpbx, QVAR, i,j,k,n, { 
        PeleC_slope_x(i,j,k,n, slarr, q);
//        slarr(i,j,k,n) =0.e0; 
    }); // */
    AMREX_PARALLEL_FOR_4D (xslpbx, QVAR, i,j,k,n, { 
        q(i,j,k,n) = slarr(i,j,k,n); 
//        slarr(i,j,k,n) =0.e0; 
    }); // */

    return; 
    std::cin.get(); 
//==================== X interp ====================================
    AsyncFab qxm(xmbx, QVAR); 
    AsyncFab qxp(xslpbx, QVAR);
    auto const& qxmarr = qxm.array(); 
    auto const& qxparr = qxp.array(); 
   
    AMREX_PARALLEL_FOR_3D (xslpbx,i,j,k, {
      PeleC_plm_x(i, j, k, qxmarr, qxparr, slarr, q, qaux(i,j,k,QC), 
                   dloga, dx, dt);
    }); 
 
/*       AMREX_PARALLEL_FOR_4D(xslpbx, QVAR, i, j, k, n,{ 
               qxmarr(i+1,j,k,n) = q(i,j,k,n); 
               qxparr(i,j,k,n) = q(i,j,k,n); 
        }); //*/

//===================== X initial fluxes ===========================
    AsyncFab fx(xflxbx, NVAR);
    auto const& fxarr = fx.array(); 
    AsyncFab qgdx(bxg2, NGDNV); 
    auto const& gdtemp = qgdx.array(); 

    //bcMaskarr at this point does nothing. 
    AMREX_PARALLEL_FOR_3D (xflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k,bclx, bchx, dlx, dhx, qxmarr, qxparr, fxarr, gdtemp, qaux,0);
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
    AMREX_PARALLEL_FOR_4D (yslpbx, QVAR, i, j, k, n,{
//        PeleC_slope_y(i, j, k, n, slarr, q);
        slarr(i,j,k,n) =0.e0; 

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
        PeleC_cmpflx(i,j,k, bcly, bchy, dly, dhy, qymarr, qyparr, fyarr, q2, qaux, 1); 
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
   });  /*/ 
       AMREX_PARALLEL_FOR_4D(tybx, QVAR, i, j, k, n,{ 
//               qmarr(i+1,j,k,n) = q(i,j,k,n); 
//               qparr(i,j,k,n) = q(i,j,k,n); 
                 qmarr(i+1,j,k,n) = qxmarr(i+1,j,k,n); 
                 qparr(i,j,k,n) = qxparr(i,j,k,n); 
        }); //*/

//===================== Final Riemann problem X ====================
    const Box& xfxbx = surroundingNodes(bx, cdir);
    AMREX_PARALLEL_FOR_3D (xfxbx, i,j,k, {      
      PeleC_cmpflx(i,j,k, bclx, bchx, dlx, dhx, qmarr, qparr, flx1, q1, qaux,0); // bcMaskarr, 0);
    }); 

//===================== Y interface corrections ====================
    cdir = 1; 
    const Box& txbx = grow(bx, cdir, 1);
    AMREX_PARALLEL_FOR_3D (txbx, i, j , k, {
        PeleC_transx(i,j,k, qmarr, qparr, qymarr, qyparr, fxarr,
                     srcQ, qaux, gdtemp, a1, vol, hdt, hdtdx);                
    }); /*/ 
       AMREX_PARALLEL_FOR_4D(txbx, QVAR, i, j, k, n,{ 
//               qmarr(i,j+1,k,n) = q(i,j,k,n); 
//               qparr(i,j,k,n) = q(i,j,k,n); 
                 qmarr(i,j+1,k,n) = qymarr(i,j+1,k,n); 
                 qparr(i,j,k,n) = qyparr(i,j,k,n); 
        });// */ 

//===================== Final Riemann problem Y ====================
    
    const Box& yfxbx = surroundingNodes(bx, cdir);
    AMREX_PARALLEL_FOR_3D (yfxbx, i, j, k, {
      PeleC_cmpflx(i,j,k, bcly, bchy, dly, dhy, qmarr, qparr, flx2, q2, qaux, 1); // bcMaskarr, 1); 
    });

//===================== Construct p div{U} =========================
    AMREX_PARALLEL_FOR_3D (bx, i, j, k, {
        PeleC_pdivu(i,j,k, pdivu, q1, q2, a1, a2, vol); 
    });
}

