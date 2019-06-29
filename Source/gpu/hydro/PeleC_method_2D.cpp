#include "PeleC_method_2D.H" 

//Host function to call gpu hydro functions
void PeleC_umeth_2D(amrex::Box const& bx, const int* bclo, const int* bchi, 
           const int* domlo, const int* domhi, 
           amrex::Array4<const amrex::Real> const &q, 
           amrex::Array4<const amrex::Real> const& qaux,
           amrex::Array4<const amrex::Real> const& srcQ,// amrex::IArrayBox const& bcMask,
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
    
//    auto const& bcMaskarr = bcMask.array();
    const Box& bxg1 = grow(bx, 1); 
    const Box& bxg2 = grow(bx, 2);
    FArrayBox slope(bxg2, QVAR);
    Elixir sleli = slope.elixir(); 
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
    FArrayBox qxm(xmbx, QVAR); 
    FArrayBox qxp(xslpbx, QVAR);
    Elixir qxmeli = qxm.elixir(), qxpeli = qxp.elixir(); 
    auto const& qxmarr = qxm.array(); 
    auto const& qxparr = qxp.array(); 
   
    AMREX_PARALLEL_FOR_3D (xslpbx,i,j,k, {
      PeleC_plm_x(i, j, k, qxmarr, qxparr, slarr, q, qaux(i,j,k,QC), 
                   dloga, dx, dt);
    }); 

//===================== X initial fluxes ===========================
    FArrayBox fx(xflxbx, NVAR);
    Elixir fxeli = fx.elixir(); 
    auto const& fxarr = fx.array(); 
    FArrayBox qgdx(bxg2, NGDNV); 
    Elixir qgxeli = qgdx.elixir(); 
    auto const& gdtemp = qgdx.array(); 

   AMREX_PARALLEL_FOR_3D (xflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k,bclx, bchx, dlx, dhx, qxmarr, qxparr, fxarr, gdtemp, qaux, 0);
    });

//==================== Y slopes ====================================
    cdir = 1; 
    const Box& yflxbx = surroundingNodes(bxg1,cdir); 
    const Box& yslpbx = grow(bxg1, cdir, 1);
    const Box& ymbx = growHi(yslpbx, 1, 1); 
    FArrayBox qym(ymbx, QVAR);
    Elixir qymeli = qym.elixir(); 
    FArrayBox qyp(yslpbx, QVAR);
    Elixir qypeli = qyp.elixir(); 
    auto const& qymarr = qym.array(); 
    auto const& qyparr = qyp.array(); 
    AMREX_PARALLEL_FOR_4D (yslpbx, QVAR, i, j, k, n,{
        PeleC_slope_y(i, j, k, n, slarr, q);
    });

//==================== Y interp ====================================
    AMREX_PARALLEL_FOR_3D (yslpbx, i,j,k, {
        PeleC_plm_y(i,j,k, qymarr, qyparr, slarr, q, qaux(i,j,k,QC), 
                    dy, dt);
    });
 
//===================== Y initial fluxes ===========================
    FArrayBox fy(yflxbx, NVAR); 
    Elixir fyeli = fy.elixir(); 
    auto const& fyarr = fy.array();
   AMREX_PARALLEL_FOR_3D (yflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, bcly, bchy, dly, dhy, qymarr, qyparr, fyarr, q2, qaux, 1); 
    }); 

//===================== X interface corrections ====================
    cdir = 0;
    FArrayBox qm(bxg2, QVAR); 
    Elixir qmeli = qm.elixir(); 
    FArrayBox qp(bxg1, QVAR);
    Elixir qpeli = qp.elixir(); 
    const Box& tybx = grow(bx, cdir, 1);
    auto const& qmarr = qm.array();
    auto const& qparr = qp.array();  
    AMREX_PARALLEL_FOR_3D (tybx, i,j,k, {
        PeleC_transy(i,j,k, qmarr, qparr, qxmarr, qxparr, fyarr,
                     srcQ, qaux, q2, a2, vol, hdt, hdtdy);
   });
    fyeli.clear(); 
    qxmeli.clear(); 
    qxpeli.clear(); 
//===================== Final Riemann problem X ====================
    const Box& xfxbx = surroundingNodes(bx, cdir);
   AMREX_PARALLEL_FOR_3D (xfxbx, i,j,k, {      
      PeleC_cmpflx(i,j,k, bclx, bchx, dlx, dhx, qmarr, qparr, flx1, q1, qaux,0); 
    }); 

//===================== Y interface corrections ====================
    cdir = 1; 
    const Box& txbx = grow(bx, cdir, 1);
    AMREX_PARALLEL_FOR_3D (txbx, i, j , k, {
        PeleC_transx(i,j,k, qmarr, qparr, qymarr, qyparr, fxarr,
                     srcQ, qaux, gdtemp, a1, vol, hdt, hdtdx);                
    }); 
    fxeli.clear(); 
    qymeli.clear(); 
    qypeli.clear(); 
//===================== Final Riemann problem Y ====================
    
    const Box& yfxbx = surroundingNodes(bx, cdir);
    AMREX_PARALLEL_FOR_3D (yfxbx, i, j, k, {
      PeleC_cmpflx(i,j,k, bcly, bchy, dly, dhy, qmarr, qparr, flx2, q2, qaux, 1);
    });

//===================== Construct p div{U} =========================
    AMREX_PARALLEL_FOR_3D (bx, i, j, k, {
        PeleC_pdivu(i,j,k, pdivu, q1, q2, a1, a2, vol); 
    });
}

