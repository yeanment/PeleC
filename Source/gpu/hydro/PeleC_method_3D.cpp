#include "PeleC_method_3D.H" 

//Host function to call gpu hydro functions
void PeleC_umeth_3D(amrex::Box const& bx, const int* bclo, const int* bchi, 
           const int* domlo, const int* domhi, 
           amrex::Array4<const amrex::Real> const &q, 
           amrex::Array4<const amrex::Real> const& qaux,
           amrex::Array4<const amrex::Real> const& srcQ,// amrex::IArrayBox const& bcMask,
           amrex::Array4<amrex::Real> const& flx1, amrex::Array4<amrex::Real> const& flx2,
           amrex::Array4<amrex::Real> const& flx3, // amrex::Array4<const amrex::Real> const& dloga,
           amrex::Array4<amrex::Real> const& q1, amrex::Array4< amrex::Real> const& q2, 
           amrex::Array4<amrex::Real> const& q3, amrex::Array4<const amrex::Real> const& a1, 
           amrex::Array4<const amrex::Real> const& a2, amrex::Array4<const amrex::Real> const& a3, 
           amrex::Array4<amrex::Real> const& pdivu, amrex::Array4<const amrex::Real> const& vol,
           const amrex::Real *del, const amrex::Real dt)
{
    amrex::Real const dx    = del[0]; 
    amrex::Real const dy    = del[1]; 
    amrex::Real const dz    = del[2]; 
    amrex::Real const hdtdx = 0.5*dt/dx; 
    amrex::Real const hdtdy = 0.5*dt/dy; 
    amrex::Real const hdtdz = 0.5*dt/dz;
    amrex::Real const cdtdx = 1.0/3.0*dt/dx; 
    amrex::Real const cdtdy = 1.0/3.0*dt/dy; 
    amrex::Real const cdtdz = 1.0/3.0*dt/dz;  
    amrex::Real const hdt   = 0.5*dt; 


    const int bclx = bclo[0]; 
    const int bcly = bclo[1];
    const int bclz = bclo[2];  
    const int bchx = bchi[0]; 
    const int bchy = bchi[1]; 
    const int bchz = bchi[2]; 
    const int dlx  = domlo[0]; 
    const int dly  = domlo[1];
    const int dlz  = domlo[2];      
    const int dhx  = domhi[0]; 
    const int dhy  = domhi[1]; 
    const int dhz  = domhi[2]; 
    
//    auto const& bcMaskarr = bcMask.array();
    const Box& bxg1 = grow(bx, 1); 
    const Box& bxg2 = grow(bx, 2);

//===================== X data  ===================================

    int cdir = 0; 
    const Box& xmbx = growHi(bxg2, cdir, 1); 
    FArrayBox qxm(xmbx, QVAR); 
    FArrayBox qxp(bxg2, QVAR);
    Elixir qxmeli = qxm.elixir(); 
    Elixir qxpeli = qxp.elixir(); 
    auto const& qxmarr = qxm.array(); 
    auto const& qxparr = qxp.array();

//===================== Y data  ===================================

    cdir = 1; 
    const Box& yflxbx = surroundingNodes(grow(bxg2, cdir, -1) ,cdir); 
    const Box& ymbx = growHi(bxg2, cdir, 1); 
    FArrayBox qym(ymbx, QVAR);
    FArrayBox qyp(bxg2, QVAR);
    Elixir qymeli = qym.elixir(); 
    Elixir qypeli = qyp.elixir(); 
    auto const& qymarr = qym.array(); 
    auto const& qyparr = qyp.array();  

//===================== Z data  ===================================

    cdir = 2; 
    const Box& zmbx = growHi(bxg2, cdir, 1); 
    const Box& zflxbx = surroundingNodes(grow(bxg2,cdir, -1),cdir);
    FArrayBox qzm(zmbx, QVAR); 
    FArrayBox qzp(bxg2, QVAR);
    Elixir qzmeli = qzm.elixir(); 
    Elixir qzpeli = qzp.elixir(); 
    auto const& qzmarr = qzm.array(); 
    auto const& qzparr = qzp.array();

/* Put the PLM and slopes in the same kernel launch to avoid unnecessary launch overhead */ 
/* Pelec_Slope_* are SIMD as well as PeleC_plm_* which loop over the same box */
    AMREX_PARALLEL_FOR_3D (bxg2,i,j,k, {
        amrex::Real slope[QVAR];
//===================== X slopes ===================================
        for(int n = 0; n < QVAR; ++n)
           slope[n] = PeleC_slope_x(i,j,k,n,q);
//==================== X interp ====================================
        PeleC_plm_x(i, j, k, qxmarr, qxparr, slope, q, qaux(i,j,k,QC), dx, dt);
//==================== Y slopes ====================================
        for(int n = 0; n < QVAR; n++)
            slope[n] =  PeleC_slope_y(i, j, k, n, q);
//==================== Y interp ====================================
        PeleC_plm_y(i,j,k, qymarr, qyparr, slope, q, qaux(i,j,k,QC),dy, dt);
//===================== Z slopes ===================================
        for(int n = 0; n < QVAR; ++n)
            slope[n] = PeleC_slope_z(i,j,k,n, q);
//==================== Z interp ====================================
        PeleC_plm_z(i, j, k, qzmarr, qzparr, slope, q, qaux(i,j,k,QC), dz, dt);
    }); 


/* These are the first flux estimates as per the corner-transport-upwind method */ 
//===================== X initial fluxes ===========================
    cdir = 0; 
    const Box& xflxbx = surroundingNodes(grow(bxg2, cdir, -1), cdir); 
    FArrayBox fx(xflxbx, NVAR);
    auto const& fxarr = fx.array(); 
    FArrayBox qgdx(xflxbx, NGDNV); 
    auto const& gdtempx = qgdx.array(); 
    Elixir fxeli = fx.elixir(), qgdxeli = qgdx.elixir(); 
    AMREX_PARALLEL_FOR_3D (xflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k,bclx, bchx, dlx, dhx, qxmarr, qxparr, fxarr, gdtempx, qaux, cdir);
    });

//===================== Y initial fluxes ===========================
    cdir = 1; 

    FArrayBox fy(yflxbx, NVAR); 
    auto const& fyarr = fy.array();
    FArrayBox qgdy(yflxbx, NGDNV); 
    auto const& gdtempy = qgdy.array();
    Elixir fyeli = fy.elixir(), qgdyeli = qgdy.elixir(); 
    AMREX_PARALLEL_FOR_3D (yflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, bcly, bchy, dly, dhy, qymarr, qyparr, fyarr, gdtempy, qaux, cdir); 
    }); 

//===================== Z initial fluxes ===========================
    cdir = 2; 
    FArrayBox fz(zflxbx, NVAR);
    auto const& fzarr = fz.array(); 
    FArrayBox qgdz(zflxbx, NGDNV); 
    auto const& gdtempz = qgdz.array(); 
    Elixir fzeli = fz.elixir(), qgdzeli = qgdz.elixir(); 
    AMREX_PARALLEL_FOR_3D (zflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k,bclz, bchz, dlz, dhz, qzmarr, qzparr, fzarr, gdtempz, qaux, cdir);
    });


//===================== X interface corrections ====================

    cdir = 0;
    const Box& txbx = grow(bxg1, cdir, 1);
    const Box& txbxm = growHi(txbx, cdir, 1); 
    FArrayBox qxym(txbxm, QVAR); 
    FArrayBox qxyp(txbx , QVAR);
    auto const& qmxy = qxym.array();
    auto const& qpxy = qxyp.array();  

    FArrayBox qxzm(txbxm, QVAR); 
    FArrayBox qxzp(txbx , QVAR);
    auto const& qmxz = qxzm.array();
    auto const& qpxz = qxzp.array(); 

/* The first interface corrections are grouped by direction, as they utilize the same box. This 
 * helps reduce launch overhead. */  
    AMREX_PARALLEL_FOR_3D (txbx, i,j,k, {
// ----------------- X|Y ------------------------------------------
        PeleC_transy1(i,j,k, qmxy, qpxy, qxmarr, qxparr, fyarr,
                      qaux, gdtempy, cdtdy);


// ---------------- X|Z ------------------------------------------
        PeleC_transz1(i,j,k, qmxz, qpxz, qxmarr, qxparr, fzarr,
                      qaux, gdtempz, cdtdz);

   });
   
   const Box& txfxbx = surroundingNodes(bxg1, cdir); 
   FArrayBox fluxxy(txfxbx, NVAR); 
   FArrayBox fluxxz(txfxbx, NVAR); 
   FArrayBox gdvxyfab(txfxbx, NGDNV); 
   FArrayBox gdvxzfab(txfxbx, NGDNV); 
   Elixir fluxxyeli = fluxxy.elixir(), gdvxyeli = gdvxyfab.elixir(); 
   Elixir fluxxzeli = fluxxz.elixir(), gdvxzeli = gdvxzfab.elixir();    
   
   auto const& flxy = fluxxy.array(); 
   auto const& flxz = fluxxz.array(); 
   auto const& qxy = gdvxyfab.array(); 
   auto const& qxz = gdvxzfab.array();  

//===================== Riemann problem X|Y X|Z ====================
/* The Interface corrected fluxes are grouped in the same launch, this 
 * helps reduce launch overhead. */  

    AMREX_PARALLEL_FOR_3D (txfxbx, i,j,k, {     
// -----------------  X|Y ---------------------------------------------------       
      PeleC_cmpflx(i,j,k, bclx, bchx, dlx, dhx, qmxy, qpxy, flxy, qxy, qaux, cdir); 
// -----------------  X|Z --------------------------------------------------
      PeleC_cmpflx(i,j,k, bclx, bchx, dlx, dhx, qmxz, qpxz, flxz, qxz, qaux, cdir); 
    }); 


//===================== Y interface corrections ====================

    cdir = 1; 
    const Box& tybx  = grow(bxg1, cdir, 1);
    const Box& tybxm = growHi(tybx, cdir, 1); 
    FArrayBox qyxm(tybxm, QVAR); 
    FArrayBox qyxp(tybx, QVAR); 
    FArrayBox qyzm(tybxm, QVAR); 
    FArrayBox qyzp(tybx, QVAR); 
    auto const& qmyx = qyxm.array(); 
    auto const& qpyx = qyxp.array(); 
    auto const& qmyz = qyzm.array(); 
    auto const& qpyz = qyzp.array(); 

    AMREX_PARALLEL_FOR_3D (tybx, i, j , k, {
//--------------------- Y|X -------------------------------------
        PeleC_transx1(i,j,k, qmyx, qpyx, qymarr, qyparr, fxarr,
                      qaux, gdtempx, cdtdx); 

//--------------------- Y|Z ------------------------------------              
        PeleC_transz2(i,j,k, qmyz, qpyz, qymarr, qyparr, fzarr, 
                      qaux, gdtempz, cdtdz); 
    }); 

//Clear used Temp Elixirs 
    fzeli.clear(); 
    qgdzeli.clear(); 

//===================== Riemann problem Y|X Y|Z  ====================

   const Box& tyfxbx = surroundingNodes(bxg1, cdir);
   FArrayBox fluxyx(tyfxbx, NVAR); 
   FArrayBox fluxyz(tyfxbx, NVAR); 
   FArrayBox gdvyxfab(tyfxbx, NGDNV); 
   FArrayBox gdvyzfab(tyfxbx, NGDNV); 
   Elixir fluxyxeli = fluxyx.elixir(), gdvyxeli = gdvyxfab.elixir(); 
   Elixir fluxyzeli = fluxyz.elixir(), gdvyxeli = gdvyxfab.elixir();    
   
   auto const& flyx = fluxyx.array(); 
   auto const& flyz = fluxyz.array(); 
   auto const& qyx = gdvyxfab.array(); 
   auto const& qyz = gdvyzfab.array();  
   
    AMREX_PARALLEL_FOR_3D (tyfxbx, i, j, k, {
//--------------------- Y|X ----------------------------------------------------
      PeleC_cmpflx(i,j,k, bcly, bchy, dly, dhy, qmyx , qpyx , flyx, qyx, qaux, cdir);

//--------------------- Y|Z ----------------------------------------------------
      PeleC_cmpflx(i,j,k, bcly, bchy, dly, dhy, qmyz , qpyz , flyz, qyz, qaux, cdir); 
    });


//===================== Z interface corrections ====================

    cdir = 2; 
    const Box& tzbx = grow(bxg1, cdir, 1);
    const Box& tzbxm = growHi(tzbx, cdir, 1);
    FArrayBox qzxm(tzbxm, QVAR); 
    FArrayBox qzxp(tzbx, QVAR); 
    FArrayBox qzym(tzbxm, QVAR); 
    FArrayBox qzyp(tzbx, QVAR); 
    auto const& qmzx = qzxm.array(); 
    auto const& qpzx = qzxp.array(); 
    auto const& qmzy = qzym.array(); 
    auto const& qpzy = qzyp.array(); 

    AMREX_PARALLEL_FOR_3D (tzbx, i, j , k, {
//------------------- Z|X -------------------------------------
        PeleC_transx2(i,j,k, qmzx, qpzx, qzmarr, qzparr, fxarr,
                      qaux, gdtempx, cdtdx);

//------------------- Z|Y -------------------------------------                
        PeleC_transy2(i,j,k, qmzy, qpzy, qzmarr, qzparr, fyarr, 
                      qaux, gdtempy,cdtdy); 
    }); 

/* Clear Elixirs */ 
    fxeli.clear(); 
    fyeli.clear(); 
    qgdxeli.clear(); 
    qgdyeli.clear(); 


//===================== Riemann problem Z|X Z|Y  ====================
    
   const Box& tzfxbx = surroundingNodes(bxg1, cdir);
   FArrayBox fluxzx(tzfxbx, NVAR); 
   FArrayBox fluxzy(tzfxbx, NVAR); 
   FArrayBox gdvzxfab(tzfxbx, NGDNV); 
   FArrayBox gdvzyfab(tzfxbx, NGDNV); 
   Elixir fluxzxeli = fluxzx.elixir(), gdvzxeli = gdvzxfab.elixir(); 
   Elixir fluxzyeli = fluxzy.elixir(), gdvzxeli - gdvzxfab.elixir();    

 
   auto const& flzx = fluxzx.array(); 
   auto const& flzy = fluxzy.array(); 
   auto const& qzx = gdvzxfab.array(); 
   auto const& qzy = gdvzyfab.array(); 

   AMREX_PARALLEL_FOR_3D (tzfxbx, i, j, k, {
//-------------------- Z|X -----------------------------------------------------
      PeleC_cmpflx(i,j,k, bclz, bchz, dlz, dhz, qmzx, qpzx, flzx, qzx, qaux, cdir);
//-------------------- Z|Y -----------------------------------------------------
      PeleC_cmpflx(i,j,k, bclz, bchz, dlz, dhz, qmzy, qpzy, flzy, qzy, qaux, cdir); 
    });


    FArrayBox qmfab(bxg2, QVAR); 
    FArrayBox qpfab(bxg1, QVAR);
    auto const& qm = qmfab.array(); 
    auto const& qp = qpfab.array();  

//==================== X| Y&Z ======================================
    cdir = 0;   
    const Box& xfxbx = surroundingNodes(bx, cdir);
    const Box& tyzbx = grow(bx, cdir, 1); 
    AMREX_PARALLEL_FOR_3D ( tyzbx, i, j, k, { 
        PeleC_transyz(i,j,k, qm, qp, qxmarr, qxparr, flyz, flzy, qyz, qzy, qaux, srcQ, hdt, hdtdy, hdtdz); 
    });

//============== Final X flux ======================================
    AMREX_PARALLEL_FOR_3D(xfxbx, i, j, k, {
        PeleC_cmpflx(i,j,k,bclx, bchx, dlx, dhx, qm, qp, flx1, q1, qaux, cdir); 
    }); 

//==================== Y| X&Z ======================================
    cdir = 1;   
    const Box& yfxbx = surroundingNodes(bx, cdir);
    const Box& txzbx = grow(bx, cdir, 1); 
    AMREX_PARALLEL_FOR_3D (txzbx, i, j, k, { 
        PeleC_transxz(i,j,k, qm, qp, qymarr, qyparr, flxz, flzx, qxz, qzx, qaux, srcQ, hdt, hdtdx, hdtdz); 
    }); 

//============== Final Y flux ======================================
    AMREX_PARALLEL_FOR_3D(yfxbx, i, j, k, {
        PeleC_cmpflx(i,j,k,bcly, bchy, dly, dhy, qm, qp, flx2, q2, qaux, cdir); 
    }); 

//==================== Z| X&Y ======================================
    cdir = 2;   
    const Box& zfxbx = surroundingNodes(bx, cdir);
    const Box& txybx = grow(bx, cdir, 1); 
    AMREX_PARALLEL_FOR_3D ( txybx, i, j, k, { 
        PeleC_transxy(i,j,k, qm, qp, qzmarr, qzparr, flxy, flyx, qxy, qyx, qaux, srcQ, hdt, hdtdx, hdtdy);
    }); 

//============== Final Z flux ======================================
    AMREX_PARALLEL_FOR_3D(zfxbx, i, j, k, {
        PeleC_cmpflx(i,j,k,bclz, bchz, dlz, dhz, qm, qp, flx3, q3, qaux, cdir); 
    }); 

//===================== Construct p div{U} =========================
    AMREX_PARALLEL_FOR_3D (bx, i, j, k, {
        PeleC_pdivu(i,j,k, pdivu, q1, q2, q3, a1, a2, a3, vol); 
    });
}

