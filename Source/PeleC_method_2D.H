#ifndef PELEC_METHOD_2D_H_
#define PELEC_METHOD_2D_H_
#include "PeleC_K.H" 
#include "PeleC_EOS.H" 
 
void PeleC_umeth2d(amrex::Box const& bx, amrex::FArrayBox &flatn,
                   amrex::FArrayBox const& quax, amrex::FArrayBox const& srcQ, 
                   amrex::FArrayBox const& bcMask, amrex::FArrayBox &flx1, 
                   amrex::FArrayBox &flx2, amrex::FArrayBox &q1, 
                   amrex::FArrayBox &q2, amrex::FArrayBox &a1, 
                   amrex::FArrayBox &a2, amrex::FArrayBox &pdiv, 
                   amrex::FArrayBox &vol, amrex::Real *dx, amrex::Real dt)
{
    amrex::Real const dxdt  = dt/dx[0]; 
    amrex::Real const hdtdx = 0.5*dtdx; 
    amrex::Real const hdtdy = 0.5*dt/dx[1]; 
    amrex::Real const hdt   = 0.5*dt; 

    const amrex::Box& bxg1 = amrex::grow(bx, 1); 
    const amrex::Box& bxg2 = amrex::grow(bx, 2); 
    Gpu::AsyncFab slope(bxg2, QVAR);

    //===================== X slopes ===================================
    int cdir = 0; 
    const amrex::Box& xslpbx = amrex::grow(bxgx1, cdir, 1); 
    AMREX_LAUNCH_DEVICE_LAMBDA(xslpbx, tbx, { 
        PeleC_slope_x(tbx, slope.fab(), q.fab());
    }); 

    //==================== X interp ====================================
    Gpu::AsyncFab qxm(xslpbx, QVAR); 
    Gpu::AsyncFab qxp(xslpbx, QVAR); 
    AMREX_LAUNCH_DEVICE_LAMBDA(xslpbx, tbx, {
        PeleC_plm_x(tbx, qxm.fab(), qxp.fab(), slope.fab(), q.fab(), quax.fab(), 
                    srcQ.fab(), dloga.fab(), dx, dt); 
    });

    //===================== X initial fluxes ===========================
    const amrex::Box& xflxbx = amrex::surroundingNodes(bxg1,cdir); 
    Gpu::AsyncFab fx(xflxbx, NUM_VAR);
    Gpu::AsyncFab q1(xflxbx, QVAR); 
    //May omit bcmask. 
    AMREX_LAUNCH_DEVICE_LAMBDA(xflxbx, tbx, {
        PeleC_cmpflx(tbx, qxm.fab(), qxp.fab(), fx.fab(), q1.fab(), qaux.fab(),
                     bcMask.fab(), 0);
    });
 
    //==================== Y slopes ====================================
    cdir = 1; 
    const amrex::Box& yslpbx = amrex::grow(bxg1, cdir, 1); 
    Gpu::AsyncFab qym(yslpbx, QVAR);
    Gpu::AsyncFab qyp(yslpbx, QVAR);  
    AMREX_LAUNCH_DEVICE_LAMBDA(yslpbx, tbx,{
        PeleC_slope_x(tbx, slope.fab(), q.fab()); 
    });

    //==================== Y interp ====================================
    AMREX_LAUNCH_DEVICE_LAMBDA(yslpbx, tbx, {
        PeleC_plm_y(tbx, qym.fab(), qyp.fab(), slope.fab(), q.fab(), dy, dt); 
    });

    //===================== Y initial fluxes ===========================
    const amrex::Box& yflxbx = amrex::surroundingNodes(bxg1,cdir); 
    Gpu::AsyncFab fy(yflxbx, NUM_VAR); 
    Gpu::AsyncFab q2(yflxbx, QVAR); 
    AMREX_LAUNCH_DEVICE_LAMBDA(yflxbx, tbx, {
        PeleC_cmpflx(tbx, qym.fab(), qyp.fab(), fy.fab(), q2.fab(), qaux.fab(), bcMask.fab(), 1); 
    }); 

    //===================== X interface corrections ====================
    cdir = 0; 
    Gpu::AsyncFab qm(bgx1, QVAR); 
    Gpu::AsyncFab qp(bgx1, QVAR); 
    //TODO think about box size for these 
    const amrex::Box& xbx = amrex::grow(bx, cdir, 1);
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_transy(tbx, qxm.fab(), qm.fab(), qxp.fab(), qp.fab(), fy.fab(),
                     q2.fab(), quax.fab(), srcQ.fab(), hdt, hdtdy); 
    }); 

    //===================== Final Riemann problem X ====================
    const amrex::Box& xfxbx = amrex::surroundingNodes(bx, cdir); 
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_cmpflx(tbx, qm.fab(), qp.fab(), flx1.fab(), qaux.fab(), 0);
    }); 
     
    //===================== Y interface corrections ====================
    cdir = 1; 
    const amrex::Box& ybx = amrex::grow(bx, cdir, 1);
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_transx(tbx, qym.fab(), qm.fab(), qyp.fab(), qp.fab(), fx.fab(),
                     quax.fab(), srcQ.fab(), hdt, hdtdx, area1, vol); 
    });
   
    //===================== Final Riemann problem Y ====================
    const amrex::Box& yfxbx = amrex::surroundingNodes(bx, cdir); 
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_cmpflx(tbx, qm.fab(), qp.fab(), flx2.fab(), q2.fab(), quax.fab() 1); 
    });
 
    //===================== Construct p div{U} =========================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_pdivu(tbx, pdivu.fab(), q1.fab(), q2.fab(), vol); 
    }); 
}

AMREX_GPU_DEVICE
inline
void
PeleC_slope_x(amrex::Box const& bx, amrex::FArrayBox& dqfab, amrex::FArrayBox& const& qfab)
{
    const auto len = amrex::length(bx);
    const auto lo  = amrex::lbound(bx);
    const auto dq = dqfab.view(lo);
    const auto  q =  qfab.view(lo);

    for         (int k = 0; k < len.z; ++k) {
        for     (int j = 0; j < len.y; ++j) {
            AMREX_PRAGMA_SIMD 
            for (int i = 0; i < len.x; ++i) {
                auto f = [] (amrex::Real dlft, amrex::Real drgt) -> amrex::Real {
                    amrex::Real dcen = 0.5*(dlft+drgt);
                    amrex::Real dsgn = std::copysign(1.0, dcen);
                    amrex::Real slop = 2.0 * amrex::min(std::abs(dlft),std::abs(drgt));
                    amrex::Real dlim = (dlft*drgt >= 0.0) ? slop : 0.0;
                    amrex::Real d = dsgn * amrex::min(dlim,std::abs(dcen));
                    return d;
                };

                amrex::Real dlft = 0.5*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/q(i,j,k,QCS) - 0.5*q(i,j,k,QRHO)*(q(i,j,k,QU) - q(i-1,j,k,QU));
                amrex::Real drgt = 0.5*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/q(i,j,k,QCS) - 0.5*q(i,j,k,QRHO)*(q(i+1,j,k,QU) - q(i,j,k,QU));
                amrex::Real d0 = f(dlft, drgt);

                amrex::Real cs2 = q(i,j,k,QCS)*q(i,j,k,QCS);
                dlft = (q(i,j,k,QRHO)-q(i-1,j,k,QRHO)) - (q(i,j,k,QPRES) - q(i-1,j,k,QPRES))/cs2;
                drgt = (q(i+1,j,k,QRHO)-q(i,j,k,QRHO)) - (q(i+1,j,k,QPRES) - q(i,j,k,QPRES))/cs2;
                amrex::Real d1 = f(dlft, drgt);

                dlft = 0.5*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/q(i,j,k,QCS) + 0.5*q(i,j,k,QRHO)*(q(i,j,k,QU) - q(i-1,j,k,QU));
                drgt = 0.5*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/q(i,j,k,QCS) + 0.5*q(i,j,k,QRHO)*(q(i+1,j,k,QU) - q(i,j,k,QU));
                amrex::Real d2 = f(dlft, drgt);

                dlft = q(i,j,k,QV) - q(i-1,j,k,QV);
                drgt = q(i+1,j,k,QV) - q(i,j,k,QV);
                amrex::Real d3 = f(dlft, drgt);

                dlft = q(i,j,k,QW) - q(i-1,j,k,QW);
                drgt = q(i+1,j,k,QW) - q(i,j,k,QW);
                amrex::Real d4 = f(dlft, drgt);

                dq(i,j,k,0) = d0;
                dq(i,j,k,1) = d1;
                dq(i,j,k,2) = d2;
                dq(i,j,k,3) = d3;
                dq(i,j,k,4) = d4;
            }
        }
    }
}

AMREX_GPU_DEVICE
inline
void
PeleC_slope_y (amrex::Box const& bx, amrex::FArrayBox& dqfab, amrex::FArrayBox const& qfab)
{
    const auto len = amrex::length(bx);
    const auto lo  = amrex::lbound(bx);
    const auto dq = dqfab.view(lo);
    const auto  q =  qfab.view(lo);

    for         (int k = 0; k < len.z; ++k) {
        for     (int j = 0; j < len.y; ++j) {
            AMREX_PRAGMA_SIMD 
            for (int i = 0; i < len.x; ++i) {
                auto f = [] (amrex::Real dlft, amrex::Real drgt) -> amrex::Real {
                    amrex::Real dcen = 0.5*(dlft+drgt);
                    amrex::Real dsgn = std::copysign(1.0, dcen);
                    amrex::Real slop = 2.0 * amrex::min(std::abs(dlft), std::abs(drgt));
                    amrex::Real dlim = (dlft*drgt >= 0.0) ? slop : 0.0;
                    amrex::Real d = dsgn * amrex::min(dlim, std::abs(dcen));
                    return d;
                };

                amrex::Real dlft = 0.5*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/q(i,j,k,QCS) - 0.5*q(i,j,k,QRHO)*(q(i,j,k,QV) - q(i,j-1,k,QV));
                amrex::Real drgt = 0.5*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/q(i,j,k,QCS) - 0.5*q(i,j,k,QRHO)*(q(i,j+1,k,QV) - q(i,j,k,QV));
                amrex::Real d0 = f(dlft, drgt);

                amrex::Real cs2 = q(i,j,k,QCS)*q(i,j,k,QCS);
                dlft = (q(i,j,k,QRHO)-q(i,j-1,k,QRHO)) - (q(i,j,k,QPRES) - q(i,j-1,k,QPRES))/cs2;
                drgt = (q(i,j+1,k,QRHO)-q(i,j,k,QRHO)) - (q(i,j+1,k,QPRES) - q(i,j,k,QPRES))/cs2;
                amrex::Real d1 = f(dlft, drgt);

                dlft = 0.5*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/q(i,j,k,QCS) + 0.5*q(i,j,k,QRHO)*(q(i,j,k,QV) - q(i,j-1,k,QV));
                drgt = 0.5*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/q(i,j,k,QCS) + 0.5*q(i,j,k,QRHO)*(q(i,j+1,k,QV) - q(i,j,k,QV));
                amrex::Real d2 = f(dlft, drgt);

                dlft = q(i,j,k,QU) - q(i,j-1,k,QU);
                drgt = q(i,j+1,k,QU) - q(i,j,k,QU);
                amrex::Real d3 = f(dlft, drgt);

                dlft = q(i,j,k,QW) - q(i,j-1,k,QW);
                drgt = q(i,j+1,k,QW) - q(i,j,k,QW);
                amrex::Real d4 = f(dlft, drgt);

                dq(i,j,k,0) = d0;
                dq(i,j,k,1) = d1;
                dq(i,j,k,2) = d2;
                dq(i,j,k,3) = d3;
                dq(i,j,k,4) = d4;
            }
        }
    }
}




AMREX_GPU_DEVICE
inline 
void 
PeleC_shock(amrex::Box const& bx, amrex::FArrayBox const &qfab, amrex::FArrayBox &shkfab, 
                 amrex::Real dx, amrex::Real dy)
{
    const auto len = amrex::length(bx); 
    const auto lo  = amrex::lbound(bx); 
    const auto q   = qfab.view(lo); 
    const auto shk = shkfab.view(lo); 

//TODO figure out the box stuff for fluxes
    for         (int k = 0; k < len.z; ++k){
        for     (int j = 0; j < len.y; ++j){
            for (int i = 0; i < len.x; ++i){
                amrex::Real divU; 
                if(coord_type == 0) divU = 0.5*(q(i+1,j,k,QU) - q(i-1,j,k,QU))/dx + 
                                    0.5*(q(i,j+1,k,QV) - q(i,j-1,k,QV))/dy; 
                else if(coord_type == 1){
                    amrex::Real rc, rm, rp; 
                    rc = double(i + 0.5e0)*dx;
                    rm = double(i - 1.e0)*dx; 
                    rp = double(i + 1.e0)*dx; 

                    divU = 0.5e0*(rp*q(i+1,j,k,QU) - rm*q(i-1,j,k,QU))/(rc*dx) + 
                           0.5e0*(q(i,j+1,k,QV) - q(i,j-1,k,QV))/dy; 
                }
                amrex::Real px_pre, px_post, py_pre, py_post; 
                if (q(i+1,j,k,QPRES) - q(i-1,j,k,QPRES) < 0.0e0){ 
                     px_pre  = q(i+1,j,k,QPRES);
                     px_post = q(i-1,j,k,QPRES);
                }
                else{
                     px_pre  = q(i-1,j,k,QPRES);
                     px_post = q(i+1,j,k,QPRES);
                }
                if (q(i,j+1,k,QPRES) - q(i,j-1,k,QPRES) < 0.0e0){
                     py_pre  = q(i,j+1,k,QPRES);
                     py_post = q(i,j-1,k,QPRES);
                }
                else{
                     py_pre  = q(i,j-1,k,QPRES);
                     py_post = q(i,j+1,k,QPRES);
                }
                amrex::Real e_x, e_y, d, p_pre, p_post; 
                e_x = pow(q(i+1,j,k,QU) - q(i-1,j,k,QU),2);
                e_y = pow(q(i,j+1,k,QV) - q(i,j-1,k,QV),2);
//TODO need small here                
                d   = 1.e0/(e_x + e_y + small); 
                p_pre = e_x*px_pre + e_y*py_pre; 
                p_post = e_x*px_post + e_y*py_post; 
                if(p_pre == 0.0e0) pjump = 0.0e0; 
//TODO need eps here
                else pjump = eps - ( p_post - p_pre)/p_pre; 
                
                if(pjump < 0.0e0 && divU < 0.0e0) shk(i,j,k) = 1.0e0; 
                else shk(i,j,k) = 0.0e0; 
            }
        }
    }
}

namespace{
//Riemann_md 
AMREX_GPU_DEVICE
inline
void
riemann (const amrex::Real rl, const amrex::Real ul, const amrex::Real vl,
         const amrex::Real v2l, const amrex::Real pl, const amrex::Real rel, 
         const amrex::Real *spl, const amrex::Real rr, 
         const amrex::Real ur, const amrex::Real vr,
         const amrex::Real v2r, const amrex::Real pr, const amrex::Real rer, 
         const amrex::Real *spr, EOS &state, EOS &gdnv_state, 
         amrex::Real &uflx_rho, amrex::Real &uflx_u, 
         amrex::Real &uflx_v, amrex::Real &uflx_v, amrex::Real &uflx_w,
         amrex::Real &uflx_eden, amrex::Real &uflx_eint,
         amrex::Real &qint_iu, amrex::Real &qint_iv1, amrex::Real &qint_iv2, 
         amrex::Real &qint_gdpres, amrex::Real &qint_gdgame,
         amrex::Real &ustar, const int idir, const amrex::Real csmall,
         const amrex::Real cav)
//TODO consider making a uflx array instead of passing each in. 
{
    amrex::Real wsmall = small_dens*csmall; 
    state.rho = rl; 
    state.p   = pl; 
#pragma unroll 
    for(int i = 0; i < NUM_SPECIES; ++i)  state.massfrac[i] = spl[i]; 
    state.eos_rp(); 
    amrex::Real csl = state.cs; 

    state.rho = rr; 
    state.p   = pr; 
#pragma unroll 
    for(int i = 0; i < NUM_SPECIES; ++i) state.massfrac[i] = spr[i]; 
    state.eos_rp(); 
    amrex::Real csr = state.cs; 

//    amrex::Real wl = amrex::max(wsmall, sqrt(abs(gamcl*pl*rl))); 
//    amrex::Real wr = amrex::max(wsmall, sqrt(abs(gamcr*pr*rr))); 

    amrex::Real wl = amrex::max(wsmall, rl*csl); 
    amrex::Real wr = amrex::max(wsmall, rr*csr); 

    amrex::Real pstar = ((wr*pl + wl*pr) + wl*wr*(ul - ur))/(wl + wr); 
    ustar = ((wl*ul + wr*ur) + (pl - pr))/(wl + wr); 
    
    pstar = amrex::max(pstar, small_pres); 
    if (abs(ustar) < smallu*0.5e0*(abs(ul) + abs(ur))) ustar = 0.e0; 
    amrex::Real ro, uo, po; 
    if(ustar > 0.e0){
        ro = rl; 
        uo = ul; 
        po = pl; 
#pragma unroll    
        for(int i = 0; i < NUM_SPECIES; ++i) 
            gdnv_state.massfrac[i] = spl[i];
    }
    else if(ustar < 0.e0){
        ro = rr; 
        uo = ur; 
        po = pr; 
#pragma unroll    
        for(int i = 0; i < NUM_SPECIES; ++i) 
            gdnv_state.massfrac[i] = spr[i];
    }
    else{
        ro = 0.5e0*(rr+rl); 
        uo = 0.5e0*(ur+ul); 
        po = 0.5e0*(pr+pl); 
#pragma unroll    
        for(int i = 0; i < NUM_SPECIES; ++i) 
            gdnv_state.massfrac[i] = 0.5e0*(spr[i] + spl[i]);
    }  
    gdnv_state.rho = ro; 
    gdnv_state.p   = po; 
    gdnv_state.eos_rp(); 
    amrex::Real reo = gdnv_state.rho*gdnv_state.e;
    amrex::Real co  = gdvn_state.cs; 

    amrex::Real drho  = (pstar - po)/(co*co); 
    amrex::Real rstar = ro + drho; 
    rstar = amrex::max(small_dens, rstar); 

    gdnv_state.rho = rstar; 
    gdnv_state.p   = pstar; 

    gdnv_state.eos_rp(); 
    amrex::Real cstar = gdnv_state.cs; 
    amrex::Real estar = gdnv_state.rho*gdnv_state.e; 

    amrex::Real sgnm = sign(ustar); 
    amrex::Real spoit = co - sgnm*uo; 
    amrex::real spin  = cstar - sgnm*ustar; 
    amrex::Real ushock = 0.5e0*(spin + spout); 

    if(pstar - po >= 0.e0){
        spin = ushock; 
        spout = ushock; 
    }
    if(spout-spin == 0.e0) scr = small*cav; 
    else scr = spout-spin; 
    
    amrex::Real frac = (1.e0 + (spout + spin)/scr)*0.5e0; 
    fram = amrex::max(0.e0, amrex::min(1.e0, frac)); 

    if(ustar > 0.e0){
        vgd = vl; 
        wgd = v2l; 
    }
    else if(ustar < 0.e0){
        vgd = vr;
        wgd = v2r; 
    }
    else{
        vgd = 0.5e0*(vl + vr); 
        wgd = 0.5e0*(v2l + v2r); 
    }
    
    amrex::Real rgd = frac*rstar + (1.e0 - frac)*ro; 
    qint_iu = frac*ustar + (1.e0 - frac)*uo; 
    qint_iv1 = vgd; 
    qint_iv2 = wgd; 
    
    qint_gdpres = frac*pstar + (1.e0 - frac)*po; 
    gdvn_state.rho = rgd; 
    gdvn_state.p = qint_gdpres; 
    gdvn_state.eos_rp(); 
    amrex::Real regd = gdvn_state.rho*gdnv_state.e; 
    
    if(spout < 0.e0){
        rgd = ro; 
        qint_iu = uo; 
        qint_gdpres = po; 
        regd = reo; 
    }
    if(spin >= 0.e0){
        rgd = rstar; 
        qint_iu = ustar; 
        qint_gdpres = pstar; 
        regd = estar; 
    }

    gdnv_state.rho = rgd; 
    gdnv_state.p = qint_gdpres; 
    gdnv_state.eos_rp(); 
    regd = gdnv_state.rho*gdnv_state.e; 

    qint_gdgame = qint_gdpres/regd + 1.e0; 
//Where is bc_test_val? 
//Ignore for now TODO fix. 
//    qint_iu     = bc_test_val * qint_iu; 

    uflx_rho    = rgd*qint_iu; 

    if( idir == 1){
        uflx_u = uflx_rho*qint_iu; 
        uflx_v = uflx_rho*qint_iv1; 
        uflx_w = uflx_rho*qint_iv2; 
        if(coord_type ==0) uflx_u += qint_gdpres; 
    }
    else if(idir == 2){
        uflx_u = uflx_rho*qint_iv1; 
        uflx_v = uflx_rho*qint_iu + qint_gdpres; 
        uflx_w = uflx_rho*qint_iv2; 
    }
    else{
        uflx_u = uflx_rho*qint_iv1; 
        uflx_v = uflx_rho*qint_iv2; 
        uflx_w = uflx_rho*qint_iu + qint_gdpres; 
    }
    amrex::Real rhoetot = regd + 0.5e0*rgd*(qint_iu*qint_iu + 
                          qint_iv1*qint_iv1 + qint_iv2*qint_iv2); 
    uflx_eden = qint_iu*(rhoetot + qint_gdpres); 
    uflx_eint = qint_iu*regd;              
}
}

AMREX_GPU_DEVICE
inline
void
PeleC_cmpflx(amrex::Box const& bx, amrex::FArrayBox const& qlfab, 
             amrex::FArrayBox const & qrfab, amrex::FArrayBox& fluxfab,
             amrex::FArrayBox const& qfab, amrex::FArrayBox const& qafab, int idir)
{
    const auto  len = amrex::length(bx);
    const auto   lo = amrex::lbound(bx);
    const auto    q =  qfab.view(lo);
    const auto   qa = qafab.view(lo); 
    const auto   ql = qlfab.view(lo); 
    const auto   qr = qrfab.view(lo); 
    const auto  flx = fluxfab.view(lo);
    EOS state, gdnv_state; 
    amrex::Real cav;
    amrex::Real spl[NUM_SPECIES];
    amrex::Real spr[NUM_SPECIES]; 
//    amrex::Real ul, ur, vl, vr, v2l, v2r, rel, rer; 
    int idx, IU, IV, IV2; 
*  if(idir == 0){
         IU = QU; 
         IV = QV; 
        IV2 = QW; 
    }
    elseif( idir == 1){
        IU = QV; 
        IV = QU; 
       IV2 = QW; 
    }
    else{
        IU = QW; 
        IV = QU; 
       IV2 = QV; 
    }
       

    for         (int k = 0; k < len.z; ++k) {
        for     (int j = 0; j < len.y; ++j) {
            for (int i = 0; i < len.x; ++i) {
                 if(idir == 0) cav = 0.5e0*(qa(i,j,k,QC) + qa(i-1,j,k,QC)); 
                 else cav = 0.5e0*(qa(i,j,k,QC) + qa(i,j-1,k,QC));
#pragma unroll 
                for(int sp = 0; sp < NUM_SPECIES; ++sp){
                    spl[sp] = ql(i,j,k,QFS+sp); 
                    spr[sp] = qr(i,j,k,QFS+sp); 
                }                 
/* TODO assess if we need this. 
//Outflow Hack --> 
                   ul = ql(i,j,k,IU); 
                   vl = ql(i,j,k,IV); 
                  v2l = ql(i,j,k,IV2); 
                  rel = ql(i,j,k,QREINT); 
                
                   ur = qr(i,j,k,IU); 
                   vr = qr(i,j,k,IV); 
                  v2r = qr(i,j,k,IV2); 
                  rer = qr(i,j,k,QREINT); 
                 idx = idir == 1 ? i : j; 
                 if(bclo[idir] == Outflow && idx == domlo[idir])
                 {
                    ul = ur; 
                    vl = vr; 
                    v2l = v2r; 
                    rel = rer; 
                 }
                 if(bchi[idir] == Outflow && idx == domhi[idir])
                 {  
                    ur  = ul; 
                    vr  = vl; 
                    v2r = v2l; 
                    rer = rel;         
                 } 
                  riemann(ql(i,j,k,QRHO), ul, vl, v2l, 
                         ql(i,j,k,QPRES), rel, spl,   
                         qr(i,j,k,QRHO), ur, vr, v2r, 
                         qr(i,j,k,QPRES), rer, spr,
                         state, gdnv_state, flx(i,j,k,URHO), flx(i,j,k,UMX), flx(i,j,k,UMY), 
                         flx(i,j,k,UMZ), flx(i,j,k,UEDEN), flx(i,j,k,UEINT), idir, csmall, cav); 
                
*/                 
                 riemann(ql(i,j,k,QRHO), ql(i,j,k,QU), ql(i,j,k,QV), ql(i,j,k,QW), 
                         ql(i,j,k,QPRES), ql(i,j,k,QRIENT), spl,   
                         qr(i,j,k,QRHO), qr(i,j,k,QU), qr(i,j,k,QV), qr(i,j,k,QW), 
                         qr(i,j,k,QPRES), qr(i,j,k,QRIENT), spr,
                         state, gdnv_state, flx(i,j,k,URHO), flx(i,j,k,UMX), flx(i,j,k,UMY), 
                         flx(i,j,k,UMZ), flx(i,j,k,UEDEN), flx(i,j,k,UEINT),
                         q(i,j,k,IU), q(i,j,k,IV1), q(i,j,k,IV2), q(i,j,k,GDPRES),
                         q(i,j,k,GDGAME), ustar, idir, csmall, cav); 

                amrex::Real flxrho = flx(i,j,k,URHO); 
                for(int ipass = 0; ipass < npassive; ++ipass){
                    int n = EOS::upass_map(ipass); 
                    int nqp = EOS::qpass_map(ipass);                     
                    if(ustar > 0) flx(i,j,k,n) = flxrho*ql(i,j,k,nqp); 
                    else if(ustar < 0) flx(i,j,k,n) = flxrho*qr(i,j,k,nqp); 
                    else flx(i,j,k,n) = flxrho*0.5e0*(ql(i,j,k,nqp) + qr(i,j,k,nqp)); 
                }
                    
            }
        }
    }
}



AMREX_GPU_DEVICE
inline 
void 
PeleC_plm_x(amrex::Box const& bx, amrex::FArrayBox qmfab, amrex::FArrayBox qpfab,
            amrex::FArrayBox const& slopefab,  amrex::FArrayBox const& qfab,
            amrex::FArrayBox const& qauxfab,   amrex::FArrayBox const& srcQfab, 
            amrex::FArrayBox const& dlogfab,   amrex::Real const dx, amrex::Real const dt)
{
    const auto len    = amrex::length(bx); 
    const auto lo     = amrex::lbound(bx); 
    const auto qm     = qmfab.view(lo); 
    const auto qp     = qpfab.view(lo); 
    const auto slope  = slopefab.view(lo); 
    const auto q      = qfab.view(lo); 
    const auto srcQ   = srcQfab.view(lo); 
    const auto qa     = qauxfab.view(lo); 
    const auto dloga  = dlogfab.view(lo); 

    amrex::Real e[3]; 
    amrex::Real dtdx = dt/dx; 
    
    for             (int k = 0; k < len.z; ++k){
        for         (int j = 0; j < len.y; ++j){
            for     (int i = 0; i < len.x; ++i){
                amrex::Real cc    = qa(i,j,k,QC); 
                amrex::Real cs    = cc*cc; 
                amrex::Real rho   = q(i,j,k,QRHO); 
                amrex::Real u     = q(i,j,k,QU); 
                amrex::Real v     = q(i,j,k,QV); 
                amrex::Real p     = q(i,j,k,QPRES); 
                amrex::Real rhoe  = q(i,j,k,QREINT); 
                amrex::Real enth  = ((rhoe+p)/rho)/cs; 
                
                amrex::Real drho  = slope(i,j,k,QRHO); 
                amrex::Real du    = slope(i,j,k,QU); 
                amrex::Real dv    = slope(i,j,k,QV); 
                amrex::Real dp    = slope(i,j,k,QPRES); 
                amrex::Real drhoe = slope(i,j,k,QREINT); 
                
                amrex::Real alpham = 0.5e0*(dp/(rho*cc) - du)*rho/cc; 
                amrex::Real alphap = 0.5e0*(dp/(rho*cc) + du)*rho/cc; 
                amrex::Real alpha0r = drho - dp/cs;
                amrex::Real alpha0e = drhoe - dp*enth;
                amrex::Real alpha0v = dv; 
                
                e[0] = u - cc; 
                e[1] = u; 
                e[2] = u + cc; 
// ------------ Construct the right state on the i-1/2 interface 
                amrex::Real rho_ref  = rho - 0.5e0*(1.e0 + dtdx*min(e[0],0.e0))*drho;                
                amrex::Real u_ref    = u - 0.5e0*(1.e0 + dtdx*min(e[0], 0.e0))*du; 
                amrex::Real v_ref    = v - 0.5e0*(1.e0 + dtdx*min(e[0], 0.e0))*dv; 
                amrex::Real p_ref    = p - 0.5e0*(1.e0 + dtdx*min(e[0], 0.e0))*dp; 
                amrex::Real rhoe_ref = rhoe - 0.5e0*(1.e0 + dtdx*min(e[0], 0.e0))*drhoe; 

                amrex::Real apright = 0.25e0*dtdx*(e[0] - e[2])*
                                      (1.e0 - std::copysign(1.0, e[2]))*alphap; 
                amrex::Real amright = 0.25e0*dtdx*(e[0] - e[2])*
                                      (1.e0 - std::copysign(1.0, e[0]))*alpham; 

                amrex::Real azrright = 0.25e0*dtdx*(e[0] - e[1])*
                                      (1.e0 - std::copysign(1.0, e[1]))*alpha0r; 
                amrex::Real azeright = 0.25e0*dtdx*(e[0] - e[1])*
                                      (1.e0 - std::copysign(1.0, e[1]))*alpha0e; 
                amrex::Real azv1rght = 0.25e0*dtdx*(e[0] - e[1])*
                                      (1.e0 - std::copysign(1.0, e[1]))*alpha0v; 

                qp(i,j,k,QRHO)   = rho_ref + apright + amright + azrright;
                qp(i,j,k,QRHO)   = amrex::max(small_dens,qp(i,j,k,QRHO));
                qp(i,j,k,QU)     = u_ref + (apright - amright)*cc/rho;
                qp(i,j,k,QV)     = v_ref + azv1rght;
                qp(i,j,k,QPRES)  = p_ref + (apright + amright)*csq;
                qp(i,j,k,QPRES)  = amrex::max(qp(i,j,k,QPRES),small_pres);
                qp(i,j,k,QREINT) = rhoe_ref + (apright + amright)*enth*csq + azeright;

// ------------ Construct the left state on the i+1/2 interface 
                
                rho_ref = rho + 0.5e0*(1.e0 - dtdx*max(e[2],0.e0))*drho;
                u_ref = u + 0.5e0*(1.e0 - dtdx*max(e[2],0.e0))*du;
                v_ref = v + 0.5e0*(1.e0 - dtdx*max(e[2],0.e0))*dv;
                p_ref = p + 0.5e0*(1.e0 - dtdx*max(e[2],0.e0))*dp;
                rhoe_ref = rhoe + 0.5e0*(1.e0 - dtdx*max(e[2],0.e0))*drhoe;

                apleft = 0.25e0*dtdx*(e[2] - e[2])*(1.e0 + std::copysign(1.e0,e[2]))*alphap;
                amleft = 0.25e0*dtdx*(e[2] - e[0])*(1.e0 + std::copysign(1.e0,e[0]))*alpham;

                azrleft = 0.25e0*dtdx*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0r;
                azeleft = 0.25e0*dtdx*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0e;
                azv1left = 0.25e0*dtdx*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0v;
  
                qm(i+1,j,k,QRHO) = rho_ref + apleft + amleft + azrleft;
                qm(i+1,j,k,QRHO) = max(qm(i+1,j,k,QRHO),small_dens);
                qm(i+1,j,k,QU) = u_ref + (apleft - amleft)*cc/rho;
                qm(i+1,j,k,QV) = v_ref + azv1left;
                qm(i+1,j,k,QPRES) = p_ref + (apleft + amleft)*csq;
                qm(i+1,j,k,QPRES) = max(qm(i+1,j,k,QPRES), small_pres);
                qm(i+1,j,k,QREINT) = rhoe_ref + (apleft + amleft)*enth*csq + azeleft;
                                
                if(dloga(i,j,k)!=0){
                    amrex::Real courn = dtdx*(cc+abs(u)); 
                    amrex::Real eta = (1.e0-courn)/(cc*dt*abs(dloga(i,j,k))); 
                    amrex::Real dlogatmp = amrex::min(eta, 1.e0)*dloga(i,j,k);
                    amrex::Real sourcr = -0.5e0*dt*rho*dlogatmp*u; 
                    amrex::Real sourcp = sourcr*cs; 
                    amrex::Real source = courcp*enth; 
                    
                    qm(i+1,j,k,QRHO) += sourcr; 
                    qm(i+1,j,k,QRHO) = amrex::max(qm(i+1,j,k,QRHO), small_dens); 
                    qm(i+1,j,k,QPRES) += sourcp; 
                    qm(i+1,j,k,QREINT) += source; 
                    
                    qp(i,j,k,QRHO) += sourcr; 
                    qp(i,j,k,QRHO) = amrex::Max(qp(i,j,k,QRHO), small_dens); 
                    qp(i,j,k,QPRES) += sourcp; 
                    qp(i,j,k,QREINT) += source; 
                }
//--------------- Upwind the passive variables --------------------------------
#pragma unroll
                for(int ipassive = 0; ipassive < npassive; ++ipassive){
                    int n = EOS::qpass_map(ipassive); 
                    //Right state 
                    u = q(i,j,k,QU); 
                    amrex::Real spzero; 
                    spzero = u > 0 ? -1.e0 : u*dtdx; 
                    amrex::Real acmprght = 0.5e0*(-1.e0 - spzero)*slope(i,j,k,n); 
                    qp(i,j,k,n) = q(i,j,k,n) + acmpright;

                    u = q(i-1,j,k,QU); 
                    spzero = u > 0 ? u*dtdx : -1.e0; 
                    amrex::Real acmpleft = 0.5e0*(1.e0 - spzero)*slope(i-1,j,k,n);  
                    qm(i+1, j, k, n) = q(i,j,k,n) + acmpleft; 
                }                            
            }
        }
    }
} 
AMREX_GPU_DEVICE
inline 
void 
PeleC_plm_y(amrex::Box const& bx, amrex::FArrayBox qmfab, amrex::FArrayBox qpfab,
            amrex::FArrayBox const& slopefab,  amrex::FArrayBox const& qfab,
            amrex::FArrayBox const& qauxfab,   amrex::FArrayBox const& srcQfab, 
            amrex::FArryaBox const& dlogafab,  amrex::Real const dy, amrex::Real const dt)
{
    const auto len    = amrex::length(bx); 
    const auto lo     = amrex::lbound(bx); 
    const auto qm     = qmfab.view(lo); 
    const auto qp     = qpfab.view(lo); 
    const auto slope  = slopefab.view(lo); 
    const auto q      = qfab.view(lo); 
    const auto srcQ   = srcQfab.view(lo); 
    const auto qa     = qauxfab.view(lo); 
    const auto dloga  = dlogafab.view(lo); 

    amrex::Real e[3]; 
    amrex::Real dtdy = dt/dy; 
    
    for             (int k = 0; k < len.z; ++k){
        for         (int j = 0; j < len.y; ++j){
            for     (int i = 0; i < len.x; ++i){
                amrex::Real cc    = qa(i,j,k,QC); 
                amrex::Real cs    = cc*cc; 
                amrex::Real rho   = q(i,j,k,QRHO); 
                amrex::Real u     = q(i,j,k,QU); 
                amrex::Real v     = q(i,j,k,QV); 
                amrex::Real p     = q(i,j,k,QPRES); 
                amrex::Real rhoe  = q(i,j,k,QREINT); 
                amrex::Real enth  = ((rhoe+p)/rho)/cs; 
                
                amrex::Real drho  = slope(i,j,k,QRHO); 
                amrex::Real du    = slope(i,j,k,QU); 
                amrex::Real dv    = slope(i,j,k,QV); 
                amrex::Real dp    = slope(i,j,k,QPRES); 
                amrex::Real drhoe = slope(i,j,k,QREINT); 
                
                amrex::Real alpham = 0.5e0*(dp/(rho*cc) - dv)*rho/cc; 
                amrex::Real alphap = 0.5e0*(dp/(rho*cc) + dv)*rho/cc; 
                amrex::Real alpha0r = drho - dp/cs;
                amrex::Real alpha0e = drhoe - dp*enth;
                amrex::Real alpha0u = du; 
                
                e[0] = v - cc; 
                e[1] = v; 
                e[2] = v + cc; 
// ------------ Construct the "right" state on the j-1/2 interface 
                amrex::Real rho_ref  = rho - 0.5e0*(1.e0 + dtdy*min(e[0],0.e0))*drho;                
                amrex::Real u_ref    = u - 0.5e0*(1.e0 + dtdy*min(e[0], 0.e0))*du; 
                amrex::Real v_ref    = v - 0.5e0*(1.e0 + dtdy*min(e[0], 0.e0))*dv; 
                amrex::Real p_ref    = p - 0.5e0*(1.e0 + dtdy*min(e[0], 0.e0))*dp; 
                amrex::Real rhoe_ref = rhoe - 0.5e0*(1.e0 + dtdy*min(e[0], 0.e0))*drhoe; 

                amrex::Real apright = 0.25e0*dtdy*(e[0] - e[2])*
                                      (1.e0 - std::copysign(1.0, e[2]))*alphap; 
                amrex::Real amright = 0.25e0*dtdy*(e[0] - e[2])*
                                      (1.e0 - std::copysign(1.0, e[0]))*alpham; 

                amrex::Real azrright = 0.25e0*dtdy*(e[0] - e[1])*
                                      (1.e0 - std::copysign(1.0, e[1]))*alpha0r; 
                amrex::Real azeright = 0.25e0*dtdy*(e[0] - e[1])*
                                      (1.e0 - std::copysign(1.0, e[1]))*alpha0e; 
                amrex::Real azu1rght = 0.25e0*dtdy*(e[0] - e[1])*
                                      (1.e0 - std::copysign(1.0, e[1]))*alpha0u; 

                qp(i,j,k,QRHO)   = rho_ref + apright + amright + azrright;
                qp(i,j,k,QRHO)   = amrex::max(small_dens,qp(i,j,k,QRHO));
                qp(i,j,k,QV)     = v_ref + (apright - amright)*cc/rho;
                qp(i,j,k,QU)     = u_ref + azu1rght;
                qp(i,j,k,QPRES)  = p_ref + (apright + amright)*csq;
                qp(i,j,k,QPRES)  = amrex::max(qp(i,j,k,QPRES),small_pres);
                qp(i,j,k,QREINT) = rhoe_ref + (apright + amright)*enth*csq + azeright;

// ------------ Construct the "left" state on the j+1/2 interface 
                
                rho_ref = rho + 0.5e0*(1.e0 - dtdy*max(e[2],0.e0))*drho;
                u_ref = u + 0.5e0*(1.e0 - dtdy*max(e[2],0.e0))*du;
                v_ref = v + 0.5e0*(1.e0 - dtdy*max(e[2],0.e0))*dv;
                p_ref = p + 0.5e0*(1.e0 - dtdy*max(e[2],0.e0))*dp;
                rhoe_ref = rhoe + 0.5e0*(1.e0 - dtdy*max(e[2],0.e0))*drhoe;

                apleft = 0.25e0*dtdy*(e[2] - e[2])*(1.e0 + std::copysign(1.e0,e[2]))*alphap;
                amleft = 0.25e0*dtdy*(e[2] - e[0])*(1.e0 + std::copysign(1.e0,e[0]))*alpham;

                azrleft = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0r;
                azeleft = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0e;
                azv1left = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0v;
  
                qm(i,j+1,k,QRHO) = rho_ref + apleft + amleft + azrleft;
                qm(i,j+1,k,QRHO) = max(qm(i,j+1,k,QRHO),small_dens);
                qm(i,j+1,k,QV) = v_ref + (apleft - amleft)*cc/rho;
                qm(i,j+1,k,QU) = u_ref + azu1left;
                qm(i,j+1,k,QPRES) = p_ref + (apleft + amleft)*csq;
                qm(i,j+1,k,QPRES) = max(qm(i,j+1,k,QPRES), small_pres);
                qm(i,j+1,k,QREINT) = rhoe_ref + (apleft + amleft)*enth*csq + azeleft;
                                
                if(dloga(i,j,k)!=0){
                    amrex::Real courn = dtdx*(cc+abs(u)); 
                    amrex::Real eta = (1.e0-courn)/(cc*dt*abs(dloga(i,j,k))); 
                    amrex::Real dlogatmp = amrex::min(eta, 1.e0)*dloga(i,j,k);
                    amrex::Real sourcr = -0.5e0*dt*rho*dlogatmp*u; 
                    amrex::Real sourcp = sourcr*cs; 
                    amrex::Real source = courcp*enth; 
                    
                    qm(i,j+1,k,QRHO)   += sourcr; 
                    qm(i,j+1,k,QRHO)    = amrex::max(qm(i+1,j,k,QRHO), small_dens); 
                    qm(i,j+1,k,QPRES)  += sourcp; 
                    qm(i,j+1,k,QREINT) += source; 
                    
                    qp(i,j,k,QRHO)   += sourcr; 
                    qp(i,j,k,QRHO)    = amrex::max(qp(i,j,k,QRHO), small_dens); 
                    qp(i,j,k,QPRES)  += sourcp; 
                    qp(i,j,k,QREINT) += source; 
                }
//--------------- Upwind the passive variables --------------------------------
#pragma unroll
                for(int ipassive = 0; ipassive < npassive; ++ipassive){
                    int n = EOS::qpass_map(ipassive); 
                    //Right state 
                    v = q(i,j,k,QV); 
                    amrex::Real spzero; 
                    spzero = v > 0 ? -1.e0 : v*dtdy; 
                    amrex::Real acmptop = 0.5e0*(-1.e0 - spzero)*slope(i,j,k,n); 
                    qp(i,j,k,n) = q(i,j,k,n) + acmptop;

                    v = q(i,j-1,k,QV); 
                    spzero = v > 0 ? v*dtdy : -1.e0; 
                    amrex::Real acmpbot = 0.5e0*(1.e0 - spzero)*slope(i-1,j,k,n);  
                    qm(i,j+1,k,n) = q(i,j,k,n) + acmpbot; 
                }                                           
            }
        }
    }
}

//================================ Corrects the left and right y states with x flux =====
AMREX_GPU_DEVICE 
inline 
void 
PeleC_transx(amrex::Box const& bx, amrex::FArrayBox &qmfab, amrex::FArrayBox &qpfab, 
             amrex::FArrayBox const& qymfab, amrex::FArrayBox const& qypfab, 
             amrex::FArrayBox const& flxxfab, amrex::FArrayBox const& srcQfab,
             amrex::FArrayBox const& qintfab, amrex::Real hdt, amrex::Real cdtdy) 
{
    const auto len    = amrex::length(bx); 
    const auto lo     = amrex::lbound(bx); 
    const auto qm     = qmfab.view(lo); 
    const auto qp     = qpfab.view(lo); 
    const auto qym    = qymfab.view(lo); 
    const auto qyp    = qypfab.view(lo); 
    const auto flxx   = flxxfab.view(lo); 
    const auto srcQ   = srcQfab.view(lo); 
    const auto qint   = qintfab.view(lo);
 
    EOS state; 
    amrex::Real compn, rr, rrnew, compo, rwl; 
    amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr; 
    amrex::Real runewr, rvnewr, renewr, rhoekinr ; 
    amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl, renewl; 
    amrex::Real rhoekinl, rwl, srcr, srce, srcpass;
    amrex::Real pggp, pggm, ugp, ugm, gegp, gegm;
    amrex::Real dAup, pav, uav, dAu, geav, dge; 
//  Local variables to avoid reading from global memory too much 
    amrex::Real flxrho, flxu, flxv, flxe; 

    for             (int k = 0; k < len.z; ++k){
        for         (int j = 0; j < len.y; ++j){
            for     (int i = 0; i < len.x; ++i){
                flxrho = cdtdy*(flxx(i+1,j,k,URHO)  - flxx(i,j,k,URHO));
                flxu   = cdtdy*(flxx(i+1,j,k,UMX)   - flxx(i,j,k,UMX)); 
                flxv   = cdtdy*(flxx(i+1,j,k,UMY)   - flxx(i,j,k,UMY)); 
                flxe   = cdtdy*(flxx(i+1,j,k,UEDEN) - flxx(i,j,k,UEDEN));  
                srcr   = srcQ(i,j,k,QRHO); 
                srce   = srcQ(i,j,k,QREINT); 
// ---------------------------  Update Passive Variables ------------------
                for (int ipass = 0; ipass < npassive; ++ipass){
                    int n   = EOS::upass_map(ipass); 
                    int nqp = EOS::qpass_map(ipass); 
                    srcpass = srcQ(i,j,k,nqp); 
                    compn = cdtdy*(flxx(i+1,j,k,n) - flxx(i,j,k,n)); 
//TODO bounds                     
                    rr = qyp(i,j,k,QRHO); 
                    rrnew = rr - flxrho; 
                    compo = rr*qyp(i,j,k,nqp) - compn; 
                    qp(i,j,k,nqp) = compo/rrnew + hdt*srcpass;  
            
                    rr = qm(i,j+1,k,QRHO); 
                    rrnew = rr - flxrho; 
                    compo = rr*qm(i,j+1,k,nqp) - compn; 
                    qm(i,j+1,k,nqp) = compo/rrnew + hdt*srcpass;                     
                }                

//------------------------------- Update hydro vars -----------------------
// TODO 
// Add area 

                pggp = qint(i+1,j,k,GDPRES); 
                pggm = qint(i,j,k,GDPRES); 
                ugp  = qint(i+1,j,k,GDU); 
                ugm  = qint(i,j,k,GDU); 
                gegp = qint(i+1,j,k,GDGAME); 
                gegm = qint(i,j,k,GDGAME); 
                dAup = pggp*ugp - pggm*ugm; //
                pav  = 0.5e0*(pggp + pggm); 
                uav  = 0.5e0*(ugp + ugm); 
                dAu  = ugp - ugm; //
                geav = 0.5e0*(gegp+gegm); 
                dge  = gegp-gegm; 
            
//========================= QP ===========================================

//Again bounds... 

//Convert to conservative 
                rrr = qyp(i,j,k,QRHO); 
                rur = qyp(i,j,k,QU);             
                rvr = qyp(i,j,k,QV); 
                rwr = qyp(i,j,k,QW); 
                ekinr = 0.5e0*rrr*(rur*rur + rvr*rvr + rwr*rwr);
                rur *= rrr; 
                rvr *= rrr; 

                rer = qyp(i,j,k,QREINT) + ekinr; 
// Add transverse predictor 
                rrnewr = rrr - flxrho; 
                runewr = rur - flxu; 
                rvnewr = rvr - flxv; 
                renewr = rer - flxe; 

// Convert back to primitive 
//                state.rho = rrnewr + hdt*srcr; 
                qp(i,j,k,QRHO) = rrnewr + hdt*srcr; //state.rho; 
                qp(i,j,k,QU)   = runewr/rrnewr; 
                qp(i,j,k,QV)   = rvnewr/rrnewr; 
                rwr           *= rrnewr; 
                rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwr*rwr )/rrnewr; 
                amrex::Real pnewr = qxp(i,j,k,QPRES) 
                                  - hdt*(dAup + pav*dAu*(qa(i,j,k,QGAMC) - 1.e0)); 
                qp(i,j,k,QPRES) = amrex::max(pnewr , small_pres); 
                qp(i,j,k,QREINT) = rernewr - rhokinr + hdt*srce; 
//======================= QM ================================================

//Conversion to Conservative 
                rrl = qym(i,j+1,k,QRHO); 
                rul = qym(i,j+1,k,QU); 
                rvl = qym(i,j+1,k,QV); 
                rwl = qym(i,j+1,k,QW);  
                ekinl = 0.5e0*rrl*(rul*rul + rvl*rvl + rwl*rwl); 
                rul *= rrl; 
                rvl *= rrl; 
                rel = qym(i,j+1,k,QREINT) + ekinl; 

// Transverse fluxes 
                rrnewl = rrl - fluxrho; 
                runewl = rul - fluxu; 
                rvnewl = rvl - fluxv; 
                renewl = rel - flxe; 
                rwl *= rrnewl; 
                
                state.rho = rrnewl + hdt*srcr; 
                qm(i,j+1,k,QRHO) = state.rho;
                qm(i,j+1,k,QU  ) = runewl/rrnewl; 
                qm(i,j+1,k,QV  ) = rvnewl/rrnewl; 
                rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwl*rwl)/rrnewl; 

                qm(i,j+1,k,QRIENT) = renewl - rhoekinl + hdt*srce;               
                amrex::Real pnewl = qym(i,j+1,k,QPRES) 
                                  - hdt*(dAup + pav*dAu*(qa(i,j,k,QGAMC) - 1.e0)); 
                qm(i,j+1,k,QPRES) = amrex::max(pnewr , small_pres); 
            }
        }
    } 
}



//================================ Corrects the left and right x states with y flux ====== 
AMREX_GPU_DEVICE 
inline 
void 
PeleC_transy(amrex::Box const& bx, amrex::FArrayBox &qmfab, amrex::FArrayBox &qpfab, 
             amrex::FArrayBox const& qxmfab, amrex::FArrayBox const& qxpfab, 
             amrex::FArrayBox const& flxyfab, amrex::FArrayBox const& srcQfab, 
             amrex::FArrayBox const& qintfab, amrex::Real hdt, amrex::Real cdtdy) 
{
    const auto len    = amrex::length(bx); 
    const auto lo     = amrex::lbound(bx); 
    const auto qm     = qmfab.view(lo); 
    const auto qp     = qpfab.view(lo); 
    const auto qxm    = qxmfab.view(lo); 
    const auto qxp    = qxpfab.view(lo); 
    const auto flxy   = flxyfab.view(lo); 
    const auto srcQ   = srcQfab.view(lo);
    const auto qint   = qintfab.view(lo); 

    EOS state; 
    amrex::Real compn, rr, rrnew, compo, rwl; 
    amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr; 
    amrex::Real runewr, rvnewr, renewr, rhoekinr; 
    amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl, renewl; 
    amrex::Real rhoekinl, rwl, srcr, srce, srcpass;
    amrex::Real pggp, pggm, ugp, ugm, gegp, gegm; 
//  Local variables to avoid reading from global memory too much 
    amrex::Real dup, pav, uav, du, pnewr, pnewl, geav, dge;  
    amrex::Real flxrho, flxu, flxv, flxe; 

    for             (int k = 0; k < len.z; ++k){
        for         (int j = 0; j < len.y; ++j){
            for     (int i = 0; i < len.x; ++i){
                flxrho = cdtdy*(flxy(i,j+1,k,URHO)  - flxy(i,j,k,URHO));
                flxu   = cdtdy*(flxy(i,j+1,k,UMX)   - flxy(i,j,k,UMX)); 
                flxv   = cdtdy*(flxy(i,j+1,k,UMY)   - flxy(i,j,k,UMY)); 
                flxe   = cdtdy*(flxy(i,j+1,k,UEDEN) - flxy(i,j,k,UEDEN));  
                srcr   = srcQ(i,j,k,QRHO); 
                srce   = srcQ(i,j,k,QREINT); 
// ---------------------------  Update Passive Variables ------------------
                for (int ipass = 0; ipass < npassive; ++ipass){
                    int n   = EOS::upass_map(ipass); 
                    int nqp = EOS::qpass_map(ipass); 
                    srcpass = srcQ(i,j,k,nqp); 
                    compn = cdtdy*(flxy(i,j+1,k,n) - flxy(i,j,k,n)); 
//TODO bounds                     
                    rr = qxp(i,j,k,QRHO); 
                    rrnew = rr - flxrho; 
                    compo = rr*qxp(i,j,k,nqp) - compn; 
                    qp(i,j,k,nqp) = compo/rrnew + hdt*srcpass;  
            
                    rr = qm(i+1,j,k,QRHO); 
                    rrnew = rr - flxrho; 
                    compo = rr*qm(i+1,j,k,nqp) - compn; 
                    qm(i+1,j,k,nqp) = compo/rrnew + hdt*srcpass;                     
                }                

//------------------------------- Update hydro vars -----------------------
// TODO 
                pggp = qint(i,j+1,k,GDPRES); 
                pggm = qint(i,j,k,GDPRES); 
                ugp  = qint(i,j+1,k,GDV); 
                ugm  = qint(i,j,k,GDV);
                gegp = qint(i,j+1,k,GDGAME); 
                gegm = qint(i,j,k,GDGAME); 
            
                dup  = pggp*ugp - pggm*ugm; 
                pav  = 0.5e0*(pggp+pggm); 
                uav  = 0.5e0*(ugp + ugm); 
                du   = ugp - ugm; 
                geav = 0.5e0*(gegp+gegm); 
                gde  = gegp - gegm; 

//========================= QP ===========================================

//Again bounds... 

//Convert to conservative 
                rrr = qxp(i,j,k,QRHO); 
                rur = qxp(i,j,k,QU);             
                rvr = qxp(i,j,k,QV); 
                rwr = qxp(i,j,k,QW); 
                ekinr = 0.5e0*rrr*(rur*rur + rvr*rvr + rwr*rwr);
                rur *= rrr; 
                rvr *= rrr; 

                rer = qxp(i,j,k,QREINT) + ekinr; 
// Add transverse predictor 
                rrnewr = rrr - flxrho; 
                runewr = rur - flxu; 
                rvnewr = rvr - flxv; 
                renewr = rer - flxe; 

// Convert back to primitive 
                state.rho = rrnewr + hdt*srcr; 
                qp(i,j,k,QRHO) = state.rho; 
                qp(i,j,k,QU)   = runewr/rrnewr; 
                qp(i,j,k,QV)   = rvnewr/rrnewr; 
                rwr           *= rrnewr; 
                rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwr*rwr )/rrnewr; 
                
                pnewr = qxp(i,j,k,QPRES) - cdtdy*(dup + pav*du*(qa(i,j,k,QGAMC) - 1.e0)); 
                qp(i,j,k,QPRES) = amrex::max(pnewr, small_pres); 
                qp(i,j,k,QREINT) = rernewr - rhoekinr + hdt*srce; 
 

//======================= QM ================================================

//Conversion to Conservative 
                rrl = qxm(i+1,j,k,QRHO); 
                rul = qxm(i+1,j,k,QU); 
                rvl = qxm(i+1,j,k,QV); 
                rwl = qxm(i+1,j,k,QW);  
                ekinl = 0.5e0*rrl*(rul*rul + rvl*rvl + rwl*rwl); 
                rul *= rrl; 
                rvl *= rrl; 
                rel = qxm(i+1,j,k,QREINT) + ekinl; 

// Transverse fluxes 
                rrnewl = rrl - fluxrho; 
                runewl = rul - fluxu; 
                rvnewl = rvl - fluxv; 
                renewl = rel - flxe; 
                rwl *= rrnewl; 
                
                
                state.rho = rrnewl + hdt*srcr; 
                qm(i+1,j,k,QRHO) = state.rho;  
                qm(i+1,j,k,QU  ) = runewl/rrnewl; 
                qm(i+1,j,k,QV  ) = rvnewl/rrnewl; 
                rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwl*rwl)/rrnewl; 

                pnewr = qxm(i+1,j,k,QPRES) - cdtdy*(dup + pav*du*(qa(i,j,k,QGAMC) - 1.e0)); 
                qm(i+1,j,k,QPRES) = amrex::max(pnewr , small_pres); 
                qm(i+1,j,k,QREINT) = rernewr - rhoekinr + hdt*srce; 
            }
        }
    } 
}


//Use interface states from Riemann solver for pdivu. 
AMREX_GPU_DEVICE 
inline
void 
PeleC_pdivu(amrex::Box const &bx, amrex::FArrayBox &pdivufab, amrex::FArrayBox const &q1fab, 
            amrex::FArrayBox const& q2fab, amrex::FArrayBox const& a1, 
            amrex::FArrayBox const& a2,  amrex::FArrayBox const& volfab)
{
    const auto len   = amrex::length(bx); 
    const auto lo    = amrex::lbound(bx); 
    const auto pdivu = pdivufab.view(lo); 
    const auto q1    = q1fab.view(lo); 
    const auto q2    = q2fab.view(lo); 
    const auto vol   = volfab.view(lo); 
    const auto area1 = a1.view(lo); 
    const auto area2 = a2.view(lo); 

    for             (int k = 0; k < len.z; ++k){
        for         (int j = 0; j < len.y; ++j){
            AMREX_PRAGMA_SIMD
            for     (int i = 0; i < len.x; ++i){
                pdivu(i,j,k) = 0.5e0*((q1(i+1,j,k,GDPRES) + q1(i,j,k,GDPRES))
                             * (q1(i+1,j,k,GDU)*area1(i+1,j,k) - q1(i,j,k,GDU)*area1(i,j,k))
                             + (q2(i,j+1,k,GDPRES) + q2(i,j,k,GDPRES))
                             * (q2(i,j+1,k,GDV)*area2(i,j+1,k) - q2(i,j,k,GDV)*area2(i,j,k)))
                             / vol(i,j,k); 
            }
        }
    }
}

#endif 
