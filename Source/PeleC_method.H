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
    Gpu::AsyncFab slope(bxg1, QVAR); 
    //===================== X slopes ===================================
    int cdir = 0; 
    const amrex::Box& xslpbx = amrex::grow(bx, cdir, 1); 
    AMREX_LAUNCH_DEVICE_LAMBDA(xslpbx, tbx, { 
        PeleC_slope_x(tbx, slope.fab(), q.fab());
    }); 
    //==================== X interp ====================================
    AMREX_LAUNCH_DEVICE_LAMBDA(xslpbx, tbx, {
        PeleC_plm_x(tbx, qxm.fab(), qxp.fab(), slope.fab(), q.fab(), quax.fab(), 
                    srcQ.fab(), dloga.fab(), dx, dt); 
    });
    const amrex::Box& xflxbx = amrex::surroundingNodes(bx,cdir); 
    //===================== X initial fluxes ===========================
    AMREX_LAUNCH_DEVICE_LAMBDA(xflxbx, tbx, {
        PeleC_cmpflx(tbx, qxm, qxp, fx, q1, qaux, bcMask, shk, 0);
    }); 
    //==================== Y slopes ====================================
    cdir = 1; 
    const amrex::Box& yslpbx = amrex::grow(bx, cdir, 1); 
    AMREX_LAUNCH_DEVICE_LAMBDA(yslpbx, tbx,{
        PeleC_slope_x(tbx, slope.fab(), q.fab()); 
    });
    //==================== Y interp ====================================
    AMREX_LAUNCH_DEVICE_LAMBDA(yslpbx, tbx, {
        PeleC_plm_y(tbx, qym.fab(), qyp.fab(), slope.fab(), q.fab(), dy, dt); 
    });
    const amrex::Box& yflxbx = amrex::surroundingNodes(bx,cdir); 
    //===================== Y initial fluxes ===========================
    AMREX_LAUNCH_DEVICE_LAMBDA(yflxbx, tbx, {
        PeleC_cmpflx(tbx, qym.fab(), qyp.fab(), fy.fab(), q2.fab(), qaux.fab(), bcMask,fab(), 1); 
    }); 
    //TODO think about box size for these 
    //===================== X interface corrections ====================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_transy(tbx, qxm.fab(), qm.fab(), qxp.fab(), qp.fab(), fy.fab(),
                     q2.fab(), quax.fab(), srcQ.fab(), hdt, hdtdy); 
    }); 
    //===================== Final Riemann problem X ====================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_cmpflx(tbx, qm.fab(), qp.fab(), flx1.fab(), q1.fab(), qaux.fab(), bcMask.fab(), 0);
    }); 
    //===================== Y interface corrections ====================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_transx(tbx, qym.fab(), qm.fab(), qyp.fab(), qp.fab(), fx.fab(),
                     q1.fab(), quax.fab(), srcQ.fab(), hdt, hdtdx, area1, vol); 
    });
    //===================== Final Riemann problem Y ====================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_cmpflx(qm, qp, flx2, q2, quax, bcMask, shk, 1); 
    }); 
    //===================== Construct p div{U} =========================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_pdivu(pdivu, q1, q2, vol); 
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
         amrex::Real &uflx_eden, amrex::Real &uflx_eint, const int idir, const int coord_type, 
         const amrex::Real csmall, const amrex::Real cav)
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
    amrex::Real ustar = ((wl*ul + wr*ur) + (pl - pr))/(wl + wr); 
    
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
    amrex::Real qint_iu = frac*ustar + (1.e0 - frac)*uo; 
    amrex::Real qint_iv1 = vgd; 
    amrex::Real qint_iv2 = wgd; 
    
    amrex::Real qint_gdpres = frac*pstar + (1.e0 - frac)*po; 
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
PeleC_cmpflx(amrex::Box const& bx, amrex::FArrayBox& fluxfab, amrex::FArrayBox const& qlfab, 
             amrex::FArrayBox const & qrfab, amrex::FArrayBox const& qfab,
             amrex::FArrayBox const& qafab, int idir)
{
    const auto  len = amrex::length(bx);
    const auto   lo = amrex::lbound(bx);
    const auto    q =  qfab.view(lo);
    const auto   qa = qafab.view(lo); 
    const auto   ql = qlfab.view(lo); 
    const auto   qr = qrfab.view(lo); 
    const auto  flx = fluxfab.view(lo);
    EOS state, gdnv_state; 
    amrex::Real cav, ul, vl, v2l, rel, ur, vr, v2r, rer; 


//TODO coord type? 
    for         (int k = 0; k < len.z; ++k) {
        for     (int j = 0; j < len.y; ++j) {
            for (int i = 0; i < len.x; ++i) {
                 if(idir == 0) cav = 0.5e0*(qa(i,j,k,QC) + qa(i-1,j,k,QC)); 
                 else cav = 0.5e0*(qa(i,j,k,QC) + qa(i,j-1,k,QC));
                 riemann(ql(i,j,k,QRHO), ql(i,j,k,QU), ql(i,j,k,QV), ql(i,j,k,QW), 
                         ql(i,j,k,QPRES), ql(i,j,k,QRIENT), spl,   
                         qr(i,j,k,QRHO), qr(i,j,k,QU), qr(i,j,k,QV), qr(i,j,k,QW), 
                         qr(i,j,k,QPRES), qr(i,j,k,QRIENT), spr,
                         state, gdnv_state, flx(i,j,k,URHO), flx(i,j,k,UMX), flx(i,j,k,UMY), 
                         flx(i,j,k,UMZ), flx(i,j,k,UEDEN), flx(i,j,k,UEINT), idir, csmall, cav); 
                    
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
