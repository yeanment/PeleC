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
    amrex::Real const hdtdx = 0.5*dtdx[0]; 
    amrex::Real const hdtdy = 0.5*dt/dx[1]; 
    amrex::Real const hdt   = 0.5*dt; 

/* TODO make necessary temporary FABs. */ 

    if(hybrid_riemann == 1)
    {
        AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
            PeleC_shock(q, shk, dx, dy); 
        });
    }
    else shk.setVal(0.0); 

    if(ppm_type == 0){
        AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
            PeleC_trace(q, qaux, flatn, qxm, qxp, qym, qyp, srcQ, dx, dy, dt); 
        }); 
    }
    else{
        AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
            PeleC_ppm(q, qaux, flatn, qxm, qxp, qym, qyp, srcQ, dx, dy, dt);
        }); 
    }
//TODO consider placing all functions into one Launch Lambda
    //===================== X initial fluxes ============================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_cmpflx(qxm, qxp, fx, q1, qaux, bcMask, shk, 0);
    }); 
    //===================== Y initial fluxes ============================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_cmpflx(qym, qyp, fy, q2, qaux, bcMask, shk, 1); 
    }); 
    //===================== X interface corrections =====================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_transy(qxm, qm, qxp, qp, fy, q2, quax, srcQ, hdt, hdtdy); 
    }); 
    //===================== Final Riemann problem X =====================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_cmpflx(qm, qp, flx1, q1, qaux, bcMask, shk, 0);
    }); 
    //===================== Y interface corrections =====================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_transx(qym, qm, qyp, qp, fx, q1, quax, srcQ, hdt, hdtdx, area1, vol); 
    });
    //===================== Final Riemann problem Y =====================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_cmpflx(qm, qp, flx2, q2, quax, bcMask, shk, 1); 
    }); 
    //===================== Construct p div{U} ==========================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_pdivu(pdivu, q1, q2, vol); 
    }); 
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
         const amrex::Real *spl, const amrex::Real gamcl, 
         const amrex::Real rr, const amrex::Real ur, const amrex::Real vr,
         const amrex::Real v2r, const amrex::Real pr, const amrex::Real rer, 
         const amrex::Real *spr, const amrex::Real gamcr, 
         EOS &state, EOS &gdnv_state, amrex::Real &uflx_rho, amrex::Real &uflx_u, 
         amrex::Real &uflx_v, amrex::Real &uflx_v, amrex::Real &uflx_w,
         amrex::Real &uflx_eden, amrex::Real &uflx_eint, const int idir, const int coord_type, 
         const amrex::Real bc_test_vall, const amrex::Real csmall, const amrex::Real cav)
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

    amrex::Real wl = amrex::max(wsmall, sqrt(abs(gamcl*pl*rl))); 
    amrex::Real wr = amrex::max(wsmall, sqrt(abs(gamcr*pr*rr))); 

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
    qint_iu     = bc_test_val * qint_iu; 

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
PeleC_riemann(amrex::Box const& bx, amrex::FArrayBox& fluxfab, amrex::FArrayBox const& dqfab,
               amrex::FArrayBox const& qfab, int idir)
{
//TODO Stolen from CNS, need to transform to work with PeleC
//CNS uses Parm object, maybe insert an eos object to get gamma at the left and right states? 
    const auto  len = amrex::length(bx);
    const auto   lo = amrex::lbound(bx);
    const auto   dq = dqfab.view(lo);
    const auto    q =  qfab.view(lo);
    EOS state; 

    if(idir == 0) {
        const auto fx = fluxfab.view(lo);
        for         (int k = 0; k < len.z; ++k) {
            for     (int j = 0; j < len.y; ++j) {
                for (int i = 0; i < len.x; ++i) {
                    amrex::Real cspeed = q(i-1,j,k,QCS);
                    amrex::Real rl = q(i-1,j,k,QRHO) + 
                    0.5 * ( (dq(i-1,j,k,0)+dq(i-1,j,k,2))/cspeed + dq(i-1,j,k,1));
                    rl = amrex::max(rl, Parm::smallr);
                    amrex::Real ul = q(i-1,j,k,QU) 
                    + 0.5 * ( (dq(i-1,j,k,2)-dq(i-1,j,k,0))/q(i-1,j,k,QRHO));
                    amrex::Real pl = q(i-1,j,k,QPRES) + 0.5 *  (dq(i-1,j,k,0)+dq(i-1,j,k,2))*cspeed;
                    pl = amrex::max(pl, Parm::smallp);
                    amrex::Real ut1l = q(i-1,j,k,QV) + 0.5 * dq(i-1,j,k,3);
                    amrex::Real ut2l = q(i-1,j,k,QW) + 0.5 * dq(i-1,j,k,4);

                    cspeed = q(i,j,k,QCS);
                    amrex::Real rr = q(i,j,k,QRHO) 
                    - 0.5 * ( (dq(i,j,k,0)+dq(i,j,k,2))/cspeed + dq(i,j,k,1));
                    rr = amrex::max(rr, Parm::smallr);
                    amrex::Real ur = q(i,j,k,QU) - 0.5 * ( (dq(i,j,k,2)-dq(i,j,k,0))/q(i,j,k,QRHO));
                    amrex::Real pr = q(i,j,k,QPRES) - 0.5 * (dq(i,j,k,0)+dq(i,j,k,2))*cspeed;
                    pr = amrex::max(pr, Parm::smallp);
                    amrex::Real ut1r = q(i,j,k,QV) - 0.5 * dq(i,j,k,3);
                    amrex::Real ut2r = q(i,j,k,QW) - 0.5 * dq(i,j,k,4);
    
                    riemann(Parm::eos_gamma, Parm::smallp, Parm::smallr,
                            rl, ul, pl, ut1l, ut2l, rr, ur, pr, ut1r, ut2r,
                            fx(i,j,k,URHO), fx(i,j,k,UMX), fx(i,j,k,UMY), fx(i,j,k,UMZ)
                            , fx(i,j,k,UEDEN));
                }
            }
        }
    }
    else if(idir == 1){
        const auto fy = fluxfab.view(lo);
        for         (int k = 0; k < len.z; ++k) {
            for     (int j = 0; j < len.y; ++j) {
                for (int i = 0; i < len.x; ++i) {
                    amrex::Real cspeed = q(i,j-1,k,QCS);
                    amrex::Real rl = q(i,j-1,k,QRHO) 
                    + 0.5 * ( (dq(i,j-1,k,0)+dq(i,j-1,k,2))/cspeed + dq(i,j-1,k,1));
                    rl = amrex::max(rl, Parm::smallr);
                    amrex::Real ul = q(i,j-1,k,QV) 
                    + 0.5 * ( (dq(i,j-1,k,2)-dq(i,j-1,k,0))/q(i,j-1,k,QRHO));
                    amrex::Real pl = q(i,j-1,k,QPRES) + 0.5 *  (dq(i,j-1,k,0)+dq(i,j-1,k,2))*cspeed;
                    pl = amrex::max(pl, Parm::smallp);
                    amrex::Real ut1l = q(i,j-1,k,QU) + 0.5 * dq(i,j-1,k,3);
                    amrex::Real ut2l = q(i,j-1,k,QW) + 0.5 * dq(i,j-1,k,4);

                    cspeed = q(i,j,k,QCS);
                    amrex::Real rr = q(i,j,k,QRHO) 
                    - 0.5 * ( (dq(i,j,k,0)+dq(i,j,k,2))/cspeed + dq(i,j,k,1));
                    rr = amrex::max(rr, Parm::smallr);
                    amrex::Real ur = q(i,j,k,QV) - 0.5 * ( (dq(i,j,k,2)-dq(i,j,k,0))/q(i,j,k,QRHO));
                    amrex::Real pr = q(i,j,k,QPRES) - 0.5 * (dq(i,j,k,0)+dq(i,j,k,2))*cspeed;
                    pr = amrex::max(pr, Parm::smallp);
                    amrex::Real ut1r = q(i,j,k,QU) - 0.5 * dq(i,j,k,3);
                    amrex::Real ut2r = q(i,j,k,QW) - 0.5 * dq(i,j,k,4);

                    riemann(Parm::eos_gamma, Parm::smallp, Parm::smallr,
                            rl, ul, pl, ut1l, ut2l, rr, ur, pr, ut1r, ut2r,
                            fy(i,j,k,URHO), fy(i,j,k,UMY), fy(i,j,k,UMX), fy(i,j,k,UMZ),
                            fy(i,j,k,UEDEN));
                }
            }
        }
    }
}

