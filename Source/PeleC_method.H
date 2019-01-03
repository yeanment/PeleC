#include "PeleC_K.H" 
 
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
        PeleC_shock(q, shk, dx, dy); 
    }
    else shk.setVal(0.0); 

    if(ppm_type == 0){
        PeleC_trace(q, qaux, flatn, qxm, qxp, qym, qyp, srcQ, dx, dy, dt); 
    }
    else{
        PeleC_ppm(q, qaux, flatn, qxm, qxp, qym, qyp, srcQ, dx, dy, dt); 
    }

    //===================== X initial fluxes ============================
    PeleC_cmpflx(qxm, qxp, fx, q1, qaux, bcMask, shk, 0); 
    //===================== Y initial fluxes ============================
    PeleC_cmpflx(qym, qyp, fy, q2, qaux, bcMask, shk, 1); 
    //===================== X interface corrections =====================
    PeleC_transy(qxm, qm, qxp, qp, fy, q2, quax, srcQ, hdt, hdtdy); 
    //===================== Final Riemann problem X =====================
    PeleC_cmpflx(qm, qp, flx1, q1, qaux, bcMask, shk, 0);
    //===================== Y interface corrections =====================
    PeleC_transx(qym, qm, qyp, qp, fx, q1, quax, srcQ, hdt, hdtdx, area1, vol); 
    //===================== Final Riemann problem Y =====================
    PeleC_cmpflx(qm, qp, flx2, q2, quax, bcMask, shk, 1); 
    //===================== Construct p div{U} ==========================
    PeleC_pdivu(pdivu, q1, q2, vol); 
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


//TODO Stolen from CNS need to transform to make PeleC 
AMREX_GPU_DEVICE
inline
void
riemann (const amrex::Real gamma, const amrex::Real smallp, const amrex::Real smallr,
         const amrex::Real rl, const amrex::Real ul, const amrex::Real pl,
         const amrex::Real ut1l, const amrex::Real ut2l,
         const amrex::Real rr, const amrex::Real ur, const amrex::Real pr,
         const amrex::Real ut1r, const amrex::Real ut2r,
         amrex::Real& flxrho, amrex::Real& flxu, amrex::Real& flxut, amrex::Real& flxutt, amrex::Real& flxe)
{
    constexpr amrex::Real weakwv = 1.e-3;
    constexpr amrex::Real small = 1.e-6;

    amrex::Real clsql = gamma*pl*rl;
    amrex::Real clsqr = gamma*pr*rr;
    amrex::Real wl = std::sqrt(clsql);
    amrex::Real wr = std::sqrt(clsqr);
    amrex::Real cleft = wl/rl;
    amrex::Real cright = wr/rr;
    amrex::Real ccsmall = small*(cleft+cright);

    amrex::Real pstar = (wl*pr + wr*pl - wr*wl*(ur-ul))/(wl+wr);
    pstar = amrex::max(pstar,smallp);
    amrex::Real pstnm1 = pstar;

    amrex::Real wlsq = (.5*(gamma-1.)*(pstar+pl)+pstar)*rl;
    amrex::Real wrsq = (.5*(gamma-1.)*(pstar+pr)+pstar)*rr;

    wl = std::sqrt(wlsq);
    wr = std::sqrt(wrsq);
    amrex::Real ustarp = ul - (pstar-pl)/wl;
    amrex::Real ustarm = ur + (pstar-pr)/wr;

    pstar = (wl*pr + wr*pl - wr*wl*(ur-ul))/(wl+wr);
    pstar = amrex::max(pstar,smallp);

    amrex::Real ustar;
    for (int iter = 0; iter < 3; ++iter)
    {
        wlsq = (.5*(gamma-1.)*(pstar+pl)+pstar)*rl;
        wrsq = (.5*(gamma-1.)*(pstar+pr)+pstar)*rr;

        wl = 1./sqrt(wlsq);
        wr = 1./sqrt(wrsq);

        amrex::Real ustnm1 = ustarm;
        amrex::Real ustnp1 = ustarp;

        ustarm = ur - (pr - pstar)*wr;
        ustarp = ul + (pl - pstar)*wl;

        amrex::Real dpditer = std::abs(pstnm1-pstar);
        amrex::Real zp = std::abs(ustarp-ustnp1);
        if (zp-weakwv*cleft < 0.0 ) {
            zp = dpditer*wl;
        }
        amrex::Real zm = std::abs(ustarm-ustnm1);
        if (zm-weakwv*cright < 0.0 ) {
            zm = dpditer*wr;
        }

        amrex::Real zz = zp+zm;
        amrex::Real denom = dpditer/ amrex::max(zz,ccsmall);
        pstnm1 = pstar;
        pstar = pstar - denom*(ustarm-ustarp);
        pstar = amrex::max(pstar,smallp);
        ustar = 0.5*(ustarm+ustarp);
    }

    amrex::Real ro, uo, po, sgnm, utrans1, utrans2;
    if (ustar > 0.) {
        ro = rl;
        uo = ul;
        po = pl;
        sgnm = 1.;
        utrans1 = ut1l;
        utrans2 = ut2l;
    } else if (ustar < 0.) {
        ro = rr;
        uo = ur;
        po = pr;
        sgnm = -1.;
        utrans1 = ut1r;
        utrans2 = ut2r;
    } else {
        uo = 0.5*(ur+ul);
        po = 0.5*(pr+pl);
        ro = 2.*(rl*rr)/(rl+rr);
        sgnm = 1.;
        utrans1 = 0.5*(ut1l+ut1r);
        utrans2 = 0.5*(ut2l+ut2r);
    }
    amrex::Real wosq = (.5*(gamma-1.)*(pstar+po)+pstar)*ro;
    amrex::Real co = std::sqrt(gamma * po / ro);
    amrex::Real wo = std::sqrt(wosq);
    amrex::Real dpjmp = pstar-po;
    amrex::Real rstar = ro/(1.-ro*dpjmp/wosq);
    amrex::Real cstar = sqrt(gamma * pstar / rstar);
    amrex::Real spout = co-sgnm*uo;
    amrex::Real spin = cstar - sgnm*uo;
    if(pstar >= po) {
        spin = wo/ro-sgnm*uo;
        spout = spin;
    }
    amrex::Real ss = amrex::max(spout-spin, spout+spin);
    amrex::Real frac = 0.5*(1.+(spin+spout)/amrex::max(ss,ccsmall));

    amrex::Real rgdnv, ugdnv, pgdnv;
    if (spout < 0.) {
        rgdnv = ro;
        ugdnv = uo;
        pgdnv = po;
    } else if(spin >= 0.) {
        rgdnv = rstar;
        ugdnv = ustar;
        pgdnv = pstar;
    } else {
        rgdnv = frac*rstar + (1. - frac)* ro;
        ugdnv = frac*ustar + (1. - frac)* uo;
        pgdnv = frac*pstar + (1. - frac)* po;
    }
    
    flxrho = rgdnv*ugdnv;
    flxu = rgdnv*ugdnv*ugdnv+pgdnv;
    flxut = rgdnv*ugdnv*utrans1;
    flxutt = rgdnv*ugdnv*utrans2;
    flxe = ugdnv*(0.5*rgdnv*(ugdnv*ugdnv+utrans1*utrans1+utrans2*utrans2) + pgdnv/(gamma -1.) + pgdnv);
}
}

AMREX_GPU_DEVICE
inline
void
PeleC_riemann(amrex::Box const& bx, amrex::FArrayBox& fluxfab, amrex::FArrayBox const& dqfab,
               amrex::FArrayBox const& qfab, int idir)
{
//TODO Stolen from CNS, need to transform to work with PeleC
    const auto len = amrex::length(bx);
    const auto lo  = amrex::lbound(bx);
    const auto dq = dqfab.view(lo);
    const auto  q =  qfab.view(lo);

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

