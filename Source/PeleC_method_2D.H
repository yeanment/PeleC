#ifndef PELEC_METHOD_2D_H_
#define PELEC_METHOD_2D_H_
#include <AMReX_FArrayBox.H>
#include "PeleC_Parameters.H"
#include "PeleC_index_macros.H"
#include "PeleC_EOS.H" 
 
using namespace amrex 
{

void PeleC_umeth2d(Box const& bx, FArrayBox const &q, 
           FArrayBox const& quax,
           FArrayBox const& srcQ, FArrayBox const& bcMask,
           FArrayBox &flx1, FArrayBox &flx2, 
           FArrayBox const& dloga, FArrayBox &q1,
           FArrayBox &q2, FArrayBox &a1, 
           FArrayBox &a2, FArrayBox &pdivu, 
           FArrayBox &vol, Real *dx, Real dt)
{
    Real const dtdx  = dt/dx[0]; 
    Real const hdtdx = 0.5*dtdx; 
    Real const hdtdy = 0.5*dt/dx[1]; 
    Real const hdt   = 0.5*dt; 

    const Box& bxg1 = grow(bx, 1); 
    const Box& bxg2 = grow(bx, 2); 
    Gpu::AsyncFab slope(bxg2, QVAR);
    Array4<Real> slfab = slope.array(), qfab  = q.array(); 
    Array4<Real> qauxfab = qaux.array(), srcQfab = srcQ.array(); 
    Array4<Real> dlogafab = dloga.array(), area1 = a1.array(), area2 = a2.array();  

//===================== X slopes ===================================
    int cdir = 0; 
    const Box& xslpbx = grow(bxg1, cdir, 1); 
    AMREX_FOR_3D (xslpbx,i,j,k, { 
        PeleC_slope_x(i,j,k, slfab, qfab);
    }); 

//==================== X interp ====================================
//Need to include dlogA.
    Gpu::AsyncFab qxm(xslpbx, QVAR); 
    Gpu::AsyncFab qxp(xslpbx, QVAR);
    Array4<Real> qxmfab = qxm.array(); 
    Array4<Real> qxpfab = qxp.array(); 

    AMREX_FOR_3D(xslpbx, i,j,k, {
        PeleC_plm_x(i, j, k, qxmfab, qxpfab, slfab, qfab, quaxfab, 
                srcQfab, dlogafab, dx, dt); 
    });

//===================== X initial fluxes ===========================
    const Box& xflxbx = surroundingNodes(bxg1,cdir); 
    Gpu::AsyncFab fx(xflxbx, NUM_VAR);
    Array4<Real> fxfab = fx.array(); 
    Array4<Real> q1fab = q1.array();  
    //May omit bcmask. 
    AMREX_FOR_3D (xflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qxmfab, qxpfab, fxfab, q1fab, qauxfab,
                 bcMaskfab, 0);
    });

//==================== Y slopes ====================================
    cdir = 1; 
    const Box& yslpbx = grow(bxg1, cdir, 1); 
    Gpu::AsyncFab qym(yslpbx, QVAR);
    Gpu::AsyncFab qyp(yslpbx, QVAR);
    Array4<Real> qymfab = qym.array(), qypfab = qyp.array();   
    AMREX_FOR_3D(yslpbx, i,j,k,{
        PeleC_slope_y(i,j,k, slfab, qfab); 
    });

//==================== Y interp ====================================
    AMREX_FOR_3D(yslpbx, i,j,k, {
        PeleC_plm_y(i,j,k, qymfab, qypfab, slfab, qfab, quaxfab, 
                srcQfab, dlogafab, dy, dt); 
    });

//===================== Y initial fluxes ===========================
    const Box& yflxbx = surroundingNodes(bxg1,cdir); 
    Gpu::AsyncFab fy(yflxbx, NUM_VAR); 
    Gpu::AsyncFab q2(yflxbx, QVAR); 
    Array4<Real> fyfab = fy.array(), q2fab = q2.array(); 
    AMREX_FOR_3D(yflxbx, i,j,k, {
        PeleC_cmpflx(i,j,k, qymfab, qypfab, fyfab, q2fab, qauxfab, bcMaskfab, 1); 
    }); 

//===================== X interface corrections ====================
    cdir = 0; 
    Gpu::AsyncFab qm(bgx1, QVAR); 
    Gpu::AsyncFab qp(bgx1, QVAR);
    Array4<Real> qmfab = qm.array(), qpfab = qp.array();  
    //TODO think about box size for these 
    const Box& xbx = grow(bx, cdir, 1);
    AMREX_FOR_3D(xbx, i,j,k, {
        PeleC_transy(i,j,k, qxmfab, qmfab, qxpfab, qpfab, fyfab,
                 q2fab, quaxfab, srcQfab, hdt, hdtdy); 
    }); 

//===================== Final Riemann problem X ====================
    const Box& xfxbx = surroundingNodes(bx, cdir); 
    Array4<Real> flx1fab = flx1.array(); 
    AMREX_FOR_3D(xfxbx, i,j,k, {
        PeleC_cmpflx(tbx, qmfab, qpfab, flx1fab, qauxfab, 0);
    }); 

//===================== Y interface corrections ====================
    cdir = 1; 
    const Box& ybx = grow(bx, cdir, 1);
    AMREX_FOR_3D(bx, tbx, {
    PeleC_transx(tbx, qymfab, qmfab, qypfab, qpfab, fxfab,
                 quaxfab, srcQfab, hdt, hdtdx, area1, vol); 
    });

//===================== Final Riemann problem Y ====================
    const Box& yfxbx = surroundingNodes(bx, cdir); 
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_cmpflx(tbx, qmfab, qpfab, flx2fab, q2fab, quaxfab 1); 
    });

//===================== Construct p div{U} =========================
    AMREX_LAUNCH_DEVICE_LAMBDA(bx, tbx, {
        PeleC_pdivu(tbx, pdivufab, q1fab, q2fab, vol); 
    }); 
}

AMREX_GPU_DEVICE
inline
void
PeleC_slope_x(const int i, const int j, const int k, 
      Array4<Real> &dq, Array4<Real> const& q)
{
        auto f = [] (Real dlft, Real drgt) -> Real {
            Real dcen = 0.5*(dlft+drgt);
            Real dsgn = std::copysign(1.0, dcen);
            Real slop = 2.0 * min(std::abs(dlft),std::abs(drgt));
            Real dlim = (dlft*drgt >= 0.0) ? slop : 0.0;
            Real d = dsgn * min(dlim,std::abs(dcen));
            return d;
        };

        Real dlft = 0.5*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/q(i,j,k,QCS) - 0.5*q(i,j,k,QRHO)*(q(i,j,k,QU) - q(i-1,j,k,QU));
        Real drgt = 0.5*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/q(i,j,k,QCS) - 0.5*q(i,j,k,QRHO)*(q(i+1,j,k,QU) - q(i,j,k,QU));
        Real d0 = f(dlft, drgt);

        Real cs2 = q(i,j,k,QCS)*q(i,j,k,QCS);
        dlft = (q(i,j,k,QRHO)-q(i-1,j,k,QRHO)) - (q(i,j,k,QPRES) - q(i-1,j,k,QPRES))/cs2;
        drgt = (q(i+1,j,k,QRHO)-q(i,j,k,QRHO)) - (q(i+1,j,k,QPRES) - q(i,j,k,QPRES))/cs2;
        Real d1 = f(dlft, drgt);

        dlft = 0.5*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/q(i,j,k,QCS) + 0.5*q(i,j,k,QRHO)*(q(i,j,k,QU) - q(i-1,j,k,QU));
        drgt = 0.5*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/q(i,j,k,QCS) + 0.5*q(i,j,k,QRHO)*(q(i+1,j,k,QU) - q(i,j,k,QU));
        Real d2 = f(dlft, drgt);

        dlft = q(i,j,k,QV) - q(i-1,j,k,QV);
        drgt = q(i+1,j,k,QV) - q(i,j,k,QV);
        Real d3 = f(dlft, drgt);

        dlft = q(i,j,k,QW) - q(i-1,j,k,QW);
        drgt = q(i+1,j,k,QW) - q(i,j,k,QW);
        Real d4 = f(dlft, drgt);

        dq(i,j,k,0) = d0;
        dq(i,j,k,1) = d1;
        dq(i,j,k,2) = d2;
        dq(i,j,k,3) = d3;
}

AMREX_GPU_DEVICE
inline
void
PeleC_slope_y (const int i, const int j, const int k,
               Array4<Real>& dq, Array4<Real> const& q)
{
        auto f = [] (Real dlft, Real drgt) -> Real {
            Real dcen = 0.5*(dlft+drgt);
            Real dsgn = std::copysign(1.0, dcen);
            Real slop = 2.0 * min(std::abs(dlft), std::abs(drgt));
            Real dlim = (dlft*drgt >= 0.0) ? slop : 0.0;
            Real d = dsgn * min(dlim, std::abs(dcen));
            return d;
        };

        Real dlft = 0.5*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/q(i,j,k,QCS) - 0.5*q(i,j,k,QRHO)*(q(i,j,k,QV) - q(i,j-1,k,QV));
        Real drgt = 0.5*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/q(i,j,k,QCS) - 0.5*q(i,j,k,QRHO)*(q(i,j+1,k,QV) - q(i,j,k,QV));
        Real d0 = f(dlft, drgt);

        Real cs2 = q(i,j,k,QCS)*q(i,j,k,QCS);
        dlft = (q(i,j,k,QRHO)-q(i,j-1,k,QRHO)) - (q(i,j,k,QPRES) - q(i,j-1,k,QPRES))/cs2;
        drgt = (q(i,j+1,k,QRHO)-q(i,j,k,QRHO)) - (q(i,j+1,k,QPRES) - q(i,j,k,QPRES))/cs2;
        Real d1 = f(dlft, drgt);

        dlft = 0.5*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/q(i,j,k,QCS) + 0.5*q(i,j,k,QRHO)*(q(i,j,k,QV) - q(i,j-1,k,QV));
        drgt = 0.5*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/q(i,j,k,QCS) + 0.5*q(i,j,k,QRHO)*(q(i,j+1,k,QV) - q(i,j,k,QV));
        Real d2 = f(dlft, drgt);

        dlft = q(i,j,k,QU) - q(i,j-1,k,QU);
        drgt = q(i,j+1,k,QU) - q(i,j,k,QU);
        Real d3 = f(dlft, drgt);

        dlft = q(i,j,k,QW) - q(i,j-1,k,QW);
        drgt = q(i,j+1,k,QW) - q(i,j,k,QW);
        Real d4 = f(dlft, drgt);

        dq(i,j,k,0) = d0;
        dq(i,j,k,1) = d1;
        dq(i,j,k,2) = d2;
        dq(i,j,k,3) = d3;
        dq(i,j,k,4) = d4;
}




AMREX_GPU_DEVICE
inline 
void 
PeleC_shock(const int i, const int j, const int k,
         Array4<Real> const &q, Array4<Real> &shk, 
         Real dx, Real dy)
{

        Real divU; 
        if(coord_type == 0) divU = 0.5*(q(i+1,j,k,QU) - q(i-1,j,k,QU))/dx + 
                            0.5*(q(i,j+1,k,QV) - q(i,j-1,k,QV))/dy; 
        else if(coord_type == 1){
            Real rc, rm, rp; 
            rc = double(i + 0.5e0)*dx;
            rm = double(i - 1.e0)*dx; 
            rp = double(i + 1.e0)*dx; 

            divU = 0.5e0*(rp*q(i+1,j,k,QU) - rm*q(i-1,j,k,QU))/(rc*dx) + 
                   0.5e0*(q(i,j+1,k,QV) - q(i,j-1,k,QV))/dy; 
        }
        Real px_pre, px_post, py_pre, py_post; 
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
        Real e_x, e_y, d, p_pre, p_post; 
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

namespace{
//Riemann_md 
AMREX_GPU_DEVICE
inline
void
riemann (const Real rl, const Real ul, const Real vl,
 const Real v2l, const Real pl, const Real rel, 
 const Real *spl, const Real rr, 
 const Real ur, const Real vr,
 const Real v2r, const Real pr, const Real rer, 
 const Real *spr, EOS &state, EOS &gdnv_state, 
 Real &uflx_rho, Real &uflx_u, 
 Real &uflx_v, Real &uflx_v, Real &uflx_w,
 Real &uflx_eden, Real &uflx_eint,
 Real &qint_iu, Real &qint_iv1, Real &qint_iv2, 
 Real &qint_gdpres, Real &qint_gdgame,
 Real &ustar, const int idir, const Real csmall,
 const Real cav)
//TODO consider making a uflx array instead of passing each in. 
{
        Real wsmall = small_dens*csmall; 
        state.rho = rl; 
        state.p   = pl; 
        #pragma unroll 
        for(int i = 0; i < NUM_SPECIES; ++i)  state.massfrac[i] = spl[i]; 
        state.eos_rp(); 
        Real csl = state.cs; 

        state.rho = rr; 
        state.p   = pr; 
        #pragma unroll 
        for(int i = 0; i < NUM_SPECIES; ++i) state.massfrac[i] = spr[i]; 
        state.eos_rp(); 
        Real csr = state.cs; 

        //    Real wl = max(wsmall, sqrt(abs(gamcl*pl*rl))); 
        //    Real wr = max(wsmall, sqrt(abs(gamcr*pr*rr))); 

        Real wl = max(wsmall, rl*csl); 
        Real wr = max(wsmall, rr*csr); 

        Real pstar = ((wr*pl + wl*pr) + wl*wr*(ul - ur))/(wl + wr); 
        ustar = ((wl*ul + wr*ur) + (pl - pr))/(wl + wr); 

        pstar = max(pstar, small_pres); 
        if (abs(ustar) < smallu*0.5e0*(abs(ul) + abs(ur))) ustar = 0.e0; 
        Real ro, uo, po; 
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
        Real reo = gdnv_state.rho*gdnv_state.e;
        Real co  = gdvn_state.cs; 

        Real drho  = (pstar - po)/(co*co); 
        Real rstar = ro + drho; 
        rstar = max(small_dens, rstar); 

        gdnv_state.rho = rstar; 
        gdnv_state.p   = pstar; 

        gdnv_state.eos_rp(); 
        Real cstar = gdnv_state.cs; 
        Real estar = gdnv_state.rho*gdnv_state.e; 

        Real sgnm = sign(ustar); 
        Real spoit = co - sgnm*uo; 
        real spin  = cstar - sgnm*ustar; 
        Real ushock = 0.5e0*(spin + spout); 

        if(pstar - po >= 0.e0){
                spin = ushock; 
                spout = ushock; 
        }
        if(spout-spin == 0.e0) scr = small*cav; 
        else scr = spout-spin; 

        Real frac = (1.e0 + (spout + spin)/scr)*0.5e0; 
        fram = max(0.e0, min(1.e0, frac)); 

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

        Real rgd = frac*rstar + (1.e0 - frac)*ro; 
        qint_iu = frac*ustar + (1.e0 - frac)*uo; 
        qint_iv1 = vgd; 
        qint_iv2 = wgd; 

        qint_gdpres = frac*pstar + (1.e0 - frac)*po; 
        gdvn_state.rho = rgd; 
        gdvn_state.p = qint_gdpres; 
        gdvn_state.eos_rp(); 
        Real regd = gdvn_state.rho*gdnv_state.e; 

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
        Real rhoetot = regd + 0.5e0*rgd*(qint_iu*qint_iu + 
                          qint_iv1*qint_iv1 + qint_iv2*qint_iv2); 
        uflx_eden = qint_iu*(rhoetot + qint_gdpres); 
        uflx_eint = qint_iu*regd;              
}


AMREX_GPU_DEVICE
inline
void
PeleC_cmpflx(const int i, const int j, const int k,
     Array4<Real> const& ql, 
     Array4<Real> const & qr, Array4<Real>& flux,
     Array4<Real> const& q, Array4<Real> const& qa, int idir)
{
    EOS state, gdnv_state; 
    Real cav;
    Real spl[NUM_SPECIES];
    Real spr[NUM_SPECIES]; 
//    Real ul, ur, vl, vr, v2l, v2r, rel, rer; 
    int idx, IU, IV, IV2; 
    if(idir == 0){
         IU = QU; 
         IV = QV; 
        IV2 = QW; 
    }
    else if( idir == 1){
        IU = QV; 
        IV = QU; 
        IV2 = QW; 
    }
    else{
        IU = QW; 
        IV = QU; 
        IV2 = QV; 
    }


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

        Real flxrho = flx(i,j,k,URHO); 
        for(int ipass = 0; ipass < npassive; ++ipass){
            int n = EOS::upass_map(ipass); 
            int nqp = EOS::qpass_map(ipass);                     
            if(ustar > 0) flx(i,j,k,n) = flxrho*ql(i,j,k,nqp); 
            else if(ustar < 0) flx(i,j,k,n) = flxrho*qr(i,j,k,nqp); 
            else flx(i,j,k,n) = flxrho*0.5e0*(ql(i,j,k,nqp) + qr(i,j,k,nqp)); 
        }
            
}



AMREX_GPU_DEVICE
inline 
void 
PeleC_plm_x(const int i, const int j, const int k,
    Array4<Real> qm, Array4<Real> qp,
    Array4<Real> const& slope,  Array4<Real> const& q,
    Array4<Real> const& qaux,   Array4<Real> const& srcQ, 
    Array4<Real> const& dlog,   Real const dx, Real const dt)
{

Real e[3]; 
Real dtdx = dt/dx; 

        Real cc    = qa(i,j,k,QC); 
        Real cs    = cc*cc; 
        Real rho   = q(i,j,k,QRHO); 
        Real u     = q(i,j,k,QU); 
        Real v     = q(i,j,k,QV); 
        Real p     = q(i,j,k,QPRES); 
        Real rhoe  = q(i,j,k,QREINT); 
        Real enth  = ((rhoe+p)/rho)/cs; 
        
        Real drho  = slope(i,j,k,QRHO); 
        Real du    = slope(i,j,k,QU); 
        Real dv    = slope(i,j,k,QV); 
        Real dp    = slope(i,j,k,QPRES); 
        Real drhoe = slope(i,j,k,QREINT); 
        
        Real alpham = 0.5e0*(dp/(rho*cc) - du)*rho/cc; 
        Real alphap = 0.5e0*(dp/(rho*cc) + du)*rho/cc; 
        Real alpha0r = drho - dp/cs;
        Real alpha0e = drhoe - dp*enth;
        Real alpha0v = dv; 
        
        e[0] = u - cc; 
        e[1] = u; 
        e[2] = u + cc; 
// ------------ Construct the right state on the i-1/2 interface 
        Real rho_ref  = rho - 0.5e0*(1.e0 + dtdx*min(e[0],0.e0))*drho;                
        Real u_ref    = u - 0.5e0*(1.e0 + dtdx*min(e[0], 0.e0))*du; 
        Real v_ref    = v - 0.5e0*(1.e0 + dtdx*min(e[0], 0.e0))*dv; 
        Real p_ref    = p - 0.5e0*(1.e0 + dtdx*min(e[0], 0.e0))*dp; 
        Real rhoe_ref = rhoe - 0.5e0*(1.e0 + dtdx*min(e[0], 0.e0))*drhoe; 

        Real apright = 0.25e0*dtdx*(e[0] - e[2])*
                              (1.e0 - std::copysign(1.0, e[2]))*alphap; 
        Real amright = 0.25e0*dtdx*(e[0] - e[2])*
                              (1.e0 - std::copysign(1.0, e[0]))*alpham; 

        Real azrright = 0.25e0*dtdx*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0r; 
        Real azeright = 0.25e0*dtdx*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0e; 
        Real azv1rght = 0.25e0*dtdx*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0v; 

        qp(i,j,k,QRHO)   = rho_ref + apright + amright + azrright;
        qp(i,j,k,QRHO)   = max(small_dens,qp(i,j,k,QRHO));
        qp(i,j,k,QU)     = u_ref + (apright - amright)*cc/rho;
        qp(i,j,k,QV)     = v_ref + azv1rght;
        qp(i,j,k,QPRES)  = p_ref + (apright + amright)*csq;
        qp(i,j,k,QPRES)  = max(qp(i,j,k,QPRES),small_pres);
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
            Real courn = dtdx*(cc+abs(u)); 
            Real eta = (1.e0-courn)/(cc*dt*abs(dloga(i,j,k))); 
            Real dlogatmp = min(eta, 1.e0)*dloga(i,j,k);
            Real sourcr = -0.5e0*dt*rho*dlogatmp*u; 
            Real sourcp = sourcr*cs; 
            Real source = courcp*enth; 
            
            qm(i+1,j,k,QRHO) += sourcr; 
            qm(i+1,j,k,QRHO) = max(qm(i+1,j,k,QRHO), small_dens); 
            qm(i+1,j,k,QPRES) += sourcp; 
            qm(i+1,j,k,QREINT) += source; 
            
            qp(i,j,k,QRHO) += sourcr; 
            qp(i,j,k,QRHO) = Max(qp(i,j,k,QRHO), small_dens); 
            qp(i,j,k,QPRES) += sourcp; 
            qp(i,j,k,QREINT) += source; 
        }
//--------------- Upwind the passive variables --------------------------------
#pragma unroll
        for(int ipassive = 0; ipassive < npassive; ++ipassive){
            int n = EOS::qpass_map(ipassive); 
            //Right state 
            u = q(i,j,k,QU); 
            Real spzero; 
            spzero = u > 0 ? -1.e0 : u*dtdx; 
            Real acmprght = 0.5e0*(-1.e0 - spzero)*slope(i,j,k,n); 
            qp(i,j,k,n) = q(i,j,k,n) + acmpright;

            u = q(i-1,j,k,QU); 
            spzero = u > 0 ? u*dtdx : -1.e0; 
            Real acmpleft = 0.5e0*(1.e0 - spzero)*slope(i-1,j,k,n);  
            qm(i+1, j, k, n) = q(i,j,k,n) + acmpleft; 
        }                            
} 

AMREX_GPU_DEVICE
inline 
void 
PeleC_plm_y(const int i, const int j, const int k, 
    Array4<Real> qmfab, Array4<Real> qpfab,
    Array4<Real> const& slope,  Array4<Real> const& q,
    Array4<Real> const& qaux,   Array4<Real> const& srcQ, 
    Array4<Real> const& dloga,  Real const dy, Real const dt)
{

Real e[3]; 
Real dtdy = dt/dy; 

        Real cc    = qa(i,j,k,QC); 
        Real cs    = cc*cc; 
        Real rho   = q(i,j,k,QRHO); 
        Real u     = q(i,j,k,QU); 
        Real v     = q(i,j,k,QV); 
        Real p     = q(i,j,k,QPRES); 
        Real rhoe  = q(i,j,k,QREINT); 
        Real enth  = ((rhoe+p)/rho)/cs; 
        
        Real drho  = slope(i,j,k,QRHO); 
        Real du    = slope(i,j,k,QU); 
        Real dv    = slope(i,j,k,QV); 
        Real dp    = slope(i,j,k,QPRES); 
        Real drhoe = slope(i,j,k,QREINT); 
        
        Real alpham = 0.5e0*(dp/(rho*cc) - dv)*rho/cc; 
        Real alphap = 0.5e0*(dp/(rho*cc) + dv)*rho/cc; 
        Real alpha0r = drho - dp/cs;
        Real alpha0e = drhoe - dp*enth;
        Real alpha0u = du; 
        
        e[0] = v - cc; 
        e[1] = v; 
        e[2] = v + cc; 
// ------------ Construct the "right" state on the j-1/2 interface 
        Real rho_ref  = rho - 0.5e0*(1.e0 + dtdy*min(e[0],0.e0))*drho;                
        Real u_ref    = u - 0.5e0*(1.e0 + dtdy*min(e[0], 0.e0))*du; 
        Real v_ref    = v - 0.5e0*(1.e0 + dtdy*min(e[0], 0.e0))*dv; 
        Real p_ref    = p - 0.5e0*(1.e0 + dtdy*min(e[0], 0.e0))*dp; 
        Real rhoe_ref = rhoe - 0.5e0*(1.e0 + dtdy*min(e[0], 0.e0))*drhoe; 

        Real apright = 0.25e0*dtdy*(e[0] - e[2])*
                              (1.e0 - std::copysign(1.0, e[2]))*alphap; 
        Real amright = 0.25e0*dtdy*(e[0] - e[2])*
                              (1.e0 - std::copysign(1.0, e[0]))*alpham; 

        Real azrright = 0.25e0*dtdy*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0r; 
        Real azeright = 0.25e0*dtdy*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0e; 
        Real azu1rght = 0.25e0*dtdy*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0u; 

        qp(i,j,k,QRHO)   = rho_ref + apright + amright + azrright;
        qp(i,j,k,QRHO)   = max(small_dens,qp(i,j,k,QRHO));
        qp(i,j,k,QV)     = v_ref + (apright - amright)*cc/rho;
        qp(i,j,k,QU)     = u_ref + azu1rght;
        qp(i,j,k,QPRES)  = p_ref + (apright + amright)*csq;
        qp(i,j,k,QPRES)  = max(qp(i,j,k,QPRES),small_pres);
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
            Real courn = dtdx*(cc+abs(u)); 
            Real eta = (1.e0-courn)/(cc*dt*abs(dloga(i,j,k))); 
            Real dlogatmp = min(eta, 1.e0)*dloga(i,j,k);
            Real sourcr = -0.5e0*dt*rho*dlogatmp*u; 
            Real sourcp = sourcr*cs; 
            Real source = courcp*enth; 
            
            qm(i,j+1,k,QRHO)   += sourcr; 
            qm(i,j+1,k,QRHO)    = max(qm(i+1,j,k,QRHO), small_dens); 
            qm(i,j+1,k,QPRES)  += sourcp; 
            qm(i,j+1,k,QREINT) += source; 
            
            qp(i,j,k,QRHO)   += sourcr; 
            qp(i,j,k,QRHO)    = max(qp(i,j,k,QRHO), small_dens); 
            qp(i,j,k,QPRES)  += sourcp; 
            qp(i,j,k,QREINT) += source; 
        }
//--------------- Upwind the passive variables --------------------------------
#pragma unroll
        for(int ipassive = 0; ipassive < npassive; ++ipassive){
            int n = EOS::qpass_map(ipassive); 
            //Right state 
            v = q(i,j,k,QV); 
            Real spzero; 
            spzero = v > 0 ? -1.e0 : v*dtdy; 
            Real acmptop = 0.5e0*(-1.e0 - spzero)*slope(i,j,k,n); 
            qp(i,j,k,n) = q(i,j,k,n) + acmptop;

            v = q(i,j-1,k,QV); 
            spzero = v > 0 ? v*dtdy : -1.e0; 
            Real acmpbot = 0.5e0*(1.e0 - spzero)*slope(i-1,j,k,n);  
            qm(i,j+1,k,n) = q(i,j,k,n) + acmpbot; 
        }                                           
}

//================================ Corrects the left and right y states with x flux =====
AMREX_GPU_DEVICE 
inline 
void 
PeleC_transx(const i, const j, const k,
     Array4<Real> &qm, Array4<Real> &qp, 
     Array4<Real> const& qym, Array4<Real> const& qyp, 
     Array4<Real> const& flxx, Array4<Real> const& srcQ,
     Array4<Real> const& qint, Real hdt, Real cdtdy) 
{

EOS state; 
Real compn, rr, rrnew, compo, rwl; 
Real rrr, rur, rvr, ekinr, rer, rrnewr; 
Real runewr, rvnewr, renewr, rhoekinr ; 
Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl, renewl; 
Real rhoekinl, rwl, srcr, srce, srcpass;
Real pggp, pggm, ugp, ugm, gegp, gegm;
Real dAup, pav, uav, dAu, geav, dge; 
//  Local variables to avoid reading from global memory too much 
Real flxrho, flxu, flxv, flxe; 

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
        Real pnewr = qxp(i,j,k,QPRES) 
                          - hdt*(dAup + pav*dAu*(qa(i,j,k,QGAMC) - 1.e0)); 
        qp(i,j,k,QPRES) = max(pnewr , small_pres); 
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
        Real pnewl = qym(i,j+1,k,QPRES) 
                          - hdt*(dAup + pav*dAu*(qa(i,j,k,QGAMC) - 1.e0)); 
        qm(i,j+1,k,QPRES) = max(pnewr , small_pres); 
}



//================================ Corrects the left and right x states with y flux ====== 
AMREX_GPU_DEVICE 
inline 
void 
PeleC_transy(const int i, const int j, const int k,
     Array4<Real> &qm, Array4<Real> &qp, 
     Array4<Real> const& qxm, Array4<Real> const& qxp, 
     Array4<Real> const& flxy, Array4<Real> const& srcQ, 
     Array4<Real> const& qint, Real hdt, Real cdtdy) 
{

EOS state; 
Real compn, rr, rrnew, compo, rwl; 
Real rrr, rur, rvr, ekinr, rer, rrnewr; 
Real runewr, rvnewr, renewr, rhoekinr; 
Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl, renewl; 
Real rhoekinl, rwl, srcr, srce, srcpass;
Real pggp, pggm, ugp, ugm, gegp, gegm; 
//  Local variables to avoid reading from global memory too much 
Real dup, pav, uav, du, pnewr, pnewl, geav, dge;  
Real flxrho, flxu, flxv, flxe; 

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
        qp(i,j,k,QPRES) = max(pnewr, small_pres); 
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
        qm(i+1,j,k,QPRES) = max(pnewr , small_pres); 
        qm(i+1,j,k,QREINT) = rernewr - rhoekinr + hdt*srce; 
}


//Use interface states from Riemann solver for pdivu. 
AMREX_GPU_DEVICE 
inline
void 
PeleC_pdivu(const int i, const int j, const int k, 
    Array4<Real> &pdivu, Array4<Real> const &q1, 
    Array4<Real> const& q2, Array4<Real> const& a1, 
    Array4<Real> const& a2,  Array4<Real> const& vol)
{
        pdivu(i,j,k) = 0.5e0*((q1(i+1,j,k,GDPRES) + q1(i,j,k,GDPRES))
                     * (q1(i+1,j,k,GDU)*a1(i+1,j,k) - q1(i,j,k,GDU)*a1(i,j,k))
                     + (q2(i,j+1,k,GDPRES) + q2(i,j,k,GDPRES))
                     * (q2(i,j+1,k,GDV)*a2(i,j+1,k) - q2(i,j,k,GDV)*a2(i,j,k)))
                     / vol(i,j,k); 
}


}// end namespace 
#endif 
