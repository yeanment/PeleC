#ifndef PELEC_METHOD_3D_H_
#define PELEC_METHOD_3D_H_ 
#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
#include "PeleC_Parameters.H" 
#include "PeleC_index_macros.H"
#include "PeleC_EOS.H" 
#include <cmath> 


AMREX_GPU_DEVICE
inline
void
PeleC_slope_x(const int i, const int j, const int k, const int n,  
      amrex::Array4< amrex::Real> const& dq, amrex::Array4<const amrex::Real> const& q)
{
    amrex::Real dlft, drgt, dcen, dfm, dfp, dlim, dsgn, dtemp;
    amrex::Real qm, qp, qi;
    qi = q(i,j,k,n); 
    qm = q(i-1,j,k,n); 
    qp = q(i+1,j,k,n); 

    dlft = qm - q(i-2,j,k,n); 
    drgt = qi - qm; 
    dcen = 0.5*(dlft+drgt);        
    dsgn = std::copysign(1.e0, dcen);  
    dlim = (dlft*drgt >= 0.0) ? 2.0*amrex::min(std::abs(dlft), std::abs(drgt)) : 0.0; 
    dfm = dsgn*amrex::min(dlim, std::abs(dcen));

    dlft = qp - qi; 
    drgt = q(i+2,j,k,n) - qp; 
    dcen = 0.5*(dlft+drgt);         
    dsgn = std::copysign(1.e0, dcen);  
    dlim = (dlft*drgt >= 0.0) ? 2.0*amrex::min(std::abs(dlft), std::abs(drgt)) : 0.0; 
    dfp = dsgn*amrex::min(dlim, std::abs(dcen)); 

    dlft = qi - qm; 
    drgt = qp - qi; 
    dcen = 0.5*(dlft+drgt);         
    dsgn = std::copysign(1.e0, dcen);  
    dlim = (dlft*drgt >= 0.0) ? 2.0*amrex::min(std::abs(dlft), std::abs(drgt)) : 0.0; 

    dtemp  = 4.0/3.0*dcen - 1.0/6.0*(dfp + dfm); 

//flattening can go here. 
    dq(i,j,k,n) = dsgn*amrex::min(dlim, std::abs(dtemp));
}

AMREX_GPU_DEVICE
inline
void
PeleC_slope_y (const int i, const int j, const int k, const int n, 
               amrex::Array4<amrex::Real> const& dq, amrex::Array4<const amrex::Real> const& q)
{
    amrex::Real dlft, drgt, dcen, dfm, dfp, dlim, dsgn, dtemp;
    amrex::Real qm, qp, qj;
    qj = q(i,j,k,n); 
    qm = q(i,j-1,k,n); 
    qp = q(i,j+1,k,n); 

    dlft = qm - q(i,j-2,k,n); 
    drgt = qj - qm; 
    dcen = 0.5*(dlft+drgt);         
    dsgn = std::copysign(1.e0, dcen);  
    dlim = (dlft*drgt >= 0.0) ? 2.0*amrex::min(std::abs(dlft), std::abs(drgt)) : 0.0; 
    dfm = dsgn*amrex::min(dlim, std::abs(dcen));

    dlft = qp - qj; 
    drgt = q(i,j+2,k,n) - qp; 
    dcen = 0.5*(dlft+drgt);         
    dsgn = std::copysign(1.e0, dcen); 
    dlim = (dlft*drgt >= 0.0) ? 2.0*amrex::min(std::abs(dlft), std::abs(drgt)) : 0.0; 
    dfp = dsgn*amrex::min(dlim, std::abs(dcen)); 

    dlft = qj - qm; 
    drgt = qp - qj; 
    dcen = 0.5*(dlft+drgt);         
    dsgn = std::copysign(1.e0, dcen);  
    dlim = (dlft*drgt >= 0.0) ? 2.0*amrex::min(std::abs(dlft), std::abs(drgt)) : 0.0; 

    dtemp  = 4.0/3.0*dcen - 1.0/6.0*(dfp + dfm); 
//flattening can go here. 
    dq(i,j,k,n) = dsgn*amrex::min(dlim, std::abs(dtemp)); 
}

AMREX_GPU_DEVICE
inline
void
PeleC_slope_z (const int i, const int j, const int k, const int n, 
               amrex::Array4<amrex::Real> const& dq, amrex::Array4<const amrex::Real> const& q)
{
    amrex::Real dlft, drgt, dcen, dfm, dfp, dlim, dsgn, dtemp;
    amrex::Real qm, qp, qk;
    qk = q(i,j,k,n); 
    qm = q(i,j,k-1,n); 
    qp = q(i,j,k+1,n); 

    dlft = qm - q(i,j,k-2,n); 
    drgt = qk - qm; 
    dcen = 0.5*(dlft+drgt);         
    dsgn = std::copysign(1.e0, dcen);  
    dlim = (dlft*drgt >= 0.0) ? 2.0*amrex::min(std::abs(dlft), std::abs(drgt)) : 0.0; 
    dfm = dsgn*amrex::min(dlim, std::abs(dcen));

    dlft = qp - qk; 
    drgt = q(i,j,k+2,n) - qp; 
    dcen = 0.5*(dlft+drgt);         
    dsgn = std::copysign(1.e0, dcen); 
    dlim = (dlft*drgt >= 0.0) ? 2.0*amrex::min(std::abs(dlft), std::abs(drgt)) : 0.0; 
    dfp = dsgn*amrex::min(dlim, std::abs(dcen)); 

    dlft = qk - qm; 
    drgt = qp - qk; 
    dcen = 0.5*(dlft+drgt);         
    dsgn = std::copysign(1.e0, dcen);  
    dlim = (dlft*drgt >= 0.0) ? 2.0*amrex::min(std::abs(dlft), std::abs(drgt)) : 0.0; 

    dtemp  = 4.0/3.0*dcen - 1.0/6.0*(dfp + dfm); 
//flattening can go here. 
    dq(i,j,k,n) = dsgn*amrex::min(dlim, std::abs(dtemp)); 
}

//Riemann_md 
AMREX_GPU_DEVICE
inline
void
riemann (const amrex::Real rl, const amrex::Real ul, const amrex::Real vl,
 const amrex::Real v2l, const amrex::Real pl, const amrex::Real rel, 
 const amrex::Real gamcl, const amrex::Real *spl, 
 const amrex::Real rr, const amrex::Real ur, const amrex::Real vr,
 const amrex::Real v2r, const amrex::Real pr, const amrex::Real rer, 
 const amrex::Real gamcr,  const amrex::Real *spr,
 EOS &gdnv_state, 
 amrex::Real &uflx_rho, amrex::Real &uflx_u, 
 amrex::Real &uflx_v, amrex::Real &uflx_w,
 amrex::Real &uflx_eden, amrex::Real &uflx_eint,
 amrex::Real &qint_iu, amrex::Real &qint_iv1, amrex::Real &qint_iv2, 
 amrex::Real &qint_gdpres, amrex::Real &qint_gdgame,
 amrex::Real &ustar, const int idir, const amrex::Real csmall,
 const amrex::Real cav)
//TODO consider making a uflx array instead of passing each in. 
{        
        amrex::Real wsmall = small_dens*csmall; 
        gdnv_state.rho = rl; 
        gdnv_state.p   = pl;

        for(int i = 0; i < NUM_SPECIES; ++i)
        {
          gdnv_state.massfrac[i] = spl[i];
        }
        
        gdnv_state.eos_rp(); 

        gdnv_state.rho = rr; 
        gdnv_state.p   = pr; 
        for(int i = 0; i < NUM_SPECIES; ++i){
             gdnv_state.massfrac[i] = spr[i]; 
        }
        gdnv_state.eos_rp(); 
        amrex::Real wl = amrex::max(wsmall, std::sqrt(std::abs(gamcl*pl*rl))); 
        amrex::Real wr = amrex::max(wsmall, std::sqrt(std::abs(gamcr*pr*rr))); 

        amrex::Real pstar = ((wr*pl + wl*pr) + wl*wr*(ul - ur))/(wl + wr); 
        ustar = ((wl*ul + wr*ur) + (pl - pr))/(wl + wr); 

        pstar = amrex::max(pstar, small_pres); 
        if (std::abs(ustar) < smallu*0.5e0*(std::abs(ul) + std::abs(ur))) ustar = 0.e0; 
        amrex::Real ro, uo, po; 
        if(ustar > 0.e0){
            ro = rl; 
            uo = ul; 
            po = pl; 
            for(int i = 0; i < NUM_SPECIES; ++i) gdnv_state.massfrac[i] = spl[i];
        }
        else if(ustar < 0.e0){
            ro = rr; 
            uo = ur; 
            po = pr; 
            for(int i = 0; i < NUM_SPECIES; ++i)  gdnv_state.massfrac[i] = spr[i];
        }
        else{
            ro = 0.5e0*(rr+rl); 
            uo = 0.5e0*(ur+ul); 
            po = 0.5e0*(pr+pl); 
            for(int i = 0; i < NUM_SPECIES; ++i)   gdnv_state.massfrac[i] = 0.5e0*(spr[i] + spl[i]);
        }  

        gdnv_state.rho = ro; 
        gdnv_state.p   = po; 
        gdnv_state.eos_rp(); 

        amrex::Real reo = gdnv_state.rho*gdnv_state.e;
        amrex::Real co  = gdnv_state.cs; 

        amrex::Real drho  = (pstar - po)/(co*co); 
        amrex::Real rstar = ro + drho; 
        rstar = amrex::max(small_dens, rstar); 

        gdnv_state.rho = rstar; 
        gdnv_state.p   = pstar; 

        gdnv_state.eos_rp(); 
        amrex::Real cstar = gdnv_state.cs; 
        amrex::Real estar = gdnv_state.rho*gdnv_state.e; 

        amrex::Real sgnm = std::copysign(1.e0, ustar);  
        amrex::Real spout = co - sgnm*uo; 
        amrex::Real spin  = cstar - sgnm*ustar; 

        amrex::Real ushock = 0.5e0*(spin + spout); 

        if(pstar - po >= 0.e0){
                spin = ushock; 
                spout = ushock; 
        }

        amrex::Real scr; 
        if(spout-spin == 0.e0) scr = small*cav; 
        else scr = spout-spin; 

        amrex::Real frac = (1.e0 + (spout + spin)/scr)*0.5e0; 
        frac = amrex::max(0.e0, amrex::min(1.e0, frac)); 
        
        amrex::Real vgd, wgd; 
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
        gdnv_state.rho = rgd; 
        gdnv_state.p = qint_gdpres; 
        gdnv_state.eos_rp(); 
        amrex::Real regd = gdnv_state.rho*gdnv_state.e; 

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

        if( idir == 0){
                uflx_u = uflx_rho*qint_iu + qint_gdpres; 
                uflx_v = uflx_rho*qint_iv1; 
                uflx_w = uflx_rho*qint_iv2; 
        }
        else if(idir == 1){
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


AMREX_GPU_DEVICE
inline
void
PeleC_cmpflx(const int i, const int j, const int k,
     const int bclo, const int bchi, const int domlo, const int domhi, 
     amrex::Array4<const amrex::Real> const& ql, 
     amrex::Array4<const amrex::Real> const & qr, amrex::Array4<amrex::Real> const& flx,
     amrex::Array4<amrex::Real> const& q, amrex::Array4<const amrex::Real> const& qa,
//     amrex::Array4<const int> const& bcMask,
     const int idir)
{
    EOS gdnv_state; 
    amrex::Real cav, ustar;
    amrex::Real spl[NUM_SPECIES];
    amrex::Real spr[NUM_SPECIES]; 
    amrex::Real ul, ur, vl, vr, v2l, v2r, rel, rer, gamcl, gamcr;
    int idx;  
    int IU, IV, IV2;
    int GU, GV, GV2;  
    if(idir == 0){
         IU = QU; 
         IV = QV; 
        IV2 = QW;
         GU = GDU; 
         GV = GDV; 
        GV2 = GDW;
        gamcl = qa(i-1,j,k,QGAMC);
        gamcr = qa(i,j,k,QGAMC);
        cav = 0.5e0*(qa(i,j,k,QC) + qa(i-1,j,k,QC));  
    }
    else if( idir == 1){
         IU = QV; 
         IV = QW; 
        IV2 = QU; 
         GU = GDV; 
         GV = GDW; 
        GV2 = GDU; 
        gamcl = qa(i,j-1,k,QGAMC);
        gamcr = qa(i,j,k,QGAMC); 
        cav = 0.5e0*(qa(i,j,k,QC) + qa(i,j-1,k,QC)); 
    }
    else{
         IU = QW; 
         IV = QU; 
        IV2 = QV; 
         GU = GDW; 
         GV = GDU; 
        GV2 = GDV; 
        gamcl = qa(i,j,k-1,QGAMC);
        gamcr = qa(i,j,k,QGAMC); 
        cav = 0.5e0*(qa(i,j,k,QC) + qa(i,j,k-1,QC)); 
    }

    for(int sp = 0; sp < NUM_SPECIES; ++sp){
            spl[sp] = ql(i,j,k,QFS+sp); 
            spr[sp] = qr(i,j,k,QFS+sp);
    }               
           ul = ql(i,j,k,IU); 
           vl = ql(i,j,k,IV); 
          v2l = ql(i,j,k,IV2); 
          rel = ql(i,j,k,QREINT); 
        
           ur = qr(i,j,k,IU); 
           vr = qr(i,j,k,IV); 
          v2r = qr(i,j,k,IV2); 
          rer = qr(i,j,k,QREINT); 

//Outflow Hack --> 
         idx = (idir == 0) ? i : j;
         if(bclo == Outflow && idx == domlo)
         {
            ul = ur; 
            vl = vr; 
            v2l = v2r; 
            rel = rer; 
         }
         if(bchi == Outflow && idx == domhi+1)
         {  
            ur  = ul; 
            vr  = vl; 
            v2r = v2l; 
            rer = rel;         
         } 
//*/
          riemann(ql(i,j,k,QRHO), ul, vl, v2l, 
                 ql(i,j,k,QPRES), rel, gamcl, spl,   
                 qr(i,j,k,QRHO), ur, vr, v2r, 
                 qr(i,j,k,QPRES), rer, gamcr, spr,
                 gdnv_state, flx(i,j,k,URHO), flx(i,j,k,UMX), flx(i,j,k,UMY), 
                 flx(i,j,k,UMZ), flx(i,j,k,UEDEN), flx(i,j,k,UEINT),
                 q(i,j,k,GU), q(i,j,k,GV), q(i,j,k,GV2), q(i,j,k,GDPRES),
                 q(i,j,k,GDGAME), ustar, idir, qa(i,j,k,QCSML), cav);
        
        amrex::Real flxrho = flx(i,j,k,URHO); 
        for(int ipass = 0; ipass < npassive; ++ipass){
            int n = EOS::upass_map(ipass); 
            int nqp = EOS::qpass_map(ipass);                     
            if(ustar > 0.e0) flx(i,j,k,n) = flxrho*ql(i,j,k,nqp); 
            else if(ustar < 0.e0) flx(i,j,k,n) = flxrho*qr(i,j,k,nqp); 
            else flx(i,j,k,n) = flxrho*0.5e0*(ql(i,j,k,nqp) + qr(i,j,k,nqp)); 
        }
            
}

AMREX_GPU_DEVICE
inline 
void 
PeleC_plm_x(const int i, const int j, const int k,
    amrex::Array4<amrex::Real> const& qm, amrex::Array4<amrex::Real> const& qp,
    amrex::Array4<const amrex::Real> const& slope,  amrex::Array4<const amrex::Real> const& q,
    amrex::Real const cc, 
    amrex::Array4<const amrex::Real> const& dloga,
    amrex::Real const dx, amrex::Real const dt)
{

        amrex::Real e[3]; 
        amrex::Real dtdx = dt/dx; 

//        amrex::Real cc    = qa(i,j,k,QC); 
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
        amrex::Real rho_ref  = rho - 0.5e0*(1.e0 + dtdx*amrex::min(e[0],0.e0))*drho;                
        amrex::Real u_ref    = u - 0.5e0*(1.e0 + dtdx*amrex::min(e[0], 0.e0))*du; 
        amrex::Real v_ref    = v - 0.5e0*(1.e0 + dtdx*amrex::min(e[0], 0.e0))*dv; 
        amrex::Real p_ref    = p - 0.5e0*(1.e0 + dtdx*amrex::min(e[0], 0.e0))*dp; 
        amrex::Real rhoe_ref = rhoe - 0.5e0*(1.e0 + dtdx*amrex::min(e[0], 0.e0))*drhoe; 

        amrex::Real apright = 0.25e0*dtdx*(e[0] - e[2])*
                              (1.e0 - std::copysign(1.0, e[2]))*alphap; 
//                  amright = 0.25e0*dtdx*(e[0] - e[0])*(1.e0 - std::copysign(1.0, e[0]))*alpham; 
        amrex::Real amright = 0.e0; 

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
        qp(i,j,k,QPRES)  = p_ref + (apright + amright)*cs;
        qp(i,j,k,QPRES)  = amrex::max(qp(i,j,k,QPRES),small_pres);
        qp(i,j,k,QREINT) = rhoe_ref + (apright + amright)*enth*cs + azeright;

// ------------ Construct the left state on the i+1/2 interface 
        
        rho_ref  = rho + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*drho;
        u_ref    = u + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*du;
        v_ref    = v + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*dv;
        p_ref    = p + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*dp;
        rhoe_ref = rhoe + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*drhoe;

//                  apleft = 0.25e0*dtdx*(e[2] - e[2])*(1.e0 + std::copysign(1.e0,e[2]))*alphap;
        amrex::Real apleft = 0.e0; 
        amrex::Real amleft = 0.25e0*dtdx*(e[2] - e[0])*(1.e0 + std::copysign(1.e0,e[0]))*alpham;

        amrex::Real azrleft = 0.25e0*dtdx*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0r;
        amrex::Real azeleft = 0.25e0*dtdx*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0e;
        amrex::Real azv1left = 0.25e0*dtdx*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0v;

        qm(i+1,j,k,QRHO) = rho_ref + apleft + amleft + azrleft;        
        qm(i+1,j,k,QRHO) = amrex::max(qm(i+1,j,k,QRHO),small_dens);
        qm(i+1,j,k,QU) = u_ref + (apleft - amleft)*cc/rho;
        qm(i+1,j,k,QV) = v_ref + azv1left;
        qm(i+1,j,k,QPRES) = p_ref + (apleft + amleft)*cs;
        qm(i+1,j,k,QPRES) = amrex::max(qm(i+1,j,k,QPRES), small_pres);
        qm(i+1,j,k,QREINT) = rhoe_ref + (apleft + amleft)*enth*cs + azeleft;
                        
        if(dloga(i,j,k)!=0){
            amrex::Real courn = dtdx*(cc+std::abs(u)); 
            amrex::Real eta = (1.e0-courn)/(cc*dt*std::abs(dloga(i,j,k))); 
            amrex::Real dlogatmp = amrex::min(eta, 1.e0)*dloga(i,j,k);
            amrex::Real sourcr = -0.5e0*dt*rho*dlogatmp*u; 
            amrex::Real sourcp = sourcr*cs; 
            amrex::Real source = sourcp*enth; 
            
            qm(i+1,j,k,QRHO) += sourcr; 
            qm(i+1,j,k,QRHO) = amrex::max(qm(i+1,j,k,QRHO), small_dens); 
            qm(i+1,j,k,QPRES) += sourcp; 
            qm(i+1,j,k,QREINT) += source; 
            
            qp(i,j,k,QRHO) += sourcr; 
            qp(i,j,k,QRHO) = amrex::max(qp(i,j,k,QRHO), small_dens); 
            qp(i,j,k,QPRES) += sourcp; 
            qp(i,j,k,QREINT) += source; 
        }
//--------------- Upwind the passive variables --------------------------------
        for(int ipassive = 0; ipassive < npassive; ++ipassive){
            int n = EOS::qpass_map(ipassive); 
            //Right state 
            u = q(i,j,k,QU); 
            amrex::Real spzero; 
            spzero = u > 0 ? -1.e0 : u*dtdx; 
            amrex::Real acmprght = 0.5e0*(-1.e0 - spzero)*slope(i,j,k,n); 
            qp(i,j,k,n) = q(i,j,k,n) + acmprght;

            spzero = u >= 0 ? u*dtdx : 1.e0; 
            amrex::Real acmpleft = 0.5e0*(1.e0 - spzero)*slope(i,j,k,n);  
            qm(i+1, j, k, n) = q(i,j,k,n) + acmpleft; 
        }                            
} 

AMREX_GPU_DEVICE
inline 
void 
PeleC_plm_y(const int i, const int j, const int k, 
    amrex::Array4<amrex::Real> const& qm         , amrex::Array4<amrex::Real> const& qp,
    amrex::Array4<const amrex::Real> const& slope, amrex::Array4<const amrex::Real> const& q,
    amrex::Real const cc, 
//    amrex::Array4<const amrex::Real> const& dloga, 
    amrex::Real const dy, amrex::Real const dt)
{

        amrex::Real e[3]; 
        amrex::Real dtdy = dt/dy; 

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
        amrex::Real rho_ref  = rho - 0.5e0*(1.e0 + dtdy*amrex::min(e[0],0.e0))*drho;                
        amrex::Real u_ref    = u - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*du; 
        amrex::Real v_ref    = v - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*dv; 
        amrex::Real p_ref    = p - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*dp; 
        amrex::Real rhoe_ref = rhoe - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*drhoe; 

        amrex::Real apright = 0.25e0*dtdy*(e[0] - e[2])*
                              (1.e0 - std::copysign(1.0, e[2]))*alphap; 
//      amright = 0.25e0*dtdy*(e[0] - e[0])*(1.e0 - std::copysign(1.0, e[0]))*alpham;  
        amrex::Real amright = 0.e0; 

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
        qp(i,j,k,QPRES)  = p_ref + (apright + amright)*cs;
        qp(i,j,k,QPRES)  = amrex::max(qp(i,j,k,QPRES),small_pres);
        qp(i,j,k,QREINT) = rhoe_ref + (apright + amright)*enth*cs + azeright;

// ------------ Construct the "left" state on the j+1/2 interface 
        
        rho_ref = rho + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*drho;
        u_ref = u + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*du;
        v_ref = v + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*dv;
        p_ref = p + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*dp;
        rhoe_ref = rhoe + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*drhoe;

//                  apleft = 0.25e0*dtdy*(e[2] - e[2])*(1.e0 + std::copysign(1.e0,e[2]))*alphap;
        amrex::Real apleft = 0.e0; 
        amrex::Real amleft = 0.25e0*dtdy*(e[2] - e[0])*(1.e0 + std::copysign(1.e0,e[0]))*alpham;

        amrex::Real azrleft = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0r;
        amrex::Real azeleft = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0e;
        amrex::Real azu1left = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0u;

        qm(i,j+1,k,QRHO) = rho_ref + apleft + amleft + azrleft;
        qm(i,j+1,k,QRHO) = amrex::max(qm(i,j+1,k,QRHO),small_dens);
        qm(i,j+1,k,QV) = v_ref + (apleft - amleft)*cc/rho;
        qm(i,j+1,k,QU) = u_ref + azu1left;
        qm(i,j+1,k,QPRES) = p_ref + (apleft + amleft)*cs;
        qm(i,j+1,k,QPRES) = amrex::max(qm(i,j+1,k,QPRES), small_pres);
        qm(i,j+1,k,QREINT) = rhoe_ref + (apleft + amleft)*enth*cs + azeleft;

/*                        
        if(dloga(i,j,k)!=0){
            amrex::Real courn = dtdy*(cc+abs(v)); 
            amrex::Real eta = (1.e0-courn)/(cc*dt*abs(dloga(i,j,k))); 
            amrex::Real dlogatmp = amrex::min(eta, 1.e0)*dloga(i,j,k);
            amrex::Real sourcr = -0.5e0*dt*rho*dlogatmp*v; 
            amrex::Real sourcp = sourcr*cs; 
            amrex::Real source = sourcp*enth; 
            
            qm(i,j+1,k,QRHO)   += sourcr; 
            qm(i,j+1,k,QRHO)    = amrex::max(qm(i,j+1,k,QRHO), small_dens); 
            qm(i,j+1,k,QPRES)  += sourcp; 
            qm(i,j+1,k,QREINT) += source; 
            
            qp(i,j,k,QRHO)   += sourcr; 
            qp(i,j,k,QRHO)    = amrex::max(qp(i,j,k,QRHO), small_dens); 
            qp(i,j,k,QPRES)  += sourcp; 
            qp(i,j,k,QREINT) += source; 
        } // */ 
//--------------- Upwind the passive variables --------------------------------
        for(int ipassive = 0; ipassive < npassive; ++ipassive){
            int n = EOS::qpass_map(ipassive); 
            //Right state 
            v = q(i,j,k,QV); 
            amrex::Real spzero; 
            spzero = v > 0 ? -1.e0 : v*dtdy; 
            amrex::Real acmptop = 0.5e0*(-1.e0 - spzero)*slope(i,j,k,n); 
            qp(i,j,k,n) = q(i,j,k,n) + acmptop;

            spzero = v >= 0 ? v*dtdy : 1.e0; 
            amrex::Real acmpbot = 0.5e0*(1.e0 - spzero)*slope(i,j,k,n);  
            qm(i,j+1,k,n) = q(i,j,k,n) + acmpbot; 
        }                                           
}

AMREX_GPU_DEVICE
inline 
void 
PeleC_plm_z(const int i, const int j, const int k, 
    amrex::Array4<amrex::Real> const& qm         , amrex::Array4<amrex::Real> const& qp,
    amrex::Array4<const amrex::Real> const& slope, amrex::Array4<const amrex::Real> const& q,
    amrex::Real const cc, 
//    amrex::Array4<const amrex::Real> const& dloga, 
    amrex::Real const dy, amrex::Real const dt)
{

        amrex::Real e[3]; 
        amrex::Real dtdy = dt/dy; 

        amrex::Real cs    = cc*cc; 
        amrex::Real rho   = q(i,j,k,QRHO); 
        amrex::Real u     = q(i,j,k,QU); 
        amrex::Real v     = q(i,j,k,QV);
        amrex::Real w     = q(i,j,k,QW);  
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
        
        e[0] = w - cc; 
        e[1] = w; 
        e[2] = w + cc; 
// ------------ Construct the "right" state on the j-1/2 interface 
        amrex::Real rho_ref  = rho - 0.5e0*(1.e0 + dtdy*amrex::min(e[0],0.e0))*drho;                
        amrex::Real u_ref    = u - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*du; 
        amrex::Real v_ref    = v - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*dv; 
        amrex::Real p_ref    = p - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*dp; 
        amrex::Real rhoe_ref = rhoe - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*drhoe; 

        amrex::Real apright = 0.25e0*dtdy*(e[0] - e[2])*
                              (1.e0 - std::copysign(1.0, e[2]))*alphap; 
//      amright = 0.25e0*dtdy*(e[0] - e[0])*(1.e0 - std::copysign(1.0, e[0]))*alpham;  
        amrex::Real amright = 0.e0; 

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
        qp(i,j,k,QPRES)  = p_ref + (apright + amright)*cs;
        qp(i,j,k,QPRES)  = amrex::max(qp(i,j,k,QPRES),small_pres);
        qp(i,j,k,QREINT) = rhoe_ref + (apright + amright)*enth*cs + azeright;

// ------------ Construct the "left" state on the j+1/2 interface 
        
        rho_ref = rho + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*drho;
        u_ref = u + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*du;
        v_ref = v + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*dv;
        p_ref = p + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*dp;
        rhoe_ref = rhoe + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*drhoe;

//                  apleft = 0.25e0*dtdy*(e[2] - e[2])*(1.e0 + std::copysign(1.e0,e[2]))*alphap;
        amrex::Real apleft = 0.e0; 
        amrex::Real amleft = 0.25e0*dtdy*(e[2] - e[0])*(1.e0 + std::copysign(1.e0,e[0]))*alpham;

        amrex::Real azrleft = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0r;
        amrex::Real azeleft = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0e;
        amrex::Real azu1left = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0u;

        qm(i,j,k+1,QRHO) = rho_ref + apleft + amleft + azrleft;
        qm(i,j,k+1,QRHO) = amrex::max(qm(i,j,k+1,QRHO),small_dens);
        qm(i,j,k+1,QV) = v_ref + (apleft - amleft)*cc/rho;
        qm(i,j,k+1,QU) = u_ref + azu1left;
        qm(i,j,k+1,QPRES) = p_ref + (apleft + amleft)*cs;
        qm(i,j,k+1,QPRES) = amrex::max(qm(i,j,k+1,QPRES), small_pres);
        qm(i,j,k+1,QREINT) = rhoe_ref + (apleft + amleft)*enth*cs + azeleft;

/*                        
        if(dloga(i,j,k)!=0){
            amrex::Real courn = dtdy*(cc+abs(v)); 
            amrex::Real eta = (1.e0-courn)/(cc*dt*abs(dloga(i,j,k))); 
            amrex::Real dlogatmp = amrex::min(eta, 1.e0)*dloga(i,j,k);
            amrex::Real sourcr = -0.5e0*dt*rho*dlogatmp*v; 
            amrex::Real sourcp = sourcr*cs; 
            amrex::Real source = sourcp*enth; 
            
            qm(i,j+1,k,QRHO)   += sourcr; 
            qm(i,j+1,k,QRHO)    = amrex::max(qm(i,j+1,k,QRHO), small_dens); 
            qm(i,j+1,k,QPRES)  += sourcp; 
            qm(i,j+1,k,QREINT) += source; 
            
            qp(i,j,k,QRHO)   += sourcr; 
            qp(i,j,k,QRHO)    = amrex::max(qp(i,j,k,QRHO), small_dens); 
            qp(i,j,k,QPRES)  += sourcp; 
            qp(i,j,k,QREINT) += source; 
        } // */ 
//--------------- Upwind the passive variables --------------------------------
        for(int ipassive = 0; ipassive < npassive; ++ipassive){
            int n = EOS::qpass_map(ipassive); 
            //Right state 
            w = q(i,j,k,QW); 
            amrex::Real spzero; 
            spzero = w > 0 ? -1.e0 : w*dtdy; 
            amrex::Real acmptop = 0.5e0*(-1.e0 - spzero)*slope(i,j,k,n); 
            qp(i,j,k,n) = q(i,j,k,n) + acmptop;

            spzero = w >= 0 ? w*dtdy : 1.e0; 
            amrex::Real acmpbot = 0.5e0*(1.e0 - spzero)*slope(i,j,k,n);  
            qm(i,j,k+1,n) = q(i,j,k,n) + acmpbot; 
        }                                           
}

#endif
