#ifndef PELEC_METHOD_3D_H_
#define PELEC_METHOD_3D_H_ 
#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
#include "PeleC_Parameters.H" 
#include "PeleC_index_macros.H"
#include "PeleC_EOS.H" 
#include <cmath> 

/*This header file contains functions and declarations for the 3D hydro portion of PeleC gpu. 
  As per the convention of AMReX, inlined functions are defined here. Where as non-inline functions 
  are declared here. */ 

AMREX_GPU_DEVICE
inline
amrex::Real 
PeleC_slope_x(const int i, const int j, const int k, const int n,  
              amrex::Array4<const amrex::Real> const& q)
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
    return dsgn*amrex::min(dlim, std::abs(dtemp));
}

AMREX_GPU_DEVICE
inline
amrex::Real 
PeleC_slope_y (const int i, const int j, const int k, const int n, 
               amrex::Array4<const amrex::Real> const& q)
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
    return dsgn*amrex::min(dlim, std::abs(dtemp)); 
}

AMREX_GPU_DEVICE
inline
amrex::Real 
PeleC_slope_z (const int i, const int j, const int k, const int n, 
               amrex::Array4<const amrex::Real> const& q)
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
    return dsgn*amrex::min(dlim, std::abs(dtemp)); 
}


//Riemannd 
AMREX_GPU_DEVICE
inline
void
riemannus (const amrex::Real rl, const amrex::Real ul, const amrex::Real vl,
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
    amrex::Real wl, wr; 
    wl = std::sqrt(std::abs(gamcl*pl*rl)); 
    wr = std::sqrt(std::abs(gamcr*pr*rr)); 
    
    amrex::Real wwinv = 1.e0/(wl + wr); 
    amrex::Real pstar = ((wr*pl + wl*pr) + wl*wr*(ul -ur))*wwinv; 
    ustar = ((wl*ul + wr*ur) + (pl - pr))*wwinv; 

    if(std::abs(ustar) < smallu*0.5e0*(std::abs(ul) + std::abs(ur))) ustar = 0.e0; 
    amrex::Real ro, uo, po, reo, gamco; 
    if(ustar > 0.e0){
        ro    = rl; 
        uo    = ul; 
        po    = pl; 
        reo   = rel; 
        gamco = gamcl; 
    }
    else if(ustar < 0.e0){
        ro    = rr; 
        uo    = ur; 
        po    = pr; 
        reo   = rer; 
        gamco = gamcr; 
    }
    else
    {
        ro    = 0.5e0*(rl + rr); 
        uo    = 0.5e0*(ul + ur); 
        po    = 0.5e0*(pl + pr); 
        reo   = 0.5e0*(rel + rer); 
        gamco = 0.5e0*(gamcl + gamcr); 
    }
    
    ro = amrex::max(small_dens, ro); 
    amrex::Real roinv = 1.e0/ro; 
    amrex::Real co = std::sqrt(std::abs(gamco*po*roinv)); 
    co = amrex::max(csmall, co); 
    amrex::Real co2inv = 1.e0/(co*co); 
    amrex::Real drho = (pstar - po)*co2inv; 
    amrex::Real rstar = ro + drho;
    rstar = amrex::max(small_dens, rstar);  
    amrex::Real entho = (reo + po)*roinv*co2inv; 
    amrex::Real estar = reo + (pstar -po)*entho; 
    amrex::Real cstar = std::sqrt(std::abs(gamco*pstar/rstar)); 
    
    amrex::Real sgnm = std::copysign(1.e0, ustar); 
    amrex::Real spout = co - sgnm*uo; 
    amrex::Real spin  = cstar - sgnm*ustar; 
    amrex::Real ushock = 0.5e0*(spin + spout); 
    
    if (pstar-po > 0.e0){
        spin = ushock; 
        spout = ushock; 
    }
    amrex::Real scr; 
    if(spout-spin == 0.e0) scr = small*cav; 
    else scr = spout-spin; 
    
    amrex::Real frac = (1.e0 + (spout + spin)/scr)*0.5e0; 
    frac = amrex::max(0.e0, amrex::min(1.e0, frac)); 

    if(ustar > 0.e0){
        qint_iv1 = vl; 
        qint_iv2 = v2l; 
    }
    else if(ustar < 0.e0)
    {
        qint_iv1 = vr; 
        qint_iv2 = v2r; 
    }
    else{
        qint_iv1 = 0.5e0*(vl + vr); 
        qint_iv2 = 0.5e0*(v2l + v2r); 
    }
    amrex::Real qint_rho = frac*rstar + (1.e0 - frac)*ro; 
    qint_iu = frac*ustar + (1.e0 - frac)*uo; 
    qint_gdpres = frac*pstar + (1.e0 - frac)*po; 
    amrex::Real regdnv = frac*estar + (1.e0 - frac)*reo; 
    if(spout < 0.e0){
        qint_rho = ro; 
        qint_iu = uo; 
        qint_gdpres = po; 
        regdnv = reo; 
    }
    if (spin >= 0.e0){
        qint_rho = rstar; 
        qint_iu = ustar; 
        qint_gdpres = pstar; 
        regdnv = estar; 
    }
    qint_gdgame = qint_gdpres/regdnv + 1.e0; 
    qint_gdpres = amrex::max(small_pres, qint_gdpres); 
//    amrex::Real u_adv = qint_iu; 
    
    //compute fluxes! 
    uflx_rho = qint_rho*qint_iu; 
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
    amrex::Real rhoetot = regdnv + 0.5e0*qint_rho*(qint_iu*qint_iu + qint_iv1*qint_iv1 + qint_iv2*qint_iv2); 
    uflx_eden = qint_iu*(rhoetot + qint_gdpres); 
    uflx_eint = qint_iu*regdnv; 
}


//Riemann_md 
AMREX_GPU_DEVICE
inline
void
riemann_md (const amrex::Real rl, const amrex::Real ul, const amrex::Real vl,
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
         IV = QU; 
        IV2 = QW; 
         GU = GDV; 
         GV = GDU; 
        GV2 = GDW; 
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
         if(idir == 2) idx = k; 
         else idx = (idir == 0) ? i : j;
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
//here we have two riemann dudes riemannus and riemann_md
       riemannus(ql(i,j,k,QRHO), ul, vl, v2l, 
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
    amrex::Real  const slope[QVAR],  amrex::Array4<const amrex::Real> const& q,
    amrex::Real const cc, 
//   amrex::Array4<const amrex::Real> const& dloga,
    amrex::Real const dx, amrex::Real const dt)
{

        amrex::Real e[3]; 
        amrex::Real dtdx = dt/dx; 

//        amrex::Real cc    = qa(i,j,k,QC); 
        amrex::Real cs    = cc*cc; 
        amrex::Real rho   = q(i,j,k,QRHO); 
        amrex::Real u     = q(i,j,k,QU); 
        amrex::Real v     = q(i,j,k,QV);
        amrex::Real w     = q(i,j,k,QW);  
        amrex::Real p     = q(i,j,k,QPRES); 
        amrex::Real rhoe  = q(i,j,k,QREINT); 
        amrex::Real enth  = ((rhoe+p)/rho)/cs; 
        
        amrex::Real drho  = slope[QRHO]; 
        amrex::Real du    = slope[QU]; 
        amrex::Real dv    = slope[QV];
        amrex::Real dw    = slope[QW];  
        amrex::Real dp    = slope[QPRES]; 
        amrex::Real drhoe = slope[QREINT]; 
        
        amrex::Real alpham = 0.5e0*(dp/(rho*cc) - du)*rho/cc; 
        amrex::Real alphap = 0.5e0*(dp/(rho*cc) + du)*rho/cc; 
        amrex::Real alpha0r = drho - dp/cs;
        amrex::Real alpha0e = drhoe - dp*enth;
        amrex::Real alpha0v = dv;
        amrex::Real alpha0w = dw;  
        
        e[0] = u - cc; 
        e[1] = u; 
        e[2] = u + cc; 
// ------------ Construct the right state on the i-1/2 interface 
        amrex::Real rho_ref  = rho - 0.5e0*(1.e0 + dtdx*amrex::min(e[0],0.e0))*drho;                
        amrex::Real u_ref    = u - 0.5e0*(1.e0 + dtdx*amrex::min(e[0], 0.e0))*du; 
        amrex::Real v_ref    = v - 0.5e0*(1.e0 + dtdx*amrex::min(e[0], 0.e0))*dv;
        amrex::Real w_ref    = w - 0.5e0*(1.e0 + dtdx*amrex::min(e[0], 0.e0))*dw;  
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
        amrex::Real azw1rght = 0.25e0*dtdx*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0w;  

        qp(i,j,k,QRHO)   = rho_ref + apright + amright + azrright;
        qp(i,j,k,QRHO)   = amrex::max(small_dens,qp(i,j,k,QRHO));
        qp(i,j,k,QU)     = u_ref + (apright - amright)*cc/rho;
        qp(i,j,k,QV)     = v_ref + azv1rght;
        qp(i,j,k,QW)     = w_ref + azw1rght; 
        qp(i,j,k,QPRES)  = p_ref + (apright + amright)*cs;
        qp(i,j,k,QPRES)  = amrex::max(qp(i,j,k,QPRES),small_pres);
        qp(i,j,k,QREINT) = rhoe_ref + (apright + amright)*enth*cs + azeright;

// ------------ Construct the left state on the i+1/2 interface 
        
        rho_ref  = rho + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*drho;
        u_ref    = u + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*du;
        v_ref    = v + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*dv;
        w_ref    = w + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*dw; 
        p_ref    = p + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*dp;
        rhoe_ref = rhoe + 0.5e0*(1.e0 - dtdx*amrex::max(e[2],0.e0))*drhoe;

//                  apleft = 0.25e0*dtdx*(e[2] - e[2])*(1.e0 + std::copysign(1.e0,e[2]))*alphap;
        amrex::Real apleft = 0.e0; 
        amrex::Real amleft = 0.25e0*dtdx*(e[2] - e[0])*(1.e0 + std::copysign(1.e0,e[0]))*alpham;

        amrex::Real azrleft  = 0.25e0*dtdx*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0r;
        amrex::Real azeleft  = 0.25e0*dtdx*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0e;
        amrex::Real azv1left = 0.25e0*dtdx*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0v;
        amrex::Real azw1left = 0.25e0*dtdx*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0w; 
        qm(i+1,j,k,QRHO) = rho_ref + apleft + amleft + azrleft;        
        qm(i+1,j,k,QRHO) = amrex::max(qm(i+1,j,k,QRHO),small_dens);
        qm(i+1,j,k,QU) = u_ref + (apleft - amleft)*cc/rho;
        qm(i+1,j,k,QV) = v_ref + azv1left;
        qm(i+1,j,k,QW) = w_ref + azw1left; 
        qm(i+1,j,k,QPRES) = p_ref + (apleft + amleft)*cs;
        qm(i+1,j,k,QPRES) = amrex::max(qm(i+1,j,k,QPRES), small_pres);
        qm(i+1,j,k,QREINT) = rhoe_ref + (apleft + amleft)*enth*cs + azeleft;
//--------------- Upwind the passive variables --------------------------------
        for(int ipassive = 0; ipassive < npassive; ++ipassive){
            int n = EOS::qpass_map(ipassive); 
            //Right state 
            u = q(i,j,k,QU); 
            amrex::Real spzero; 
            spzero = u > 0 ? -1.e0 : u*dtdx; 
            amrex::Real acmprght = 0.5e0*(-1.e0 - spzero)*slope[n]; 
            qp(i,j,k,n) = q(i,j,k,n) + acmprght;

            spzero = u >= 0 ? u*dtdx : 1.e0; 
            amrex::Real acmpleft = 0.5e0*(1.e0 - spzero)*slope[n];  
            qm(i+1, j, k, n) = q(i,j,k,n) + acmpleft; 
        }                            
} 

AMREX_GPU_DEVICE
inline 
void 
PeleC_plm_y(const int i, const int j, const int k, 
    amrex::Array4<amrex::Real> const& qm         , amrex::Array4<amrex::Real> const& qp,
    amrex::Real const slope[QVAR], amrex::Array4<const amrex::Real> const& q,
    amrex::Real const cc, 
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
        
        amrex::Real drho  = slope[QRHO]; 
        amrex::Real du    = slope[QU]; 
        amrex::Real dv    = slope[QV];
        amrex::Real dw    = slope[QW];  
        amrex::Real dp    = slope[QPRES]; 
        amrex::Real drhoe = slope[QREINT]; 
        
        amrex::Real alpham = 0.5e0*(dp/(rho*cc) - dv)*rho/cc; 
        amrex::Real alphap = 0.5e0*(dp/(rho*cc) + dv)*rho/cc; 
        amrex::Real alpha0r = drho - dp/cs;
        amrex::Real alpha0e = drhoe - dp*enth;
        amrex::Real alpha0u = du; 
        amrex::Real alpha0w = dw; 
        
        e[0] = v - cc; 
        e[1] = v; 
        e[2] = v + cc; 
// ------------ Construct the "right" state on the j-1/2 interface 
        amrex::Real rho_ref  = rho - 0.5e0*(1.e0 + dtdy*amrex::min(e[0],0.e0))*drho;                
        amrex::Real u_ref    = u - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*du; 
        amrex::Real v_ref    = v - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*dv; 
        amrex::Real w_ref    = w - 0.5e0*(1.e0 + dtdy*amrex::min(e[0], 0.e0))*dw; 
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
        amrex::Real azw1rght = 0.25e0*dtdy*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0w;  

        qp(i,j,k,QRHO)   = rho_ref + apright + amright + azrright;
        qp(i,j,k,QRHO)   = amrex::max(small_dens,qp(i,j,k,QRHO));
        qp(i,j,k,QU)     = u_ref + azu1rght;
        qp(i,j,k,QV)     = v_ref + (apright - amright)*cc/rho;
        qp(i,j,k,QW)     = w_ref + azw1rght; 
        qp(i,j,k,QPRES)  = p_ref + (apright + amright)*cs;
        qp(i,j,k,QPRES)  = amrex::max(qp(i,j,k,QPRES),small_pres);
        qp(i,j,k,QREINT) = rhoe_ref + (apright + amright)*enth*cs + azeright;

// ------------ Construct the "left" state on the j+1/2 interface 
        
        rho_ref = rho + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*drho;
        u_ref = u + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*du;
        v_ref = v + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*dv;
        w_ref = w + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*dw; 
        p_ref = p + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*dp;
        rhoe_ref = rhoe + 0.5e0*(1.e0 - dtdy*amrex::max(e[2],0.e0))*drhoe;

//                  apleft = 0.25e0*dtdy*(e[2] - e[2])*(1.e0 + std::copysign(1.e0,e[2]))*alphap;
        amrex::Real apleft = 0.e0; 
        amrex::Real amleft = 0.25e0*dtdy*(e[2] - e[0])*(1.e0 + std::copysign(1.e0,e[0]))*alpham;

        amrex::Real azrleft = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0r;
        amrex::Real azeleft = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0e;
        amrex::Real azu1left = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0u;
        amrex::Real azw1left = 0.25e0*dtdy*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0w; 

        qm(i,j+1,k,QRHO) = rho_ref + apleft + amleft + azrleft;
        qm(i,j+1,k,QRHO) = amrex::max(qm(i,j+1,k,QRHO),small_dens);
        qm(i,j+1,k,QU) = u_ref + azu1left;
        qm(i,j+1,k,QV) = v_ref + (apleft - amleft)*cc/rho;
        qm(i,j+1,k,QW) = w_ref + azw1left; 
        qm(i,j+1,k,QPRES) = p_ref + (apleft + amleft)*cs;
        qm(i,j+1,k,QPRES) = amrex::max(qm(i,j+1,k,QPRES), small_pres);
        qm(i,j+1,k,QREINT) = rhoe_ref + (apleft + amleft)*enth*cs + azeleft;

//--------------- Upwind the passive variables --------------------------------
        for(int ipassive = 0; ipassive < npassive; ++ipassive){
            int n = EOS::qpass_map(ipassive); 
            //Right state 
            v = q(i,j,k,QV); 
            amrex::Real spzero; 
            spzero = v > 0 ? -1.e0 : v*dtdy; 
            amrex::Real acmptop = 0.5e0*(-1.e0 - spzero)*slope[n]; 
            qp(i,j,k,n) = q(i,j,k,n) + acmptop;

            spzero = v >= 0 ? v*dtdy : 1.e0; 
            amrex::Real acmpbot = 0.5e0*(1.e0 - spzero)*slope[n];  
            qm(i,j+1,k,n) = q(i,j,k,n) + acmpbot; 
        }                                           
}

AMREX_GPU_DEVICE
inline 
void 
PeleC_plm_z(const int i, const int j, const int k, 
    amrex::Array4<amrex::Real> const& qm         , amrex::Array4<amrex::Real> const& qp,
    amrex::Real const slope[QVAR], amrex::Array4<const amrex::Real> const& q,
    amrex::Real const cc, 
    amrex::Real const dz, amrex::Real const dt)
{

        amrex::Real e[3]; 
        amrex::Real dtdz = dt/dz; 

        amrex::Real cs    = cc*cc; 
        amrex::Real rho   = q(i,j,k,QRHO); 
        amrex::Real u     = q(i,j,k,QU); 
        amrex::Real v     = q(i,j,k,QV);
        amrex::Real w     = q(i,j,k,QW);  
        amrex::Real p     = q(i,j,k,QPRES); 
        amrex::Real rhoe  = q(i,j,k,QREINT); 
        amrex::Real enth  = ((rhoe+p)/rho)/cs; 
        
        amrex::Real drho  = slope[QRHO]; 
        amrex::Real du    = slope[QU]; 
        amrex::Real dv    = slope[QV];
        amrex::Real dw    = slope[QW];  
        amrex::Real dp    = slope[QPRES]; 
        amrex::Real drhoe = slope[QREINT]; 
        
        amrex::Real alpham = 0.5e0*(dp/(rho*cc) - dv)*rho/cc; 
        amrex::Real alphap = 0.5e0*(dp/(rho*cc) + dv)*rho/cc; 
        amrex::Real alpha0r = drho - dp/cs;
        amrex::Real alpha0e = drhoe - dp*enth;
        amrex::Real alpha0u = du; 
        amrex::Real alpha0v = dv; 
        
        e[0] = w - cc; 
        e[1] = w; 
        e[2] = w + cc; 

// ------------ Construct the "right" state on the j-1/2 interface 
        amrex::Real rho_ref  = rho - 0.5e0*(1.e0 + dtdz*amrex::min(e[0],0.e0))*drho;                
        amrex::Real u_ref    = u - 0.5e0*(1.e0 + dtdz*amrex::min(e[0], 0.e0))*du; 
        amrex::Real v_ref    = v - 0.5e0*(1.e0 + dtdz*amrex::min(e[0], 0.e0))*dv; 
        amrex::Real w_ref    = w - 0.5e0*(1.e0 + dtdz*amrex::min(e[0], 0.e0))*dw; 
        amrex::Real p_ref    = p - 0.5e0*(1.e0 + dtdz*amrex::min(e[0], 0.e0))*dp; 
        amrex::Real rhoe_ref = rhoe - 0.5e0*(1.e0 + dtdz*amrex::min(e[0], 0.e0))*drhoe; 

        amrex::Real apright = 0.25e0*dtdz*(e[0] - e[2])*
                              (1.e0 - std::copysign(1.0, e[2]))*alphap; 
//      amright = 0.25e0*dtdz*(e[0] - e[0])*(1.e0 - std::copysign(1.0, e[0]))*alpham;  
        amrex::Real amright = 0.e0; 

        amrex::Real azrright = 0.25e0*dtdz*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0r; 
        amrex::Real azeright = 0.25e0*dtdz*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0e; 
        amrex::Real azu1rght = 0.25e0*dtdz*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0u; 
        amrex::Real azv1rght = 0.25e0*dtdz*(e[0] - e[1])*
                              (1.e0 - std::copysign(1.0, e[1]))*alpha0v; 

        qp(i,j,k,QRHO)   = rho_ref + apright + amright + azrright;
        qp(i,j,k,QRHO)   = amrex::max(small_dens,qp(i,j,k,QRHO));
        qp(i,j,k,QU)     = u_ref + azu1rght;
        qp(i,j,k,QV)     = v_ref + azv1rght; 
        qp(i,j,k,QW)     = w_ref + (apright - amright)*cc/rho; 
        qp(i,j,k,QPRES)  = p_ref + (apright + amright)*cs;
        qp(i,j,k,QPRES)  = amrex::max(qp(i,j,k,QPRES),small_pres);
        qp(i,j,k,QREINT) = rhoe_ref + (apright + amright)*enth*cs + azeright;

// ------------ Construct the "left" state on the j+1/2 interface 
        
        rho_ref  = rho + 0.5e0*(1.e0 - dtdz*amrex::max(e[2],0.e0))*drho;
        u_ref    = u + 0.5e0*(1.e0 - dtdz*amrex::max(e[2],0.e0))*du;
        v_ref    = v + 0.5e0*(1.e0 - dtdz*amrex::max(e[2],0.e0))*dv;
        w_ref    = w + 0.5e0*(1.e0 - dtdz*amrex::max(e[2],0.e0))*dw; 
        p_ref    = p + 0.5e0*(1.e0 - dtdz*amrex::max(e[2],0.e0))*dp;
        rhoe_ref = rhoe + 0.5e0*(1.e0 - dtdz*amrex::max(e[2],0.e0))*drhoe;

//                  apleft = 0.25e0*dtdz*(e[2] - e[2])*(1.e0 + std::copysign(1.e0,e[2]))*alphap;
        amrex::Real apleft = 0.e0; 
        amrex::Real amleft = 0.25e0*dtdz*(e[2] - e[0])*(1.e0 + std::copysign(1.e0,e[0]))*alpham;

        amrex::Real azrleft = 0.25e0*dtdz*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0r;
        amrex::Real azeleft = 0.25e0*dtdz*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0e;
        amrex::Real azu1left = 0.25e0*dtdz*(e[2] - e[1])*(1.e0 + std::copysign(1.e0,e[1]))*alpha0u;
        amrex::Real azv1left = 0.25e0*dtdz*(e[2] - e[1])*(1.e0 + std::copysign(1.e0, e[1]))*alpha0v; 

        qm(i,j,k+1,QRHO) = rho_ref + apleft + amleft + azrleft;
        qm(i,j,k+1,QRHO) = amrex::max(qm(i,j,k+1,QRHO),small_dens);
        qm(i,j,k+1,QU) = u_ref + azu1left;
        qm(i,j,k+1,QV) = v_ref + azv1left; 
        qm(i,j,k+1,QW) = w_ref + (apleft - amleft)*cc/rho; 
        qm(i,j,k+1,QPRES) = p_ref + (apleft + amleft)*cs;
        qm(i,j,k+1,QPRES) = amrex::max(qm(i,j,k+1,QPRES), small_pres);
        qm(i,j,k+1,QREINT) = rhoe_ref + (apleft + amleft)*enth*cs + azeleft;

//--------------- Upwind the passive variables --------------------------------
        for(int ipassive = 0; ipassive < npassive; ++ipassive){
            int n = EOS::qpass_map(ipassive); 
            //Right state 
            w = q(i,j,k,QW); 
            amrex::Real spzero; 
            spzero = w > 0 ? -1.e0 : w*dtdz; 
            amrex::Real acmptop = 0.5e0*(-1.e0 - spzero)*slope[n]; 
            qp(i,j,k,n) = q(i,j,k,n) + acmptop;

            spzero = w >= 0 ? w*dtdz : 1.e0; 
            amrex::Real acmpbot = 0.5e0*(1.e0 - spzero)*slope[n];  
            qm(i,j,k+1,n) = q(i,j,k,n) + acmpbot; 
        }                                           
}


/* First Transverse Correction for Predicted Y-states, using X-Flux */ 
AMREX_GPU_DEVICE
inline 
void 
PeleC_transx1(const int i, const int j, const int k,
        amrex::Array4<amrex::Real> const& qm        , amrex::Array4<amrex::Real> const& qp,
        amrex::Array4<const amrex::Real> const& qym , amrex::Array4<const amrex::Real> const& qyp,
        amrex::Array4<const amrex::Real> const& flxx,
        amrex::Array4<const amrex::Real> const& qa  , amrex::Array4<const amrex::Real> const& qint,
        const amrex::Real cdtdx)
{
        //  Local variables to avoid reading from global memory too much 
        amrex::Real compn, rr, rrnew, compo, rwr, c;
        amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr;
        amrex::Real runewr, rvnewr, rwnewr, renewr, rhoekinr ;
        amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl,rwnewl, renewl;
        amrex::Real rhoekinl, rwl;
        amrex::Real pggp, pggm, ugp, ugm;
        amrex::Real dAup, pav,dAu;
        amrex::Real flxrho, flxu, flxv, flxw, flxe;

        flxrho = cdtdx*(flxx(i+1,j,k,URHO)  - flxx(i,j,k,URHO));
        flxu   = cdtdx*(flxx(i+1,j,k,UMX)   - flxx(i,j,k,UMX));
        flxv   = cdtdx*(flxx(i+1,j,k,UMY)   - flxx(i,j,k,UMY));
        flxw   = cdtdx*(flxx(i+1,j,k,UMZ)   - flxx(i,j,k,UMZ));
        flxe   = cdtdx*(flxx(i+1,j,k,UEDEN) - flxx(i,j,k,UEDEN));
        c      = qa(i,j,k,QGAMC);
// ---------------------------  Update Passive Variables ------------------
        for (int ipass = 0; ipass < npassive; ++ipass){
            int n   = EOS::upass_map(ipass);
            int nqp = EOS::qpass_map(ipass);
            compn = cdtdx*(flxx(i+1,j,k,n) - flxx(i,j,k,n));
            rr = qyp(i,j,k,QRHO);
            rrnew = rr - flxrho;
            compo = rr*qyp(i,j,k,nqp) - compn;
            qp(i,j,k,nqp) = compo/rrnew;

            rr = qym(i,j+1,k,QRHO);
            rrnew = rr - flxrho;
            compo = rr*qym(i,j+1,k,nqp) - compn;
            qm(i,j+1,k,nqp) = compo/rrnew;
        }
//------------------------------- Update hydro vars -----------------------

        pggp = qint(i+1,j,k,GDPRES); 
        pggm = qint(i,j,k,GDPRES); 
        ugp  = qint(i+1,j,k,GDU); 
        ugm  = qint(i,j,k,GDU); 
    
        dAup = pggp*ugp - pggm*ugm; //
        pav  = 0.5e0*(pggp + pggm); 
        dAu  = ugp - ugm; //
    
//========================= QP ===========================================

//Convert to conservative 
        rrr = qyp(i,j,k,QRHO); 
        rur = qyp(i,j,k,QU);             
        rvr = qyp(i,j,k,QV); 
        rwr = qyp(i,j,k,QW); 
        ekinr = 0.5e0*rrr*(rur*rur + rvr*rvr + rwr*rwr);
        rur *= rrr; 
        rvr *= rrr; 
        rwr *= rrr; 

        rer = qyp(i,j,k,QREINT) + ekinr; 
// Add transverse predictor 
        rrnewr = rrr - flxrho; 
        runewr = rur - flxu; 
        rvnewr = rvr - flxv;
        rwnewr = rwr - flxw;  
        renewr = rer - flxe; 

// Convert back to primitive 
        qp(i,j,k,QRHO) = rrnewr;
        qp(i,j,k,QU)   = runewr/rrnewr; 
        qp(i,j,k,QV)   = rvnewr/rrnewr;
        qp(i,j,k,QW)   = rwnewr/rrnewr;  
        rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwnewr*rwnewr)/rrnewr; 
        amrex::Real pnewr = qyp(i,j,k,QPRES) 
                          - cdtdx*(dAup + pav*dAu*(c - 1.e0));
        qp(i,j,k,QPRES) = amrex::max(pnewr, small_pres); 
        qp(i,j,k,QREINT) = renewr - rhoekinr;
//======================= QM ================================================

//Conversion to Conservative 
        rrl = qym(i,j+1,k,QRHO); 
        rul = qym(i,j+1,k,QU); 
        rvl = qym(i,j+1,k,QV); 
        rwl = qym(i,j+1,k,QW);  
        ekinl = 0.5e0*rrl*(rul*rul + rvl*rvl + rwl*rwl); 
        rul *= rrl; 
        rvl *= rrl; 
        rwl *= rrl; 
        rel = qym(i,j+1,k,QREINT) + ekinl; 

// Transverse fluxes 
        rrnewl = rrl - flxrho; 
        runewl = rul - flxu; 
        rvnewl = rvl - flxv;
        rwnewl = rwl - flxw;  
        renewl = rel - flxe; 
               
        qm(i,j+1,k,QRHO) = rrnewl;
        qm(i,j+1,k,QU  ) = runewl/rrnewl; 
        qm(i,j+1,k,QV  ) = rvnewl/rrnewl;
        qm(i,j+1,k,QW  ) = rwnewl/rrnewl;  
        rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwnewl*rwnewl)/rrnewl; 

        qm(i,j+1,k,QREINT) = renewl - rhoekinl;               
        amrex::Real pnewl = qym(i,j+1,k,QPRES) 
                          - cdtdx*(dAup + pav*dAu*(c - 1.e0)); 
        qm(i,j+1,k,QPRES) = amrex::max(pnewl, small_pres); 
}



//================================ Corrects the left and right x states with y flux ====== 
AMREX_GPU_DEVICE 
inline 
void 
PeleC_transy1(const int i, const int j, const int k,
     amrex::Array4<amrex::Real> const& qm, amrex::Array4<amrex::Real> const& qp, 
     amrex::Array4<const amrex::Real> const& qxm, amrex::Array4<const amrex::Real> const& qxp, 
     amrex::Array4<const amrex::Real> const& flxy, 
     amrex::Array4<const amrex::Real> const& qa,  amrex::Array4<const amrex::Real> const& qint,
     const amrex::Real cdtdy) 
{

        amrex::Real compn, rr, rrnew, compo, rwr, c; 
        amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr; 
        amrex::Real runewr, rvnewr, rwnewr, renewr, rhoekinr; 
        amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl, rwnewl, renewl; 
        amrex::Real rhoekinl, rwl;
        amrex::Real pggp, pggm, ugp, ugm;
        //  Local variables to avoid reading from global memory too much 
        amrex::Real dup, pav, du, pnewr, pnewl;
        amrex::Real flxrho, flxu, flxv, flxw, flxe;

        flxrho = cdtdy*(flxy(i,j+1,k,URHO)  - flxy(i,j,k,URHO));
        flxu   = cdtdy*(flxy(i,j+1,k,UMX)   - flxy(i,j,k,UMX)); 
        flxv   = cdtdy*(flxy(i,j+1,k,UMY)   - flxy(i,j,k,UMY)); 
        flxw   = cdtdy*(flxy(i,j+1,k,UMZ)   - flxy(i,j,k,UMZ)); 
        flxe   = cdtdy*(flxy(i,j+1,k,UEDEN) - flxy(i,j,k,UEDEN));  
        c      = qa(i,j,k,QGAMC);  
// ---------------------------  Update Passive Variables ------------------
        for (int ipass = 0; ipass < npassive; ++ipass){
            int n   = EOS::upass_map(ipass); 
            int nqp = EOS::qpass_map(ipass); 
            compn = cdtdy*(flxy(i,j+1,k,n) - flxy(i,j,k,n)); 
            rr = qxp(i,j,k,QRHO); 
            rrnew = rr - flxrho; 
            compo = rr*qxp(i,j,k,nqp) - compn; 
            qp(i,j,k,nqp) = compo/rrnew;  
    
            rr = qxm(i+1,j,k,QRHO); 
            rrnew = rr - flxrho; 
            compo = rr*qxm(i+1,j,k,nqp) - compn; 
            qm(i+1,j,k,nqp) = compo/rrnew;                     
        }                

//------------------------------- Update hydro vars -----------------------
// pressure divergence and such.  
        pggp = qint(i,j+1,k,GDPRES); 
        pggm = qint(i,j,k,GDPRES); 
        ugp  = qint(i,j+1,k,GDV); 
        ugm  = qint(i,j,k,GDV);
    
        dup  = pggp*ugp - pggm*ugm; 
        pav  = 0.5e0*(pggp + pggm); 
        du   = ugp - ugm; 

//========================= QP ===========================================

//Convert to conservative 
        rrr = qxp(i,j,k,QRHO); 
        rur = qxp(i,j,k,QU);             
        rvr = qxp(i,j,k,QV); 
        rwr = qxp(i,j,k,QW); 
        ekinr = 0.5e0*rrr*(rur*rur + rvr*rvr + rwr*rwr);
        rur *= rrr; 
        rvr *= rrr;
        rwr *= rrr;  

        rer = qxp(i,j,k,QREINT) + ekinr; 
// Add transverse predictor 
        rrnewr = rrr - flxrho; 
        runewr = rur - flxu; 
        rvnewr = rvr - flxv; 
        rwnewr = rwr - flxw; 
        renewr = rer - flxe; 

// Convert back to primitive 
        qp(i,j,k,QRHO) = rrnewr;// + hdt*srcr; 
        qp(i,j,k,QU)   = runewr/rrnewr; 
        qp(i,j,k,QV)   = rvnewr/rrnewr; 
        qp(i,j,k,QW)   = rwnewr/rrnewr; 
        rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwnewr*rwnewr )/rrnewr; 
        
        pnewr = qxp(i,j,k,QPRES) - cdtdy*(dup + pav*du*(c - 1.e0)); 
        qp(i,j,k,QPRES) = amrex::max(pnewr, small_pres); 
        qp(i,j,k,QREINT) = renewr - rhoekinr; 

//======================= QM ================================================

//Conversion to Conservative 
        rrl = qxm(i+1,j,k,QRHO); 
        rul = qxm(i+1,j,k,QU); 
        rvl = qxm(i+1,j,k,QV); 
        rwl = qxm(i+1,j,k,QW);  
        ekinl = 0.5e0*rrl*(rul*rul + rvl*rvl + rwl*rwl); 
        rul *= rrl; 
        rvl *= rrl; 
        rwl *= rrl; 
        rel = qxm(i+1,j,k,QREINT) + ekinl; 

// Transverse fluxes 
        rrnewl = rrl - flxrho; 
        runewl = rul - flxu; 
        rvnewl = rvl - flxv; 
        rwnewl = rwl - flxw; 
        renewl = rel - flxe; 
        
        
        
        qm(i+1,j,k,QRHO) = rrnewl; 
        qm(i+1,j,k,QU  ) = runewl/rrnewl; 
        qm(i+1,j,k,QV  ) = rvnewl/rrnewl; 
        qm(i+1,j,k,QW  ) = rwnewl/rrnewl; 
        rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwnewl*rwnewl)/rrnewl; 

        pnewl = qxm(i+1,j,k,QPRES) - cdtdy*(dup + pav*du*(c - 1.e0)); 
        qm(i+1,j,k,QPRES) = amrex::max(pnewl, small_pres); 
        qm(i+1,j,k,QREINT) = renewl - rhoekinl; 
}

/* Transverse Correction for Predicted Z-states, using X-Flux */ 
AMREX_GPU_DEVICE
inline 
void 
PeleC_transx2(const int i, const int j, const int k,
        amrex::Array4<amrex::Real> const& qm        , amrex::Array4<amrex::Real> const& qp,
        amrex::Array4<const amrex::Real> const& qzm , amrex::Array4<const amrex::Real> const& qzp,
        amrex::Array4<const amrex::Real> const& flxx, 
        amrex::Array4<const amrex::Real> const& qa  , amrex::Array4<const amrex::Real> const& qint,
        const amrex::Real cdtdx)
{
        //  Local variables to avoid reading from global memory too much 
        amrex::Real compn, rr, rrnew, compo, rwr, c;
        amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr;
        amrex::Real runewr, rvnewr, rwnewr, renewr, rhoekinr ;
        amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl, rwnewl, renewl;
        amrex::Real rhoekinl, rwl;
        amrex::Real pggp, pggm, ugp, ugm;
        amrex::Real dAup, pav,dAu;
        amrex::Real flxrho, flxu, flxv, flxw, flxe;

        flxrho = cdtdx*(flxx(i+1,j,k,URHO)  - flxx(i,j,k,URHO));
        flxu   = cdtdx*(flxx(i+1,j,k,UMX)   - flxx(i,j,k,UMX));
        flxv   = cdtdx*(flxx(i+1,j,k,UMY)   - flxx(i,j,k,UMY));
        flxw   = cdtdx*(flxx(i+1,j,k,UMZ)   - flxx(i,j,k,UMZ)); 
        flxe   = cdtdx*(flxx(i+1,j,k,UEDEN) - flxx(i,j,k,UEDEN));
        c      = qa(i,j,k,QGAMC);
// ---------------------------  Update Passive Variables ------------------
        for (int ipass = 0; ipass < npassive; ++ipass){
            int n   = EOS::upass_map(ipass);
            int nqp = EOS::qpass_map(ipass);
            compn = cdtdx*(flxx(i+1,j,k,n) - flxx(i,j,k,n));
            rr = qzp(i,j,k,QRHO);
            rrnew = rr - flxrho;
            compo = rr*qzp(i,j,k,nqp) - compn;
            qp(i,j,k,nqp) = compo/rrnew;

            rr = qzm(i,j,k+1,QRHO);
            rrnew = rr - flxrho;
            compo = rr*qzm(i,j,k+1,nqp) - compn;
            qm(i,j,k+1,nqp) = compo/rrnew;
        }
//------------------------------- Update hydro vars -----------------------

        pggp = qint(i+1,j,k,GDPRES); 
        pggm = qint(i,j,k,GDPRES); 
        ugp  = qint(i+1,j,k,GDU); 
        ugm  = qint(i,j,k,GDU); 
    
        dAup = pggp*ugp - pggm*ugm; //
        pav  = 0.5e0*(pggp + pggm); 
        dAu  = ugp - ugm; //
    
//========================= QP ===========================================

//Convert to conservative 
        rrr = qzp(i,j,k,QRHO); 
        rur = qzp(i,j,k,QU);             
        rvr = qzp(i,j,k,QV); 
        rwr = qzp(i,j,k,QW); 
        ekinr = 0.5e0*rrr*(rur*rur + rvr*rvr + rwr*rwr);
        rur *= rrr; 
        rvr *= rrr; 
        rwr *= rrr; 
        
        rer = qzp(i,j,k,QREINT) + ekinr; 
// Add transverse predictor 
        rrnewr = rrr - flxrho; 
        runewr = rur - flxu; 
        rvnewr = rvr - flxv; 
        rwnewr = rwr - flxw; 
        renewr = rer - flxe; 

// Convert back to primitive 
        qp(i,j,k,QRHO) = rrnewr;
        qp(i,j,k,QU)   = runewr/rrnewr; 
        qp(i,j,k,QV)   = rvnewr/rrnewr;
        qp(i,j,k,QW)   = rwnewr/rrnewr;  
        rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwnewr*rwnewr)/rrnewr; 
        amrex::Real pnewr = qzp(i,j,k,QPRES) 
                          - cdtdx*(dAup + pav*dAu*(c - 1.e0));
        qp(i,j,k,QPRES) = amrex::max(pnewr, small_pres); 
        qp(i,j,k,QREINT) = renewr - rhoekinr;
//======================= QM ================================================

//Conversion to Conservative 
        rrl = qzm(i,j,k+1,QRHO); 
        rul = qzm(i,j,k+1,QU); 
        rvl = qzm(i,j,k+1,QV); 
        rwl = qzm(i,j,k+1,QW);  
        ekinl = 0.5e0*rrl*(rul*rul + rvl*rvl + rwl*rwl); 
        rul *= rrl; 
        rvl *= rrl;
        rwl *= rrl;  
        rel = qzm(i,j,k+1,QREINT) + ekinl; 

// Transverse fluxes 
        rrnewl = rrl - flxrho; 
        runewl = rul - flxu; 
        rvnewl = rvl - flxv; 
        rwnewl = rwl - flxw; 
        renewl = rel - flxe; 
               
        qm(i,j,k+1,QRHO) = rrnewl;
        qm(i,j,k+1,QU  ) = runewl/rrnewl; 
        qm(i,j,k+1,QV  ) = rvnewl/rrnewl;
        qm(i,j,k+1,QW  ) = rwnewl/rrnewl;  
        rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwnewl*rwnewl)/rrnewl; 

        qm(i,j,k+1,QREINT) = renewl - rhoekinl;               
        amrex::Real pnewl = qzm(i,j,k+1,QPRES) 
                          - cdtdx*(dAup + pav*dAu*(c - 1.e0)); 
        qm(i,j,k+1,QPRES) = amrex::max(pnewl , small_pres); 
}



//================================ Corrects the left and right z states with y flux ====== 
AMREX_GPU_DEVICE 
inline 
void 
PeleC_transy2(const int i, const int j, const int k,
        amrex::Array4<amrex::Real> const& qm        , amrex::Array4<amrex::Real> const& qp,
        amrex::Array4<const amrex::Real> const& qzm , amrex::Array4<const amrex::Real> const& qzp,
        amrex::Array4<const amrex::Real> const& flxy,
        amrex::Array4<const amrex::Real> const& qa  , amrex::Array4<const amrex::Real> const& qint,
        const amrex::Real cdtdy)
{
        //  Local variables to avoid reading from global memory too much 
        amrex::Real compn, rr, rrnew, compo, rwr, c;
        amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr;
        amrex::Real runewr, rvnewr, rwnewr, renewr, rhoekinr ;
        amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl, rwnewl, renewl;
        amrex::Real rhoekinl, rwl;
        amrex::Real pggp, pggm, ugp, ugm;
        amrex::Real dAup, pav,dAu;
        amrex::Real flxrho, flxu, flxv, flxw, flxe;

        flxrho = cdtdy*(flxy(i,j+1,k,URHO)  - flxy(i,j,k,URHO));
        flxu   = cdtdy*(flxy(i,j+1,k,UMX)   - flxy(i,j,k,UMX));
        flxv   = cdtdy*(flxy(i,j+1,k,UMY)   - flxy(i,j,k,UMY));
        flxw   = cdtdy*(flxy(i,j+1,k,UMZ)   - flxy(i,j,k,UMZ)); 
        flxe   = cdtdy*(flxy(i,j+1,k,UEDEN) - flxy(i,j,k,UEDEN));
        c      = qa(i,j,k,QGAMC);
// ---------------------------  Update Passive Variables ------------------
        for (int ipass = 0; ipass < npassive; ++ipass){
            int n   = EOS::upass_map(ipass);
            int nqp = EOS::qpass_map(ipass);
            compn = cdtdy*(flxy(i,j+1,k,n) - flxy(i,j,k,n));
            rr = qzp(i,j,k,QRHO);
            rrnew = rr - flxrho;
            compo = rr*qzp(i,j,k,nqp) - compn;
            qp(i,j,k,nqp) = compo/rrnew;

            rr = qzm(i,j,k+1,QRHO);
            rrnew = rr - flxrho;
            compo = rr*qzm(i,j,k+1,nqp) - compn;
            qm(i,j,k+1,nqp) = compo/rrnew;
        }
//------------------------------- Update hydro vars -----------------------

        pggp = qint(i,j+1,k,GDPRES); 
        pggm = qint(i,j,k,GDPRES); 
        ugp  = qint(i,j+1,k,GDV); 
        ugm  = qint(i,j,k,GDV); 
    
        dAup = pggp*ugp - pggm*ugm; //
        pav  = 0.5e0*(pggp + pggm); 
        dAu  = ugp - ugm; //
    
//========================= QP ===========================================

//Convert to conservative 
        rrr = qzp(i,j,k,QRHO); 
        rur = qzp(i,j,k,QU);             
        rvr = qzp(i,j,k,QV); 
        rwr = qzp(i,j,k,QW); 
        ekinr = 0.5e0*rrr*(rur*rur + rvr*rvr + rwr*rwr);
        rur *= rrr; 
        rvr *= rrr; 
        rwr *= rrr; 
        
        rer = qzp(i,j,k,QREINT) + ekinr; 
// Add transverse predictor 
        rrnewr = rrr - flxrho; 
        runewr = rur - flxu; 
        rvnewr = rvr - flxv; 
        rwnewr = rwr - flxw; 
        renewr = rer - flxe; 

// Convert back to primitive 
        qp(i,j,k,QRHO) = rrnewr;
        qp(i,j,k,QU)   = runewr/rrnewr; 
        qp(i,j,k,QV)   = rvnewr/rrnewr;
        qp(i,j,k,QW)   = rwnewr/rrnewr;  
        rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwnewr*rwnewr)/rrnewr; 
        amrex::Real pnewr = qzp(i,j,k,QPRES) 
                          - cdtdy*(dAup + pav*dAu*(c - 1.e0));
        qp(i,j,k,QPRES) = amrex::max(pnewr, small_pres); 
        qp(i,j,k,QREINT) = renewr - rhoekinr;
//======================= QM ================================================

//Conversion to Conservative 
        rrl = qzm(i,j,k+1,QRHO); 
        rul = qzm(i,j,k+1,QU); 
        rvl = qzm(i,j,k+1,QV); 
        rwl = qzm(i,j,k+1,QW);  
        ekinl = 0.5e0*rrl*(rul*rul + rvl*rvl + rwl*rwl); 
        rul *= rrl; 
        rvl *= rrl;
        rwl *= rrl;  
        rel = qzm(i,j,k+1,QREINT) + ekinl; 

// Transverse fluxes 
        rrnewl = rrl - flxrho; 
        runewl = rul - flxu; 
        rvnewl = rvl - flxv; 
        rwnewl = rwl - flxw; 
        renewl = rel - flxe; 
               
        qm(i,j,k+1,QRHO) = rrnewl;
        qm(i,j,k+1,QU  ) = runewl/rrnewl; 
        qm(i,j,k+1,QV  ) = rvnewl/rrnewl;
        qm(i,j,k+1,QW  ) = rwnewl/rrnewl;  
        rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwnewl*rwnewl)/rrnewl; 

        qm(i,j,k+1,QREINT) = renewl - rhoekinl;               
        amrex::Real pnewl = qzm(i,j,k+1,QPRES) 
                          - cdtdy*(dAup + pav*dAu*(c - 1.e0)); 
        qm(i,j,k+1,QPRES) = amrex::max(pnewl , small_pres); 
}

//================================ Corrects the left and right x states with z flux ====== 
AMREX_GPU_DEVICE 
inline 
void 
PeleC_transz1(const int i, const int j, const int k,
     amrex::Array4<amrex::Real> const& qm, amrex::Array4<amrex::Real> const& qp, 
     amrex::Array4<const amrex::Real> const& qxm, amrex::Array4<const amrex::Real> const& qxp, 
     amrex::Array4<const amrex::Real> const& flxz, 
     amrex::Array4<const amrex::Real> const& qa,  amrex::Array4<const amrex::Real> const& qint,
     const amrex::Real cdtdz) 
{

        amrex::Real compn, rr, rrnew, compo, rwr, c; 
        amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr; 
        amrex::Real runewr, rvnewr, rwnewr, renewr, rhoekinr; 
        amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl, rwnewl,  renewl; 
        amrex::Real rhoekinl, rwl;
        amrex::Real pggp, pggm, ugp, ugm;
        //  Local variables to avoid reading from global memory too much 
        amrex::Real dup, pav, du, pnewr, pnewl;
        amrex::Real flxrho, flxu, flxv, flxw, flxe; 

        flxrho = cdtdz*(flxz(i,j,k+1,URHO)  - flxz(i,j,k,URHO));
        flxu   = cdtdz*(flxz(i,j,k+1,UMX)   - flxz(i,j,k,UMX)); 
        flxv   = cdtdz*(flxz(i,j,k+1,UMY)   - flxz(i,j,k,UMY)); 
        flxw   = cdtdz*(flxz(i,j,k+1,UMZ)   - flxz(i,j,k,UMZ)); 
        flxe   = cdtdz*(flxz(i,j,k+1,UEDEN) - flxz(i,j,k,UEDEN));  
        c      = qa(i,j,k,QGAMC);  
// ---------------------------  Update Passive Variables ------------------
        for (int ipass = 0; ipass < npassive; ++ipass){
            int n   = EOS::upass_map(ipass); 
            int nqp = EOS::qpass_map(ipass); 
            compn = cdtdz*(flxz(i,j,k+1,n) - flxz(i,j,k,n)); 
            rr = qxp(i,j,k,QRHO); 
            rrnew = rr - flxrho; 
            compo = rr*qxp(i,j,k,nqp) - compn; 
            qp(i,j,k,nqp) = compo/rrnew;  
    
            rr = qxm(i+1,j,k,QRHO); 
            rrnew = rr - flxrho; 
            compo = rr*qxm(i+1,j,k,nqp) - compn; 
            qm(i+1,j,k,nqp) = compo/rrnew;                     
        }                

//------------------------------- Update hydro vars -----------------------
// pressure divergence and such.  
        pggp = qint(i,j,k+1,GDPRES); 
        pggm = qint(i,j,k,GDPRES); 
        ugp  = qint(i,j,k+1,GDW); 
        ugm  = qint(i,j,k,GDW);
    
        dup  = pggp*ugp - pggm*ugm; 
        pav  = 0.5e0*(pggp + pggm); 
        du   = ugp - ugm; 

//========================= QP ===========================================

//Convert to conservative 
        rrr = qxp(i,j,k,QRHO); 
        rur = qxp(i,j,k,QU);             
        rvr = qxp(i,j,k,QV); 
        rwr = qxp(i,j,k,QW); 
        ekinr = 0.5e0*rrr*(rur*rur + rvr*rvr + rwr*rwr);
        rur *= rrr; 
        rvr *= rrr;
        rwr *= rrr;  

        rer = qxp(i,j,k,QREINT) + ekinr; 
// Add transverse predictor 
        rrnewr = rrr - flxrho; 
        runewr = rur - flxu; 
        rvnewr = rvr - flxv; 
        rwnewr = rwr - flxw; 
        renewr = rer - flxe; 

// Convert back to primitive 
        qp(i,j,k,QRHO) = rrnewr;// + hdt*srcr; 
        qp(i,j,k,QU)   = runewr/rrnewr; 
        qp(i,j,k,QV)   = rvnewr/rrnewr; 
        qp(i,j,k,QW)   = rvnewr/rrnewr; 
        rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwnewr*rwnewr )/rrnewr; 
        
        pnewr = qxp(i,j,k,QPRES) - cdtdz*(dup + pav*du*(c - 1.e0)); 
        qp(i,j,k,QPRES) = amrex::max(pnewr, small_pres); 
        qp(i,j,k,QREINT) = renewr - rhoekinr; 

//======================= QM ================================================

//Conversion to Conservative 
        rrl = qxm(i+1,j,k,QRHO); 
        rul = qxm(i+1,j,k,QU); 
        rvl = qxm(i+1,j,k,QV); 
        rwl = qxm(i+1,j,k,QW);  
        ekinl = 0.5e0*rrl*(rul*rul + rvl*rvl + rwl*rwl); 
        rul *= rrl; 
        rvl *= rrl; 
        rwl *= rrl; 
        rel = qxm(i+1,j,k,QREINT) + ekinl; 

// Transverse fluxes 
        rrnewl = rrl - flxrho; 
        runewl = rul - flxu; 
        rvnewl = rvl - flxv; 
        rwnewl = rwl - flxw; 
        renewl = rel - flxe; 
        
        
        
        qm(i+1,j,k,QRHO) = rrnewl; 
        qm(i+1,j,k,QU  ) = runewl/rrnewl; 
        qm(i+1,j,k,QV  ) = rvnewl/rrnewl; 
        qm(i+1,j,k,QW  ) = rwnewl/rrnewl; 
        rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwnewl*rwnewl)/rrnewl; 

        pnewl = qxm(i+1,j,k,QPRES) - cdtdz*(dup + pav*du*(c - 1.e0)); 
        qm(i+1,j,k,QPRES) = amrex::max(pnewl, small_pres); 
        qm(i+1,j,k,QREINT) = renewl - rhoekinl; 
}

 /* Transverse Correction for Predicted Y-states, using Z-Flux */ 
AMREX_GPU_DEVICE
inline 
void 
PeleC_transz2(const int i, const int j, const int k,
        amrex::Array4<amrex::Real> const& qm        , amrex::Array4<amrex::Real> const& qp,
        amrex::Array4<const amrex::Real> const& qym , amrex::Array4<const amrex::Real> const& qyp,
        amrex::Array4<const amrex::Real> const& flxz,
        amrex::Array4<const amrex::Real> const& qa  , amrex::Array4<const amrex::Real> const& qint,
        const amrex::Real cdtdz)
{
        //  Local variables to avoid reading from global memory too much 
        amrex::Real compn, rr, rrnew, compo, rwr, c;
        amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr;
        amrex::Real runewr, rvnewr, rwnewr, renewr, rhoekinr ;
        amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl,rwnewl, renewl;
        amrex::Real rhoekinl, rwl;
        amrex::Real pggp, pggm, ugp, ugm;
        amrex::Real dAup, pav,dAu;
        amrex::Real flxrho, flxu, flxv, flxw, flxe;

        flxrho = cdtdz*(flxz(i,j,k+1,URHO)  - flxz(i,j,k,URHO));
        flxu   = cdtdz*(flxz(i,j,k+1,UMX)   - flxz(i,j,k,UMX)); 
        flxv   = cdtdz*(flxz(i,j,k+1,UMY)   - flxz(i,j,k,UMY)); 
        flxw   = cdtdz*(flxz(i,j,k+1,UMZ)   - flxz(i,j,k,UMZ)); 
        flxe   = cdtdz*(flxz(i,j,k+1,UEDEN) - flxz(i,j,k,UEDEN));  
        c      = qa(i,j,k,QGAMC);
// ---------------------------  Update Passive Variables ------------------
        for (int ipass = 0; ipass < npassive; ++ipass){
            int n   = EOS::upass_map(ipass);
            int nqp = EOS::qpass_map(ipass);
            compn = cdtdz*(flxz(i,j,k+1,n) - flxz(i,j,k,n));
            rr = qyp(i,j,k,QRHO);
            rrnew = rr - flxrho;
            compo = rr*qyp(i,j,k,nqp) - compn;
            qp(i,j,k,nqp) = compo/rrnew;

            rr = qym(i,j+1,k,QRHO);
            rrnew = rr - flxrho;
            compo = rr*qym(i,j+1,k,nqp) - compn;
            qm(i,j+1,k,nqp) = compo/rrnew;
        }
//------------------------------- Update hydro vars -----------------------

        pggp = qint(i,j,k+1,GDPRES); 
        pggm = qint(i,j,k,GDPRES); 
        ugp  = qint(i,j,k+1,GDW); 
        ugm  = qint(i,j,k,GDW); 
    
        dAup = pggp*ugp - pggm*ugm; //
        pav  = 0.5e0*(pggp + pggm); 
        dAu  = ugp - ugm; //
    
//========================= QP ===========================================

//Convert to conservative 
        rrr = qyp(i,j,k,QRHO); 
        rur = qyp(i,j,k,QU);             
        rvr = qyp(i,j,k,QV); 
        rwr = qyp(i,j,k,QW); 
        ekinr = 0.5e0*rrr*(rur*rur + rvr*rvr + rwr*rwr);
        rur *= rrr; 
        rvr *= rrr; 
        rwr *= rrr; 
        rer = qyp(i,j,k,QREINT) + ekinr; 
// Add transverse predictor 
        rrnewr = rrr - flxrho; 
        runewr = rur - flxu; 
        rvnewr = rvr - flxv;
        rwnewr = rwr - flxw;  
        renewr = rer - flxe; 

// Convert back to primitive 
        qp(i,j,k,QRHO) = rrnewr;
        qp(i,j,k,QU)   = runewr/rrnewr; 
        qp(i,j,k,QV)   = rvnewr/rrnewr;
        qp(i,j,k,QW)   = rwnewr/rrnewr;  
        rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwnewr*rwnewr)/rrnewr; 
        amrex::Real pnewr = qyp(i,j,k,QPRES) 
                          - cdtdz*(dAup + pav*dAu*(c - 1.e0));
        qp(i,j,k,QPRES) = amrex::max(pnewr, small_pres); 
        qp(i,j,k,QREINT) = renewr - rhoekinr;
//======================= QM ================================================

//Conversion to Conservative 
        rrl = qym(i,j+1,k,QRHO); 
        rul = qym(i,j+1,k,QU); 
        rvl = qym(i,j+1,k,QV); 
        rwl = qym(i,j+1,k,QW);  
        ekinl = 0.5e0*rrl*(rul*rul + rvl*rvl + rwl*rwl); 
        rul *= rrl; 
        rvl *= rrl; 
        rwl *= rrl; 
        rel = qym(i,j+1,k,QREINT) + ekinl; 

// Transverse fluxes 
        rrnewl = rrl - flxrho; 
        runewl = rul - flxu; 
        rvnewl = rvl - flxv;
        rwnewl = rwl - flxw;  
        renewl = rel - flxe; 
               
        qm(i,j+1,k,QRHO) = rrnewl;
        qm(i,j+1,k,QU  ) = runewl/rrnewl; 
        qm(i,j+1,k,QV  ) = rvnewl/rrnewl;
        qm(i,j+1,k,QW  ) = rwnewl/rrnewl;  
        rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwnewl*rwnewl)/rrnewl; 

        qm(i,j+1,k,QREINT) = renewl - rhoekinl;               
        amrex::Real pnewl = qym(i,j+1,k,QPRES) 
                          - cdtdz*(dAup + pav*dAu*(c - 1.e0)); 
        qm(i,j+1,k,QPRES) = amrex::max(pnewl, small_pres); 
}


//Z corrected from xy 
AMREX_GPU_DEVICE
inline 
void 
PeleC_transxy(const int i, const int j, const int k, 
         amrex::Array4<amrex::Real> const &qm        , amrex::Array4<amrex::Real> const &qp, 
         amrex::Array4<const amrex::Real> const& qzm , amrex::Array4<const amrex::Real> const& qzp,
         amrex::Array4<const amrex::Real> const& flxx, amrex::Array4<const amrex::Real> const& flxy, 
         amrex::Array4<const amrex::Real> const& qx  , amrex::Array4<const amrex::Real> const& qy,
         amrex::Array4<const amrex::Real> const& qa  , amrex::Array4<const amrex::Real> const& srcq,
         const amrex::Real hdt, const amrex::Real cdtdx, 
         const amrex::Real cdtdy)
{
        amrex::Real compn, rr, rrnew, compo, rwr, c;        
        amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr;
        amrex::Real runewr, rvnewr, rwnewr, renewr, rhoekinr ;
        amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl,rwnewl, renewl;
        amrex::Real rhoekinl, rwl;
        amrex::Real pggpx, pggmx, ugpx, ugmx;
        amrex::Real dAupx, pavx, dAux;
        amrex::Real pggpy, pggmy, ugpy, ugmy;
        amrex::Real dAupy, pavy, dAuy;

        amrex::Real flxrho, flxu, flxv, flxw, flxe;
        amrex::Real srcpass, srcrho, srcu, srcv, srcw, srce, srcp; 

        flxrho = cdtdx*(flxx(i+1,j,k,URHO)  - flxx(i,j,k,URHO)) 
               + cdtdy*(flxy(i,j+1,k,URHO)  - flxy(i,j,k,URHO)); 
        flxu   = cdtdx*(flxx(i+1,j,k,UMX)   - flxx(i,j,k,UMX)) 
               + cdtdy*(flxy(i,j+1,k,UMX)   - flxy(i,j,k,UMX)); 
        flxv   = cdtdx*(flxx(i+1,j,k,UMY)   - flxx(i,j,k,UMY)) 
               + cdtdy*(flxy(i,j+1,k,UMY)   - flxy(i,j,k,UMY)); 
        flxw   = cdtdx*(flxx(i+1,j,k,UMZ)   - flxx(i,j,k,UMZ)) 
               + cdtdy*(flxy(i,j+1,k,UMZ)   - flxy(i,j,k,UMZ)); 
        flxe   = cdtdx*(flxx(i+1,j,k,UEDEN) - flxx(i,j,k,UEDEN)) 
               + cdtdy*(flxy(i,j+1,k,UEDEN) - flxy(i,j,k,UEDEN)); 
        c      = qa(i,j,k,QGAMC);
        srcrho = srcq(i,j,k,QRHO);  
        srcu   = srcq(i,j,k,QU); 
        srcv   = srcq(i,j,k,QV); 
        srcw   = srcq(i,j,k,QW); 
        srce   = srcq(i,j,k,QREINT);  
        srcp   = srcq(i,j,k,QPRES); 
// ---------------------------  Update Passive Variables ------------------
        rrr = qzp(i,j,k  ,QRHO); 
        rrl = qzm(i,j,k+1,QRHO); 
        rrnewl = rrl - flxrho; 
        rrnewr = rrr - flxrho; 
        for (int ipass = 0; ipass < npassive; ++ipass){
            int n   = EOS::upass_map(ipass);
            int nqp = EOS::qpass_map(ipass);
            srcpass = srcq(i,j,k,nqp); 
            compn = cdtdx*(flxx(i+1,j,k,n) - flxx(i,j,k,n))
                  + cdtdy*(flxy(i,j+1,k,n) - flxy(i,j,k,n));
            //qp
            compo = rrr*qzp(i,j,k,nqp) - compn;
            qp(i,j,k,nqp) = compo/rrnewr + hdt*srcpass;          
            //qm
            compo = rrl*qzm(i,j,k+1,nqp) - compn;
            qm(i,j,k+1,nqp) = compo/rrnewl + hdt*srcpass;
        }

 //------------------------------- Update hydro vars -----------------------

        pggpx = qx(i+1,j,k,GDPRES); 
        pggmx = qx(i  ,j,k,GDPRES); 
        ugpx  = qx(i+1,j,k,GDU); 
        ugmx  = qx(i  ,j,k,GDU); 
    
        dAupx = pggpx*ugpx - pggmx*ugmx; //
        pavx  = 0.5e0*(pggpx + pggmx); 
        dAux  = ugpx - ugmx; //
       
        pggpy = qy(i,j+1,k,GDPRES); 
        pggmy = qy(i,j,k,GDPRES); 
        ugpy  = qy(i,j+1,k,GDV); 
        ugmy  = qy(i,j,k,GDV); 
    
        dAupy = pggpy*ugpy - pggmy*ugmy; //
        pavy  = 0.5e0*(pggpy + pggmy); 
        dAuy  = ugpy - ugmy; //
        amrex::Real pxnew = cdtdx*(dAupx + pavx*dAux*(c - 1.e0)); 
        amrex::Real pynew = cdtdy*(dAupy + pavy*dAuy*(c - 1.e0)); 

// ---------------------- qp state ------------------------------------- 
        rur = rrr*qzp(i,j,k,QU); 
        rvr = rrr*qzp(i,j,k,QV); 
        rwr = rrr*qzp(i,j,k,QW); 
        ekinr = 0.5*(rur*rur + rvr*rvr + rwr*rwr)/rrr; 
        rer = qzp(i,j,k,QREINT) + ekinr; 

        runewr = rur - flxu; 
        rvnewr = rvr - flxv; 
        rwnewr = rwr - flxw; 
        renewr = rer - flxe; 

        qp(i,j,k,QRHO) = rrnewr + hdt*srcrho; 
        qp(i,j,k,QU  ) = runewr/rrnewr + hdt*srcu; 
        qp(i,j,k,QV  ) = rvnewr/rrnewr + hdt*srcv; 
        qp(i,j,k,QW  ) = rwnewr/rrnewr + hdt*srcw;
        rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwnewr*rwnewr)/rrnewr; 
        qp(i,j,k,QREINT) = renewr - rhoekinr + hdt*srce;
        amrex::Real temppres = qzp(i,j,k,QPRES) - pxnew - pynew + hdt*srcp;  
        qp(i,j,k,QPRES) = amrex::max(temppres, small_pres);  

// --------------------- qm state ----------------------------------- 
        rul = rrl*qzm(i,j,k+1,QU); 
        rvl = rrl*qzm(i,j,k+1,QV); 
        rwl = rrl*qzm(i,j,k+1,QW); 
        ekinl = 0.5*(rul*rul + rvl*rvl + rwl*rwl)/rrl;
        rel = qzm(i,j,k+1,QREINT) + ekinl; 

        runewl = rul - flxu; 
        rvnewl = rvl - flxv; 
        rwnewl = rwl - flxw; 
        renewl = rel - flxe; 
        
        qm(i,j,k+1,QRHO) = rrnewl + hdt*srcrho; 
        qm(i,j,k+1,QU  ) = runewl/rrnewl + hdt*srcu; 
        qm(i,j,k+1,QV  ) = rvnewl/rrnewl + hdt*srcv; 
        qm(i,j,k+1,QW  ) = rwnewl/rrnewl + hdt*srcw; 
        rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwnewl*rwnewl)/rrnewl; 
        qm(i,j,k+1,QREINT) = renewl - rhoekinl + hdt*srce; 
        temppres = qzm(i,j,k+1,QPRES) - pxnew - pynew + hdt*srcp; 
        qm(i,j,k+1,QPRES) = amrex::max(temppres, small_pres);  
}

//Y corrected by x and z 
AMREX_GPU_DEVICE
inline 
void 
PeleC_transxz(const int i, const int j, const int k, 
         amrex::Array4<amrex::Real> const &qm        , amrex::Array4<amrex::Real> const &qp, 
         amrex::Array4<const amrex::Real> const& qym , amrex::Array4<const amrex::Real> const& qyp,
         amrex::Array4<const amrex::Real> const& flxx, amrex::Array4<const amrex::Real> const& flxz, 
         amrex::Array4<const amrex::Real> const& qx  , amrex::Array4<const amrex::Real> const& qz, 
         amrex::Array4<const amrex::Real> const& qa  , amrex::Array4<const amrex::Real> const& srcq, 
         const amrex::Real hdt, const amrex::Real cdtdx, 
         const amrex::Real cdtdz)
{
        amrex::Real compn, rr, rrnew, compo, rwr, c;
        amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr;
        amrex::Real runewr, rvnewr, rwnewr, renewr, rhoekinr ;
        amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl,rwnewl, renewl;
        amrex::Real rhoekinl, rwl;
        amrex::Real pggpx, pggmx, ugpx, ugmx;
        amrex::Real dAupx, pavx, dAux;
        amrex::Real pggpz, pggmz, ugpz, ugmz;
        amrex::Real dAupz, pavz, dAuz;

        amrex::Real flxrho, flxu, flxv, flxw, flxe;
        amrex::Real srcpass, srcrho, srcu, srcv, srcw, srce, srcp; 

        flxrho = cdtdx*(flxx(i+1,j,k,URHO)  - flxx(i,j,k,URHO)) 
               + cdtdz*(flxz(i,j,k+1,URHO)  - flxz(i,j,k,URHO)); 
        flxu   = cdtdx*(flxx(i+1,j,k,UMX)   - flxx(i,j,k,UMX)) 
               + cdtdz*(flxz(i,j,k+1,UMX)   - flxz(i,j,k,UMX)); 
        flxv   = cdtdx*(flxx(i+1,j,k,UMY)   - flxx(i,j,k,UMY)) 
               + cdtdz*(flxz(i,j,k+1,UMY)   - flxz(i,j,k,UMY)); 
        flxw   = cdtdx*(flxx(i+1,j,k,UMZ)   - flxx(i,j,k,UMZ)) 
               + cdtdz*(flxz(i,j,k+1,UMZ)   - flxz(i,j,k,UMZ)); 
        flxe   = cdtdx*(flxx(i+1,j,k,UEDEN) - flxx(i,j,k,UEDEN)) 
               + cdtdz*(flxz(i,j,k+1,UEDEN) - flxz(i,j,k,UEDEN)); 
        c      = qa(i,j,k,QGAMC);
        srcrho = srcq(i,j,k,QRHO);  
        srcu   = srcq(i,j,k,QU); 
        srcv   = srcq(i,j,k,QV); 
        srcw   = srcq(i,j,k,QW); 
        srce   = srcq(i,j,k,QREINT);  
        srcp   = srcq(i,j,k,QPRES); 
// ---------------------------  Update Passive Variables ------------------
        rrr = qyp(i,j,k  ,QRHO); 
        rrl = qym(i,j+1,k,QRHO); 
        rrnewl = rrl - flxrho; 
        rrnewr = rrr - flxrho; 
        for (int ipass = 0; ipass < npassive; ++ipass){
            int n   = EOS::upass_map(ipass);
            int nqp = EOS::qpass_map(ipass);
            srcpass = srcq(i,j,k,nqp); 
            compn = cdtdx*(flxx(i+1,j,k,n) - flxx(i,j,k,n))
                  + cdtdz*(flxz(i,j,k+1,n) - flxz(i,j,k,n));
            //qp
            compo = rrr*qyp(i,j,k,nqp) - compn;
            qp(i,j,k,nqp) = compo/rrnewr + hdt*srcpass;          
            //qm
            compo = rrl*qym(i,j+1,k,nqp) - compn;
            qm(i,j+1,k,nqp) = compo/rrnewl + hdt*srcpass;
        }

 //------------------------------- Update hydro vars -----------------------

        pggpx = qx(i+1,j,k,GDPRES); 
        pggmx = qx(i,j,k,GDPRES); 
        ugpx  = qx(i+1,j,k,GDU); 
        ugmx  = qx(i,j,k,GDU); 
    
        dAupx = pggpx*ugpx - pggmx*ugmx; //
        pavx  = 0.5e0*(pggpx + pggmx); 
        dAux  = ugpx - ugmx; //
       
        pggpz = qz(i,j,k+1,GDPRES); 
        pggmz = qz(i,j,k,GDPRES); 
        ugpz  = qz(i,j,k+1,GDW); 
        ugmz  = qz(i,j,k,GDW); 
    
        dAupz = pggpz*ugpz - pggmz*ugmz; //
        pavz  = 0.5e0*(pggpz + pggmz); 
        dAuz  = ugpz - ugmz; //
        amrex::Real pxnew = cdtdx*(dAupx + pavx*dAux*(c - 1.e0)); 
        amrex::Real pznew = cdtdz*(dAupz + pavz*dAuz*(c - 1.e0)); 

// ---------------------- qp state ------------------------------------- 
        rur = rrr*qyp(i,j,k,QU); 
        rvr = rrr*qyp(i,j,k,QV); 
        rwr = rrr*qyp(i,j,k,QW); 
        ekinr = 0.5*(rur*rur + rvr*rvr + rwr*rwr)/rrr; 
        rer = qyp(i,j,k,QREINT) + ekinr; 

        runewr = rur - flxu; 
        rvnewr = rvr - flxv; 
        rwnewr = rwr - flxw; 
        renewr = rer - flxe; 

        qp(i,j,k,QRHO) = rrnewr + hdt*srcrho; 
        qp(i,j,k,QU  ) = runewr/rrnewr + hdt*srcu; 
        qp(i,j,k,QV  ) = rvnewr/rrnewr + hdt*srcv; 
        qp(i,j,k,QW  ) = rwnewr/rrnewr + hdt*srcw;
        rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwnewr*rwnewr)/rrnewr; 
        qp(i,j,k,QREINT) = renewr - rhoekinr + hdt*srce;
        amrex::Real temppres = qyp(i,j,k,QPRES) - pxnew - pznew + hdt*srcp;  
        qp(i,j,k,QPRES) = amrex::max(temppres, small_pres);  

// --------------------- qm state ----------------------------------- 
        rul = rrl*qym(i,j+1,k,QU); 
        rvl = rrl*qym(i,j+1,k,QV); 
        rwl = rrl*qym(i,j+1,k,QW); 
        ekinl = 0.5*(rul*rul + rvl*rvl + rwl*rwl)/rrl; 
        rel = qym(i,j+1,k,QREINT) + ekinl; 

        runewl = rul - flxu; 
        rvnewl = rvl - flxv; 
        rwnewl = rwl - flxw; 
        renewl = rel - flxe; 
        
        qm(i,j+1,k,QRHO) = rrnewl + hdt*srcrho; 
        qm(i,j+1,k,QU  ) = runewl/rrnewl + hdt*srcu; 
        qm(i,j+1,k,QV  ) = rvnewl/rrnewl + hdt*srcv; 
        qm(i,j+1,k,QW  ) = rwnewl/rrnewl + hdt*srcw; 
        rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwnewl*rwnewl)/rrnewl; 
        qm(i,j+1,k,QREINT) = renewl - rhoekinl + hdt*srce; 
        temppres = qym(i,j+1,k,QPRES) - pxnew - pznew + hdt*srcp; 
        qm(i,j+1,k,QPRES) = amrex::max(temppres, small_pres);  
}


//x corrected by y and z 
AMREX_GPU_DEVICE
inline 
void 
PeleC_transyz(const int i, const int j, const int k, 
         amrex::Array4<amrex::Real> const &qm        , amrex::Array4<amrex::Real> const &qp, 
         amrex::Array4<const amrex::Real> const& qxm , amrex::Array4<const amrex::Real> const& qxp,
         amrex::Array4<const amrex::Real> const& flxy, amrex::Array4<const amrex::Real> const& flxz, 
         amrex::Array4<const amrex::Real> const& qy  , amrex::Array4<const amrex::Real> const& qz, 
         amrex::Array4<const amrex::Real> const& qa  , amrex::Array4<const amrex::Real> const& srcq, 
         const amrex::Real hdt, const amrex::Real cdtdy, 
         const amrex::Real cdtdz)
{
        amrex::Real compn, rr, rrnew, compo, rwr, c;
        amrex::Real rrr, rur, rvr, ekinr, rer, rrnewr;
        amrex::Real runewr, rvnewr, rwnewr, renewr, rhoekinr ;
        amrex::Real rrl, rul, rvl, ekinl, rel, rrnewl, runewl, rvnewl,rwnewl, renewl;
        amrex::Real rhoekinl, rwl;
        amrex::Real pggpz, pggmz, ugpz, ugmz;
        amrex::Real dAupz, pavz, dAuz;
        amrex::Real pggpy, pggmy, ugpy, ugmy;
        amrex::Real dAupy, pavy, dAuy;

        amrex::Real flxrho, flxu, flxv, flxw, flxe;
        amrex::Real srcpass, srcrho, srcu, srcv, srcw, srce, srcp; 

        flxrho = cdtdy*(flxy(i,j+1,k,URHO)  - flxy(i,j,k,URHO)) 
               + cdtdz*(flxz(i,j,k+1,URHO)  - flxz(i,j,k,URHO)); 
        flxu   = cdtdy*(flxy(i,j+1,k,UMX)   - flxy(i,j,k,UMX)) 
               + cdtdz*(flxz(i,j,k+1,UMX)   - flxz(i,j,k,UMX)); 
        flxv   = cdtdy*(flxy(i,j+1,k,UMY)   - flxy(i,j,k,UMY)) 
               + cdtdz*(flxz(i,j,k+1,UMY)   - flxz(i,j,k,UMY)); 
        flxw   = cdtdy*(flxy(i,j+1,k,UMZ)   - flxy(i,j,k,UMZ)) 
               + cdtdz*(flxz(i,j,k+1,UMZ)   - flxz(i,j,k,UMZ)); 
        flxe   = cdtdy*(flxy(i,j+1,k,UEDEN) - flxy(i,j,k,UEDEN)) 
               + cdtdz*(flxz(i,j,k+1,UEDEN) - flxz(i,j,k,UEDEN)); 
        c      = qa(i,j,k,QGAMC);
        srcrho = srcq(i,j,k,QRHO);  
        srcu   = srcq(i,j,k,QU); 
        srcv   = srcq(i,j,k,QV); 
        srcw   = srcq(i,j,k,QW); 
        srce   = srcq(i,j,k,QREINT);  
        srcp   = srcq(i,j,k,QPRES); 
// ---------------------------  Update Passive Variables ------------------
        rrr = qxp(i  ,j,k,QRHO); 
        rrl = qxm(i+1,j,k,QRHO); 
        rrnewl = rrl - flxrho; 
        rrnewr = rrr - flxrho;

        for (int ipass = 0; ipass < npassive; ++ipass){
            int n   = EOS::upass_map(ipass);
            int nqp = EOS::qpass_map(ipass);
            srcpass = srcq(i,j,k,nqp); 
            compn = cdtdy*(flxy(i,j+1,k,n) - flxy(i,j,k,n))
                  + cdtdz*(flxz(i,j,k+1,n) - flxz(i,j,k,n));
            //qp
            compo = rrr*qxp(i,j,k,nqp) - compn;
            qp(i,j,k,nqp) = compo/rrnewr + hdt*srcpass;          
            //qm
            compo = rrl*qxm(i+1,j,k,nqp) - compn;
            qm(i+1,j,k,nqp) = compo/rrnewl + hdt*srcpass;
        }

 //------------------------------- Update hydro vars -----------------------

        pggpz = qz(i,j,k+1,GDPRES); 
        pggmz = qz(i,j,k,GDPRES); 
        ugpz  = qz(i,j,k+1,GDW); 
        ugmz  = qz(i,j,k,GDW); 
    
        dAupz = pggpz*ugpz - pggmz*ugmz; //
        pavz  = 0.5e0*(pggpz + pggmz); 
        dAuz  = ugpz - ugmz; //
       
        pggpy = qy(i,j+1,k,GDPRES); 
        pggmy = qy(i,j,k,GDPRES); 
        ugpy  = qy(i,j+1,k,GDV); 
        ugmy  = qy(i,j,k,GDV); 
    
        dAupy = pggpy*ugpy - pggmy*ugmy; //
        pavy  = 0.5e0*(pggpy + pggmy); 
        dAuy  = ugpy - ugmy; //
        amrex::Real pznew = cdtdz*(dAupz + pavz*dAuz*(c - 1.e0)); 
        amrex::Real pynew = cdtdy*(dAupy + pavy*dAuy*(c - 1.e0)); 

// ---------------------- qp state ------------------------------------- 
        rur = rrr*qxp(i,j,k,QU); 
        rvr = rrr*qxp(i,j,k,QV); 
        rwr = rrr*qxp(i,j,k,QW); 
        ekinr = 0.5*(rur*rur + rvr*rvr + rwr*rwr)/rrr; 
        rer = qxp(i,j,k,QREINT) + ekinr; 

        runewr = rur - flxu; 
        rvnewr = rvr - flxv; 
        rwnewr = rwr - flxw; 
        renewr = rer - flxe; 

        qp(i,j,k,QRHO) = rrnewr + hdt*srcrho; 
        qp(i,j,k,QU  ) = runewr/rrnewr + hdt*srcu; 
        qp(i,j,k,QV  ) = rvnewr/rrnewr + hdt*srcv; 
        qp(i,j,k,QW  ) = rwnewr/rrnewr + hdt*srcw;
        rhoekinr = 0.5e0*(runewr*runewr + rvnewr*rvnewr + rwnewr*rwnewr)/rrnewr; 
        qp(i,j,k,QREINT) = renewr - rhoekinr + hdt*srce;
        amrex::Real temppres = qxp(i,j,k,QPRES) - pznew - pynew + hdt*srcp;  
        qp(i,j,k,QPRES) = amrex::max(temppres, small_pres);  

// --------------------- qm state ----------------------------------- 
        rul = rrl*qxm(i+1,j,k,QU); 
        rvl = rrl*qxm(i+1,j,k,QV); 
        rwl = rrl*qxm(i+1,j,k,QW); 
        ekinl = 0.5*(rul*rul + rvl*rvl + rwl*rwl)/rrl; 
        rel = qxm(i+1,j,k,QREINT) + ekinl; 

        runewl = rul - flxu; 
        rvnewl = rvl - flxv; 
        rwnewl = rwl - flxw; 
        renewl = rel - flxe; 
        
        qm(i+1,j,k,QRHO) = rrnewl + hdt*srcrho; 
        qm(i+1,j,k,QU  ) = runewl/rrnewl + hdt*srcu; 
        qm(i+1,j,k,QV  ) = rvnewl/rrnewl + hdt*srcv; 
        qm(i+1,j,k,QW  ) = rwnewl/rrnewl + hdt*srcw; 
        rhoekinl = 0.5e0*(runewl*runewl + rvnewl*rvnewl + rwnewl*rwnewl)/rrnewl; 
        qm(i+1,j,k,QREINT) = renewl - rhoekinl + hdt*srce; 
        temppres = qxm(i+1,j,k,QPRES) - pznew - pynew + hdt*srcp; 
        qm(i+1,j,k,QPRES) = amrex::max(temppres, small_pres);  
}

//Use interface states from Riemann solver for pdivu. 
AMREX_GPU_DEVICE 
inline
void 
PeleC_pdivu(const int i, const int j, const int k, 
    amrex::Array4<amrex::Real> const& pdivu, amrex::Array4<const amrex::Real> const& q1, 
    amrex::Array4<const amrex::Real> const& q2, amrex::Array4<const amrex::Real> const& q3,
    amrex::Array4<const amrex::Real> const& a1, amrex::Array4<const amrex::Real> const& a2,
    amrex::Array4<const amrex::Real> const& a3, amrex::Array4<const amrex::Real> const& vol)
{
        pdivu(i,j,k) = 0.5e0*((q1(i+1,j,k,GDPRES) + q1(i,j,k,GDPRES))
                     * (q1(i+1,j,k,GDU)*a1(i+1,j,k) - q1(i,j,k,GDU)*a1(i,j,k))
                     + (q2(i,j+1,k,GDPRES) + q2(i,j,k,GDPRES))
                     * (q2(i,j+1,k,GDV)*a2(i,j+1,k) - q2(i,j,k,GDV)*a2(i,j,k))
                     + (q3(i,j,k+1,GDPRES) + q3(i,j,k,GDPRES))
                     * (q3(i,j,k+1,GDW)*a3(i,j,k+1) - q3(i,j,k,GDW)*a3(i,j,k)))
                     / vol(i,j,k); 
}

// ====================================== Consup Support Functions =======================================
// Designed for CPU or GPU 
AMREX_GPU_DEVICE
inline 
void PeleC_artif_visc(const int i, const int j, const int k,
                      amrex::Array4<amrex::Real> const& flx,
                      amrex::Array4<const amrex::Real> const& div, 
                      amrex::Array4<const amrex::Real> const &u,
                      amrex::Real const dx, amrex::Real const difmag,  const int cdir)
{
    amrex::Real div1;
    if(cdir==0){ //X direction 
        div1 = 0.25e0*(div(i,j,k) + div(i,j+1,k) + div(i,j,k+1) + div(i,j+1,k+1)); 
        div1 = difmag*amrex::min(0.0, div1); 

#pragma unroll
        for(int n=0; n < NVAR; ++n){
            if(n != UTEMP){
// Passing in difmag TODO not pass in. 
                flx(i,j,k,n) += dx*div1*(u(i,j,k,n) - u(i-1,j,k,n)); 
            }
        }
    }
    else if(cdir==1){ //Y direction 
        div1 = 0.25e0*(div(i,j,k) + div(i+1,j,k) + div(i,j,k+1) + div(i+1,j,k+1)); 
        div1 = difmag*amrex::min(0.0, div1); 
#pragma unroll 
        for(int n=0; n < NVAR; ++n){
            if(n  != UTEMP){
                flx(i,j,k,n) += dx*div1*(u(i,j,k,n) - u(i,j-1,k,n)); //Here dx is dy 
            }
        }
    }
    else{ // Z direction
        div1 = 0.25e0*(div(i,j,k) + div(i+1,j,k) + div(i,j+1,k) + div(i+1,j+1,k)); 
        div1 = difmag*amrex::min(0.0, div1); 
#pragma unroll 
        for(int n=0; n < NVAR; ++n){
            if(n  != UTEMP){
                flx(i,j,k,n) += dx*div1*(u(i,j,k,n) - u(i,j,k-1,n)); //Here dx is dz
            }
        }
    }
    flx(i,j,k,UTEMP) = 0.e0; 
 
}




// =============================== Host Functions! ========================================
void PeleC_umeth_3D(amrex::Box const& bx, const int* bclo, const int* bchi, 
           const int* domlo, const int* domhi, 
           amrex::Array4<const amrex::Real> const &q, 
           amrex::Array4<const amrex::Real> const& qaux,
           amrex::Array4<const amrex::Real> const& srcQ,// amrex::IArrayBox const& bcMask,
           amrex::Array4<amrex::Real> const& flx1, amrex::Array4<amrex::Real> const& flx2,
           amrex::Array4<amrex::Real> const& flx3,// amrex::Array4<const amrex::Real> const& dloga,
           amrex::Array4<amrex::Real> const& q1, amrex::Array4< amrex::Real> const& q2, 
           amrex::Array4<amrex::Real> const& q3, amrex::Array4<const amrex::Real> const& a1, 
           amrex::Array4<const amrex::Real> const& a2, amrex::Array4<const amrex::Real> const& a3, 
           amrex::Array4<amrex::Real> const& pdivu, amrex::Array4<const amrex::Real> const& vol,
           const amrex::Real *del, const amrex::Real dt); 

#endif
