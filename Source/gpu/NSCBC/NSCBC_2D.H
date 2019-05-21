#ifndef PELEC_NSCBC_2D_H
#define PELEC_NSCBC_2D_H
#include <PeleC_index_macros.H> 
#include <PeleC_EOS.H> 


/*
    This header file contains inlined device functions and declarations for host functions for 2D NSCBC. 
    While the NSCBC routines do not compute on much data, it is better to keep the data on the GPU.
    Eventhough these functions would probably be faster on the CPU, the data migration costs outweigh the 
    loss in speed by using the GPU. 
*/ 

AMREX_GPU_DEVICE 
inline void PeleC_normal_deriv(const int i, const int j, const int k, const int isgn, 
                               const amrex::Real delta, amrex::Real &dp, amrex::Real &du, 
                               amrex::Real &dv, amrex::Real &drho, 
                               const amrex::Array4<const amrex::Real> &q) 
{
    //Computes Second Order derivatives for normal direction derivatives. 
    amrex::Real dinv = 1.e0/delta; 
    amrex::Real threehalves = 3.e0/2.e0; //multiplication is cheaper than division.  
    if(idir == 0){
        if(isgn == 1){ 
            dp = ((-threehalves)*q(i,j,k,QPRES) + 2.e0*q(i+1,j,k,QPRES) - 0.5e0*q(i+2,j,k, QPRES))*dinv; 
            du = ((-threehalves)*q(i,j,k,QU) + 2.e0*q(i+1,j,k,QU) - 0.5e0*q(i+2,j,k,QU))*dinv;
            dv = ((-threehalves)*q(i,j,k,QV) + 2.e0*q(i+1,j,k,QV) - 0.5e0*q(i+2,j,k,QV))*dinv; 
            drho = ((-threehalves)*q(i,j,k,QRHO) + 2.e0*q(i+1,j,k,QRHO) - 0.5e0*q(i+2,j,k,QRHO))*dinv; 
        }
        else{
            dp = ((threehalves)*q(i,j,k,QPRES) - 2.e0*q(i-1,j,k,QPRES) + 0.5e0*q(i-2,j,k,QPRES))*dinv; 
            du = ((threehalves)*q(i,j,k,QU) - 2.e0*q(i-1,j,k,QU) + 0.5e0*q(i-2,j,k,QU))*dinv;
            dv = ((threehalves)*q(i,j,k,QV) - 2.e0*q(i-1,j,k,QV) + 0.5e0*q(i-2,j,k,QV))*dinv; 
            drho = ((threehalves)*q(i,j,k,QRHO) - 2.e0*q(i-1,j,k,QRHO) + 0.5e0*q(i-2,j,k,QRHO))*dinv; 
        }
    }
    else if(idir == 1){
        if(isgn == 1){ 
            dp = ((-threehalves)*q(i,j,k,QPRES) + 2.e0*q(i,j+1,k,QPRES) - 0.5e0*q(i,j+2,k, QPRES))*dinv; 
            du = ((-threehalves)*q(i,j,k,QU) + 2.e0*q(i,j+1,k,QU) - 0.5e0*q(i,j+2,k,QU))*dinv;
            dv = ((-threehalves)*q(i,j,k,QV) + 2.e0*q(i,j+1,k,QV) - 0.5e0*q(i,j+2,k,QV))*dinv; 
            drho = ((-threehalves)*q(i,j,k,QRHO) + 2.e0*q(i,j+1,k,QRHO) - 0.5e0*q(i,j+2,k,QRHO))*dinv; 
        }
        else{
            dp = ((threehalves)*q(i,j,k,QPRES) - 2.e0*q(i,j-1,k,QPRES) + 0.5e0*q(i,j-2,k, QPRES))*dinv; 
            du = ((threehalves)*q(i,j,k,QU) - 2.e0*q(i,j-1,k,QU) + 0.5e0*q(i,j-2,k,QU))*dinv;
            dv = ((threehalves)*q(i,j,k,QV) - 2.e0*q(i,j-1,k,QV) + 0.5e0*q(i,j-2,k,QV))*dinv; 
            drho = ((threehalves)*q(i,j,k,QRHO) - 2.e0*q(i,j-1,k,QRHO) + 0.5e0*q(i,j-2,k,QRHO))*dinv; 
        }
    }
}

AMREX_GPU_DEVICE
inline void PeleC_tang_deriv(const int i , const int j, const int k, const int idir, 
                             const int delta, amrex::Real &dp, amrex::Real &du, amrex::Real &dv, 
                             amrex::Real &drho, const amrex::Array4<const amrex::Real> &q)
{
    //Idir refers to the normal direction, so we compute the tangential to idir. 
    //Hence delta will be dy for idir =0 and dx for idir =1
    //dinv is Delta inverse over 2 for central differences
    dinv = 0.5e0/delta; 
    if(idir == 0){
        dp = (q(i,j+1,k, QPRES) - q(i,j-1,k,QPRES))*dinv; 
        du = (q(i,j+1,k, QU) - q(i,j-1,k,QU))*dinv;
        dv = (q(i,j+1,k, QV) - q(i,j-1,k,QV))*dinv; 
        drho = (q(i,j+1,k, QRHO) - q(i,j-1,k,QRHO))*dinv; 
    }
    else if(idir == 1){
        dp = (q(i+1,j,k, QPRES) - q(i-1,j,k,QPRES))*dinv; 
        du = (q(i+1,j,k, QU) - q(i-1,j,k,QU))*dinv;
        dv = (q(i+1,j,k, QV) - q(i-1,j,k,QV))*dinv; 
        drho = (q(i+1,j,k, QRHO) - q(i-1,j,k,QRHO))*dinv; 
    }
}

AMREX_GPU_DEVICE 
inline void PeleC_update_gc(const int i,const int j, const int k, const int bc_type, const int isgn, 
                            const amrex::Real delta, const int dlo, const int dhi, const int bc_type, 
                            const amrex::Real L1, const amrex::Real L2, const amrex::Real L3,
                            const amrex::Real L4, const amrex::Array4<amrex::Real> &u,
                            const amrex::Array4<amrex::Real> &q, const amrex::Array4<amrex::Real> &qaux)
{
    EOS eos; 
    int local_id, idx_gc1, idx_gc2, idx_gc3, idx_gc4, idx_int1, idx_int2, idx_int3, idx_start, idx_end, bndy_loc; 
    //maybe these can be translated into 2-Byte ints? 

    amrex::Real, drho, du, dv, dp, wall_sign; 
    const amrex::Real c = qaux(i,j,k,QC); 
    if(idir == 0){
        local_id = i; 
        drho = (L2 + 0.5e0*(L1+L4))/(c*c); 
        du   = (L4-L1)/(2.e0*c*q(i,j,k,QRHO)); 
        dv   = L3; 
        dp   = 0.5e0*(L1+L4); 
    }
    else if(idir == 1){ 
        local_id = j; 
        drho = (L3 + 0.5e0*(L1+L4))/(c*c); 
        du   = L2; 
        dv   = (L4-L1)/(2.e0*c*q(i,j,k,QRHO)); 
        dp   = 0.5e0*(L1+L4); 
    }
    
    if(isgn==1){
        idx_gc1 = local_id-1;
        idx_gc2 = local_id-2;
        idx_gc3 = local_id-3;
        idx_gc4 = local_id-4;
        idx_int1 = local_id+1;
        idx_int2 = local_id+2;
        idx_int3 = local_id+3;
        idx_start = dlo-1;
        idx_end   = dlo-4;
        bndy_loc = dlo;
    }
    else if(isgn==-1){
        idx_gc1 = local_id+1;
        idx_gc2 = local_id+2;
        idx_gc3 = local_id+3;
        idx_gc4 = local_id+4;
        idx_int1 = local_id-1;
        idx_int2 = local_id-2;
        idx_int3 = local_id-3;
        idx_start = dlo+1;
        idx_end   = dlo+4;
        bndy_loc = dhi;
    }
    if(idir==0){
            amrex::Real qu    = q(i,j,k,QU); 
            amrex::Real qv    = q(i,j,k,QV); 
            amrex::Real qp    = q(i,j,k,QPRES); 
            amrex::Real qrho  = q(i,j,k,QRHO); 

            amrex::Real qu1   = q(idx_int1,j,k,QU); 
            amrex::Real qv1   = q(idx_int1,j,k,QV); 
            amrex::Real qrho1 = q(idx_int1,j,k,QRHO); 
            amrex::Real qp1   = q(idx_int1,j,k,QPRES); 

            q(idx_gc1,j,k,QU)    = qu1 - 2.e0*delta*du*isgn;
            q(idx_gc1,j,k,QV)    = qv1 - 2.e0*delta*dv*isgn;
            q(idx_gc1,j,k,QRHO)  = qrho1  - 2.e0*delta*drho*isgn;
            q(idx_gc1,j,k,QPRES) = qp1 - 2.e0*delta*dp*isgn;

            q(idx_gc2,j,k,QU)    = -2.e0*qu1 - 3.e0*qu + 6.e0*q(idx_gc1,j,k,QU) + 6.e0*delta*du*isgn;
            q(idx_gc2,j,k,QV)    = -2.e0*qv1 - 3.e0*qv + 6.e0*q(idx_gc1,j,k,QV) + 6.e0*delta*dv*isgn;
            q(idx_gc2,j,k,QRHO)  = -2.e0*qrho1 - 3.e0*qrho + 6.e0*q(idx_gc1,j,k,QRHO) + 6.e0*delta*drho*isgn;
            q(idx_gc2,j,k,QPRES) = -2.e0*qp1 - 3.e0*qp + 6.e0*q(idx_gc1,j,k,QPRES) + 6.e0*delta*dp*isgn;

            q(idx_gc3,j,k,QU)    = 3.e0*qu1 +10.e0*qu - 18.e0*q(idx_gc1,j,k,QU) + 6.e0*q(idx_gc2,j,k,QU) - 12.e0*delta*du*isgn;
            q(idx_gc3,j,k,QV)    = 3.e0*qv1 +10.e0*qv - 18.e0*q(idx_gc1,j,k,QV) + 6.e0*q(idx_gc2,j,k,QV) - 12.e0*delta*dv*isgn;
            q(idx_gc3,j,k,QRHO)  = 3.e0*qrho1 +10.e0*qrho - 18.e0*q(idx_gc1,j,k,QRHO) + 6.e0*q(idx_gc2,j,k,QRHO) - 12.e0*delta*drho*isgn;
            q(idx_gc3,j,k,QPRES) = 3.e0*qp1 +10.e0*qp - 18.e0*q(idx_gc1,j,k,QPRES) + 6.e0*q(idx_gc2,j,k,QPRES) - 12.e0*delta*dp*isgn;

            q(idx_gc4,j,k,QU)    = -2.e0*qu1 - 13.e0*qu + 24.e0*q(idx_gc1,j,k,QU) - 12.e0*q(idx_gc2,j,k,QU)  
                                 + 4.e0*q(idx_gc3,j,k,QU) + 12.e0*delta*du*isgn;
            q(idx_gc4,j,k,QV)    = -2.e0*qv1 - 13.e0*qv + 24.e0*q(idx_gc1,j,k,QV) - 12.e0*q(idx_gc2,j,k,QV) 
                                 + 4.e0*q(idx_gc3,j,k,QV) + 12.e0*delta*dv*isgn;
            q(idx_gc4,j,k,QRHO)  = -2.e0*qrho1 - 13.e0*qrho + 24.e0*q(idx_gc1,j,k,QRHO) - 12.e0*q(idx_gc2,j,k,QRHO) 
                                 + 4.e0*q(idx_gc3,j,k,QRHO) + 12.e0*delta*drho*isgn;
            q(idx_gc4,j,k,QPRES) = -2.e0*qp1 - 13.e0*qp + 24.e0*q(idx_gc1,j,k,QPRES) - 12.e0*q(idx_gc2,j,k,QPRES) 
                                 + 4.e0*q(idx_gc3,j,k,QPRES) + 12.e0*delta*dp*isgn;

            if(bc_type==NoSlipWall || bc_type==SlipWall){
                if(bc_type==NoSlipWall) wall_sign = -1.e0; 
                else if(bc_type==SlipWall) wall_sign = 1.e0; 
                q(idx_gc1,j,k,QU)    = -qu;
                q(idx_gc2,j,k,QU)    = -qu1;
                q(idx_gc3,j,k,QU)    = -q(idx_int2,j,k,QU);
                q(idx_gc4,j,k,QU)    = -q(idx_int3,j,k,QU);

                q(idx_gc1,j,k,QV)    = wall_sign*qv;
                q(idx_gc2,j,k,QV)    = wall_sign*qv1;
                q(idx_gc3,j,k,QV)    = wall_sign*q(idx_int2,j,k,QV);
                q(idx_gc4,j,k,QV)    = wall_sign*q(idx_int3,j,k,QV);

                q(idx_gc1,j,k,QRHO)  = qrho;
                q(idx_gc2,j,k,QRHO)  = qrho1;
                q(idx_gc3,j,k,QRHO)  = q(idx_int2,j,k,QRHO);
                q(idx_gc4,j,k,QRHO)  = q(idx_int3,j,k,QRHO);

                q(idx_gc1,j,k,QPRES)  = qp;
                q(idx_gc2,j,k,QPRES)  = qp1;
                q(idx_gc3,j,k,QPRES)  = q(idx_int2,j,k,QPRES);
                q(idx_gc4,j,k,QPRES)  = q(idx_int3,j,k,QPRES);                            
         }

        if(isgn==1){ 
            for(int hop = idx_start; hop > idx_end; hop--){
                eos.p   = q(hop,j,k,QPRES); 
                eos.rho = q(hop,j,k,QRHO); 
                for(int n = 0; n < NUM_SPECIES; n++) eos.massfrac[i] = q(bndy_loc,j,k,QFS+n); 
                for(int n = 0; n < NUM_AUXILIARY; n++) eos.aux[i] = q(bndy_loc,j,k,QFX+n); 
                eos.eos_rp(); 
                q(hop, j, k, QTEMP)  = eos.T; 
                q(hop, j, k, QREINT) = eos.e*q(hop, j, k, QRHO); 
                q(hop, j, k, QGAME)  = q(hop, j, k, QPRES)/q(hop, j, k, QREINT) + 1.e0; 

                qaux(hop, j, k, QDPDR) = eos.dpdr_e; 
                qaux(hop, j, k, QDPDE) = eos.dpde; 
                qaux(hop, j, k, QC   ) = eos.cs; 
                qaux(hop, j, k, QSCML) = amrex::max(small, small*qaux(hop,j, k,QC)); 

                u(hop, j, k, URHO) = eos.rho; 
                u(hop, j, k, UMX) = q(hop,j,k, QU)*eos.rho; 
                u(hop, j, k, UMY) = q(hop, j, k, QV)*eos.rho; 
                u(hop, j, k, UMZ) = 0.e0; 
                u(hop, j, k, UEINT) = eos.rho*eos.e; 
                u(hop, j, k, UEDEN) = eos.rho *(eoe.e + 0.5e0*(q(hop,j,k,QU)*q(hop,j,k,QU) + q(hop,j,k,QV)*q(hop,j,k,QV)); 
                u(hop, j, k, UTEMP) = eos.T; 
                for(int n = 0; n < NUM_SPECIES; n++) 
                    u(hop, j, k, UFS + n) = eos.rho*eos.massfrac[n]; 

            }
        }
        else if(isgn==-1) 
        {
            for(int hop = idx_start; hop < idx_end; hop++){
                eos.p   = q(hop,j,k,QPRES); 
                eos.rho = q(hop,j,k,QRHO); 
                for(int n = 0; n < NUM_SPECIES; n++) eos.massfrac[i] = q(bndy_loc,j,k,QFS+n); 
                for(int n = 0; n < NUM_AUXILIARY; n++) eos.aux[i] = q(bndy_loc,j,k,QFX+n); 
                eos.eos_rp(); 
                q(hop, j, k, QTEMP)  = eos.T; 
                q(hop, j, k, QREINT) = eos.e*q(hop, j, k, QRHO); 
                q(hop, j, k, QGAME)  = q(hop, j, k, QPRES)/q(hop, j, k, QREINT) + 1.e0; 

                qaux(hop, j, k, QDPDR) = eos.dpdr_e; 
                qaux(hop, j, k, QDPDE) = eos.dpde; 
                qaux(hop, j, k, QC   ) = eos.cs; 
                qaux(hop, j, k, QSCML) = amrex::max(small, small*qaux(hop,j, k,QC)); 

                u(hop, j, k, URHO) = eos.rho; 
                u(hop, j, k, UMX) = q(hop,j,k, QU)*eos.rho; 
                u(hop, j, k, UMY) = q(hop, j, k, QV)*eos.rho; 
                u(hop, j, k, UMZ) = 0.e0; 
                u(hop, j, k, UEINT) = eos.rho*eos.e; 
                u(hop, j, k, UEDEN) = eos.rho *(eoe.e + 0.5e0*(q(hop,j,k,QU)*q(hop,j,k,QU) + q(hop,j,k,QV)*q(hop,j,k,QV)); 
                u(hop, j, k, UTEMP) = eos.T; 
                for(int n = 0; n < NUM_SPECIES; n++) 
                    u(hop, j, k, UFS + n) = eos.rho*eos.massfrac[n]; 

            }
        }           
    }
    else if(idir==1) 
    {
        amrex::Real qu    = q(i,j,k,QU); 
        amrex::Real qv    = q(i,j,k,QV); 
        amrex::Real qp    = q(i,j,k,QPRES); 
        amrex::Real qrho  = q(i,j,k,QRHO); 

        amrex::Real qu1   = q(i,idx_int1,k,QU); 
        amrex::Real qv1   = q(i,idx_int1,k,QV); 
        amrex::Real qrho1 = q(i,idx_int1,k,QRHO); 
        amrex::Real qp1   = q(i,idx_int1,k,QPRES); 

        q(i,idx_gc1,k,QU)    = qu1 - 2.e0*delta*du*isgn;
        q(i,idx_gc1,k,QV)    = qv1 - 2.e0*delta*dv*isgn;
        q(i,idx_gc1,k,QRHO)  = qrho1  - 2.e0*delta*drho*isgn;
        q(i,idx_gc1,k,QPRES) = qp1 - 2.e0*delta*dp*isgn;

        q(i,idx_gc2,k,QU)    = -2.e0*qu1 - 3.e0*qu + 6.e0*q(i,idx_gc1,k,QU) + 6.e0*delta*du*isgn;
        q(i,idx_gc2,k,QV)    = -2.e0*qv1 - 3.e0*qv + 6.e0*q(i,idx_gc1,k,QV) + 6.e0*delta*dv*isgn;
        q(i,idx_gc2,k,QRHO)  = -2.e0*qrho1 - 3.e0*qrho + 6.e0*q(i,idx_gc1,k,QRHO) + 6.e0*delta*drho*isgn;
        q(i,idx_gc2,k,QPRES) = -2.e0*qp1 - 3.e0*qp + 6.e0*q(i,idx_gc1,k,QPRES) + 6.e0*delta*dp*isgn;

        q(i,idx_gc3,k,QU)    = 3.e0*qu1 +10.e0*qu - 18.e0*q(i,idx_gc1,k,QU) + 6.e0*q(i,idx_gc2,k,QU) - 12.e0*delta*du*isgn;
        q(i,idx_gc3,k,QV)    = 3.e0*qv1 +10.e0*qv - 18.e0*q(i,idx_gc1,k,QV) + 6.e0*q(i,idx_gc2,k,QV) - 12.e0*delta*dv*isgn;
        q(i,idx_gc3,k,QRHO)  = 3.e0*qrho1 +10.e0*qrho - 18.e0*q(i,idx_gc1,k,QRHO) + 6.e0*q(i,idx_gc2,k,QRHO) - 12.e0*delta*drho*isgn;
        q(i,idx_gc3,k,QPRES) = 3.e0*qp1 +10.e0*qp - 18.e0*q(i,idx_gc1,k,QPRES) + 6.e0*q(i,idx_gc2,k,QPRES) - 12.e0*delta*dp*isgn;

        q(i,idx_gc4,k,QU)    = -2.e0*qu1 - 13.e0*qu + 24.e0*q(i,idx_gc1,k,QU) - 12.e0*q(i,idx_gc2,k,QU) 
                             + 4.e0*q(i,idx_gc3,k,QU) + 12.e0*delta*du*isgn;
        q(i,idx_gc4,k,QV)    = -2.e0*qv1 - 13.e0*qv + 24.e0*q(i,idx_gc1,k,QV) - 12.e0*q(i,idx_gc2,k,QV)
                             + 4.e0*q(i,idx_gc3,k,QV) + 12.e0*delta*dv*isgn;
        q(i,idx_gc4,k,QRHO)  = -2.e0*qrho1 - 13.e0*qrho + 24.e0*q(i,idx_gc1,k,QRHO) - 12.e0*q(i,idx_gc2,k,QRHO)
                             + 4.e0*q(i,idx_gc3,k,QRHO) + 12.e0*delta*drho*isgn;
        q(i,idx_gc4,k,QPRES) = -2.e0*qp1 - 13.e0*qp + 24.e0*q(i,idx_gc1,k,QPRES) - 12.e0*q(i,idx_gc2,k,QPRES) 
                             + 4.e0*q(i,idx_gc3,k,QPRES) + 12.e0*delta*dp*isgn;

        if(bc_type==NoSlipWall || bc_type==SlipWall){
            wall_sign = (bc_type==NoSlipWall)? -1.e0 : 1.e0; 
            q(i,idx_gc1,k,QU)    = wall_sign*qu;
            q(i,idx_gc2,k,QU)    = wall_sign*qu1;
            q(i,idx_gc3,k,QU)    = wall_sign*q(i,idx_int2,k,QU);
            q(i,idx_gc4,k,QU)    = wall_sign*q(i,idx_int3,k,QU);

            q(i,idx_gc1,k,QV)    = -qv;
            q(i,idx_gc2,k,QV)    = -qv1;
            q(i,idx_gc3,k,QV)    = -q(i,idx_int2,k,QV);
            q(i,idx_gc4,k,QV)    = -q(i,idx_int3,k,QV);

            q(i,idx_gc1,k,QRHO)  = qrho;
            q(i,idx_gc2,k,QRHO)  = qrho1;
            q(i,idx_gc3,k,QRHO)  = q(i,idx_int2,k,QRHO);
            q(i,idx_gc4,k,QRHO)  = q(i,idx_int3,k,QRHO);

            q(i,idx_gc1,k,QPRES)  = qp;
            q(i,idx_gc2,k,QPRES)  = qp1;
            q(i,idx_gc3,k,QPRES)  = q(i,idx_int2,k,QPRES);
            q(i,idx_gc4,k,QPRES)  = q(i,idx_int3,k,QPRES);
           
        }
        if(isgn==1){ 
            for(int hop = idx_start; hop > idx_end; hop--){
                eos.p   = q(i,hop,k,QPRES); 
                eos.rho = q(i,hop,k,QRHO); 
                for(int n = 0; n < NUM_SPECIES; n++) eos.massfrac[i] = q(i,bndy_loc,k,QFS+n); 
                for(int n = 0; n < NUM_AUXILIARY; n++) eos.aux[i] = q(i,bndy_loc,k,QFX+n); 
                eos.eos_rp(); 
                q(i, hop, k, QTEMP)  = eos.T; 
                q(i, hop, k, QREINT) = eos.e*q(i, hop, k, QRHO); 
                q(i, hop, k, QGAME)  = q(i, hop, k, QPRES)/q(i, hop, k, QREINT) + 1.e0; 

                qaux(i, hop, k, QDPDR) = eos.dpdr_e; 
                qaux(i, hop, k, QDPDE) = eos.dpde; 
                qaux(i, hop, k, QC   ) = eos.cs; 
                qaux(i, hop, k, QSCML) = amrex::max(small, small*qaux(i,hop, k,QC)); 

                u(i, hop, k, URHO) = eos.rho; 
                u(i, hop, k, UMX) = q(i,hop,k, QU)*eos.rho; 
                u(i, hop, k, UMY) = q(i, hop, k, QV)*eos.rho; 
                u(i, hop, k, UMZ) = 0.e0; 
                u(i, hop, k, UEINT) = eos.rho*eos.e; 
                u(i, hop, k, UEDEN) = eos.rho *(eoe.e + 0.5e0*(q(i,hop,k,QU)*q(i,hop,k,QU) + q(i,hop,k,QV)*q(i,hop,k,QV)); 
                u(i, hop, k, UTEMP) = eos.T; 
                for(int n = 0; n < NUM_SPECIES; n++) 
                    u(i, hop, k, UFS + n) = eos.rho*eos.massfrac[n]; 

            }
        }
        else if(isgn==-1) 
        {
            for(int hop = idx_start; hop < idx_end; hop++){
                eos.p   = q(i,hop,k,QPRES); 
                eos.rho = q(i,hop,k,QRHO); 
                for(int n = 0; n < NUM_SPECIES; n++) eos.massfrac[i] = q(i,bndy_loc,k,QFS+n); 
                for(int n = 0; n < NUM_AUXILIARY; n++) eos.aux[i] = q(i,bndy_loc,k,QFX+n); 
                eos.eos_rp(); 
                q(i, hop, k, QTEMP)  = eos.T; 
                q(i, hop, k, QREINT) = eos.e*q(i, hop, k, QRHO); 
                q(i, hop, k, QGAME)  = q(i, hop, k, QPRES)/q(i, hop, k, QREINT) + 1.e0; 

                qaux(i, hop, k, QDPDR) = eos.dpdr_e; 
                qaux(i, hop, k, QDPDE) = eos.dpde; 
                qaux(i, hop, k, QC   ) = eos.cs; 
                qaux(i, hop, k, QSCML) = amrex::max(small, small*qaux(i,hop, k,QC)); 

                u(i, hop, k, URHO) = eos.rho; 
                u(i, hop, k, UMX) = q(i,hop,k, QU)*eos.rho; 
                u(i, hop, k, UMY) = q(i, hop, k, QV)*eos.rho; 
                u(i, hop, k, UMZ) = 0.e0; 
                u(i, hop, k, UEINT) = eos.rho*eos.e; 
                u(i, hop, k, UEDEN) = eos.rho *(eoe.e + 0.5e0*(q(i,hop,k,QU)*q(i,hop,k,QU) + q(i,hop,k,QV)*q(i,hop,k,QV)); 
                u(i, hop, k, UTEMP) = eos.T; 
                for(int n = 0; n < NUM_SPECIES; n++) 
                    u(i, hop, k, UFS + n) = eos.rho*eos.massfrac[n]; 

            }
        }                           
    }
}


AMREX_GPU_DEVICE 
inline void PeleC_compute_waves(const int i, const int j, const int k, const int idir, const int isgn, 
                                const int bc_type); 


#endif //End of header 
