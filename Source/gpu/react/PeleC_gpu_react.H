#ifndef PeleC_react_H_
#define PeleC_react_H_ 
#include <PeleC_index_macros.H> 
#include <AMReX_FArrayBox.H> 
#include <PeleC_EOS.H> 

#if defined(AMREX_USE_GPU)
#define AMREX_GPU_CONST __constant__
#else 
#define AMREX_GPU_CONST const
#endif 

AMREX_GPU_CONST amrex::Real alpha_rk64[6] = 
        {
            3296351145737.e0/15110423921029.e0, 
            1879360555526.e0/ 7321162733569.e0, 
            10797097731880.e0/20472212111779.e0, 
            754636544611.e0/15563872110659.e0, 
            3260218886217.e0/ 2618290685819.e0, 
            5069185909380.e0/12292927838509.e0
        };

AMREX_GPU_CONST amrex::Real beta_rk64[6] = 
        {
           -1204558336989.e0/10607789004752.e0,  
           -3028468927040.e0/14078136890693.e0,  
           -455570672869.e0/ 8930094212428.e0,  
           -17275898420483.e0/15997285579755.e0,  
           -2453906524165.e0/ 9868353053862.e0,  
            0.e0
        };

AMREX_GPU_CONST amrex::Real err_rk64[6] = 
        {
           -530312978447.e0/ 9560368366154.e0,     
            473021958881.e0/ 2984707536468.e0,    
           -947229622805.e0/10456009803779.e0,    
           -2921473878215.e0/13334914072261.e0,    
            1519535112975.e0/ 9264196100452.e0,    
            167623581683.e0/ 3930932046784.e0
        };

//Timestep adapter 
AMREX_GPU_DEVICE 
inline
void adapt_timestep(const amrex::Real rk_err[], const amrex::Real dt_rk4max, amrex::Real &dt, const amrex::Real dt_rk4min, const amrex::Real tol)
{
    const amrex::Real safety_fac = 1e4; 
    const amrex::Real exp1 = 0.25; 
    const amrex::Real exp2 = 0.2; 
    amrex::Real change_factor; 

    amrex::Real err = amrex::max(rk_err[0], rk_err[1]); 
#pragma unroll        
    for(int n = 2; n < NVAR; n++) err = amrex::max(rk_err[n], err); 
    
    if(err < tol){
        err = amrex::max(err, tol/safety_fac);         
        change_factor = std::pow(tol/err, exp1); 
        dt = amrex::min(dt_rk4max, dt*change_factor); 
    }
    else 
    {
        change_factor = std::pow(tol/err, exp2); 
        dt = amrex::max(dt_rk4min, dt*change_factor); 
    }
} 


//Do the reactions, here uout and IR change
//Rk integrator
AMREX_GPU_DEVICE
inline void PeleC_expl_reactions(const int i, const int j, const int k, 
                                 amrex::Array4<const amrex::Real> const& uold,
                                 amrex::Array4<amrex::Real> const& unew, 
                                 amrex::Array4<const amrex::Real> const& asrc, 
                                 amrex::Array4<amrex::Real> const& IR,
                                 const amrex::Real dt_react,  
                                 const int nsteps_min, const int nsteps_max, 
                                 const int nsteps_guess, const int do_update) 
{
    EOS eos; 
    //compute rhoe_ext/rhoy_ext 
    amrex::Real rhoE_old = uold(i,j,k,UEDEN); 
    amrex::Real rhou = uold(i,j,k,UMX), rhov = uold(i,j,k,UMY), rhow = uold(i,j,k,UMZ); 
    amrex::Real rhoInv = 1e0/uold(i,j,k,URHO); 
    amrex::Real rhoKE_old = 0.5e0*(rhou*rhou + rhov*rhov + rhow*rhow)*rhoInv; 

    amrex::Real rho = 0.; 
    for(int n = UFS; n < UFS+NUM_SPECIES; n++) rho+= uold(i,j,k,n); 

    amrex::Real nrg = (rhoE_old - rhoKE_old)*rhoInv; 
    amrex::Real re_old = nrg*uold(i,j,k,URHO);
    rhou = unew(i,j,k,UMX), rhov = unew(i,j,k,UMY), rhow = unew(i,j,k, UMZ);  
    rhoInv = 1.e0/unew(i,j,k,URHO); 
    amrex::Real rhoKE_new = 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)*rhoInv; 
    amrex::Real rhoedot_ext = ( (unew(i,j,k,UEDEN) - rhoKE_new)
                               - rho*nrg)/dt_react; 
    amrex::Real rhoydot_ext[NUM_SPECIES]; 
    for(int n = 0; n < NUM_SPECIES; n++) rhoydot_ext[n] = asrc(i,j,k,UFS+n); 

    //RK dts 
    amrex::Real dt_rk = dt_react/nsteps_guess; 
    amrex::Real dt_min = dt_react/nsteps_max; 
    amrex::Real dt_max = dt_react/nsteps_min; 
    amrex::Real updt_time = 0.e0; 
    int steps = 0; 
    amrex::Real urk[NVAR];
    amrex::Real urk_old[NVAR] = {};
    amrex::Real urk_err[NVAR] = {}; 
    amrex::Real wdot[NVAR] = {};  
    for(int n = 0; n < NVAR; ++n) urk[n] = uold(i,j,k,n);  

    //Do the RK! 
    while(updt_time < dt_react && steps < nsteps_max){ 
#pragma unroll 
        for(int n = 0; n < NVAR; n++)  urk_old[n] = urk[n]; 
        
        updt_time+= dt_rk; 
        steps++; 
        for(int stage = 0; stage < 6; stage++)
        {
            rhoInv = 1.e0/urk[URHO]; 
#pragma unroll 
            for(int n = 0; n < NUM_SPECIES; ++n){
                 eos.massfrac[n] = urk[UFS+n]*rhoInv; 
                 wdot[n] = 0.e0; } 
            
            eos.rho = urk[URHO]; 
            eos.T = urk[UTEMP];
            eos.eos_mpr2wdot(wdot);  
            for(int n = 0; n < NUM_SPECIES; ++n) wdot[n] += rhoydot_ext[n]; 
        
            amrex::Real tempsrc = rhoedot_ext; 
#pragma unroll 
            for(int n = 0; n < NUM_SPECIES; ++n){
                 tempsrc -= wdot[n]*eos.ei[n];      
            }
            tempsrc /= (urk[URHO]*eos.cv); 


 /*================== Update urk_err =================== */
            //Species
#pragma unroll
            for(int n = 0; n < NUM_SPECIES; ++n)
                 urk_err[UFS+n] += err_rk64[stage]*dt_rk*wdot[n]; 
            //Temperature
            urk_err[UTEMP] += err_rk64[stage]*dt_rk*tempsrc; 

/*================== Update Stage solution =================== */ 
            //Species
#pragma unroll 
            for(int n = 0; n < NUM_SPECIES; ++n){
                urk[UFS+n] = urk_old[UFS+n] + alpha_rk64[stage]*dt_rk*wdot[n]; 
             }
            //Temperature
            urk[UTEMP] = urk_old[UTEMP] + alpha_rk64[stage]*dt_rk*tempsrc; 

/*================== Update urk_old =========================== */ 
             //Species
#pragma unroll 
            for(int n = 0; n < NUM_SPECIES; ++n) 
                urk_old[UFS+n] = urk[UFS+n] + beta_rk64[stage]*dt_rk*wdot[n]; 
            //Temperature
            urk_old[UTEMP] = urk[UTEMP] + beta_rk64[stage]*dt_rk*tempsrc; 

/*================= Update urk[rho] ========================= */ 
           urk[URHO] = 0.e0; 
           for(int n = 0; n < NUM_SPECIES; ++n)
                urk[URHO] += urk[UFS+n];

/*================ Adapt Time step! ======================== */ 
        }//end rk stages
        adapt_timestep(urk_err, dt_max, dt_rk, dt_min, 1.e-15); 
    }//end timestep loop 

//Add drhoY/dt to reactions MultiFab and update unew if needed
    amrex::Real umnew = uold(i,j,k,UMX) + dt_react*asrc(i,j,k,UMX); 
    amrex::Real vmnew = uold(i,j,k,UMY) + dt_react*asrc(i,j,k,UMY); 
    amrex::Real wmnew = uold(i,j,k,UMZ) + dt_react*asrc(i,j,k,UMZ); 

    if(do_update){
        unew(i,j,k,URHO) = urk[URHO]; 
        unew(i,j,k,UMX) = umnew;  
        unew(i,j,k,UMY) = vmnew; 
        unew(i,j,k,UMZ) = wmnew; 
        unew(i,j,k,UTEMP) = urk[UTEMP]; 
#pragma unroll 
        for(int n=0; n <NUM_SPECIES; ++n){
             unew(i,j,k, UFS+n) = urk[UFS+n];         
        }
    }

#pragma unroll     
    for(int n =0; n < NUM_SPECIES; ++n){
         IR(i,j,k,n) = (urk[UFS+n] - uold(i,j,k,UFS+n))/dt_react  - asrc(i,j,k,UFS+n);
    }
    IR(i,j,k,NUM_SPECIES) = (re_old + dt_react*rhoedot_ext + 0.5e0*(umnew*umnew + vmnew*vmnew + wmnew*wmnew)/urk[URHO] 
                             - uold(i,j,k,UEDEN))/dt_react - asrc(i,j,k,UEDEN);
}




#endif 
