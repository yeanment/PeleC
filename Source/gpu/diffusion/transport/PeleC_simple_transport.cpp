#include "actual_transport.H"
#include "mechanism.h"
#include <AMReX.H>
#include <AMReX_Gpu.H>
#include "trans_params.H"
#include <cmath>

#if defined(BL_FORT_USE_UPPERCASE)
#define ckcvms CKCVMS
#elif defined(BL_FORT_USE_LOWERCASE)
#define ckcvms ckcvms
#elif defined(BL_FORT_USE_UNDERSCORE)
#define ckcvms ckcvms_
#endif


extern "C" {
void ckcvms(amrex::Real* Tloc, amrex::Real* cvk );
}

using namespace amrex;
using namespace trans_params;
 
void  actual_transport(bool wtr_get_xi, bool wtr_get_mu, bool wtr_get_lam, bool wtr_get_Ddiag, amrex::Real& Tloc,
              amrex::Real& rholoc, amrex::Real* Yloc,
              amrex::Real* Ddiag, amrex::Real& mu, amrex::Real& xi, amrex::Real& lam)

{

//  need to set Ru

  amrex::Real trace = 1.e-15;
  amrex::Real Patm = 1.01325e6;
  amrex::Real wbar, pscale;

  amrex::GpuArray<amrex::Real,NUM_SPECIES>  Xloc;
  amrex::GpuArray<amrex::Real,NUM_SPECIES>  muloc;
  amrex::GpuArray<amrex::Real,NUM_SPECIES>  xiloc;
  amrex::GpuArray<amrex::Real,NUM_SPECIES>  lamloc;
  amrex::GpuArray<amrex::Real,NUM_FIT-1>  logT;

  logT[0] = std::log(Tloc);
  logT[1] = logT[0]*logT[0];
  logT[2] = logT[0]*logT[1];

  int nspec = NUM_SPECIES;

  amrex::Real sum = 0.;

  for (int i = 0; i < nspec ; ++i){

     sum = sum+Yloc[i];

  }

  wbar = 0.;
  
  amrex::Real real_nspec = NUM_SPECIES;

  for (int i = 0; i < NUM_SPECIES ; ++i){
       
      Yloc[i] = Yloc[i] + trace*(sum/real_nspec-Yloc[i]); 

  }

  for (int i = 0; i < NUM_SPECIES ; ++i){
       
      wbar = wbar + Yloc[i]*iwt[i];

  }

   wbar = 1.0/ wbar;

  for (int i = 0; i < NUM_SPECIES ; ++i){
       
     Xloc[i] = Yloc[i]*wbar*iwt[i];

  }
 

    if (wtr_get_mu == true){

     for (int i = 0; i < NUM_SPECIES ; ++i){
       
            muloc[i] = fitmu[4*i]+fitmu[1+4*i]*logT[0]+ fitmu[2+4*i]*logT[1] 
                        + fitmu[3+4*i]*logT[2];
            muloc[i] = std::exp(muloc[i]);

      }

      mu = 0.0;

      for (int i = 0; i < NUM_SPECIES ; ++i){
       
            mu = mu+Xloc[i]* std::pow(muloc[i],6.0);

       }



       mu = std::pow(mu,1.0/6.0);


//  assumption that we only get bulk viscosity if we are already getting shear viscosity

    if (wtr_get_xi == true) {

       comp_pure_bulk(Tloc, muloc.data(), xiloc.data());

       xi = 0.0;

       for (int i = 0; i < NUM_SPECIES ; ++i){
       
            xi = xi+Xloc[i]*std::pow( xiloc[i], 0.750);

        }

       xi = std::pow(xi,4.0/3.0);


    }

    }

    if ( wtr_get_lam == true){

       for (int i = 0; i < NUM_SPECIES ; ++i){

            lamloc[i] = fitlam[4*i]+fitlam[1+4*i]*logT[0]+ fitlam[2+4*i]*logT[1] 
                        + fitlam[3+4*i]*logT[2];
            lamloc[i] = std::exp(lamloc[i]);
           
       }

       lam = 0.;

       for (int i = 0; i < NUM_SPECIES ; ++i){

            lam = lam+ Xloc[i]*std::pow(lamloc[i],0.25);

       }


       lam = std::pow(lam,4.);


    }


    if (wtr_get_Ddiag == true) {

//       for (int i = 0; i < NUM_SPECIES ; ++i){
//          for (int j = 0; j < i-1 ; ++j){
//               dbinloc[i+NUM_SPECIES*j] = fitdbin[i+NUM_SPECIES*j]+fitdbin[1+4*(i+NUM_SPECIES*j)]*logT[1]  
//                   + fitdbin[2+4*(i+NUM_SPECIES*j)]*logT[2]+ fitdbin[3+4*(i+NUM_SPECIES*j)]*logT[3];
//               dbinloc[i+NUM_SPECIES*j] = std::exp(dbinloc[i+NUM_SPECIES*j]);
           
//           }

//          dbinloc(i+NUM_SPECIES*i) = 0.0;

//       }

       amrex::Real term1, term2, dbintemp;

       for (int i = 0; i < NUM_SPECIES ; ++i){

          term1 = 0.0;
          term2 = 0.0;

          for (int j = 0; j < NUM_SPECIES ; ++j){
  
              if(i != j) {

                 dbintemp = fitdbin[4*(i+NUM_SPECIES*j)]+fitdbin[1+4*(i+NUM_SPECIES*j)]*logT[0]  
                   + fitdbin[2+4*(i+NUM_SPECIES*j)]*logT[1]+ fitdbin[3+4*(i+NUM_SPECIES*j)]*logT[2];
                 term1 = term1 + Yloc[j];
                 term2 = term2 + Xloc[j]/std::exp(dbintemp);

               }


           }

           Ddiag[i] = wt[i]* term1/term2 / wbar;

        }

       amrex::Real Ru =  8.31451e+07;

        pscale = Patm * wbar/ (Ru *  Tloc *  rholoc);

       for (int i = 0; i < NUM_SPECIES ; ++i){

             Ddiag[i] = rholoc*pscale*Ddiag[i];

       }


  }

}

  void comp_pure_bulk(amrex::Real Tloc,  amrex::Real* muloc,
          amrex::Real* xiloc )

{
  amrex::GpuArray<amrex::Real,NUM_SPECIES>  cvk;
  amrex::GpuArray<amrex::Real,NUM_SPECIES>  cvkint;
  amrex::GpuArray<amrex::Real,NUM_SPECIES>  cvkrot;
  amrex::GpuArray<amrex::Real,NUM_SPECIES>  FofT;
  amrex::GpuArray<amrex::Real,NUM_SPECIES>  Fnorm;

  amrex::Real epskoverT, epskoverTstd;
  amrex::Real pi = 3.141592653589793238;



  amrex::Real Ru =  8.31451e+07;
  ckcvms(  &Tloc, cvk.data() );

  for (int i = 0 ; i < NUM_SPECIES; ++i){
 
        if(nlin[i] ==  0){

           cvkint[i] = 0.0;
           cvkrot[i] = 0.0;

         } else if(nlin[i] == 1) {

           cvkint[i] = cvk[i] * wt[i]/Ru - 1.50;
           cvkrot[i] = 1.0;

         } else {

           cvkint[i] = cvk[i] * wt[i]/Ru - 1.50;
           cvkrot[i] = 1.5;

         }

  }

  for (int i = 0 ; i < NUM_SPECIES; ++i){

      epskoverTstd = eps[i]/298.0;
      epskoverT = eps[i]/ Tloc;

      Fnorm[i] = 1.0 + 0.50*std::pow(pi,1.5)*std::sqrt(epskoverTstd) + (2.0+.50*pi*pi)*epskoverTstd
                  +std::pow(pi*epskoverTstd,1.5) ;

      FofT[i] = 1.0 + 0.50*std::pow(pi,1.5)*std::sqrt(epskoverT) + (2.0+.50*pi*pi)*epskoverT
                  +std::pow(pi*epskoverT,1.5) ;

   }
      
   for (int i = 0 ; i < NUM_SPECIES; ++i){

      if(nlin[i] == 0){

          xiloc[i] = 0.0;

      } else {

//   zrot/crot approximately zint / cint by assuming vibrational internal energy is small
//   cvkrot is scaled by wk / Ru = mk / kb relative to standard specific cv

        xiloc[i] = 0.250*pi*std::pow(cvkint[i]/(cvkint[i]+1.50),2)* zrot[i]/cvkrot[i]* 
                          Fnorm[i]/FofT[i] * muloc[i];

      }

   }

}

   
