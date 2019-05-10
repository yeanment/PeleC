#include <PeleC.H>
#include <PeleC_F.H>

using std::string;
using namespace amrex;

#include <Transport_F.H>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <PeleC_K.H>
#include "PeleC_diffusion.H" 
#if AMREX_SPACEDIM==2 
#include "PeleC_gradutil_2D.H"
#include "PeleC_diffterm_2D.H" 
#elif AMREX_SPACEDIM==3
#include "PeleC_gradutil_3D.H"
#include "PeleC_diffterm_3D.H" 
#endif 


// **********************************************************************************************
void
PeleC::getMOLSrcTermGPU(const amrex::MultiFab& S,
                     amrex::MultiFab&       MOLSrcTerm,
                     amrex::Real            time,
                     amrex::Real            dt,
                     amrex::Real            flux_factor) {
  BL_PROFILE("PeleC::getMOLSrcTerm()");
  BL_PROFILE_VAR_NS("diffusion_stuff", diff);
  if (diffuse_temp == 0
      && diffuse_enth == 0
      && diffuse_spec == 0
      && diffuse_vel  == 0
      && do_hydro == 0)
  {
    MOLSrcTerm.setVal(0,0,NUM_STATE,MOLSrcTerm.nGrow());
    return;
  }
  /**
     Across all conserved state components, compute the method of lines rhs
     = -Div(Flux).  The input state, S, contained the conserved variables, and
     is "fill patched" in the usual AMReX way, where values at Dirichlet boundaries
     actually are assumed to live on the inflow face.

     1. Convert S to Q, primitive variables (since the transport coefficients typically
     depend on mass fractions and temperature). Q then will be face-centered on
     Dirichlet faces.

     2. Evaluate transport coefficients (these also will be face-centered, if Q is).

     3. Evaluate the diffusion operator over all components

     a. Evaluate tangential derivatives for strain terms, over all cells
     b. Replace these with versions that avoid covered cells, if present
     c. Evaluate face-centered diffusion fluxes, and their divergence
     d. Zero fluxes and divergence after the fact if subset of components have diffuse shut off
     (this allows that T be diffused alone, in order to support simple tests)

     4. Replace divergence of face-centered fluxes with hybrid divergence operator, and
     weigthed redistribution.

     Extra notes:

     A. The face-based transport coefficients that are computed with face-based
     Fill-Patched data at Dirichlet boundaries.

     B. Within the routine that computes diffusion fluxes, there is a need for computing
     species enthalpies at cell faces.  In the EGLib model, species enthalpies
     are a function of temperature.  At the moment, in diffterm, we evaluate
     enthalpies at cell centers and then take the face values to be the arithmetic
     average of cell values on either side.  Similarly, mass fractions at cell faces
     are needed to compute the barodiffusion and correction velocity expressions.
     Arithmetic averages are used there as well.  Thus, these face values are thermodynamically
     inconsistent.  Note sure what are the consequences of that.

     The non-GPU version has EB, this does not at the moment. The Techniques for EB will need to be 
     carefully considered before implementation for accelerators. 
  */
  int nCompTr = dComp_lambda + 1;
  int do_harmonic = 1;  // TODO: parmparse this
  const Real* dx = geom.CellSize();

  Real dx1 = dx[0];
  for (int d=1; d < BL_SPACEDIM; ++d) {
    dx1 *= dx[d];
  }
  std::array<Real, BL_SPACEDIM> dxD = {D_DECL(dx1, dx1, dx1)};
  const Real *dxDp = &(dxD[0]);


#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    IArrayBox bcMask[BL_SPACEDIM];
    FArrayBox dm_as_fine(Box::TheUnitBox(), NUM_STATE);
    FArrayBox fab_drho_as_crse(Box::TheUnitBox(), NUM_STATE);
    IArrayBox fab_rrflag_as_crse(Box::TheUnitBox());
    

    int flag_nscbc_isAnyPerio = (geom.isAnyPeriodic()) ? 1 : 0; 
    int flag_nscbc_perio[BL_SPACEDIM]; // For 3D, we will know which corners have a periodicity
    for (int d=0; d<BL_SPACEDIM; ++d) {
        flag_nscbc_perio[d] = (Geometry::isPeriodic(d)) ? 1 : 0;
    }
	  const int*  domain_lo = geom.Domain().loVect();
	  const int*  domain_hi = geom.Domain().hiVect();

    for (MFIter mfi(S, MFItInfo().EnableTiling(hydro_tile_size).SetDynamic(true));
         mfi.isValid(); ++mfi) {

      const Box  vbox = mfi.tilebox();
      int ng = S.nGrow();
      const Box  gbox = amrex::grow(vbox,ng);
      const Box  cbox = amrex::grow(vbox,ng-1);
      const Box& dbox = geom.Domain();
      
      const int* lo = vbox.loVect();
      const int* hi = vbox.hiVect();

      const FArrayBox& Sfab = S[mfi];

      BL_PROFILE_VAR_START(diff);
      int nqaux = NQAUX > 0 ? NQAUX : 1;
      Gpu::AsyncFab q(gbox, QVAR), qaux(gbox, nqaux); 
      Gpu::AsyncFab coeff_cc(gbox, nCompTr); 
      auto const& s = S.array(mfi); 
      auto const& qar = q.array(); 
      auto const& qauxar = qaux.array(); 
      

      // Get primitives, Q, including (Y, T, p, rho) from conserved state
      // required for D term
      {
          BL_PROFILE("PeleC::ctoprim");
          AMREX_PARALLEL_FOR_3D(gbox, i, j, k, 
              {
                  PeleC_ctoprim(i,j,k, s, qar, qauxar);                  
              });
          Gpu::Device::streamSynchronize();
      }
      
      
      
      for (int i = 0; i < BL_SPACEDIM ; i++)  {
		    const Box& bxtmp = amrex::surroundingNodes(vbox,i);
        Box TestBox(bxtmp);
        for(int d=0; d<BL_SPACEDIM; ++d) {
          if (i!=d) TestBox.grow(d,1);
        }
        
		    bcMask[i].resize(TestBox,1);
        bcMask[i].setVal(0);
	    }
      
      // Becase bcMask is read in the Riemann solver in any case,
      // here we put physbc values in the appropriate faces for the non-nscbc case
      set_bc_mask(lo, hi, domain_lo, domain_hi,
                  D_DECL(BL_TO_FORTRAN(bcMask[0]),
	                       BL_TO_FORTRAN(bcMask[1]),
                         BL_TO_FORTRAN(bcMask[2])));

      if (nscbc_diff == 1)
      {
        impose_NSCBC(lo, hi, domain_lo, domain_hi,
                     BL_TO_FORTRAN(Sfab),
                     BL_TO_FORTRAN(q.fab()),
                     BL_TO_FORTRAN(qaux.fab()),
                     D_DECL(BL_TO_FORTRAN(bcMask[0]),
	                          BL_TO_FORTRAN(bcMask[1]),
                            BL_TO_FORTRAN(bcMask[2])),
                     &flag_nscbc_isAnyPerio, flag_nscbc_perio, 
                     &time, dx, &dt);
      }
      
      // Compute transport coefficients, coincident with Q
      {
        BL_PROFILE("PeleC::get_transport_coeffs call");        
        get_transport_coeffs(ARLIM_3D(gbox.loVect()),
                             ARLIM_3D(gbox.hiVect()),
                             BL_TO_FORTRAN_N_3D(q.fab(), cQFS),
                             BL_TO_FORTRAN_N_3D(q.fab(), cQTEMP),
                             BL_TO_FORTRAN_N_3D(q.fab(), cQRHO),
                             BL_TO_FORTRAN_N_3D(coeff_cc.fab(), dComp_rhoD),
                             BL_TO_FORTRAN_N_3D(coeff_cc.fab(), dComp_mu),
                             BL_TO_FORTRAN_N_3D(coeff_cc.fab(), dComp_xi),
                             BL_TO_FORTRAN_N_3D(coeff_cc.fab(), dComp_lambda));
      }
        Gpu::AsyncFab flux_ec[AMREX_SPACEDIM] = 
        {
            Gpu::AsyncFab(amrex::surroundingNodes(cbox,0), NUM_STATE)
#if AMREX_SPACEDIM > 1 
           , Gpu::AsyncFab(amrex::surroundingNodes(cbox,1), NUM_STATE)
#if AMREX_SPACEDIM > 2 
           , Gpu::AsyncFab(amrex::surroundingNodes(cbox,2), NUM_STATE)
#endif
#endif
        }; 

      // Container on grown region, for hybrid divergence & redistribution
      Gpu::AsyncFab Dterm(cbox, NUM_STATE); 
      auto const &coecc = coeff_cc.array(); 
      for (int d=0; d<BL_SPACEDIM; ++d) {
        (flux_ec[d].fab()).setVal(0.); 
        Box ebox = amrex::surroundingNodes(cbox,d);
        Gpu::AsyncFab coeff_ec(ebox, nCompTr); 
        auto const &coeec = coeff_ec.array();
        auto const &flxec = (flux_ec[d]).array();  
        const amrex::Real del = dx[d]; 
        // Get face-centered transport coefficients
        {          
          BL_PROFILE("PeleC::pc_move_transport_coeffs_to_ec call");
          AMREX_PARALLEL_FOR_4D (cbox, nCompTr, i, j, k, n, { 
                PeleC_move_transcoefs_to_ec(i,j,k,n, coecc, coeec, d, do_harmonic); 
          });  
        }
#if (BL_SPACEDIM > 1)
        int nCompTan = AMREX_D_PICK(1, 2, 6);
        Gpu::AsyncFab tander_ec(ebox, nCompTan);
        (tander_ec.fab()).setVal(0.);          
        auto const &td = tander_ec.array(); 
        // Tangential derivatives on faces only needed for velocity diffusion
        if (diffuse_vel == 0) {
          (tander_ec.fab()).setVal(0);
        } 
        else {
          {
            BL_PROFILE("PeleC::pc_compute_tangential_vel_derivs call");
#if (AMREX_SPACEDIM == 2) 
            amrex::Real del2 = (d == 0)? dx[1] : dx[0]; 
#endif            
            AMREX_PARALLEL_FOR_3D(cbox, i, j, k, {
                PeleC_compute_tangential_vel_derivs(i,j,k,td, qar, d, del2); 
            }); 
          }
        }  // diffuse_vel
        //Compute Extensive diffusion fluxes and flux divergence 
        auto const& a1 = (area[d]).array(mfi); 
        BL_PROFILE("PeleC::diffusion_flux()"); 
        AMREX_PARALLEL_FOR_3D(ebox, i, j, k, {
            PeleC_diffusion_flux(i,j,k, qar, coeec, td, a1, flxec, del, d); 
        }); 
#endif
      }  // loop over dimension

      // Compute extensive diffusion fluxes, F.A and (1/Vol).Div(F.A)
      {
        BL_PROFILE("PeleC::pc_diffup()");
        auto const D_DECL(&flx1 = flux_ec[0].array(), &flx2 = flux_ec[1].array(), &flx3 = flux_ec[2].array());  
        auto const &vol = volume.array(mfi); 
        auto const &Dif = Dterm.array(); 
        AMREX_PARALLEL_FOR_4D(cbox, NVAR, i , j, k ,n, {
            PeleC_diffup(i,j,k,n, D_DECL(flx1, flx1, flx3), vol, Dif); 
        }); 
      }  

      // Shut off unwanted diffusion after the fact
      //    ick! Under normal conditions, you either have diffusion on all or
      //      none, so this shouldn't be done this way.  However, the regression
      //      test for diffusion works by diffusing only temperature through
      //      this process.  Ideally, we'd redo that test to diffuse a passive
      //      scalar instead....
          
      if (diffuse_temp == 0 && diffuse_enth == 0) {
        (Dterm.fab()).setVal(0, Eden);
        (Dterm.fab()).setVal(0, Eint);
        for (int d = 0; d < BL_SPACEDIM; d++) {
          (flux_ec[d].fab()).setVal(0, Eden);
          (flux_ec[d].fab()).setVal(0, Eint);
        }
      }
      if (diffuse_spec == 0) {
        (Dterm.fab()).setVal(0, (Dterm.fab()).box(), FirstSpec, NumSpec);
        for (int d = 0; d < BL_SPACEDIM ; d++) {
          (flux_ec[d].fab()).setVal(0, (flux_ec[d].fab()).box(), FirstSpec, NumSpec);
        }
      }

      if (diffuse_vel  == 0) {
        (Dterm.fab()).setVal(0, (Dterm.fab()).box(), Xmom, 3);
        for (int d = 0; d < BL_SPACEDIM; d++) {
          (flux_ec[d].fab()).setVal(0, (flux_ec[d].fab()).box(), Xmom, 3);
        }
      }


      BL_PROFILE_VAR_STOP(diff);


      MOLSrcTerm[mfi].setVal(0, vbox, 0, NUM_STATE);
      MOLSrcTerm[mfi].copy(Dterm.fab(), vbox, 0, vbox, 0, NUM_STATE);

        if (do_reflux && flux_factor != 0)  // no eb in problem
        {
          for (int d = 0; d < BL_SPACEDIM ; d++) {
            (flux_ec[d].fab()).mult(flux_factor);
          }

          if (level < parent->finestLevel()) {
            getFluxReg(level+1).CrseAdd(mfi,
                                       {D_DECL(&flux_ec[0].fab(), &flux_ec[1].fab(), &flux_ec[2].fab())},
                                        dxDp, dt);
          }

          if (level > 0) {
            getFluxReg(level).FineAdd(mfi,
                                     {D_DECL(&flux_ec[0].fab(), &flux_ec[1].fab(), &flux_ec[2].fab())},
                                      dxDp, dt);
          }
        }

    }  // End of MFIter scope
  }  // End of OMP scope

  // Extrapolate to ghost cells
  if (MOLSrcTerm.nGrow() > 0) {
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(MOLSrcTerm, hydro_tile_size); mfi.isValid(); ++mfi) {
      BL_PROFILE("PeleC::diffextrap calls");

      const Box& bx = mfi.validbox();
      pc_diffextrap(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
                    BL_TO_FORTRAN_N_3D(MOLSrcTerm[mfi], Xmom), &amrex::SpaceDim);

      int nspec = NumSpec;
      pc_diffextrap(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
                    BL_TO_FORTRAN_N_3D(MOLSrcTerm[mfi], FirstSpec), &nspec);

      const int one = 1;
      pc_diffextrap(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
                    BL_TO_FORTRAN_N_3D(MOLSrcTerm[mfi], Eden), &one);
    }
  }
}
