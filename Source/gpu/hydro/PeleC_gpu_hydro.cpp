#include "PeleC.H"
/* This is for the NSCBC stuff, TODO C++ it */ 
#include "PeleC_F.H" 
#include "PeleC_K.H"
#include <AMReX_Gpu.H>
using namespace amrex;

/**
 *  Set up the source terms to go into the hydro.
 */
void
PeleC::construct_gpu_hydro_source(const MultiFab& S, Real time, Real dt, int amr_iteration, int amr_ncycle, int sub_iteration, int sub_ncycle)
{
#ifdef PELEC_USE_MOL

    if (verbose && ParallelDescriptor::IOProcessor()) {
        std::cout << "... Zeroing Godunov-based hydro advance" << std::endl;
    }
    hydro_source.setVal(0);

#else 

    if (verbose && ParallelDescriptor::IOProcessor()) {
        std::cout << "... Computing hydro advance" << std::endl;
    }

    BL_ASSERT(S.nGrow() == NUM_GROW);
    sources_for_hydro.setVal(0.0);
    int ng = 0; // TODO: This is currently the largest ngrow of the source data...maybe this needs fixing?
    for (int n = 0; n < src_list.size(); ++n)
    {
      MultiFab::Saxpy(sources_for_hydro,0.5,*new_sources[src_list[n]],0,0,NUM_STATE,ng);
      MultiFab::Saxpy(sources_for_hydro,0.5,*old_sources[src_list[n]],0,0,NUM_STATE,ng);
    }
#ifdef REACTIONS
    // Add I_R terms to advective forcing
    if (do_react == 1) {

      MultiFab::Add(sources_for_hydro, get_new_data(Reactions_Type), 0, FirstSpec, NumSpec, ng);
      MultiFab::Add(sources_for_hydro, get_new_data(Reactions_Type), NumSpec, Eden, 1, ng);

    }
#endif
    sources_for_hydro.FillBoundary(geom.periodicity());
    hydro_source.setVal(0);

    int finest_level = parent->finestLevel();

    const Real *dx = geom.CellSize();

    Real dx1 = dx[0];
    for (int d=1; d<BL_SPACEDIM; ++d) {
        dx1 *= dx[d];
    }

    std::array<Real,BL_SPACEDIM> dxD = {D_DECL(dx1, dx1, dx1)};
    const Real *dxDp = &(dxD[0]);

    Real courno    = -1.0e+200;

    MultiFab& S_new = get_new_data(State_Type);

    // note: the radiation consup currently does not fill these
    Real E_added_flux    = 0.;
    Real mass_added_flux = 0.;
    Real xmom_added_flux = 0.;
    Real ymom_added_flux = 0.;
    Real zmom_added_flux = 0.;
    Real mass_lost       = 0.;
    Real xmom_lost       = 0.;
    Real ymom_lost       = 0.;
    Real zmom_lost       = 0.;
    Real eden_lost       = 0.;
    Real xang_lost       = 0.;
    Real yang_lost       = 0.;
    Real zang_lost       = 0.;

    BL_PROFILE_VAR("PeleC::advance_hydro_pc_umdrv()", PC_UMDRV);

#ifdef _OPENMP
#pragma omp parallel reduction(+:E_added_flux,mass_added_flux)		\
    reduction(+:xmom_added_flux,ymom_added_flux,zmom_added_flux)	\
    reduction(+:mass_lost,xmom_lost,ymom_lost,zmom_lost)		\
    reduction(+:eden_lost,xang_lost,yang_lost,zang_lost) 		\
    reduction(max:courno)
#endif
    {
	FArrayBox pradial(Box::TheUnitBox(),1);
	IArrayBox bcMask[BL_SPACEDIM];

	Real cflLoc = -1.0e+200;
	int is_finest_level = (level == finest_level) ? 1 : 0;
    int flag_nscbc_isAnyPerio = (geom.isAnyPeriodic()) ? 1 : 0; 
    int flag_nscbc_perio[BL_SPACEDIM]; // For 3D, we will know which corners have a periodicity
    for (int d=0; d<BL_SPACEDIM; ++d) {
        flag_nscbc_perio[d] = (DefaultGeometry().isPeriodic(d)) ? 1 : 0;
    }

	const int*  domain_lo = geom.Domain().loVect();
	const int*  domain_hi = geom.Domain().hiVect();

    /* Temporary Fabs needed for Hydro Computation */ 
    FArrayBox q, qaux, src_q, D_DECL(flx1, flx2, flx3); 
    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi) 	
	{

	    const Box& bx  = mfi.tilebox();
	    const Box& qbx = amrex::grow(bx, NUM_GROW);
	    const int* lo = bx.loVect();
	    const int* hi = bx.hiVect();

        const Box& xfbx = surroundingNodes(bx, 0); 
        flx1.resize(xfbx, NVAR);
        Elixir flx1eli = flx1.elixir(); 
#if (AMREX_SPACEDIM >=2)
        const Box& yfbx = surroundingNodes(bx, 1);
        flx2.resize(yfbx, NVAR); 
        Elixir flx2eli = flx2.elixir(); 
#if (AMREX_SPACEDIM ==3) 
        const Box& zfbx = surroundingNodes(bx, 2); 
        flx3.resize(zfbx, NVAR); 
        Elixir flx3eli = flx3.elixir(); 
#endif
#endif

        auto const& s = S.array(mfi);     
        auto const& hyd_src = hydro_source.array(mfi); 
// Resize Temporary Fabs 
        q.resize(qbx, QVAR);
        qaux.resize(qbx, NQAUX); 
        src_q.resize(qbx, QVAR);
//Use Elixer Construct to steal the Fabs metadata 
        Elixir qeli = q.elixir(); 
        Elixir qauxeli = qaux.elixir(); 
        Elixir src_qeli = src_q.elixir(); 
//Get Arrays to pass to the gpu. 
        auto const& qarr = q.array(); 
        auto const& qauxar = qaux.array(); 
        auto const& srcqarr = src_q.array(); 

        BL_PROFILE_VAR("PeleC::ctoprim()", ctop); 
        AMREX_PARALLEL_FOR_3D(qbx, i, j, k, 
            {
                 PeleC_ctoprim(i,j,k, s, qarr, qauxar);                 

             });
        BL_PROFILE_VAR_STOP(ctop);

//TODO GPUize NCSCBC 
      // Imposing Ghost-Cells Navier-Stokes Characteristic BCs if "UserBC" are used
      // For the theory, see Motheau et al. AIAA J. Vol. 55, No. 10 : pp. 3399-3408, 2017. 
      //
      // The user should provide a bcnormal routine in bc_fill_module with additional optional arguments
      // to temporary fill ghost-cells for EXT_DIR and to provide target BC values.
      // See the examples.
      
      // Allocate fabs for bcMask. Note that we grow in the opposite direction
      // because the Riemann solver wants a face value in a ghost-cell
#if 0 
      for (int i = 0; i < BL_SPACEDIM ; i++)  {
        const Box& bxtmp = amrex::surroundingNodes(bx,i);
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

      if (nscbc_adv == 1)
      {
        impose_NSCBC(lo, hi, domain_lo, domain_hi,
                     BL_TO_FORTRAN(*statein),
                     BL_TO_FORTRAN(q.fab()),
                     BL_TO_FORTRAN(qaux.fab()),
                     D_DECL(BL_TO_FORTRAN(bcMask[0]),
                            BL_TO_FORTRAN(bcMask[1]),
                            BL_TO_FORTRAN(bcMask[2])),
                     &flag_nscbc_isAnyPerio, flag_nscbc_perio, 
                     &time, dx, &dt);
      }
#endif
        BL_PROFILE_VAR("PeleC::srctoprim()", srctop);  
        const auto& src_in  = sources_for_hydro.array(mfi);
                AMREX_PARALLEL_FOR_3D(qbx,i, j, k,{
                      PeleC_srctoprim(i, j, k, qarr, qauxar, src_in, srcqarr); 
                });
        BL_PROFILE_VAR_STOP(srctop); 

        if (!DefaultGeometry().IsCartesian()) {
            pradial.resize(amrex::surroundingNodes(bx,0),1);
        }

//#if (AMREX_SPACEDIM < 3)
//                amrex::FArrayBox *prad = &pradial; 
//#endif
#ifdef AMREX_USE_CUDA        
    auto run = RunOn::Gpu; 
#else
    auto run = RunOn::Cpu;
#endif
            BL_PROFILE_VAR("PeleC::umdrv()", purm); 
                         PeleC_umdrv
                        (is_finest_level, time,
                         bx, 
                         domain_lo, domain_hi,
                         phys_bc.lo(), phys_bc.hi(), 
                         s, hyd_src, qarr,
                         qauxar, srcqarr,
                         dx, dt,
                         D_DECL( flx1.array(),
                                 flx2.array(), 
                                 flx3.array()), 
                         D_DECL(area[0].array(mfi),
                                area[1].array(mfi),
                                area[2].array(mfi)),
                #if (BL_SPACEDIM < 3)
                         dLogArea[0].array(mfi),
                #endif
                         volume.array(mfi),
                         cflLoc); 
            BL_PROFILE_VAR_STOP(purm); 
            
            BL_PROFILE_VAR("courno + flux reg", crno); 
            courno = std::max(courno,cflLoc);
                    if (do_reflux  && sub_iteration == sub_ncycle-1 )
                    {
                      if (level < finest_level)
                      {
                        getFluxReg(level+1).CrseAdd(mfi,
                        {D_DECL(&(flx1),
                                &(flx2),
                                &(flx3))}, dxDp,dt, run);

                        if (!DefaultGeometry().IsCartesian()) {
                            amrex::Abort("Flux registers not r-z compatible yet");
                            //getPresReg(level+1).CrseAdd(mfi,pradial, dx,dt);
                        }
                      }

                      if (level > 0)
                      {
                        getFluxReg(level).FineAdd(mfi, 
                        {D_DECL(&(flx1),
                                &(flx2),
                                &(flx3))}, dxDp,dt, run);

                        if (!DefaultGeometry().IsCartesian()) {
                            amrex::Abort("Flux registers not r-z compatible yet");
                            //getPresReg(level).FineAdd(mfi,pradial, dx,dt);
                        }
                      }
                    }
            BL_PROFILE_VAR_STOP(crno); 
         } // MFIter loop
    } // end of OMP parallel region

    BL_PROFILE_VAR_STOP(PC_UMDRV);

    // Flush Fortran output


    if (track_grid_losses)
    {
	material_lost_through_boundary_temp[0] += mass_lost;
	material_lost_through_boundary_temp[1] += xmom_lost;
	material_lost_through_boundary_temp[2] += ymom_lost;
	material_lost_through_boundary_temp[3] += zmom_lost;
	material_lost_through_boundary_temp[4] += eden_lost;
	material_lost_through_boundary_temp[5] += xang_lost;
	material_lost_through_boundary_temp[6] += yang_lost;
	material_lost_through_boundary_temp[7] += zang_lost;
    }

    if (print_energy_diagnostics)
    {
	Real foo[5] = {E_added_flux, xmom_added_flux, ymom_added_flux,
		       zmom_added_flux, mass_added_flux};

#ifdef BL_LAZY
	Lazy::QueueReduction( [=] () mutable {
#endif
		ParallelDescriptor::ReduceRealSum(foo, 5, ParallelDescriptor::IOProcessorNumber());

		if (ParallelDescriptor::IOProcessor())
		{
		    E_added_flux    = foo[0];
		    xmom_added_flux = foo[1];
		    ymom_added_flux = foo[2];
		    zmom_added_flux = foo[3];
		    mass_added_flux = foo[4];
#ifdef AMREX_DEBUG
		    std::cout << "mass added from fluxes                      : " <<
			mass_added_flux << std::endl;
		    std::cout << "xmom added from fluxes                      : " <<
			xmom_added_flux << std::endl;
		    std::cout << "ymom added from fluxes                      : " <<
			ymom_added_flux << std::endl;
		    std::cout << "zmom added from fluxes                      : " <<
			zmom_added_flux << std::endl;
		    std::cout << "(rho E) added from fluxes                   : " <<
			E_added_flux << std::endl;
#endif
		}
#ifdef BL_LAZY
	    });
#endif
    }

    if (courno > 1.0) {
	std::cout << "WARNING -- EFFECTIVE CFL AT THIS LEVEL " << level << " IS " << courno << '\n';
	if (hard_cfl_limit == 1)
	    amrex::Abort("CFL is too high at this level -- go back to a checkpoint and restart with lower cfl number");
    }
#endif // PELEC_USE_MOL
}
