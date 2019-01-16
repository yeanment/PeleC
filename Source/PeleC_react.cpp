
#include "PeleC.H"
#include "PeleC_F.H"

#include "AMReX_DistributionMapping.H"

#if defined(USE_DVODE) || defined(USE_FORTRAN_CVODE) || defined(USE_SDC_FORTRAN)
#else
#include <actual_Creactor.h>
#endif

using std::string;
using namespace amrex;

void
PeleC::react_state(Real time, Real dt, bool react_init, MultiFab* A_aux)
{
  /*
    Update I_R, and recompute S_new
   */
    BL_PROFILE("PeleC::react_state()");

    const Real strt_time = ParallelDescriptor::second();

    BL_ASSERT(do_react == 1);

    if (verbose && ParallelDescriptor::IOProcessor()) {
        if (react_init) {
            std::cout << "... Initializing reactions, using interval dt = " << dt << std::endl;
        }
        else {
            std::cout << "... Computing reactions for dt = " << dt << std::endl;
        }
    }

    MultiFab& S_new = get_new_data(State_Type);

    // Build the burning mask, in case the state has ghost zones.

    const int ng = S_new.nGrow();
    auto interior_mask = build_interior_boundary_mask(ng);

    // Create a MultiFab with all of the non-reacting source terms.

    MultiFab Atmp, *Ap;
#if defined(USE_DVODE) || defined(USE_FORTRAN_CVODE) || defined(USE_SDC_FORTRAN)
#else
    MultiFab rY, rYs;
    MultiFab rE, rEs;
    rY.define(grids, dmap, NumSpec+1, ng, MFInfo(), Factory());
    rYs.define(grids, dmap, NumSpec, ng, MFInfo(), Factory());
    rE.define(grids, dmap, 1, ng, MFInfo(), Factory());
    rEs.define(grids, dmap, 1, ng, MFInfo(), Factory());
    rY.setVal(0);
    rYs.setVal(0);
    rE.setVal(0);
    rEs.setVal(0);
    int reInit = 1;
#endif

    if (A_aux == nullptr || react_init)
    {
      Atmp.define(grids, dmap, NUM_STATE, ng, MFInfo(), Factory());
      Atmp.setVal(0);
      Ap = &Atmp;
    }

    if (!react_init)
    {
      // Build non-reacting source term, and an S_new that does not include reactions

      if (A_aux == nullptr)
      {
        for (int n = 0; n < src_list.size(); ++n)
        {
          MultiFab::Saxpy(Atmp,0.5,*new_sources[src_list[n]],0,0,NUM_STATE,ng);
          MultiFab::Saxpy(Atmp,0.5,*old_sources[src_list[n]],0,0,NUM_STATE,ng);
        }
        if (do_hydro && !do_mol_AD)
        {
          MultiFab::Add(Atmp,hydro_source,0,0,NUM_STATE,ng);
        }
      }
      else
      {
        Ap = A_aux;
      }

      MultiFab& S_old = get_old_data(State_Type);
      MultiFab::Copy(S_new,S_old,0,0,NUM_STATE,ng);
      MultiFab::Saxpy(S_new,dt,*Ap,0,0,NUM_STATE,ng);
    }

    MultiFab& reactions = get_new_data(Reactions_Type);
    reactions.setVal(0.0);

    if (use_reactions_work_estimate) {
	amrex::Abort("Need to implement redistribution of chemistry work");
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {

        FArrayBox w;
        for (MFIter mfi(S_new, true); mfi.isValid(); ++mfi)
        {

          const Box& bx = mfi.growntilebox(ng);

          const FArrayBox& uold = react_init ? S_new[mfi] : get_old_data(State_Type)[mfi];
          FArrayBox& unew       = S_new[mfi];
          FArrayBox& a          = (*Ap)[mfi];
          const IArrayBox& m    = (*interior_mask)[mfi];
          w.resize(bx,1);
          FArrayBox& I_R        = reactions[mfi];
#if defined(USE_DVODE) || defined(USE_FORTRAN_CVODE) || defined(USE_SDC_FORTRAN)
#else
          FArrayBox& Y          = rY[mfi];
          FArrayBox& Ys         = rYs[mfi];
          FArrayBox& E          = rE[mfi];
          FArrayBox& Es         = rEs[mfi];
#endif
          int do_update         = react_init ? 0 : 1;  // TODO: Update here? Or just get reaction source?

#if defined(USE_DVODE) || defined(USE_FORTRAN_CVODE) || defined(USE_SDC_FORTRAN)
          pc_react_state(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
                         uold.dataPtr(),  ARLIM_3D(uold.loVect()),  ARLIM_3D(uold.hiVect()),
                         unew.dataPtr(),  ARLIM_3D(unew.loVect()),  ARLIM_3D(unew.hiVect()),
                         a.dataPtr(),     ARLIM_3D(a.loVect()),     ARLIM_3D(a.hiVect()),
                         m.dataPtr(),     ARLIM_3D(m.loVect()),     ARLIM_3D(m.hiVect()),
                         w.dataPtr(),     ARLIM_3D(w.loVect()),     ARLIM_3D(w.hiVect()),
                         I_R.dataPtr(),   ARLIM_3D(I_R.loVect()),   ARLIM_3D(I_R.hiVect()),
                         time, dt, do_update);
#else
          pc_prereact_state(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
                         uold.dataPtr(),  ARLIM_3D(uold.loVect()),  ARLIM_3D(uold.hiVect()),
                         unew.dataPtr(),  ARLIM_3D(unew.loVect()),  ARLIM_3D(unew.hiVect()),
                         a.dataPtr(),     ARLIM_3D(a.loVect()),     ARLIM_3D(a.hiVect()),
                         m.dataPtr(),     ARLIM_3D(m.loVect()),     ARLIM_3D(m.hiVect()),
                         Y.dataPtr(),     ARLIM_3D(Y.loVect()),     ARLIM_3D(Y.hiVect()),
                         Ys.dataPtr(),     ARLIM_3D(Ys.loVect()),     ARLIM_3D(Ys.hiVect()),
                         E.dataPtr(),     ARLIM_3D(E.loVect()),     ARLIM_3D(E.hiVect()),
                         Es.dataPtr(),     ARLIM_3D(Es.loVect()),     ARLIM_3D(Es.hiVect()),
                         time, dt);

	  printf("BEFORE BoxIterator \n" );
	  for (BoxIterator bit(bx); bit.ok(); ++bit) {
		  double tmp_vect[NumSpec+1];
		  double tmp_src_vect[NumSpec];
		  double tmp_vect_energy[1];
		  double tmp_src_vect_energy[1];
		  for (int i=0;i<NumSpec; i++){
			  tmp_vect[i] = Y(bit(),i);
			  tmp_src_vect[i] = Ys(bit(),i);
		  }
		  tmp_vect[NumSpec] = Y(bit(),NumSpec);   
		  tmp_vect_energy[0] = E(bit(),0);   
		  tmp_src_vect_energy[0] = Es(bit(),0);   
		  double plo = 1013250.0;
		  printf("time, dt %14.6e %14.6e \n",time, dt );
		  int cost = actual_cReact(tmp_vect, tmp_src_vect, 
				  tmp_vect_energy, tmp_src_vect_energy,
				  &plo, &dt, &time, &reInit);
		  for (int i=0;i<NumSpec+1; i++){
			  Y(bit(),i) = tmp_vect[i];
		  }
		  E(bit(),0) = tmp_vect_energy[0];   
		  w(bit(),0) = cost; 
                  //reInit                = 0;
	  }

          pc_postreact_state(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
                         uold.dataPtr(),  ARLIM_3D(uold.loVect()),  ARLIM_3D(uold.hiVect()),
                         unew.dataPtr(),  ARLIM_3D(unew.loVect()),  ARLIM_3D(unew.hiVect()),
                         a.dataPtr(),     ARLIM_3D(a.loVect()),     ARLIM_3D(a.hiVect()),
                         Y.dataPtr(),     ARLIM_3D(Y.loVect()),     ARLIM_3D(Y.hiVect()),
                         E.dataPtr(),     ARLIM_3D(E.loVect()),     ARLIM_3D(E.hiVect()),
                         m.dataPtr(),     ARLIM_3D(m.loVect()),     ARLIM_3D(m.hiVect()),
                         w.dataPtr(),     ARLIM_3D(w.loVect()),     ARLIM_3D(w.hiVect()),
                         I_R.dataPtr(),   ARLIM_3D(I_R.loVect()),   ARLIM_3D(I_R.hiVect()),
                         time, dt, do_update);

#endif

          if (do_react_load_balance || do_mol_load_balance)
          {
            get_new_data(Work_Estimate_Type)[mfi].plus(w);
          }
        }
    }

    if (ng > 0)
        S_new.FillBoundary(geom.periodicity());

    if (verbose > 1) {

        const int IOProc   = ParallelDescriptor::IOProcessorNumber();
        Real      run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
	Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time, IOProc);

	if (ParallelDescriptor::IOProcessor())
	  std::cout << "PeleC::react_state() time = " << run_time << "\n";
#ifdef BL_LAZY
	});
#endif

    }
}
