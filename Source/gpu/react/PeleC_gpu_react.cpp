#include <PeleC.H>
#include <AMReX_DistributionMapping.H>
#include "PeleC_gpu_react.H" 


using std::string;
using namespace amrex;

void
PeleC::react_state_gpu(Real time, Real dt, bool react_init, MultiFab* A_aux)
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
    const int ng = S_new.nGrow(); 
    prefetchToDevice(S_new); 

    // Create a MultiFab with all of the non-reacting source terms.

    MultiFab Atmp, *Ap;

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
    prefetchToDevice(reactions); 
    if (use_reactions_work_estimate) {
    	amrex::Abort("Need to implement redistribution of chemistry work");
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
//        FArrayBox w;
        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {

            const Box& bx = mfi.growntilebox(ng);

            auto const& uold = react_init ? S_new.array(mfi) : get_old_data(State_Type).array(mfi);
            auto const& unew       = S_new.array(mfi);
            auto const& a          = Ap->array(mfi);
//            w.resize(bx,1);
            auto const& I_R        = reactions.array(mfi);
            int do_update         = react_init ? 0 : 1;  // TODO: Update here? Or just get reaction source?

            if(chem_integrator==1)
            {
                amrex::Abort("Implicit Chemistry is not implemented yet on GPU only explicit."); 
/*                pc_react_state(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
                        uold.dataPtr(),  ARLIM_3D(uold.loVect()),  ARLIM_3D(uold.hiVect()),
                        unew.dataPtr(),  ARLIM_3D(unew.loVect()),  ARLIM_3D(unew.hiVect()),
                        a.dataPtr(),     ARLIM_3D(a.loVect()),     ARLIM_3D(a.hiVect()),
                        m.dataPtr(),     ARLIM_3D(m.loVect()),     ARLIM_3D(m.hiVect()),
                        w.dataPtr(),     ARLIM_3D(w.loVect()),     ARLIM_3D(w.hiVect()),
                        I_R.dataPtr(),   ARLIM_3D(I_R.loVect()),   ARLIM_3D(I_R.hiVect()),
                        time, dt, do_update); */ 
            }
            else
            {
               const int nsubsteps_min=adaptrk_nsubsteps_min;
               const int nsubsteps_max=adaptrk_nsubsteps_max;
               const int nsubsteps_guess=adaptrk_nsubsteps_guess;  
               const amrex::Real errtol = adaptrk_errtol; 

                AMREX_PARALLEL_FOR_3D(bx, i, j, k, {
                    PeleC_expl_reactions(i, j, k, uold, unew, 
                                         a, I_R, dt, nsubsteps_min, 
                                         nsubsteps_max, nsubsteps_guess,
                                         errtol, do_update);
                    });
             }

            if (do_react_load_balance || do_mol_load_balance)
            {
                amrex::Abort("Reaction Load Balacing not implemented yet for GPU"); 
//                get_new_data(Work_Estimate_Type)[mfi].plus(w);
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
