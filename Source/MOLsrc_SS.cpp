#include "Diffusion.H"

#ifdef PELEC_USE_EB
void
PeleC::getMOLSrcTerm_SS(
  const amrex::MultiFab& S,
  amrex::MultiFab& MOLSrcTerm,
  amrex::Real /*time*/,
  amrex::Real dt,
  amrex::Real flux_factor)
{
  BL_PROFILE("PeleC::getMOLSrcTerm()");
  BL_PROFILE_VAR_NS("diffusion_stuff", diff);
  if (
    diffuse_temp == 0 && diffuse_enth == 0 && diffuse_spec == 0 &&
    diffuse_vel == 0 && do_hydro == 0) 
  {
    MOLSrcTerm.setVal(0, 0, NVAR, MOLSrcTerm.nGrow());
    return;
  }

  const int nCompTr = dComp_lambda + 1;
  const int do_harmonic = 1; // TODO: parmparse this
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

  amrex::Real dx1 = dx[0];
  for (int dir = 1; dir < AMREX_SPACEDIM; ++dir) 
  {
    dx1 *= dx[dir];
  }
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxD = {
    {AMREX_D_DECL(dx1, dx1, dx1)}};

  // Fetch some gpu arrays
  prefetchToDevice(S);
  prefetchToDevice(MOLSrcTerm);

  auto const& fact =
    dynamic_cast<amrex::EBFArrayBoxFactory const&>(S.Factory());
  auto const& flags = fact.getMultiEBCellFlagFab();
  // amrex::Elixir flags_eli = flags.elixir();
  amrex::MultiFab* cost = nullptr;

  if (do_mol_load_balance) 
  {
    cost = &(get_new_data(Work_Estimate_Type));
  }

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  {
    for (amrex::MFIter mfi(MOLSrcTerm, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) 
    {
      const amrex::Box vbox = mfi.tilebox();
      int ng = S.nGrow();
      const amrex::Box gbox = amrex::grow(vbox, ng);
      const amrex::Box cbox = amrex::grow(vbox, ng - 1);
      auto const& MOLSrc = MOLSrcTerm.array(mfi);

      amrex::Real wt = amrex::ParallelDescriptor::second();
      const auto& flag_fab = flags[mfi];
      // amrex::Elixir flag_fab_eli = flag_fab.elixir();
      amrex::FabType typ = flag_fab.getType(vbox);
      if (typ == amrex::FabType::covered) 
      {
        //set molsrc to 0
        setV(vbox, NVAR, MOLSrc, 0);
        if (do_mol_load_balance && cost) 
        {
          wt = (amrex::ParallelDescriptor::second() - wt) / vbox.d_numPts();
          (*cost)[mfi].plus<amrex::RunOn::Device>(wt, vbox);
        }
        continue;
      }
      // Note on typ: if interior cells (vbox) are all covered, no need to
      // do anything. But otherwise, we need to do EB stuff if there are any
      // cut cells within 1 grow cell (cbox) due to fix_div_and_redistribute
      typ = flag_fab.getType(cbox);

      // const int* lo = vbox.loVect();
      // const int* hi = vbox.hiVect();

      BL_PROFILE_VAR_START(diff);
      int nqaux = NQAUX > 0 ? NQAUX : 1;
      amrex::FArrayBox q(gbox, QVAR);
      amrex::FArrayBox qaux(gbox, nqaux);
      amrex::FArrayBox coeff_cc(gbox, nCompTr);
      amrex::Elixir qeli = q.elixir();
      amrex::Elixir qauxeli = qaux.elixir();
      amrex::Elixir coefeli = coeff_cc.elixir();
      auto const& s = S.array(mfi);
      auto const& qar = q.array();
      auto const& qauxar = qaux.array();

      // Get primitives, Q, including (Y, T, p, rho) from 
      // conserved state
      // required for D term
      {
        BL_PROFILE("PeleC::ctoprim()");
        PassMap const* lpmap = d_pass_map;
        amrex::ParallelFor(
          gbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
          {
            pc_ctoprim(i, j, k, s, qar, qauxar, *lpmap);
          });
      }
      
      // Compute transport coefficients, coincident with Q
      auto const& coe_cc = coeff_cc.array();
      {
        auto const& qar_yin = q.array(QFS);
        auto const& qar_Tin = q.array(QTEMP);
        auto const& qar_rhoin = q.array(QRHO);
        auto const& coe_rhoD = coeff_cc.array(dComp_rhoD);
        auto const& coe_mu = coeff_cc.array(dComp_mu);
        auto const& coe_xi = coeff_cc.array(dComp_xi);
        auto const& coe_lambda = coeff_cc.array(dComp_lambda);
        BL_PROFILE("PeleC::get_transport_coeffs()");
        // Get Transport coefs on GPU.
        pele::physics::transport::TransParm const* ltransparm =
          pele::physics::transport::trans_parm_g;
        amrex::launch(gbox, [=] AMREX_GPU_DEVICE(amrex::Box const& tbx) 
        {
          auto trans = pele::physics::PhysicsType::transport();
          trans.get_transport_coeffs(
            tbx, qar_yin, qar_Tin, qar_rhoin, coe_rhoD, coe_mu, coe_xi,
            coe_lambda, ltransparm);
        });
      }

      amrex::FArrayBox flux_ec[AMREX_SPACEDIM];
      amrex::Elixir flux_eli[AMREX_SPACEDIM];
      const amrex::Box eboxes[AMREX_SPACEDIM] = {AMREX_D_DECL(
      amrex::surroundingNodes(cbox, 0), amrex::surroundingNodes(cbox, 1),
      amrex::surroundingNodes(cbox, 2))};
      amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx;
      const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM>
      areafab{{AMREX_D_DECL(area[0].array(mfi), area[1].array(mfi), area[2].array(mfi))}};
      
      for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
      {
        flux_ec[dir].resize(eboxes[dir], NVAR);
        flux_eli[dir] = flux_ec[dir].elixir();
        flx[dir] = flux_ec[dir].array();
        setV(eboxes[dir], NVAR, flx[dir], 0);
      }

      amrex::FArrayBox Dfab(cbox, NVAR);
      amrex::Elixir Dfab_eli = Dfab.elixir();
      auto const& Dterm = Dfab.array();
      setV(cbox, NVAR, Dterm, 0.0);

      pc_compute_diffusion_flux_SS
      (
        cbox, qar, coe_cc, flx, a, dx, do_harmonic,
        typ, Ncut, d_sv_eb_bndry_geom, flags.array(mfi),
        captured_eb_isothermal,captured_eb_boundary_T
      );

      // Compute flux divergence (1/Vol).Div(F.A)
      {
        BL_PROFILE("PeleC::pc_flux_div()");
        auto const& vol = volume.array(mfi);
        amrex::ParallelFor(
          cbox, NVAR,
          [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept 
          {
            pc_flux_div(
              i, j, k, n, AMREX_D_DECL(flx[0], flx[1], flx[2]), vol, Dterm);
          });
      }

      if (diffuse_temp == 0 && diffuse_enth == 0) 
      {
        setC(cbox, Eden, Eint, Dterm, 0.0);
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
        {
          setC(eboxes[dir], Eden, Eint, flx[dir], 0.0);
        }
      }
      if (diffuse_spec == 0) 
      {
        setC(cbox, FirstSpec, FirstSpec + NUM_SPECIES, Dterm, 0.0);
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
        {
          setC(eboxes[dir], FirstSpec, FirstSpec + NUM_SPECIES, flx[dir], 0.0);
        }
      }
      if (diffuse_vel == 0) 
      {
        setC(cbox, Xmom, Xmom + 3, Dterm, 0.0);
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
        {
          setC(eboxes[dir], Xmom, Xmom + 3, flx[dir], 0.0);
        }
      }
      BL_PROFILE_VAR_STOP(diff);

      if (do_hydro && do_mol) 
      {
        // save off the diffusion source term and fluxes (don't want to filter
        // these)
        amrex::FArrayBox diffusion_flux[AMREX_SPACEDIM];
        amrex::Elixir diffusion_flux_eli[AMREX_SPACEDIM];
        amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM>
          diffusion_flux_arr;
        if (use_explicit_filter) 
        {
          for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
          {
            diffusion_flux[dir].resize(flux_ec[dir].box(), NVAR);
            diffusion_flux_eli[dir] = diffusion_flux[dir].elixir();
            diffusion_flux_arr[dir] = diffusion_flux[dir].array();
            copy_array4(
              flux_ec[dir].box(), flux_ec[dir].nComp(), flx[dir],
              diffusion_flux_arr[dir]);
          }
        }

        { 
          // Get face-centered hyperbolic fluxes and their divergences.
          // Get hyp flux at EB wall
          BL_PROFILE("PeleC::pc_hyp_mol_flux()");
          
          // auto const& vol = volume.array(mfi);
          pc_compute_hyp_mol_flux_SS(
            cbox, qar, qauxar, flx, a, dx, plm_iorder,
            eb_small_vfrac, vfrac.array(mfi), flags.array(mfi),
            d_sv_eb_bndry_geom, Ncut, d_eb_flux_thdlocal, nFlux
          );
        }

        // Filter hydro source term and fluxes here
        if (use_explicit_filter) 
        {
          // Get the hydro term
          amrex::FArrayBox hydro_flux[AMREX_SPACEDIM];
          amrex::Elixir hydro_flux_eli[AMREX_SPACEDIM];
          amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM>
            hydro_flux_arr;
          for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
          {
            hydro_flux[dir].resize(flux_ec[dir].box(), NVAR);
            hydro_flux_eli[dir] = hydro_flux[dir].elixir();
            hydro_flux_arr[dir] = hydro_flux[dir].array();
            lincomb_array4(
              flux_ec[dir].box(), Density, NVAR, flx[dir],
              diffusion_flux_arr[dir], 1.0, -1.0, hydro_flux_arr[dir]);
          }

          // Filter
          const amrex::Box fbox = amrex::grow(cbox, -nGrowF);
          for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
          {
            const amrex::Box& bxtmp = amrex::surroundingNodes(fbox, dir);
            amrex::FArrayBox filtered_hydro_flux;
            filtered_hydro_flux.resize(bxtmp, NVAR);
            amrex::Elixir filtered_hydro_flux_eli =
              filtered_hydro_flux.elixir();
            les_filter.apply_filter(
              bxtmp, hydro_flux[dir], filtered_hydro_flux, Density, NVAR);

            setV(bxtmp, hydro_flux[dir].nComp(), hydro_flux_arr[dir], 0.0);
            copy_array4(
              bxtmp, hydro_flux[dir].nComp(), filtered_hydro_flux.array(),
              hydro_flux_arr[dir]);
          }

          // Combine with diffusion
          for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
          {
            lincomb_array4(
              diffusion_flux[dir].box(), Density, NVAR, diffusion_flux_arr[dir],
              hydro_flux_arr[dir], 1.0, 1.0, flx[dir]);
          }
        }

        // Compute flux divergence (1/Vol).Div(F.A)
        {
          BL_PROFILE("PeleC::pc_flux_div()");
          auto const& vol = volume.array(mfi);
          amrex::ParallelFor(
            cbox, NVAR,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept 
            {
              pc_flux_div(
                i, j, k, n, AMREX_D_DECL(flx[0], flx[1], flx[2]), vol, Dterm);
            });
        }
      }

#ifdef AMREX_USE_GPU
      auto device = amrex::RunOn::Gpu;
#else
      auto device = amrex::RunOn::Cpu;
#endif

      if (do_reflux && flux_factor != 0)
      {
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
        {
          amrex::ParallelFor(
            eboxes[dir], NVAR,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept 
            {
              flx[dir](i, j, k, n) *= flux_factor;
            });
        }

        if (level < parent->finestLevel()) 
        {
          getFluxReg(level + 1).CrseAdd(
            mfi, {{AMREX_D_DECL(&flux_ec[0], &flux_ec[1], &flux_ec[2])}},
            dxD.data(), dt, device);
        }

        if (level > 0) {
          getFluxReg(level).FineAdd(
            mfi, {{AMREX_D_DECL(&flux_ec[0], &flux_ec[1], &flux_ec[2])}},
            dxD.data(), dt, device);
        }
      }

      // Extrapolate to GhostCells
      if (MOLSrcTerm.nGrow() > 0) 
      {
        BL_PROFILE("PeleC::diffextrap()");
        const int mg = MOLSrcTerm.nGrow();
        const amrex::Box bx = mfi.tilebox();
        const auto* low = bx.loVect();
        const auto* high = bx.hiVect();
        auto dlo = Dterm.begin;
        auto dhi = Dterm.end;
        const int AMREX_D_DECL(lx = low[0], ly = low[1], lz = low[2]);
        const int AMREX_D_DECL(hx = high[0], hy = high[1], hz = high[2]);
        amrex::ParallelFor(
          bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
          {
            pc_diffextrap(
              i, j, k, Dterm, mg, UMX, UMZ + 1, AMREX_D_DECL(lx, ly, lz),
              AMREX_D_DECL(hx, hy, hz), dlo, dhi);
            pc_diffextrap(
              i, j, k, Dterm, mg, UFS, UFS + NUM_SPECIES,
              AMREX_D_DECL(lx, ly, lz), AMREX_D_DECL(hx, hy, hz), dlo, dhi);
            pc_diffextrap(
              i, j, k, Dterm, mg, UEDEN, UEDEN + 1, AMREX_D_DECL(lx, ly, lz),
              AMREX_D_DECL(hx, hy, hz), dlo, dhi);
         });
      }

      copy_array4(vbox, NVAR, Dterm, MOLSrc);

#ifdef PELEC_USE_EB
      if (do_mol_load_balance && cost) 
      {
        amrex::Gpu::streamSynchronize();
        wt = (amrex::ParallelDescriptor::second() - wt) / vbox.d_numPts();
        (*cost)[mfi].plus<amrex::RunOn::Device>(wt, vbox);
      }
#endif
    }
  }
}
