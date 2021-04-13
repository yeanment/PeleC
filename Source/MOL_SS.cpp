#include "MOL.H"

#ifdef PELEC_USE_EB
void
pc_compute_hyp_mol_flux_SS(
        const amrex::Box& cbox,
        const amrex::Array4<const amrex::Real>& q,
        const amrex::Array4<const amrex::Real>& qaux,
        const amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
        const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> 
        area,
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> del,
        const int plm_iorder,
        const amrex::Array4<const amrex::Real>& vfrac,
        const amrex::Array4<amrex::EBCellFlag const>& flags)
{
    const int R_RHO = 0;
    const int R_UN = 1;
    const int R_UT1 = 2;
    const int R_UT2 = 3;
    const int R_P = 4;
    const int R_Y = 5;
    const int bc_test_val = 1;

    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
    {
        amrex::FArrayBox dq_fab(cbox, QVAR);
        amrex::Elixir dq_fab_eli = dq_fab.elixir();
        auto const& dq = dq_fab.array();
        setV(cbox, QVAR, dq, 0.0);

        // dimensional indexing
        const amrex::GpuArray<const int, 3> bdim{{dir == 0, dir == 1, dir == 2}};
        const amrex::GpuArray<const int, 3> q_idx{
            {bdim[0] * QU + bdim[1] * QV + bdim[2] * QW,
                bdim[0] * QV + bdim[1] * QU + bdim[2] * QU,
                bdim[0] * QW + bdim[1] * QW + bdim[2] * QV}};
        const amrex::GpuArray<const int, 3> f_idx{
            {bdim[0] * UMX + bdim[1] * UMY + bdim[2] * UMZ,
                bdim[0] * UMY + bdim[1] * UMX + bdim[2] * UMX,
                bdim[0] * UMZ + bdim[1] * UMZ + bdim[2] * UMY}};

        if (plm_iorder != 1) 
        {
            amrex::ParallelFor(
                    cbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
                    {
                    mol_slope_SS(i, j, k, bdim, q_idx, q, qaux, dq,flags);
                    });
        }

        const amrex::Box tbox = amrex::grow(cbox, dir, -1);
        const amrex::Box ebox = amrex::surroundingNodes(tbox, dir);
        amrex::ParallelFor(
                ebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
        {
                const int ii = i - bdim[0];
                const int jj = j - bdim[1];
                const int kk = k - bdim[2];

                if(flags(i,j,k).isRegular() && flags(ii,jj,kk).isRegular())
                {

                amrex::Real qtempl[5 + NUM_SPECIES] = {0.0};
                qtempl[R_UN] =
                q(ii, jj, kk, q_idx[0]) +
                0.5 * ((dq(ii, jj, kk, 1) - dq(ii, jj, kk, 0)) / q(ii, jj, kk, QRHO));
                qtempl[R_P] =
                q(ii, jj, kk, QPRES) +
                0.5 * (dq(ii, jj, kk, 0) + dq(ii, jj, kk, 1)) * qaux(ii, jj, kk, QC);
                qtempl[R_UT1] = q(ii, jj, kk, q_idx[1]) + 0.5 * dq(ii, jj, kk, 2);
                qtempl[R_UT2] = q(ii, jj, kk, q_idx[2]) + 0.5 * dq(ii, jj, kk, 3);
                qtempl[R_RHO] = 0.0;

                for (int n = 0; n < NUM_SPECIES; n++) 
                {
                    qtempl[R_Y + n] = q(ii, jj, kk, QFS + n) * q(ii, jj, kk, QRHO) +
                        0.5 * (dq(ii, jj, kk, 4 + n) +
                                q(ii, jj, kk, QFS + n) *
                                (dq(ii, jj, kk, 0) + dq(ii, jj, kk, 1)) /
                                qaux(ii, jj, kk, QC));
                    qtempl[R_RHO] += qtempl[R_Y + n];
                }

                for (int n = 0; n < NUM_SPECIES; n++) 
                {
                    qtempl[R_Y + n] = qtempl[R_Y + n] / qtempl[R_RHO];
                }

                amrex::Real qtempr[5 + NUM_SPECIES] = {0.0};
                qtempr[R_UN] =
                    q(i, j, k, q_idx[0]) -
                    0.5 * ((dq(i, j, k, 1) - dq(i, j, k, 0)) / q(i, j, k, QRHO));
                qtempr[R_P] = q(i, j, k, QPRES) - 0.5 *
                    (dq(i, j, k, 0) + dq(i, j, k, 1)) *
                    qaux(i, j, k, QC);
                qtempr[R_UT1] = q(i, j, k, q_idx[1]) - 0.5 * dq(i, j, k, 2);
                qtempr[R_UT2] = q(i, j, k, q_idx[2]) - 0.5 * dq(i, j, k, 3);
                qtempr[R_RHO] = 0.0;

                for (int n = 0; n < NUM_SPECIES; n++) 
                {
                    qtempr[R_Y + n] =
                        q(i, j, k, QFS + n) * q(i, j, k, QRHO) -
                        0.5 * (dq(i, j, k, 4 + n) + q(i, j, k, QFS + n) *
                                (dq(i, j, k, 0) + dq(i, j, k, 1)) /
                                qaux(i, j, k, QC));
                    qtempr[R_RHO] += qtempr[R_Y + n];
                }
                for (int n = 0; n < NUM_SPECIES; n++) 
                {
                    qtempr[R_Y + n] = qtempr[R_Y + n] / qtempr[R_RHO];
                }

                const amrex::Real cavg =
                    0.5 * (qaux(i, j, k, QC) + qaux(ii, jj, kk, QC));
                const amrex::Real csmall = amrex::min<amrex::Real>(
                        qaux(i, j, k, QCSML), qaux(ii, jj, kk, QCSML));

                amrex::Real eos_state_rho;
                amrex::Real eos_state_p;
                amrex::Real eos_state_e;
                amrex::Real eos_state_cs;
                amrex::Real eos_state_gamma;
                amrex::Real eos_state_T;

                eos_state_rho = qtempl[R_RHO];
                eos_state_p = qtempl[R_P];
                amrex::Real spl[NUM_SPECIES];
                for (int n = 0; n < NUM_SPECIES; n++) 
                {
                    spl[n] = qtempl[R_Y + n];
                }
                auto eos = pele::physics::PhysicsType::eos();
                eos.RYP2T(eos_state_rho, spl, eos_state_p, eos_state_T);
                eos.RTY2E(eos_state_rho, eos_state_T, spl, eos_state_e);
                eos.RTY2G(eos_state_rho, eos_state_T, spl, eos_state_gamma);
                eos.RTY2Cs(eos_state_rho, eos_state_T, spl, eos_state_cs);

                const amrex::Real rhoe_l = eos_state_rho * eos_state_e;
                const amrex::Real gamc_l = eos_state_gamma;

                eos_state_rho = qtempr[R_RHO];
                eos_state_p = qtempr[R_P];
                amrex::Real spr[NUM_SPECIES];
                for (int n = 0; n < NUM_SPECIES; n++) 
                {
                    spr[n] = qtempr[R_Y + n];
                }
                eos.RYP2T(eos_state_rho, spr, eos_state_p, eos_state_T);
                eos.RTY2E(eos_state_rho, eos_state_T, spr, eos_state_e);
                eos.RTY2G(eos_state_rho, eos_state_T, spr, eos_state_gamma);
                eos.RTY2Cs(eos_state_rho, eos_state_T, spr, eos_state_cs);

                const amrex::Real rhoe_r = eos_state_rho * eos_state_e;
                const amrex::Real gamc_r = eos_state_gamma;

                amrex::Real flux_tmp[NVAR] = {0.0};
                amrex::Real ustar = 0.0;

                amrex::Real tmp0;
                amrex::Real tmp1;
                amrex::Real tmp2;
                amrex::Real tmp3;
                amrex::Real tmp4;
                riemann(
                        qtempl[R_RHO], qtempl[R_UN], qtempl[R_UT1], qtempl[R_UT2],
                        qtempl[R_P], rhoe_l, spl, gamc_l, qtempr[R_RHO], qtempr[R_UN],
                        qtempr[R_UT1], qtempr[R_UT2], qtempr[R_P], rhoe_r, spr, gamc_r,
                        bc_test_val, csmall, cavg, ustar, flux_tmp[URHO], flux_tmp[f_idx[0]],
                        flux_tmp[f_idx[1]], flux_tmp[f_idx[2]], flux_tmp[UEDEN],
                        flux_tmp[UEINT], tmp0, tmp1, tmp2, tmp3, tmp4);

                for (int n = 0; n < NUM_SPECIES; n++) 
                {
                    flux_tmp[UFS + n] = (ustar > 0.0) ? flux_tmp[URHO] * qtempl[R_Y + n]
                        : flux_tmp[URHO] * qtempr[R_Y + n];
                    flux_tmp[UFS + n] =
                        (ustar == 0.0)
                        ? flux_tmp[URHO] * 0.5 * (qtempl[R_Y + n] + qtempr[R_Y + n])
                        : flux_tmp[UFS + n];
                }

                flux_tmp[UTEMP] = 0.0;
                for (int n = UFX; n < UFX + NUM_AUX; n++) 
                {
                    flux_tmp[n] = (NUM_AUX > 0) ? 0.0 : flux_tmp[n];
                }
                for (int n = UFA; n < UFA + NUM_ADV; n++) 
                {
                    flux_tmp[n] = (NUM_ADV > 0) ? 0.0 : flux_tmp[n];
                }

                for (int ivar = 0; ivar < NVAR; ivar++) 
                {
                    flx[dir](i, j, k, ivar) += flux_tmp[ivar] * area[dir](i, j, k);
                }
                }
                else
                {
                    amrex::Real qtempl[5 + NUM_SPECIES] = {0.0};
                    amrex::Real qtempr[5 + NUM_SPECIES] = {0.0};
                    amrex::Real spl[NUM_SPECIES]={0.0};
                    amrex::Real spr[NUM_SPECIES]={0.0};
                    amrex::Real eos_state_rho;
                    amrex::Real eos_state_p;
                    amrex::Real eos_state_e;
                    amrex::Real eos_state_cs;
                    amrex::Real eos_state_gamma;
                    amrex::Real eos_state_T;
                    amrex::Real cavg;
                    amrex::Real csmall;
                    amrex::Real rhoe_l,rhoe_r;
                    amrex::Real gamc_l,gamc_r;
                    amrex::Real ustar = 0.0;
                    amrex::Real flux_tmp[NVAR] = {0.0};
                    amrex::Real tmp0;
                    amrex::Real tmp1;
                    amrex::Real tmp2;
                    amrex::Real tmp3;
                    amrex::Real tmp4;

                    //at least one cell should be regular
                    if(flags(i,j,k).isRegular() || flags(ii,jj,kk).isRegular())
                    {
                        if(flags(i,j,k).isRegular() && !flags(ii,jj,kk).isRegular())
                        {

                            qtempr[R_UN]  = q(i, j, k, q_idx[0]);
                            qtempr[R_P]   = q(i, j, k, QPRES);
                            qtempr[R_UT1] = q(i, j, k, q_idx[1]);
                            qtempr[R_UT2] = q(i, j, k, q_idx[2]);
                            qtempr[R_RHO] = 0.0;

                            for (int n = 0; n < NUM_SPECIES; n++) 
                            {
                                qtempr[R_Y + n] =
                                    q(i, j, k, QFS + n) * q(i, j, k, QRHO);
                                qtempr[R_RHO] += qtempr[R_Y + n];
                            }
                            for (int n = 0; n < NUM_SPECIES; n++) 
                            {
                                qtempr[R_Y + n] = qtempr[R_Y + n] / qtempr[R_RHO];
                            }

                            cavg   = qaux(i, j, k, QC);
                            csmall = qaux(i, j, k, QCSML);

                            eos_state_rho = qtempr[R_RHO];
                            eos_state_p = qtempr[R_P];
                            for (int n = 0; n < NUM_SPECIES; n++) 
                            {
                                spr[n] = qtempr[R_Y + n];
                            }

                            auto eos = pele::physics::PhysicsType::eos();
                            eos.RYP2T(eos_state_rho, spr, eos_state_p, eos_state_T);
                            eos.RTY2E(eos_state_rho, eos_state_T, spr, eos_state_e);
                            eos.RTY2G(eos_state_rho, eos_state_T, spr, eos_state_gamma);
                            eos.RTY2Cs(eos_state_rho, eos_state_T, spr, eos_state_cs);

                            const amrex::Real rhoe_r = eos_state_rho * eos_state_e;
                            const amrex::Real gamc_r = eos_state_gamma;

                            //copy over to left
                            for(int n=0;n<(5+NUM_SPECIES);n++)
                            {
                                qtempl[n]=qtempr[n];
                            }
                            for(int n=0;n<NUM_SPECIES;n++)
                            {
                                spl[n]=spr[n];
                            }
                            //reflect velocities
                            qtempl[R_UN]  = -qtempr[R_UN];
                            qtempl[R_UT1] = -qtempr[R_UT1];
                            qtempl[R_UT2] = -qtempr[R_UT2];

                            rhoe_l=rhoe_r;
                            gamc_l=gamc_r;
                        }
                        else if(!flags(i,j,k).isRegular() && flags(ii,jj,kk).isRegular())
                        {
                            qtempl[R_UN]  = q(ii, jj, kk, q_idx[0]);
                            qtempl[R_P]   = q(ii, jj, kk, QPRES);
                            qtempl[R_UT1] = q(ii, jj, kk, q_idx[1]);
                            qtempl[R_UT2] = q(ii, jj, kk, q_idx[2]);
                            qtempl[R_RHO] = 0.0;

                            for (int n = 0; n < NUM_SPECIES; n++) 
                            {
                                qtempl[R_Y + n] =
                                    q(ii, jj, kk, QFS + n) * q(ii, jj, kk, QRHO);
                                qtempl[R_RHO] += qtempl[R_Y + n];
                            }
                            for (int n = 0; n < NUM_SPECIES; n++) 
                            {
                                qtempl[R_Y + n] = qtempl[R_Y + n] / qtempl[R_RHO];
                            }

                            cavg   = qaux(ii, jj, kk, QC);
                            csmall = qaux(ii, jj, kk, QCSML);

                            eos_state_rho = qtempl[R_RHO];
                            eos_state_p = qtempl[R_P];
                            for (int n = 0; n < NUM_SPECIES; n++) 
                            {
                                spl[n] = qtempl[R_Y + n];
                            }

                            auto eos = pele::physics::PhysicsType::eos();
                            eos.RYP2T(eos_state_rho, spl, eos_state_p, eos_state_T);
                            eos.RTY2E(eos_state_rho, eos_state_T, spl, eos_state_e);
                            eos.RTY2G(eos_state_rho, eos_state_T, spl, eos_state_gamma);
                            eos.RTY2Cs(eos_state_rho, eos_state_T, spl, eos_state_cs);

                            const amrex::Real rhoe_l = eos_state_rho * eos_state_e;
                            const amrex::Real gamc_l = eos_state_gamma;

                            //copy over to right
                            for(int n=0;n<(5+NUM_SPECIES);n++)
                            {
                                qtempr[n]=qtempl[n];
                            }
                            for(int n=0;n<NUM_SPECIES;n++)
                            {
                                spr[n]=spl[n];
                            }
                            //reflect velocities
                            qtempr[R_UN]  = -qtempl[R_UN];
                            qtempr[R_UT1] = -qtempl[R_UT1];
                            qtempr[R_UT2] = -qtempl[R_UT2];

                            rhoe_r=rhoe_l;
                            gamc_r=gamc_l;

                        }

                        riemann(
                                qtempl[R_RHO], qtempl[R_UN], qtempl[R_UT1], qtempl[R_UT2],
                                qtempl[R_P], rhoe_l, spl, gamc_l, qtempr[R_RHO], qtempr[R_UN],
                                qtempr[R_UT1], qtempr[R_UT2], qtempr[R_P], rhoe_r, spr, gamc_r,
                                bc_test_val, csmall, cavg, ustar, flux_tmp[URHO], flux_tmp[f_idx[0]],
                                flux_tmp[f_idx[1]], flux_tmp[f_idx[2]], flux_tmp[UEDEN],
                                flux_tmp[UEINT], tmp0, tmp1, tmp2, tmp3, tmp4);

                        for (int n = 0; n < NUM_SPECIES; n++) 
                        {
                            flux_tmp[UFS + n] = (ustar > 0.0) ? flux_tmp[URHO] * qtempl[R_Y + n]
                                : flux_tmp[URHO] * qtempr[R_Y + n];
                            flux_tmp[UFS + n] =
                                (ustar == 0.0)
                                ? flux_tmp[URHO] * 0.5 * (qtempl[R_Y + n] + qtempr[R_Y + n])
                                : flux_tmp[UFS + n];
                        }

                        flux_tmp[UTEMP] = 0.0;
                        for (int n = UFX; n < UFX + NUM_AUX; n++) 
                        {
                            flux_tmp[n] = (NUM_AUX > 0) ? 0.0 : flux_tmp[n];
                        }
                        for (int n = UFA; n < UFA + NUM_ADV; n++) 
                        {
                            flux_tmp[n] = (NUM_ADV > 0) ? 0.0 : flux_tmp[n];
                        }
                        for (int ivar = 0; ivar < NVAR; ivar++) 
                        {
                            flx[dir](i, j, k, ivar) += flux_tmp[ivar] * area[dir](i, j, k);
                        }
                    }
                }
        });
    }

}
#endif
