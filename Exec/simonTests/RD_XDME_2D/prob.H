#ifndef PROB_H
#define PROB_H

#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

#include "PeleC.H"
#include "IndexDefines.H"
#include "PelePhysics.H"
#include "Tagging.H"
#include "ProblemDerive.H"
#include "prob_parm.H"
#include "Constants.H"

#include "kinsolma.H"

// // SUNDIALS for NONLINEAR ALGEBRIA EQUATIONS
// #include "sundials/sundials_types.h"
// #include "nvector/nvector_serial.h"
// #include "sunmatrix/sunmatrix_dense.h"
// #include "sunlinsol/sunlinsol_dense.h"
// #include "sunnonlinsol/sunnonlinsol_newton.h"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_initdata(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& state,
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm)
{

  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* prob_hi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();
  const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
  const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
  const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

  amrex::Vector<amrex::Real> pos_c = {x, y, z};

  const amrex::Real pamb = prob_parm.pamb;
  const amrex::Real Tamb = prob_parm.Tamb;
  const amrex::Real p_hot = prob_parm.p_hot;
  const amrex::Real T_hot = prob_parm.T_hot;

  amrex::Real u[3] = {0.0};
  amrex::Real ek;
  amrex::Real T;
  amrex::Real p;
  amrex::Real rho;
  amrex::Real e;
  amrex::Real molefrac[NUM_SPECIES] = {0.0};
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  // amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {{0.0}};

  auto eos = pele::physics::PhysicsType::eos();
  for (int n = 0; n < NUM_SPECIES; n++) {
    massfrac[n] = prob_parm.inlet_state[n];
  }

  T = Tamb;
  p = pamb;
  eos.PYT2RE(p, massfrac, T, rho, e);

  // Determine whether u is required to update
  if (prob_parm.inlet_type == 3) {
    // Update u based on inlet boundary conditions
    {
      amrex::Real pw = pamb; // Internal head wall pressure
      if (prob_parm.p_0 <= pw) {
        // Nothing injected
        u[0] = 0.;
        u[1] = 0.;
        u[2] = 0.;
      } else if (prob_parm.p_2 <= pw) {
        // Subsonic inlet
        amrex::Real gamma, Cs, Ma_e, u_e;
        if (prob_parm.gamma_ref_temp == -1) {
          eos.TY2G(T, massfrac, gamma);
        } else {
          gamma = prob_parm.gamma;
        }
        Ma_e = std::sqrt(
          2.0 / (gamma - 1) *
          (std::pow(prob_parm.p_0 / pw, (gamma - 1) / gamma) - 1.));

        p = pw;
        T = prob_parm.T_0 / (1 + (gamma - 1) * Ma_e * Ma_e / 2.);
        eos.PYT2RE(p, massfrac, T, rho, e);
        if (prob_parm.gamma_ref_temp == -1) {
          eos.RTY2Cs(rho, T, massfrac, Cs);
        } else {
          Cs = std::sqrt(gamma * p / rho);
        }
        u[0] = Ma_e * Cs;
        u[1] = 0.;
        u[2] = 0.;
      } else if (prob_parm.p_3 <= pw) {
        // Normal shock wave
        amrex::Real gamma, Cs, Ma_e, u_e;
        if (prob_parm.gamma_ref_temp == -1) {
          eos.TY2G(T, massfrac, gamma);
        } else {
          gamma = prob_parm.gamma;
        }
        // Iterate to get Ma_e
        amrex::Real Ma_1, Ma_2;
        amrex::Real pre_ratio = prob_parm.area_ratio * pw / prob_parm.p_0;

        int func_sol = 1;
        int retval =
          pele::kinsolma::kinsol_getma(Ma_1, Ma_2, func_sol, gamma, pre_ratio);
        Ma_e = Ma_1;

        p = pw;
        T = prob_parm.T_0 / (1 + (gamma - 1) * Ma_e * Ma_e / 2.);
        eos.PYT2RE(p, massfrac, T, rho, e);
        if (prob_parm.gamma_ref_temp == -1) {
          eos.RTY2Cs(rho, T, massfrac, Cs);
        } else {
          Cs = std::sqrt(gamma * p / rho);
        }
        u[0] = Ma_e * Cs;
        u[1] = 0.;
        u[2] = 0.;
      } else {
        // Supersonic inlet
        amrex::Real gamma, Cs;
        if (prob_parm.gamma_ref_temp == -1) {
          eos.TY2G(T, massfrac, gamma);
        } else {
          gamma = prob_parm.gamma;
        }
        amrex::Real Ma_1 = prob_parm.Ma_1;

        p =
          prob_parm.p_0 /
          std::pow((1 + (gamma - 1) * Ma_1 * Ma_1 / 2.), (gamma / (gamma - 1)));
        T = prob_parm.T_0 / (1 + (gamma - 1) * Ma_1 * Ma_1 / 2.);
        eos.PYT2RE(p, massfrac, T, rho, e);
        if (prob_parm.gamma_ref_temp == -1) {
          eos.RTY2Cs(rho, T, massfrac, Cs);
        } else {
          Cs = std::sqrt(gamma * p / rho);
        }
        u[0] = Ma_1 * Cs;
        u[1] = 0.;
        u[2] = 0.;
      }
    }
  } else if (prob_parm.inlet_type == 4) {
    {
      amrex::Real pw = p; // Internal head wall pressure
      amrex::Real p_cr = prob_parm.p_cr;
      if (prob_parm.p_0 <= pw) {
        // Nothing injected
        u[0] = 0.;
        u[1] = 0.;
        u[2] = 0.;
      } else if (prob_parm.p_cr <= pw) {
        // Subsonic inlet
        // Setup inlet states
        amrex::Real gamma, Cs, Ma_e, u_e;
        p = pw;
        if (prob_parm.gamma_ref_temp == -1) {
          eos.TY2G(T, massfrac, gamma);
        } else {
          gamma = prob_parm.gamma;
        }
        Ma_e = std::sqrt(
          2.0 / (gamma - 1) *
          (std::pow(prob_parm.p_0 / pw, (gamma - 1) / gamma) - 1.));

        p = pw;
        T = prob_parm.T_0 * std::pow(p / prob_parm.p_0, (gamma - 1) / gamma);
        // T = prob_parm.T_0 / (1 + (gamma - 1)* Ma_e * Ma_e /2.);
        eos.PYT2RE(p, massfrac, T, rho, e);
        if (prob_parm.gamma_ref_temp == -1) {
          eos.RTY2Cs(rho, T, massfrac, Cs);
        } else {
          Cs = std::sqrt(gamma * p / rho);
        }
        u[0] = Ma_e * Cs;
        u[1] = 0.;
        u[2] = 0.;
      } else {
        // Chocked
        p = prob_parm.p_cr;
        amrex::Real gamma, Cs, Ma_e, u_e;
        if (prob_parm.gamma_ref_temp == -1) {
          eos.TY2G(T, massfrac, gamma);
        } else {
          gamma = prob_parm.gamma;
        }
        Ma_e = std::sqrt(
          2.0 / (gamma - 1) *
          (std::pow(prob_parm.p_0 / p, (gamma - 1) / gamma) - 1.));

        T = prob_parm.T_0 * std::pow(p / prob_parm.p_0, (gamma - 1) / gamma);
        // T = prob_parm.T_0 / (1 + (gamma - 1)* Ma_e * Ma_e /2.);
        eos.PYT2RE(p, massfrac, T, rho, e);
        if (prob_parm.gamma_ref_temp == -1) {
          eos.RTY2Cs(rho, T, massfrac, Cs);
        } else {
          Cs = std::sqrt(gamma * p / rho);
        }
        u[0] = Ma_e * Cs;
        u[1] = 0.;
        u[2] = 0.;
      }
    }
  }

  bool is_in_hot = true;
  bool is_in_air = false;
  if (
    prob_parm.air_fill_outside_x == 1 &&
    (pos_c[0] < prob_parm.pos_hot[0] - prob_parm.rad_hot[0] ||
     pos_c[0] > prob_parm.pos_hot[0] + prob_parm.rad_hot[0])) {
    // set massfrac to air
    molefrac[O2_ID] = 0.21;
    molefrac[CH3OCH3_ID] = 0.;
    molefrac[N2_ID] = 0.79;
    auto eos = pele::physics::PhysicsType::eos();
    eos.X2Y(molefrac, massfrac);
  } else if (
    prob_parm.air_fill_outside_x == 2 &&
    (pos_c[1] > prob_hi[1] -
                  (prob_hi[1] - prob_parm.pos_hot[1] - prob_parm.rad_hot[1]) *
                    pos_c[0] / (prob_parm.pos_hot[0] + prob_parm.rad_hot[0]) ||
     pos_c[0] > prob_parm.pos_hot[0] + prob_parm.rad_hot[0])) {
    // set massfrac to air
    molefrac[O2_ID] = 0.21;
    molefrac[CH3OCH3_ID] = 0.;
    molefrac[N2_ID] = 0.79;
    auto eos = pele::physics::PhysicsType::eos();
    eos.X2Y(molefrac, massfrac);
  }

  for (int i = 0; i < AMREX_SPACEDIM; i++) {
    if (
      pos_c[i] < prob_parm.pos_hot[i] - prob_parm.rad_hot[i] ||
      pos_c[i] > prob_parm.pos_hot[i] + prob_parm.rad_hot[i]) {
      is_in_hot = false;
      break;
    }
  }

  if (is_in_hot) {
    T = T_hot;
    p = p_hot;
    if (prob_parm.init_u_hot == 1) {
      u[0] = prob_parm.u_hot[0];
      u[1] = prob_parm.u_hot[1];
      u[2] = prob_parm.u_hot[2];
    }
  }
  eos.PYT2RE(p, massfrac, T, rho, e);

  ek = 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

  state(i, j, k, URHO) = rho;
  state(i, j, k, UMX) = rho * u[0];
  state(i, j, k, UMY) = rho * u[1];
  state(i, j, k, UMZ) = rho * u[2];
  state(i, j, k, UEINT) = rho * e;
  state(i, j, k, UEDEN) = rho * (e + ek);
  state(i, j, k, UTEMP) = T;
  for (int n = 0; n < NUM_SPECIES; n++) {
    state(i, j, k, UFS + n) = rho * massfrac[n];
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real x[AMREX_SPACEDIM],
  const amrex::Real s_int[NVAR],
  amrex::Real s_ext[NVAR],
  const int idir, // x, y, z
  const int sgn,  // -1: right; 1: left
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm)
{

  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* prob_hi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();

  // amrex::Print() << "TIME = " << time
  //                << ", x = " << x[0]
  //                << ", y = " << x[1]
  //                << ", idir = " << idir
  //                << ", sgn = " << sgn << '\n';

  amrex::Real u[3] = {0.0};
  amrex::Real /* pert, */ ek, T, p, rho, e;

  amrex::Real molefrac[NUM_SPECIES] = {0.0};
  amrex::Real massfrac[NUM_SPECIES] = {0.0};

  auto eos = pele::physics::PhysicsType::eos();

  // Default, FOExtrap, use internal field
  u[0] = s_int[UMX] / s_int[URHO];
  u[1] = s_int[UMY] / s_int[URHO];
  u[2] = s_int[UMZ] / s_int[URHO];

  T = s_int[UTEMP];
  rho = s_int[URHO];
  e = s_int[UEINT] / rho;

  for (int n = 0; n < NUM_SPECIES; n++) {
    massfrac[n] = s_int[UFS + n] / s_int[URHO];
  }
  eos.Y2X(massfrac, molefrac);

  eos.RTY2P(s_int[URHO], s_int[UTEMP], massfrac, p);
  // eos.PYT2RE(p, massfrac, T, rho, e);

  if ((idir == 0) && (sgn == 1)) {
    // Inlet types:
    //    0 - FOExtrap pressure (recommended)
    //    1 - Enforce constant pressure
    //    2 - Characteristic extrapolation of pressure (Blazek Eqn 8.32 - 8.36)
    //    3 - For rotation detonation (Converged-diverged nozzle)
    //        refs:
    //          - doi:10.1007/s00193-008-0178-2
    //          - doi:10.1016/j.proci.2014.09.010
    //          - doi:10.1016/j.applthermaleng.2016.07.166
    //    4 - For rotation detonation (Isentroic expansion)
    //        refs:
    //          - doi:10.1016/j.fuel.2020.117986

    // Inlet conditions - use interior mass fraction

    // Apply different inlet types for pressure
    if (prob_parm.inlet_type == 0) {
      eos.RTY2P(rho, T, massfrac, p);
    } else if (prob_parm.inlet_type == 1) {
      p = prob_parm.p_hot;
    } else if (prob_parm.inlet_type == 2) {
      amrex::Abort("Invalid inlet boundary type specified");
    } else if (prob_parm.inlet_type == 3) {
      amrex::Real pw = p; // Internal head wall pressure
      if (prob_parm.p_0 <= pw) {
        // Nothing injected
        u[0] = 0.;
        // u[1] = 0.;
        u[2] = 0.;
      } else if (prob_parm.p_2 <= pw) {
        // Subsonic inlet
        // Setup inlet states
        for (int n = 0; n < NUM_SPECIES; n++) {
          massfrac[n] = prob_parm.inlet_state[n];
        }
        amrex::Real gamma, Cs, Ma_e, u_e;
        if (prob_parm.gamma_ref_temp == -1) {
          eos.TY2G(T, massfrac, gamma);
        } else {
          gamma = prob_parm.gamma;
        }
        Ma_e = std::sqrt(
          2.0 / (gamma - 1) *
          (std::pow(prob_parm.p_0 / pw, (gamma - 1) / gamma) - 1.));

        p = pw;
        T = prob_parm.T_0 / (1 + (gamma - 1) * Ma_e * Ma_e / 2.);
        eos.PYT2RE(p, massfrac, T, rho, e);
        if (prob_parm.gamma_ref_temp == -1) {
          eos.RTY2Cs(rho, T, massfrac, Cs);
        } else {
          Cs = std::sqrt(gamma * p / rho);
        }
        u[0] = Ma_e * Cs;
        if (prob_parm.inlet_slip_type == 0) {
          u[1] = 0.;
        }
        u[2] = 0.;
      } else if (prob_parm.p_3 <= pw) {
        // Normal shock wave
        for (int n = 0; n < NUM_SPECIES; n++) {
          massfrac[n] = prob_parm.inlet_state[n];
        }
        amrex::Real gamma, Cs, Ma_e, u_e;
        if (prob_parm.gamma_ref_temp == -1) {
          eos.TY2G(T, massfrac, gamma);
        } else {
          gamma = prob_parm.gamma;
        }
        // Iterate to get Ma_e
        amrex::Real Ma_1, Ma_2;
        amrex::Real pre_ratio = prob_parm.area_ratio * pw / prob_parm.p_0;

        int func_sol = 1;
        int retval =
          pele::kinsolma::kinsol_getma(Ma_1, Ma_2, func_sol, gamma, pre_ratio);
        Ma_e = Ma_1;

        p = pw;
        T = prob_parm.T_0 / (1 + (gamma - 1) * Ma_e * Ma_e / 2.);
        eos.PYT2RE(p, massfrac, T, rho, e);
        if (prob_parm.gamma_ref_temp == -1) {
          eos.RTY2Cs(rho, T, massfrac, Cs);
        } else {
          Cs = std::sqrt(gamma * p / rho);
        }
        u[0] = Ma_e * Cs;
        if (prob_parm.inlet_slip_type == 0) {
          u[1] = 0.;
        }
        u[2] = 0.;
      } else {
        // Supersonic inlet
        for (int n = 0; n < NUM_SPECIES; n++) {
          massfrac[n] = prob_parm.inlet_state[n];
        }
        amrex::Real gamma, Cs;
        if (prob_parm.gamma_ref_temp == -1) {
          eos.TY2G(T, massfrac, gamma);
        } else {
          gamma = prob_parm.gamma;
        }
        amrex::Real Ma_1 = prob_parm.Ma_1;

        p =
          prob_parm.p_0 /
          std::pow((1 + (gamma - 1) * Ma_1 * Ma_1 / 2.), (gamma / (gamma - 1)));
        T = prob_parm.T_0 / (1 + (gamma - 1) * Ma_1 * Ma_1 / 2.);
        eos.PYT2RE(p, massfrac, T, rho, e);
        if (prob_parm.gamma_ref_temp == -1) {
          eos.RTY2Cs(rho, T, massfrac, Cs);
        } else {
          Cs = std::sqrt(gamma * p / rho);
        }
        u[0] = Ma_1 * Cs;
        if (prob_parm.inlet_slip_type == 0) {
          u[1] = 0.;
        }
        u[2] = 0.;
        // amrex::Print() << "Inlet type 3: "
        //                << "TIME = " << time
        //                << ", T = " << T
        //                << ", p = " << p
        //                << ", Ma_e = " << Ma_1
        //                << ", u = " << u[0] << "/" << u[1]
        //                << ", u = " << s_int[UMX] / s_int[URHO]
        //                << ", u = " << s_int[UMY] / s_int[URHO]
        //                << ", rho = " << rho
        //                << ", e = " << e << std::endl;
      }
    } else if (prob_parm.inlet_type == 4) {
      amrex::Real pw = p; // Internal head wall pressure
      amrex::Real p_cr = prob_parm.p_cr;
      if (prob_parm.p_0 <= pw) {
        // Nothing injected
        u[0] = 0.;
        u[1] = 0.;
        u[2] = 0.;
      } else if (prob_parm.p_cr <= pw) {
        // Subsonic inlet
        // Setup inlet states
        for (int n = 0; n < NUM_SPECIES; n++) {
          massfrac[n] = prob_parm.inlet_state[n];
        }
        amrex::Real gamma, Cs, Ma_e, u_e;
        p = pw;
        if (prob_parm.gamma_ref_temp == -1) {
          eos.TY2G(T, massfrac, gamma);
        } else {
          gamma = prob_parm.gamma;
        }
        Ma_e = std::sqrt(
          2.0 / (gamma - 1) *
          (std::pow(prob_parm.p_0 / pw, (gamma - 1) / gamma) - 1.));

        p = pw;
        T = prob_parm.T_0 * std::pow(p / prob_parm.p_0, (gamma - 1) / gamma);
        // T = prob_parm.T_0 / (1 + (gamma - 1)* Ma_e * Ma_e /2.);
        eos.PYT2RE(p, massfrac, T, rho, e);
        if (prob_parm.gamma_ref_temp == -1) {
          eos.RTY2Cs(rho, T, massfrac, Cs);
        } else {
          Cs = std::sqrt(gamma * p / rho);
        }
        u[0] = Ma_e * Cs;
        if (prob_parm.inlet_slip_type == 0) {
          u[1] = 0.;
        }
        u[2] = 0.;
      } else {
        // Chocked
        for (int n = 0; n < NUM_SPECIES; n++) {
          massfrac[n] = prob_parm.inlet_state[n];
        }
        amrex::Real gamma, Cs, Ma_e, u_e;
        p = prob_parm.p_cr;
        if (prob_parm.gamma_ref_temp == -1) {
          eos.TY2G(T, massfrac, gamma);
        } else {
          gamma = prob_parm.gamma;
        }
        Ma_e = std::sqrt(
          2.0 / (gamma - 1) *
          (std::pow(prob_parm.p_0 / p, (gamma - 1) / gamma) - 1.));

        T = prob_parm.T_0 * std::pow(p / prob_parm.p_0, (gamma - 1) / gamma);
        // T = prob_parm.T_0 / (1 + (gamma - 1)* Ma_e * Ma_e /2.);
        eos.PYT2RE(p, massfrac, T, rho, e);
        if (prob_parm.gamma_ref_temp == -1) {
          eos.RTY2Cs(rho, T, massfrac, Cs);
        } else {
          Cs = std::sqrt(gamma * p / rho);
        }
        u[0] = Ma_e * Cs;
        if (prob_parm.inlet_slip_type == 0) {
          u[1] = 0.;
        }
        u[2] = 0.;
      }

    } else {
      amrex::Abort("Invalid inlet boundary type specified");
    }
  } else if ((idir == 0) && (sgn == -1)) {
    // outlet types:
    //    0 - FOExtrap everything
    //        Supersonic outflow
    //    1 - Characteristic Pressure Outlet (recommended)
    //        ref RegTests/EB-ConvergingNozzle
    //    2 - Fixed Pressure
    //    3 - For rotation detonation (Fixed pressure)
    //        refs:
    //          - doi:10.1007/s00193-008-0178-2
    //          - doi:10.1016/j.applthermaleng.2016.07.166
    //          - doi:10.1016/j.fuel.2021.121481
    //          - doi:10.1016/j.proci.2022.08.075
    //    4 - For rotation detonation
    //        Fixed pressure for subsonic outlet
    //        Zero gradient for supersonic outlet
    //        refs:
    //          - doi:10.1016/j.proci.2014.09.010
    //          - doi:10.1016/j.applthermaleng.2016.07.166
    //          - doi:10.1016/j.fuel.2021.121481
    //          - doi:10.1016/j.proci.2022.08.075

    // Initialize all quantities with FOExtrap values
    // amrex::Real Ma = 0.0;
    // eos.RTY2Cs(rho, T, massfrac, Ma);
    // Ma = u[0]/Ma;
    // amrex::Print() << "TIME = " << time
    //                << ", Ma = " << Ma << std::endl;

    // if (Ma < 1.0)
    {
      // If using a pressure-based outlet, compute that
      if (prob_parm.outlet_type == 0) {
        // Nothing further to be done
      } else if (prob_parm.outlet_type == 1) {
        // Following Blazek p 279, eq. 8.23 - reimplemented from EB-C10
        // NOTE: it is assumed that mass fractions are constant across the
        // boundary Interior state (point d)
        const amrex::Real xd = prob_hi[0] - 0.5 * dx[0];
        const amrex::Real u_inter = u[0];
        const amrex::Real v_inter = u[1];
        const amrex::Real w_inter = u[2];

        const amrex::Real T_inter = T;
        const amrex::Real rho_inter = rho;
        amrex::Real p_inter = 0.0, cs_inter = 0.0;
        eos.RTY2P(rho_inter, T_inter, massfrac, p_inter);
        eos.RTY2Cs(rho_inter, T_inter, massfrac, cs_inter);

        // Boundary state (point b)
        const amrex::Real xb = prob_hi[0];
        const amrex::Real pb = prob_parm.pamb;
        const amrex::Real rhob =
          s_int[URHO] + (pb - p_inter) / (cs_inter * cs_inter);
        const amrex::Real ub =
          u_inter + (p_inter - pb) / (rho_inter * cs_inter);
        const amrex::Real vb = v_inter;
        const amrex::Real wb = w_inter;

        // Update rho & p
        // Ghost state (point a). Linear extrapolation from d and b
        rho = (rhob - rho_inter) / (xb - xd) * (x[0] - xd) + rho_inter;
        p = (pb - p_inter) / (xb - xd) * (x[0] - xd) + p_inter;

        // Update e & T
        eos.RYP2E(rho, massfrac, p, e);
        eos.REY2T(rho, e, massfrac, T);

        // Update u
        u[0] = (ub - u_inter) / (xb - xd) * (x[0] - xd) + u_inter;
        u[1] = (vb - v_inter) / (xb - xd) * (x[0] - xd) + v_inter;
        u[2] = (wb - w_inter) / (xb - xd) * (x[0] - xd) + w_inter;

        if (rho < 0) {
          // Negative rho then FOExtrap
          rho = rho_inter;
          p = p_inter;
          eos.RYP2E(rho, massfrac, p, e);
          eos.REY2T(rho, e, massfrac, T);
          u[0] = u_inter;
          u[1] = v_inter;
          u[2] = w_inter;
        }

      } else if (prob_parm.outlet_type == 2) {
        // Update p, rho & e
        p = prob_parm.pamb;
        eos.PYT2RE(p, massfrac, T, rho, e);
      } else if (prob_parm.outlet_type == 3) {
        // Update Y
        for (int n = 0; n < NUM_SPECIES; n++) {
          massfrac[n] = massfrac[n] * (1 - prob_parm.damp_ratio) +
                        prob_parm.outlet_state[n] * prob_parm.damp_ratio;
        }
        // Specific pressure
        p = prob_parm.pamb;
        eos.PYT2RE(p, massfrac, T, rho, e);
      } else if (prob_parm.outlet_type == 4) {
        // Check Ma,
        amrex::Real cs;
        eos.RTY2Cs(rho, T, massfrac, cs);
        if (u[0] < cs) {
          // Specific pressure
          p = p * (1 - prob_parm.damp_ratio) +
              prob_parm.pamb * prob_parm.damp_ratio;
          eos.PYT2RE(p, massfrac, T, rho, e);
        }
      } else {
        amrex::Abort("Invalid outlet boundary type specified");
      }
    }
  }

  ek = 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

  s_ext[URHO] = rho;
  s_ext[UMX] = rho * u[0];
  s_ext[UMY] = rho * u[1];
  s_ext[UMZ] = rho * u[2];
  s_ext[UEINT] = rho * e;
  s_ext[UEDEN] = rho * (e + ek);
  s_ext[UTEMP] = T;
  for (int n = 0; n < NUM_SPECIES; n++) {
    s_ext[UFS + n] = rho * massfrac[n];
  }
}

void pc_prob_close();

using ProblemTags = EmptyProbTagStruct;
using ProblemDerives = EmptyProbDeriveStruct;

#endif
