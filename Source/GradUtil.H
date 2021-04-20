#ifndef _GRADUTIL_H_
#define _GRADUTIL_H_

#include <AMReX_FArrayBox.H>
#include "IndexDefines.H"
#ifdef PELEC_USE_EB
#include "EB.H"
#endif

// This header file contains functions and declarations for Gradient Utilities
// in 3D for PeleC GPU. As per the convention of AMReX, inlined functions are
// defined here. Where as non-inline functions are declared here.

namespace GradUtils {
const int nCompTan = AMREX_D_PICK(0, 2, 6);
} // namespace GradUtils

#ifdef PELEC_USE_EB
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_compute_tangential_vel_derivs_SS(
  const int i,
  const int j,
  const int k,
  const amrex::Array4<const amrex::Real>& q,
  const int dir,
  const amrex::Real dx1,
  const amrex::Real dx2,
  const amrex::Array4<amrex::EBCellFlag const>& flags,
  const amrex::Array4<amrex::Real>& td)
{
  // dx1 and dx2 will be the trangential grid spacing
  const amrex::Real dx1inv = 1.0 / dx1;
  const amrex::Real dx2inv = 1.0 / dx2;
  // This is because it is a tangential velocity derivative.

  amrex::Real u[8] = {0.0};
  amrex::Real v[8] = {0.0};
  amrex::Real w[8] = {0.0};

  if (dir == 0) {
    // dx1 = dy, dx2 = dz
    u[0] = flags(i, j + 1, k).isRegular() ? q(i, j + 1, k, QU)
                                          : -q(i - 1, j + 1, k, QU);
    u[1] = flags(i - 1, j + 1, k).isRegular() ? q(i - 1, j + 1, k, QU)
                                              : -q(i, j + 1, k, QU);
    u[2] = flags(i, j - 1, k).isRegular() ? q(i, j - 1, k, QU)
                                          : -q(i - 1, j - 1, k, QU);
    u[3] = flags(i - 1, j - 1, k).isRegular() ? q(i - 1, j - 1, k, QU)
                                              : -q(i, j - 1, k, QU);

    u[4] = flags(i, j, k + 1).isRegular() ? q(i, j, k + 1, QU)
                                          : -q(i - 1, j, k + 1, QU);
    u[5] = flags(i - 1, j, k + 1).isRegular() ? q(i - 1, j, k + 1, QU)
                                              : -q(i, j, k + 1, QU);
    u[6] = flags(i, j, k - 1).isRegular() ? q(i, j, k - 1, QU)
                                          : -q(i - 1, j, k - 1, QU);
    u[7] = flags(i - 1, j, k - 1).isRegular() ? q(i - 1, j, k - 1, QU)
                                              : -q(i, j, k - 1, QU);

    v[0] = flags(i, j + 1, k).isRegular() ? q(i, j + 1, k, QV)
                                          : -q(i - 1, j + 1, k, QV);
    v[1] = flags(i - 1, j + 1, k).isRegular() ? q(i - 1, j + 1, k, QV)
                                              : -q(i, j + 1, k, QV);
    v[2] = flags(i, j - 1, k).isRegular() ? q(i, j - 1, k, QV)
                                          : -q(i - 1, j - 1, k, QV);
    v[3] = flags(i - 1, j - 1, k).isRegular() ? q(i - 1, j - 1, k, QV)
                                              : -q(i, j - 1, k, QV);

    v[4] = flags(i, j, k + 1).isRegular() ? q(i, j, k + 1, QV)
                                          : -q(i - 1, j, k + 1, QV);
    v[5] = flags(i - 1, j, k + 1).isRegular() ? q(i - 1, j, k + 1, QV)
                                              : -q(i, j, k + 1, QV);
    v[6] = flags(i, j, k - 1).isRegular() ? q(i, j, k - 1, QV)
                                          : -q(i - 1, j, k - 1, QV);
    v[7] = flags(i - 1, j, k - 1).isRegular() ? q(i - 1, j, k - 1, QV)
                                              : -q(i, j, k - 1, QV);

    w[0] = flags(i, j + 1, k).isRegular() ? q(i, j + 1, k, QW)
                                          : -q(i - 1, j + 1, k, QW);
    w[1] = flags(i - 1, j + 1, k).isRegular() ? q(i - 1, j + 1, k, QW)
                                              : -q(i, j + 1, k, QW);
    w[2] = flags(i, j - 1, k).isRegular() ? q(i, j - 1, k, QW)
                                          : -q(i - 1, j - 1, k, QW);
    w[3] = flags(i - 1, j - 1, k).isRegular() ? q(i - 1, j - 1, k, QW)
                                              : -q(i, j - 1, k, QW);

    w[4] = flags(i, j, k + 1).isRegular() ? q(i, j, k + 1, QW)
                                          : -q(i - 1, j, k + 1, QW);
    w[5] = flags(i - 1, j, k + 1).isRegular() ? q(i - 1, j, k + 1, QW)
                                              : -q(i, j, k + 1, QW);
    w[6] = flags(i, j, k - 1).isRegular() ? q(i, j, k - 1, QW)
                                          : -q(i - 1, j, k - 1, QW);
    w[7] = flags(i - 1, j, k - 1).isRegular() ? q(i - 1, j, k - 1, QW)
                                              : -q(i, j, k - 1, QW);

    td(i, j, k, 0) = 0.25 * dx1inv * (u[0] + u[1] - u[2] - u[3]); // dudy
    td(i, j, k, 1) = 0.25 * dx1inv * (v[0] + v[1] - v[2] - v[3]); // dvdy
    td(i, j, k, 2) = 0.25 * dx1inv * (w[0] + w[1] - w[2] - w[3]); // dwdy
    td(i, j, k, 3) = 0.25 * dx2inv * (u[4] + u[5] - u[6] - u[7]); // dudz
    td(i, j, k, 4) = 0.25 * dx2inv * (v[4] + v[5] - v[6] - v[7]); // dvdz
    td(i, j, k, 5) = 0.25 * dx2inv * (w[4] + w[5] - w[6] - w[7]); // dwdz
  } else if (dir == 1) {
    // dx1 = dx, dx2 = dz
    u[0] = flags(i + 1, j, k).isRegular() ? q(i + 1, j, k, QU)
                                          : -q(i + 1, j - 1, k, QU);
    u[1] = flags(i + 1, j - 1, k).isRegular() ? q(i + 1, j - 1, k, QU)
                                              : -q(i + 1, j, k, QU);
    u[2] = flags(i - 1, j, k).isRegular() ? q(i - 1, j, k, QU)
                                          : -q(i - 1, j - 1, k, QU);
    u[3] = flags(i - 1, j - 1, k).isRegular() ? q(i - 1, j - 1, k, QU)
                                              : -q(i - 1, j, k, QU);

    u[4] = flags(i, j, k + 1).isRegular() ? q(i, j, k + 1, QU)
                                          : -q(i, j - 1, k + 1, QU);
    u[5] = flags(i, j - 1, k + 1).isRegular() ? q(i, j - 1, k + 1, QU)
                                              : -q(i, j, k + 1, QU);
    u[6] = flags(i, j, k - 1).isRegular() ? q(i, j, k - 1, QU)
                                          : -q(i, j - 1, k - 1, QU);
    u[7] = flags(i, j - 1, k - 1).isRegular() ? q(i, j - 1, k - 1, QU)
                                              : -q(i, j, k - 1, QU);

    v[0] = flags(i + 1, j, k).isRegular() ? q(i + 1, j, k, QV)
                                          : -q(i + 1, j - 1, k, QV);
    v[1] = flags(i + 1, j - 1, k).isRegular() ? q(i + 1, j - 1, k, QV)
                                              : -q(i + 1, j, k, QV);
    v[2] = flags(i - 1, j, k).isRegular() ? q(i - 1, j, k, QV)
                                          : -q(i - 1, j - 1, k, QV);
    v[3] = flags(i - 1, j - 1, k).isRegular() ? q(i - 1, j - 1, k, QV)
                                              : -q(i - 1, j, k, QV);

    v[4] = flags(i, j, k + 1).isRegular() ? q(i, j, k + 1, QV)
                                          : -q(i, j - 1, k + 1, QV);
    v[5] = flags(i, j - 1, k + 1).isRegular() ? q(i, j - 1, k + 1, QV)
                                              : -q(i, j, k + 1, QV);
    v[6] = flags(i, j, k - 1).isRegular() ? q(i, j, k - 1, QV)
                                          : -q(i, j - 1, k - 1, QV);
    v[7] = flags(i, j - 1, k - 1).isRegular() ? q(i, j - 1, k - 1, QV)
                                              : -q(i, j, k - 1, QV);

    w[0] = flags(i + 1, j, k).isRegular() ? q(i + 1, j, k, QW)
                                          : -q(i + 1, j - 1, k, QW);
    w[1] = flags(i + 1, j - 1, k).isRegular() ? q(i + 1, j - 1, k, QW)
                                              : -q(i + 1, j, k, QW);
    w[2] = flags(i - 1, j, k).isRegular() ? q(i - 1, j, k, QW)
                                          : -q(i - 1, j - 1, k, QW);
    w[3] = flags(i - 1, j - 1, k).isRegular() ? q(i - 1, j - 1, k, QW)
                                              : -q(i - 1, j, k, QW);

    w[4] = flags(i, j, k + 1).isRegular() ? q(i, j, k + 1, QW)
                                          : -q(i, j - 1, k + 1, QW);
    w[5] = flags(i, j - 1, k + 1).isRegular() ? q(i, j - 1, k + 1, QW)
                                              : -q(i, j, k + 1, QW);
    w[6] = flags(i, j, k - 1).isRegular() ? q(i, j, k - 1, QW)
                                          : -q(i, j - 1, k - 1, QW);
    w[7] = flags(i, j - 1, k - 1).isRegular() ? q(i, j - 1, k - 1, QW)
                                              : -q(i, j, k - 1, QW);

    td(i, j, k, 0) = 0.25 * dx1inv * (u[0] + u[1] - u[2] - u[3]); // dudx
    td(i, j, k, 1) = 0.25 * dx1inv * (v[0] + v[1] - v[2] - v[3]); // dvdx
    td(i, j, k, 2) = 0.25 * dx1inv * (w[0] + w[1] - w[2] - w[3]); // dwdx
    td(i, j, k, 3) = 0.25 * dx2inv * (u[4] + u[5] - u[6] - u[7]); // dudz
    td(i, j, k, 4) = 0.25 * dx2inv * (v[4] + v[5] - v[6] - v[7]); // dvdz
    td(i, j, k, 5) = 0.25 * dx2inv * (w[4] + w[5] - w[6] - w[7]); // dwdz
  } else if (dir == 2) {
    // dx1 = dx, dx2 = dy
    u[0] = flags(i + 1, j, k).isRegular() ? q(i + 1, j, k, QU)
                                          : -q(i + 1, j, k - 1, QU);
    u[1] = flags(i + 1, j, k - 1).isRegular() ? q(i + 1, j, k - 1, QU)
                                              : -q(i + 1, j, k, QU);
    u[2] = flags(i - 1, j, k).isRegular() ? q(i - 1, j, k, QU)
                                          : -q(i - 1, j, k - 1, QU);
    u[3] = flags(i - 1, j, k - 1).isRegular() ? q(i - 1, j, k - 1, QU)
                                              : -q(i - 1, j, k, QU);

    u[4] = flags(i, j + 1, k).isRegular() ? q(i, j + 1, k, QU)
                                          : -q(i, j + 1, k - 1, QU);
    u[5] = flags(i, j + 1, k - 1).isRegular() ? q(i, j + 1, k - 1, QU)
                                              : -q(i, j + 1, k, QU);
    u[6] = flags(i, j - 1, k).isRegular() ? q(i, j - 1, k, QU)
                                          : -q(i, j - 1, k - 1, QU);
    u[7] = flags(i, j - 1, k - 1).isRegular() ? q(i, j - 1, k - 1, QU)
                                              : -q(i, j - 1, k, QU);

    v[0] = flags(i + 1, j, k).isRegular() ? q(i + 1, j, k, QV)
                                          : -q(i + 1, j, k - 1, QV);
    v[1] = flags(i + 1, j, k - 1).isRegular() ? q(i + 1, j, k - 1, QV)
                                              : -q(i + 1, j, k, QV);
    v[2] = flags(i - 1, j, k).isRegular() ? q(i - 1, j, k, QV)
                                          : -q(i - 1, j, k - 1, QV);
    v[3] = flags(i - 1, j, k - 1).isRegular() ? q(i - 1, j, k - 1, QV)
                                              : -q(i - 1, j, k, QV);

    v[4] = flags(i, j + 1, k).isRegular() ? q(i, j + 1, k, QV)
                                          : -q(i, j + 1, k - 1, QV);
    v[5] = flags(i, j + 1, k - 1).isRegular() ? q(i, j + 1, k - 1, QV)
                                              : -q(i, j + 1, k, QV);
    v[6] = flags(i, j - 1, k).isRegular() ? q(i, j - 1, k, QV)
                                          : -q(i, j - 1, k - 1, QV);
    v[7] = flags(i, j - 1, k - 1).isRegular() ? q(i, j - 1, k - 1, QV)
                                              : -q(i, j - 1, k, QV);

    w[0] = flags(i + 1, j, k).isRegular() ? q(i + 1, j, k, QW)
                                          : -q(i + 1, j, k - 1, QW);
    w[1] = flags(i + 1, j, k - 1).isRegular() ? q(i + 1, j, k - 1, QW)
                                              : -q(i + 1, j, k, QW);
    w[2] = flags(i - 1, j, k).isRegular() ? q(i - 1, j, k, QW)
                                          : -q(i - 1, j, k - 1, QW);
    w[3] = flags(i - 1, j, k - 1).isRegular() ? q(i - 1, j, k - 1, QW)
                                              : -q(i - 1, j, k, QW);

    w[4] = flags(i, j + 1, k).isRegular() ? q(i, j + 1, k, QW)
                                          : -q(i, j + 1, k - 1, QW);
    w[5] = flags(i, j + 1, k - 1).isRegular() ? q(i, j + 1, k - 1, QW)
                                              : -q(i, j + 1, k, QW);
    w[6] = flags(i, j - 1, k).isRegular() ? q(i, j - 1, k, QW)
                                          : -q(i, j - 1, k - 1, QW);
    w[7] = flags(i, j - 1, k - 1).isRegular() ? q(i, j - 1, k - 1, QW)
                                              : -q(i, j - 1, k, QW);

    td(i, j, k, 0) = 0.25 * dx1inv * (u[0] + u[1] - u[2] - u[3]); // dudx
    td(i, j, k, 1) = 0.25 * dx1inv * (v[0] + v[1] - v[2] - v[3]); // dvdx
    td(i, j, k, 2) = 0.25 * dx1inv * (w[0] + w[1] - w[2] - w[3]); // dwdx
    td(i, j, k, 3) = 0.25 * dx2inv * (u[4] + u[5] - u[6] - u[7]); // dudy
    td(i, j, k, 4) = 0.25 * dx2inv * (v[4] + v[5] - v[6] - v[7]); // dvdy
    td(i, j, k, 5) = 0.25 * dx2inv * (w[4] + w[5] - w[6] - w[7]); // dwdy
  }
}
#endif

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_compute_tangential_vel_derivs(
  const int i,
  const int j,
  const int k,
  const amrex::Array4<const amrex::Real>& q,
  const int dir,
  const amrex::Real dx1,
  const amrex::Real dx2,
  const amrex::Array4<amrex::Real>& td)
{
  // dx1 and dx2 will be the trangential grid spacing
  const amrex::Real dx1inv = 1.0 / dx1;
  const amrex::Real dx2inv = 1.0 / dx2;
  // This is because it is a tangential velocity derivative.
  if (dir == 0) {
    // dx1 = dy, dx2 = dz
#if AMREX_SPACEDIM > 1
    td(i, j, k, 0) = 0.25 * dx1inv *
                     (q(i, j + 1, k, QU) + q(i - 1, j + 1, k, QU) -
                      q(i, j - 1, k, QU) - q(i - 1, j - 1, k, QU)); // dudy
    td(i, j, k, 1) = 0.25 * dx1inv *
                     (q(i, j + 1, k, QV) + q(i - 1, j + 1, k, QV) -
                      q(i, j - 1, k, QV) - q(i - 1, j - 1, k, QV)); // dvdy
#endif
#if AMREX_SPACEDIM > 2
    td(i, j, k, 2) = 0.25 * dx1inv *
                     (q(i, j + 1, k, QW) + q(i - 1, j + 1, k, QW) -
                      q(i, j - 1, k, QW) - q(i - 1, j - 1, k, QW)); // dwdy
    td(i, j, k, 3) = 0.25 * dx2inv *
                     (q(i, j, k + 1, QU) + q(i - 1, j, k + 1, QU) -
                      q(i, j, k - 1, QU) - q(i - 1, j, k - 1, QU)); // dudz
    td(i, j, k, 4) = 0.25 * dx2inv *
                     (q(i, j, k + 1, QV) + q(i - 1, j, k + 1, QV) -
                      q(i, j, k - 1, QV) - q(i - 1, j, k - 1, QV)); // dvdz
    td(i, j, k, 5) = 0.25 * dx2inv *
                     (q(i, j, k + 1, QW) + q(i - 1, j, k + 1, QW) -
                      q(i, j, k - 1, QW) - q(i - 1, j, k - 1, QW)); // dwdz
#endif
  } else if (dir == 1) {
    // dx1 = dx, dx2 = dz
    td(i, j, k, 0) = 0.25 * dx1inv *
                     (q(i + 1, j, k, QU) + q(i + 1, j - 1, k, QU) -
                      q(i - 1, j, k, QU) - q(i - 1, j - 1, k, QU)); // dudx
    td(i, j, k, 1) = 0.25 * dx1inv *
                     (q(i + 1, j, k, QV) + q(i + 1, j - 1, k, QV) -
                      q(i - 1, j, k, QV) - q(i - 1, j - 1, k, QV)); // dvdx
#if AMREX_SPACEDIM > 2
    td(i, j, k, 2) = 0.25 * dx1inv *
                     (q(i + 1, j, k, QW) + q(i + 1, j - 1, k, QW) -
                      q(i - 1, j, k, QW) - q(i - 1, j - 1, k, QW)); // dwdx
    td(i, j, k, 3) = 0.25 * dx2inv *
                     (q(i, j, k + 1, QU) + q(i, j - 1, k + 1, QU) -
                      q(i, j, k - 1, QU) - q(i, j - 1, k - 1, QU)); // dudz
    td(i, j, k, 4) = 0.25 * dx2inv *
                     (q(i, j, k + 1, QV) + q(i, j - 1, k + 1, QV) -
                      q(i, j, k - 1, QV) - q(i, j - 1, k - 1, QV)); // dvdz
    td(i, j, k, 5) = 0.25 * dx2inv *
                     (q(i, j, k + 1, QW) + q(i, j - 1, k + 1, QW) -
                      q(i, j, k - 1, QW) - q(i, j - 1, k - 1, QW)); // dwdz
#endif
  } else if (dir == 2) {
    // dx1 = dx, dx2 = dy
    td(i, j, k, 0) = 0.25 * dx1inv *
                     (q(i + 1, j, k, QU) + q(i + 1, j, k - 1, QU) -
                      q(i - 1, j, k, QU) - q(i - 1, j, k - 1, QU)); // dudx
    td(i, j, k, 1) = 0.25 * dx1inv *
                     (q(i + 1, j, k, QV) + q(i + 1, j, k - 1, QV) -
                      q(i - 1, j, k, QV) - q(i - 1, j, k - 1, QV)); // dvdx
    td(i, j, k, 2) = 0.25 * dx1inv *
                     (q(i + 1, j, k, QW) + q(i + 1, j, k - 1, QW) -
                      q(i - 1, j, k, QW) - q(i - 1, j, k - 1, QW)); // dwdx
    td(i, j, k, 3) = 0.25 * dx2inv *
                     (q(i, j + 1, k, QU) + q(i, j + 1, k - 1, QU) -
                      q(i, j - 1, k, QU) - q(i, j - 1, k - 1, QU)); // dudy
    td(i, j, k, 4) = 0.25 * dx2inv *
                     (q(i, j + 1, k, QV) + q(i, j + 1, k - 1, QV) -
                      q(i, j - 1, k, QV) - q(i, j - 1, k - 1, QV)); // dvdy
    td(i, j, k, 5) = 0.25 * dx2inv *
                     (q(i, j + 1, k, QW) + q(i, j + 1, k - 1, QW) -
                      q(i, j - 1, k, QW) - q(i, j - 1, k - 1, QW)); // dwdy
  }
}

#ifdef PELEC_USE_EB
void pc_compute_tangential_vel_derivs_eb(
  const amrex::Box&,
  const int,
  const amrex::Real,
  const amrex::Real,
  const EBBndryGeom*,
  const int,
  const amrex::Array4<const amrex::Real>&,
  const amrex::Array4<amrex::EBCellFlag const>&,
  const amrex::Array4<amrex::Real>&);
#endif
#endif
