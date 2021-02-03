#include "prob.H"

void
pc_prob_close()
{
}

extern "C" {
void
amrex_probinit(
  const int* init,
  const int* name,
  const int* namelen,
  const amrex_real* problo,
  const amrex_real* probhi)
{
  // Parse params
  amrex::ParmParse pp("prob");
  pp.query("p0", PeleC::prob_parm_device->p0);
  pp.query("T0", PeleC::prob_parm_device->T0);
}
}

void
PeleC::problem_post_timestep()
{
}

void
PeleC::problem_post_init()
{
}

void
PeleC::problem_post_restart()
{
}

void
EBsCO2Combustor(const amrex::Geometry& geom, const int max_level)
{
  int max_coarsening_level = max_level; // Because there are no mg solvers here

  amrex::EB2::CylinderIF polys(0.5 * 7.542, 100.0, 0, {0, 0, 0}, true);

  auto gshop = amrex::EB2::makeShop(polys);
  amrex::EB2::Build(
    gshop, geom, max_coarsening_level, max_coarsening_level, 4, false);
}
