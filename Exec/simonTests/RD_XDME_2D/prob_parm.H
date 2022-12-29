#ifndef PROB_PARM_H
#define PROB_PARM_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuMemory.H>

struct ProbParmDevice
{
  // Define the unburnt mixtures
  amrex::Real pamb = 1013250.0;
  amrex::Real Tamb = 300.0;

  // Define the init
  amrex::Real p_hot = 1013250.0 * 100;
  amrex::Real T_hot = 2000.0;
  // Region for the hot spot area: x_i - rad_i : x_i + rad_i
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos_hot = {{0.0}};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rad_hot = {{0.5}};
  // Define the area with air along x direction
  int init_u_hot = 0;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> u_hot = {{0.0}};
  // Define the area with air along x direction
  int air_fill_outside_x = 0;

  // amrex::GpuArray<amrex::Real, NVAR> inlet_state = {{0.0}};
  // Define the inlet/outlet state, total pressure and total temperature
  amrex::GpuArray<amrex::Real, NUM_SPECIES> inlet_state = {{0.0}};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> outlet_state = {{0.0}};
  amrex::Real p_0 = 1013250.0 * 100;
  amrex::Real T_0 = 300.0;
  amrex::Real damp_ratio = 0.05;
  // Define the area ratio for the Laval nozzle flow, \sigma_e/sigma_*
  amrex::Real area_ratio = 10.0;
  amrex::Real gamma = 1.4;
  amrex::Real p_1 = 0.0;
  amrex::Real p_2 = 0.0;
  amrex::Real p_3 = 0.0;
  amrex::Real Ma_1 = 3.922551820933724;
  amrex::Real Ma_2 = 0.057987202922874;
  amrex::Real p_cr = 0.0;
  // Define the inlet Mach Number
  amrex::Real Ma_in = 0.2;
  // Fix the gamma inside the nozzle with gamma_ref_temp
  // - -1: use wall temperature
  // -  0: use T_0
  // -  1: use Tamb 
  int gamma_ref_temp = 0;
  int inlet_type = 0;
  int inlet_slip_type = 1; // Zero/non-zero tangential velocity
  // Initial test with outlet conditions
  int outlet_type = 1;

  // amrex::Real* d_pmf_X = nullptr;
  // amrex::Real* d_pmf_Y = nullptr;
};

struct ProbParmHost
{
  // amrex::Vector<amrex::Real> h_pmf_X;
  // amrex::Vector<amrex::Real> h_pmf_Y;
  // amrex::Gpu::DeviceVector<amrex::Real> pmf_X;
  // amrex::Gpu::DeviceVector<amrex::Real> pmf_Y;
  // ProbParmHost() : pmf_X(0), pmf_Y(0) {}
};

#endif
