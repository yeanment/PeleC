#include "prob.H"

void
init_bc()
{
  // Initialize of the inlet & outlet state
  amrex::Real massfrac[NUM_SPECIES] = {0.0};

  // Replace with mole fractions
  auto eos = pele::physics::PhysicsType::eos();
  for (int n = 0; n < NUM_SPECIES; n++) {
    massfrac[n] = PeleC::h_prob_parm_device->inlet_state[n] ;
  }

  const amrex::Real p_0 = PeleC::h_prob_parm_device->p_0;
  const amrex::Real T_0 = PeleC::h_prob_parm_device->T_0;
  const amrex::Real area_ratio = PeleC::h_prob_parm_device->area_ratio;
  amrex::Real gamma;
  // Determine the gamma to use
  if (PeleC::h_prob_parm_device->gamma_ref_temp == -1) {
    eos.TY2G(T_0, massfrac, gamma);
    // Default initialize with T_0
    PeleC::h_prob_parm_device->gamma = gamma;
  } else if (PeleC::h_prob_parm_device->gamma_ref_temp == 0) {
    eos.TY2G(T_0, massfrac, gamma);
    // Default initialize with T_0
    PeleC::h_prob_parm_device->gamma = gamma;
  } else if (PeleC::h_prob_parm_device->gamma_ref_temp == 1) {
    const amrex::Real Tamb = PeleC::h_prob_parm_device->Tamb;
    eos.TY2G(Tamb, massfrac, gamma);
    // Default initialize with T_amb
    PeleC::h_prob_parm_device->gamma = gamma;
  } else {
    amrex::Abort("Invalid gamma_ref_temp specified");
  }

  // Obtain Ma_1 & Ma_2
  amrex::Real Ma_1 = 10.0, Ma_2 = 0.1;
  amrex::Real p_1, p_2, p_3;
  int func_sol = 0;
  int retval =
    pele::kinsolma::kinsol_getma(Ma_1, Ma_2, func_sol, gamma, area_ratio);
  PeleC::h_prob_parm_device->Ma_1 = Ma_1;
  PeleC::h_prob_parm_device->Ma_2 = Ma_2;
  amrex::Print() << "The area_ratio = " << area_ratio << std::endl;
  amrex::Print() << "The damp_ratio = " << PeleC::h_prob_parm_device->damp_ratio
                 << std::endl;
  amrex::Print() << "The critical Ma = " << Ma_1 << "/" << Ma_2 << std::endl;
  // Obtain p_1, p_2 & p_3
  p_1 =
    p_0 / std::pow((1 + (gamma - 1) * Ma_1 * Ma_1 / 2.), (gamma) / (gamma - 1));
  p_2 =
    p_0 / std::pow((1 + (gamma - 1) * Ma_2 * Ma_2 / 2.), (gamma) / (gamma - 1));
  p_3 = p_1 * (1 + 2 * gamma / (gamma + 1) * (Ma_1 * Ma_1 - 1));
  PeleC::h_prob_parm_device->p_1 = p_1;
  PeleC::h_prob_parm_device->p_2 = p_2;
  PeleC::h_prob_parm_device->p_3 = p_3;
  amrex::Print() << "The critical p = " << PeleC::h_prob_parm_device->p_1 << "/"
                 << PeleC::h_prob_parm_device->p_2 << "/"
                 << PeleC::h_prob_parm_device->p_3 << std::endl;
  amrex::Real p_cr;
  p_cr = p_0 * std::pow((2. / (gamma + 1)), (gamma) / (gamma - 1));
  PeleC::h_prob_parm_device->p_cr = p_cr;
  amrex::Print() << "The critical p_cr = " << PeleC::h_prob_parm_device->p_cr
                 << std::endl;
}

void
pc_prob_close()
{
}

extern "C" {
void
amrex_probinit(
  const int* /*init*/,
  const int* /*name*/,
  const int* /*namelen*/,
  const amrex::Real* problo,
  const amrex::Real* probhi)
{
  std::string pmf_datafile;

  amrex::ParmParse pp("prob");
  pp.query("pamb", PeleC::h_prob_parm_device->pamb);
  pp.query("Tamb", PeleC::h_prob_parm_device->Tamb);
  pp.query("p_hot", PeleC::h_prob_parm_device->p_hot);
  pp.query("T_hot", PeleC::h_prob_parm_device->T_hot);
  amrex::Vector<amrex::Real> pos_hot(AMREX_SPACEDIM, 0);
  amrex::Vector<amrex::Real> rad_hot(AMREX_SPACEDIM, 0);
  amrex::Vector<amrex::Real> u_hot(AMREX_SPACEDIM, 0);
  pp.queryarr("pos_hot", pos_hot, 0, AMREX_SPACEDIM);
  pp.queryarr("rad_hot", rad_hot, 0, AMREX_SPACEDIM);
  pp.queryarr("u_hot", u_hot, 0, AMREX_SPACEDIM);
  for (int i = 0; i < AMREX_SPACEDIM; i++) {
    PeleC::h_prob_parm_device->pos_hot[i] = pos_hot[i];
    PeleC::h_prob_parm_device->rad_hot[i] = rad_hot[i];
    PeleC::h_prob_parm_device->u_hot[i] = u_hot[i];
  }
  pp.query("init_u_hot", PeleC::h_prob_parm_device->init_u_hot);
  pp.query("air_fill_outside_x", PeleC::h_prob_parm_device->air_fill_outside_x);

  pp.query("p_0", PeleC::h_prob_parm_device->p_0);
  pp.query("T_0", PeleC::h_prob_parm_device->T_0);
  pp.query("damp_ratio", PeleC::h_prob_parm_device->damp_ratio);
  pp.query("area_ratio", PeleC::h_prob_parm_device->area_ratio);
  pp.query("gamma_ref_temp", PeleC::h_prob_parm_device->gamma_ref_temp);

  pp.query("inlet_type", PeleC::h_prob_parm_device->inlet_type);
  pp.query("outlet_type", PeleC::h_prob_parm_device->outlet_type);
  pp.query("inlet_slip_type", PeleC::h_prob_parm_device->inlet_slip_type);


  // Initialize of the inlet & outlet state
  amrex::Real molefrac[NUM_SPECIES] = {0.0};
  amrex::Real massfrac[NUM_SPECIES] = {0.0};

  // Obtain the spec_names
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Vector<std::string> m_spec_names;
  pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
    m_spec_names);
  amrex::Real sum  = 0;
  amrex::Print() << "Reading the mole fractions for initialize." << std::endl;
  for (int n = 0; n < NUM_SPECIES; n++) {
    pp.query(m_spec_names[n].c_str(), molefrac[n]);
    if (molefrac[n] > 0 ){
      amrex::Print() << m_spec_names[n] << ": " << molefrac[n] << "; ";
      sum += molefrac[n];
    } else if (molefrac[n] < 0 ){
      amrex::Print() << std::endl;
      amrex::Print() << m_spec_names[n] << ": " << molefrac[n] << "; ";
      amrex::Abort("Invalid mole fractions specified");
    }
  }
  amrex::Print() << std::endl;
  if (sum <= 0) {
    amrex::Abort("Negative sum mole fractions specified");
  }
  amrex::Print() << "Normalize the mole fractions." << std::endl;
  for (int n = 0; n < NUM_SPECIES; n++) {
    if (molefrac[n] > 0 ){
      molefrac[n] /= sum;
      amrex::Print() << m_spec_names[n] << ": " << molefrac[n] << "; ";
    }
  }
  amrex::Print() << std::endl;
  eos.X2Y(molefrac, massfrac);
  for (int n = 0; n < NUM_SPECIES; n++) {
    PeleC::h_prob_parm_device->inlet_state[n] = massfrac[n];
    PeleC::h_prob_parm_device->outlet_state[n] = massfrac[n];
  }  

  init_bc();
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
