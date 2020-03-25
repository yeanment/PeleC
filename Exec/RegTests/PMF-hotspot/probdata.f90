module probdata_module

  use pmf_module
  
  ! HIT parameters
  character(len=255), save :: iname
  logical, save            :: binfmt
  logical, save            :: restart
  double precision, save   :: urms0
  integer, save            :: inres
  integer, save            :: rescale_hit
  double precision, save   :: uin_norm
  double precision, save   :: L_x, L_y, L_z
  double precision, save   :: probxmin, probymin, probzmin
  double precision, save   :: Linput
  double precision, save, dimension(:,:,:), allocatable :: xinput, yinput, zinput, &
       uinput, vinput, winput
  double precision, save, dimension(:), allocatable :: xarray, xdiff
  double precision, save   :: k0, rho0, tau, p0, T0, eint0
  character(len=255), save :: velocity_profile
  double precision, save :: pamb, tamb, thot, dhot, phi_in, L(3), ltrans, strain
  character(len=255), save :: fuelname
  character(len=72), save :: pmf_datafile
  character(len=255), save :: hotspot_type
  
end module probdata_module
