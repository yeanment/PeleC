module probdata_module

  double precision, save   :: u0,v0,w0,p0,T0,bulk_visc_0,visc_0,diff_0,cond_0,molec_wt_0,u_jet,press_jet,temp_jet,XO2_jet,XN2_jet
  integer, save            :: diff_direction
  double precision, save, dimension(:), allocatable :: rinput, uinput, uprinput, uptinput, upzinput

contains

 subroutine jet_setup(x,time,u_ext,u_int,dir,sgn)

    use eos_type_module
    use meth_params_module, only : URHO, UMX, UMY, UMZ, UTEMP, UEDEN, UEINT, UFS,UFX
    !use chemistry_module, only : nspecies, get_species_index
    use network, only : naux
    use amrex_constants_module, only: M_PI, HALF
    use eos_module

    double precision :: u_ext(*),u_int(*),x(3),time
    double precision :: xc,yc,zc,radius,theta,ujet_pipeflow,vjet_pipeflow,wjet_pipeflow,XCO2
    integer          :: dir,sgn,iO2,iN2
    type (eos_t) :: eos_state

    call build(eos_state)

    call destroy(eos_state)

 end subroutine jet_setup

 end module probdata_module
