module pc_prob_module

  implicit none

  private

  public :: amrex_probinit, pc_initdata, pc_prob_close

contains

  subroutine amrex_probinit (init,name,namlen,problo,probhi) bind(C, name = "amrex_probinit")

    use probdata_module
    use amrex_fort_module
    use eos_module
    
    
    implicit none
    integer :: init, namlen
    integer :: name(namlen)
    double precision :: problo(3), probhi(3), L_x, L_y, L_z

    integer :: untin,i

    type(eos_t) :: eos_state
    namelist /fortin/ u0, v0, w0, p0, T0,u_jet,press_jet,temp_jet,XO2_jet,XN2_jet,diff_direction

    ! Build "probin" filename -- the name of file containing fortin namelist.
    integer, parameter :: maxlen = 256
    character :: probin*(maxlen)

    ! Local                                                                                                                                                          
    double precision, dimension(:), allocatable :: data
    integer(kind=8) :: nr

    if (namlen .gt. maxlen) then
       call bl_error('probin file name too long')
    end if

    do i = 1, namlen
       probin(i:i) = char(name(i))
    end do

    ! set namelist defaults
    u0 = 0.0_amrex_real
    v0 = 0.0_amrex_real
    w0 = 0.0_amrex_real
    p0      =  101325000.00_amrex_real
    T0       =      600.00_amrex_real

    u_jet        =     5000.00_amrex_real
    press_jet    =  101325000.00_amrex_real
    temp_jet     =      600.00_amrex_real

    XO2_jet     =     0.2095d0
    XN2_jet     =     0.7905d0

    diff_direction = 1
    
    ! Read namelists
    untin = 9
    open(untin,file=probin(1:namlen),form='formatted',status='old')
    read(untin,fortin)
    close(unit=untin)

    ! set local variable defaults
    L_x = probhi(1) - problo(1)
    L_y = probhi(2) - problo(2)
    L_z = probhi(3) - problo(3)

  end subroutine amrex_probinit


  ! ::: -----------------------------------------------------------
  ! ::: This routine is called at problem setup time and is used
  ! ::: to initialize data on each grid.
  ! :::
  ! ::: NOTE:  all arrays have one cell of ghost zones surrounding
  ! :::        the grid interior.  Values in these cells need not
  ! :::        be set here.
  ! :::
  ! ::: INPUTS/OUTPUTS:
  ! :::
  ! ::: level     => amr level of grid
  ! ::: time      => time at which to init data
  ! ::: lo,hi     => index limits of grid interior (cell centered)
  ! ::: nvar      => number of state components.
  ! ::: state     <= scalar array
  ! ::: delta     => cell size
  ! ::: xlo, xhi  => physical locations of lower left and upper
  ! :::              right hand corner of grid.  (does not include
  ! :::		   ghost region).
  ! ::: -----------------------------------------------------------
  subroutine pc_initdata(level,time,lo,hi,nvar, &
       state,state_lo,state_hi, &
       delta,xlo,xhi) bind(C, name="pc_initdata")
    use amrex_paralleldescriptor_module, only: amrex_pd_ioprocessor
    use probdata_module
    use meth_params_module, only : URHO, UMX, UMY, UMZ, UEDEN, UEINT, UTEMP, UFS, UFX
    use amrex_constants_module, only: M_PI, HALF,ZERO
    use eos_type_module
    use eos_module
    !use chemistry_module, only: nspecies, get_species_index
    use network, only: nspecies
    implicit none

    integer :: level, nvar
    integer :: lo(3), hi(3)
    integer :: state_lo(3),state_hi(3)
    double precision :: rho0, eint0, umag0, rho_jet, eint_jet
    double precision :: xlo(3), xhi(3), time, delta(3)
    double precision :: state(state_lo(1):state_hi(1), &
         state_lo(2):state_hi(2), &
         state_lo(3):state_hi(3),nvar)

    integer :: i,j,k,iO2,iN2
    integer, parameter :: out_unit=20
    double precision :: yp, u, v, w
    double precision :: xvec(3)
    type(eos_t) :: eos_state, eos_state_jet

    !  Get species indexes and assign to EOS                                                                                                                 

    !iO2 = get_species_index("O2")
    !iN2 = get_species_index("N2")

    call build(eos_state)

    ! Set the equation of state variables: outside of jet
    eos_state % p = p0
    eos_state % T = T0
    eos_state % molefrac    = 1.d0
    !eos_state % molefrac    = 0.d0
    !eos_state % molefrac(iO2)= XO2_jet
    !eos_state % molefrac(iN2)= XN2_jet
    call eos_xty(eos_state) ! get mass fractions from mole fractions  
    call eos_tp(eos_state)
    rho0 = eos_state % rho
    umag0 = sqrt(u0**2 + v0**2 + w0**2)
    eint0 = eos_state % e

    ! Set the equation of state variables: inside of jet
    eos_state_jet % p = press_jet
    eos_state_jet % T = temp_jet
    eos_state_jet % molefrac    = 1.d0
    !eos_state_jet % molefrac    = 0.d0
    !eos_state_jet % molefrac(iO2)= XO2_jet
    !eos_state_jet % molefrac(iN2)= XN2_jet
    call eos_xty(eos_state_jet) ! get mass fractions from mole fractions  
    call eos_tp(eos_state_jet)
    rho_jet = eos_state_jet % rho
    eint_jet = eos_state_jet % e

    do k = lo(3), hi(3)
       xvec(3) = xlo(3) + delta(3)*(dble(k-lo(3)) + HALF)
       do j = lo(2), hi(2)
          xvec(2) = xlo(2) + delta(2)*(dble(j-lo(2)) + HALF)
          do i = lo(1), hi(1)
             xvec(1) = xlo(1) + delta(1)*(dble(i-lo(1)) + HALF)

             ! Use a constant velocity to initialize the velocities

             if (any(xvec .gt.0)) then

                state(i,j,k,URHO)            = rho0
                state(i,j,k,UFS:UFS+nspecies-1) = rho0 * eos_state % massfrac(1:nspecies)
                state(i,j,k,UMX)             = rho0 * u0
                state(i,j,k,UMY)             = rho0 * v0
                state(i,j,k,UMZ)             = rho0 * w0
                
                state(i,j,k,UEINT)           = rho0 * eint0
                state(i,j,k,UEDEN)           = rho0 * (eint0 + HALF * (umag0*umag0))
                state(i,j,k,UTEMP)           = T0
                state(i,j,k,UFX:UFX+naux-1)  = rho0 * 0.0d+0 
                
             else 

                state(i,j,k,URHO)            = rho_jet
                state(i,j,k,UFS:UFS+nspecies-1) = rho_jet * eos_state_jet % massfrac(1:nspecies)
                state(i,j,k,UMX)             = rho_jet * u0
                state(i,j,k,UMY)             = rho_jet * v0
                state(i,j,k,UMZ)             = rho_jet * w0
                
                state(i,j,k,UEINT)           = rho_jet * eint_jet
                state(i,j,k,UEDEN)           = rho_jet * (eint_jet + HALF * (umag0*umag0))
                state(i,j,k,UTEMP)           = temp_jet
                state(i,j,k,UFX:UFX+naux-1)  = rho_jet * 1.0d+0 

             end if
             
          enddo
       enddo
    enddo

  end subroutine pc_initdata


  subroutine pc_prob_close() &
       bind(C, name="pc_prob_close")
  end subroutine pc_prob_close

end module pc_prob_module
