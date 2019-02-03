module drag_module

  implicit none

  public

contains

  subroutine update_particles(np, lev, particles, state, state_lo, state_hi, &
!HK                              source, source_lo, source_hi, & 
                              domlo, domhi, plo, phi, reflect_lo, reflect_hi, &
                              dx, dt, do_move) &
       bind(c,name='update_particles')

    use iso_c_binding
    use amrex_error_module
    use network
    use amrex_fort_module, only : amrex_real
    use meth_params_module, only : NVAR, URHO, UMX, UMY, UMZ, UEDEN, UEINT, UTEMP, UFS
    use particle_mod      , only: particle_t

    implicit none

    integer,          intent(in   )        :: np
    type(particle_t), intent(inout)        :: particles(np)
    integer,          intent(in   )        :: state_lo(2), state_hi(2)

    integer,          intent(in   )        :: domlo(2), domhi(2)
    real(amrex_real), intent(in   )        :: state &
         (state_lo(1):state_hi(1),state_lo(2):state_hi(2),NVAR)

    real(amrex_real), intent(in   )        :: plo(2),phi(2),dx(2),dt
    integer,          intent(in   )        :: do_move, lev
    integer,          intent(in   )        :: reflect_lo(2), reflect_hi(2)

    integer          :: i, j, i2, j2, n, nc, nf,  iloc, jloc,is_to_skip
    real(amrex_real) :: wx_lo, wy_lo, wx_hi, wy_hi
    real(amrex_real) :: lx, ly, lx2, ly2
    real(amrex_real) :: half_dt
    real(amrex_real) :: inv_dx(2), inv_vol, fluid_vel(2)
    real(amrex_real) :: fluid_dens, visc, force(2), drag_coef

    real(amrex_real) :: diff_u, diff_v, diff_velmag, reyn, drag, pmass

    real(amrex_real) :: coef_ll, coef_hl, coef_lh, coef_hh

    integer :: is, ie, L, M
    integer :: lo(3), hi(3)


    real(kind=dp_t), dimension(np) :: inv_tau


    real*8, parameter :: pi = 3.1415926535897932d0
    real*8, parameter :: half_pi = 0.5d0*Pi
    real*8, parameter :: pi_six = Pi/6.0d0
    real*8, parameter :: one_third = 1.0d0/3.0d0


    inv_dx = 1.0d0/dx
    inv_vol = inv_dx(1) * inv_dx(2)
    half_dt = 0.5d0 * dt


   !Eventually, when PelePhysics kernels are ported to GPU
   !we can call various thermodynamics kernels w.r.t wos_state
   !For now hard coding gas phase properties; set to air at STP
   fluid_dens = 1.1798e-3  !g/cm^3
   visc = fluid_dens * 1.5753e-1 ! CGS units

    do n = 1, np

     if ((particles(n)%id.eq.-1).or. particles(n)%id.gt.1e5.or. & 
           (particles(n)%pos(1).ne.particles(n)%pos(1))) then 
    !     print *,'PARTICLE ID ', lev,particles(n)%id,' BUST ',&
    !        particles(n)%pos(1),particles(n)%pos(2)
     else
    !     print *,'REAL ', lev,particles(n)%id, &
    !        particles(n)%pos(1),particles(n)%pos(2),particles(n)%diam

       ! ****************************************************
       ! Compute the forcing term at the particle locations
       ! ****************************************************

       lx = (particles(n)%pos(1) - plo(1))*inv_dx(1) + 0.5d0
       ly = (particles(n)%pos(2) - plo(2))*inv_dx(2) + 0.5d0

       i = floor(lx)
       j = floor(ly)

       if (i-1 .lt. state_lo(1) .or. i .gt. state_hi(1) .or. &
           j-1 .lt. state_lo(2) .or. j .gt. state_hi(2)) then
          print *,'PARTICLE ID ', particles(n)%id,' REACHING OUT OF BOUNDS AT (I,J) = ',i,j
          print *,'Array bounds are ', state_lo(:), state_hi(:)
          print *,'(x,y) are ', particles(n)%pos(1), particles(n)%pos(2)
          call amrex_error('Aborting in update_particles')
       !else
       !   print *,'PARTICLE ID ', particles(n)%id,' at grid point: ',i,j
       !   print *,'Array bounds are ', state_lo(:), state_hi(:)
       !   print *,'(x,y) are ', particles(n)%pos(1), particles(n)%pos(2)
       end if

       wx_hi = lx - i
       wy_hi = ly - j

       wx_lo = 1.0d0 - wx_hi
       wy_lo = 1.0d0 - wy_hi

       coef_ll = wx_lo * wy_lo
       coef_hl = wx_hi * wy_lo
       coef_lh = wx_lo * wy_hi
       coef_hh = wx_hi * wy_hi

       ! Compute the velocity of the fluid at the particle
       do nc = 1, 2
          nf = UMX + (nc-1)
          fluid_vel(nc) = &
                coef_ll*state(i-1, j-1, nf)/state(i-1,j-1,URHO) + &
                coef_lh*state(i-1, j,   nf)/state(i-1,j  ,URHO) + &
                coef_hl*state(i,   j-1, nf)/state(i  ,j-1,URHO) + &
                coef_hh*state(i,   j,   nf)/state(i  ,j  ,URHO) 
       end do


       ! ****************************************************
       ! Source terms by individual drop
       ! ****************************************************

       pmass = pi_six*particles(n)%density*particles(n)%diam**3

       diff_u = fluid_vel(1)-particles(n)%vel(1)
       diff_v = fluid_vel(2)-particles(n)%vel(2)

       diff_velmag = sqrt( diff_u**2 + diff_v**2 )

       ! Local Reynolds number = (Density * Relative Velocity) * (Particle Diameter) / (Viscosity)
       reyn = fluid_dens * diff_velmag * particles(n)%diam / visc

       drag_coef = 1.0d0+0.15d0*reyn**(0.687d0)

       ! Time constant.
!HK    inv_tau(n) = (18.0d0*mu_skin(n))/(particles(n)%density*particles(n)%diam**2)
       inv_tau(n) = (18.0d0*   visc   )/(particles(n)%density*particles(n)%diam**2)

       ! Drag coefficient =  (pi / 8) * (Density * Relative Velocity) * (Particle Diameter)**2
       ! drag = 0.125d0 * pi * (particles(n)%diam)**2 * fluid_dens * diff_velmag
       drag = inv_tau(n)*drag_coef*pmass

       force(1) = drag*diff_u
       force(2) = drag*diff_v


       ! ****************************************************
       ! Now apply the forcing term to the particles
       ! ****************************************************

       do nc = 1, 2
          ! Update velocity by half dt
          particles(n)%vel(nc) = particles(n)%vel(nc) + half_dt * force(nc) / pmass

          ! Update position by full dt
          if (do_move .eq. 1) &
             particles(n)%pos(nc) =particles(n)%pos(nc) + dt * particles(n)%vel(nc) 
       end do


     endif

    end do


       if (do_move .eq. 1) then

          ! If at a reflecting boundary (Symmetry or Wall),     
          ! flip the position back into the domain and flip the sign of the normal velocity 

          do nc = 1, 2

             if (reflect_lo(nc) .eq. 1) then
                do n = 1, np
                   if (particles(n)%pos(nc) .lt. plo(nc)) then
                      particles(n)%pos(nc) = 2.d0*plo(nc) - particles(n)%pos(nc) 
                      particles(n)%vel(nc) = -particles(n)%vel(nc) 
                   end if
                end do
             end if

             if (reflect_hi(nc) .eq. 1) then
                do n = 1, np
                   if (particles(n)%pos(nc) .lt. plo(nc)) then
                      particles(n)%pos(nc) = 2.d0*phi(nc)-particles(n)%pos(nc) 
                      particles(n)%vel(nc) = -particles(n)%vel(nc) 
                   end if
                end do
             end if

          end do

       end if


  end subroutine update_particles

end module drag_module
