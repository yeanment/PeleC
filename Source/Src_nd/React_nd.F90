module reactions_module

  implicit none

  public

contains

#if defined(USE_DVODE) || defined(USE_FORTRAN_CVODE) 
  subroutine pc_react_state(lo,hi, &
                            uold,uo_lo,uo_hi, &
                            unew,un_lo,un_hi, &
                            asrc,as_lo,as_hi, &
                            mask,m_lo,m_hi, &
                            cost,c_lo,c_hi, &
                            IR,IR_lo,IR_hi, &
                            time,dt_react,do_update) bind(C, name="pc_react_state")

    use network           , only : nspec
    use meth_params_module, only : NVAR, URHO, UMX, UMZ, UEDEN, UEINT, UTEMP, &
                                   UFS
    use react_type_module
#ifdef USE_DVODE
    use reactor_module, only : react
#elif USE_FORTRAN_CVODE
    use reactor_module, only : react_cvode
#endif
    use bl_constants_module, only : HALF
    use react_type_module

    implicit none

    integer          ::    lo(3),    hi(3)
    integer          :: uo_lo(3), uo_hi(3)
    integer          :: un_lo(3), un_hi(3)
    integer          :: as_lo(3), as_hi(3)
    integer          ::  m_lo(3),  m_hi(3)
    integer          ::  c_lo(3),  c_hi(3)
    integer          :: IR_lo(3), IR_hi(3)
    double precision :: uold(uo_lo(1):uo_hi(1),uo_lo(2):uo_hi(2),uo_lo(3):uo_hi(3),NVAR)
    double precision :: unew(un_lo(1):un_hi(1),un_lo(2):un_hi(2),un_lo(3):un_hi(3),NVAR)
    double precision :: asrc(as_lo(1):as_hi(1),as_lo(2):as_hi(2),as_lo(3):as_hi(3),NVAR)
    integer          :: mask(m_lo(1):m_hi(1),m_lo(2):m_hi(2),m_lo(3):m_hi(3))
    double precision :: cost(c_lo(1):c_hi(1),c_lo(2):c_hi(2),c_lo(3):c_hi(3))
    double precision :: IR(IR_lo(1):IR_hi(1),IR_lo(2):IR_hi(2),IR_lo(3):IR_hi(3),nspec+1)
    double precision :: time, dt_react
    integer          :: do_update

    integer          :: i, j, k
    double precision :: rho_e_K_old,rho_e_K_new, rhoE_old, rhoE_new, rho_new, mom_new(3)

    type (react_t) :: react_state_in, react_state_out
    type (reaction_stat_t)  :: stat

    call build(react_state_in)
    call build(react_state_out)

    do k = lo(3), hi(3)
       do j = lo(2), hi(2)
          do i = lo(1), hi(1)

             if (mask(i,j,k) .eq. 1) then

                rhoE_old                        = uold(i,j,k,UEDEN)
                rho_e_K_old                     = HALF * sum(uold(i,j,k,UMX:UMZ)**2) / uold(i,j,k,URHO)
                react_state_in %            rho = sum(uold(i,j,k,UFS:UFS+nspec-1))
                react_state_in %              e = (rhoE_old - rho_e_K_old) / uold(i,j,k,URHO)
                react_state_in %              T = uold(i,j,k,UTEMP)
                react_state_in %        rhoY(:) = uold(i,j,k,UFS:UFS+nspec-1)
                !react_state_in %    rhoedot_ext = asrc(i,j,k,UEINT)

                ! rho.e source term computed using (rho.E,rho.u,rho)_new rather than pulling from UEINT comp of asrc
                rho_e_K_new = HALF * sum(unew(i,j,k,UMX:UMZ)**2) / unew(i,j,k,URHO)
                react_state_in % rhoedot_ext = ( (unew(i,j,k,UEDEN) - rho_e_K_new) &
                     -                           (react_state_in % rho  *  react_state_in % e) ) / dt_react

                react_state_in % rhoYdot_ext(:) = asrc(i,j,k,UFS:UFS+nspec-1)
                react_state_in % i = i
                react_state_in % j = j
                react_state_in % k = k
#ifdef USE_DVODE
                stat = react(react_state_in, react_state_out, dt_react, time)
#elif USE_FORTRAN_CVODE
                stat = react_cvode(react_state_in, react_state_out, dt_react, time)
#endif

                rho_new = sum(react_state_out % rhoY(:))
                mom_new = uold(i,j,k,UMX:UMZ) + dt_react*asrc(i,j,k,UMX:UMZ)
                rhoe_new = rho_new  *  react_state_out % e
                rho_e_K_new = HALF * sum(mom_new**2) / rho_new
                rhoE_new    = rhoe_new + rho_e_K_new
                
                if (do_update .eq. 1) then

                   unew(i,j,k,URHO)            = rho_new
                   unew(i,j,k,UMX:UMZ)         = mom_new
                   unew(i,j,k,UEINT)           = rhoe_new
                   unew(i,j,k,UEDEN)           = rhoE_new
                   unew(i,j,k,UTEMP)           = react_state_out % T
                   unew(i,j,k,UFS:UFS+nspec-1) = react_state_out % rhoY(:)
                endif

                ! Add drhoY/dt to reactions MultiFab, but be
                ! careful because the reactions and state MFs may
                ! not have the same number of ghost cells.
                ! Also, be careful because the container for uold might be the same as that for unew
                if ( i .ge. IR_lo(1) .and. i .le. IR_hi(1) .and. &
                     j .ge. IR_lo(2) .and. j .le. IR_hi(2) .and. &
                     k .ge. IR_lo(3) .and. k .le. IR_hi(3) ) then

                   IR(i,j,k,1:nspec) = (react_state_out % rhoY(:) - react_state_in % rhoY(:)) / dt_react - asrc(i,j,k,UFS:UFS+nspec-1)
                   IR(i,j,k,nspec+1) = (        rhoE_new          -         rhoE_old        ) / dt_react - asrc(i,j,k,UEDEN          )

                endif

                ! Record cost of reactions
                cost(i,j,k) = stat % cost_value

             end if
          end do
       enddo
    enddo

    call destroy(react_state_in)
    call destroy(react_state_out)

  end subroutine pc_react_state
#else

  subroutine pc_prereact_state(lo,hi, &
                            uold,uo_lo,uo_hi, &
                            unew,un_lo,un_hi, &
                            asrc,as_lo,as_hi, &
                            mask,m_lo,m_hi, &
                            rY,rY_lo,rY_hi, &
                            rY_source_ext,rYs_lo,rYs_hi, &
                            rE,rE_lo,rE_hi, &
                            rE_source_ext,rEs_lo,rEs_hi, &
                            time,dt_react) bind(C, name="pc_prereact_state")

    use network           , only : nspec
    use meth_params_module, only : NVAR, URHO, UMX, UMZ, UEDEN, UEINT, UTEMP, &
                                   UFS
    use react_type_module
    use bl_constants_module, only : HALF
    use react_type_module

    implicit none

    integer          ::    lo(3),    hi(3)
    integer          :: uo_lo(3), uo_hi(3)
    integer          :: un_lo(3), un_hi(3)
    integer          :: as_lo(3), as_hi(3)
    integer          ::  m_lo(3),  m_hi(3)
    integer          ::  rY_lo(3),  rY_hi(3)
    integer          ::  rYs_lo(3),  rYs_hi(3)
    integer          ::  rE_lo(3),  rE_hi(3)
    integer          ::  rEs_lo(3),  rEs_hi(3)
    double precision :: uold(uo_lo(1):uo_hi(1),uo_lo(2):uo_hi(2),uo_lo(3):uo_hi(3),NVAR)
    double precision :: unew(un_lo(1):un_hi(1),un_lo(2):un_hi(2),un_lo(3):un_hi(3),NVAR)
    double precision :: asrc(as_lo(1):as_hi(1),as_lo(2):as_hi(2),as_lo(3):as_hi(3),NVAR)
    integer          :: mask(m_lo(1):m_hi(1),m_lo(2):m_hi(2),m_lo(3):m_hi(3))
    double precision :: rY(rY_lo(1):rY_hi(1),rY_lo(2):rY_hi(2),rY_lo(3):rY_hi(3),nspec+1)
    double precision :: rY_source_ext(rYs_lo(1):rYs_hi(1),rYs_lo(2):rYs_hi(2),&
                        rYs_lo(3):rYs_hi(3),nspec)
    double precision :: rE(rE_lo(1):rE_hi(1),rE_lo(2):rE_hi(2),rE_lo(3):rE_hi(3),1)
    double precision :: rE_source_ext(rEs_lo(1):rEs_hi(1),rEs_lo(2):rEs_hi(2),&
                        rEs_lo(3):rEs_hi(3),1)
    double precision :: time, dt_react

    integer          :: i, j, k
    double precision :: rho_e_K_old,rho_e_K_new, rhoE_old, rhoE_new, rho, mom_new(3)

    type (react_t) :: react_state_in !, react_state_out

    call build(react_state_in)

    do k = lo(3), hi(3)
       do j = lo(2), hi(2)
          do i = lo(1), hi(1)

             if (mask(i,j,k) .eq. 1) then

                rhoE_old                        = uold(i,j,k,UEDEN)
                rho_e_K_old                     = HALF * sum(uold(i,j,k,UMX:UMZ)**2) / uold(i,j,k,URHO)
                react_state_in %            rho = sum(uold(i,j,k,UFS:UFS+nspec-1))
                react_state_in %              e = (rhoE_old - rho_e_K_old) / uold(i,j,k,URHO)
                react_state_in %              T = uold(i,j,k,UTEMP)
                react_state_in %        rhoY(:) = uold(i,j,k,UFS:UFS+nspec-1)

                rho_e_K_new = HALF * sum(unew(i,j,k,UMX:UMZ)**2) / unew(i,j,k,URHO)
                react_state_in % rhoedot_ext = ( (unew(i,j,k,UEDEN) - rho_e_K_new) &
                     -                           (react_state_in % rho  *  react_state_in % e) ) / dt_react

                react_state_in % rhoYdot_ext(:) = asrc(i,j,k,UFS:UFS+nspec-1)
                react_state_in % i = i
                react_state_in % j = j
                react_state_in % k = k

                rY(i,j,k,1:nspec)            = react_state_in % rhoY(:)
                rY(i,j,k,nspec+1)            = react_state_in % T 
                rY_source_ext(i,j,k,1:nspec) = react_state_in % rhoYdot_ext(:)
                rE(i,j,k,1)                  = react_state_in % e * react_state_in % rho 
                rE_source_ext(i,j,k,1)       = react_state_in % rhoedot_ext

             end if
          end do
       enddo
    enddo

    call destroy(react_state_in)

  end subroutine pc_prereact_state

  subroutine pc_postreact_state(lo,hi, &
                            uold,uo_lo,uo_hi, &
                            unew,un_lo,un_hi, &
                            asrc,as_lo,as_hi, &
                            rY,rY_lo,rY_hi, &
                            rE,rE_lo,rE_hi, &
                            mask,m_lo,m_hi, &
                            cost,c_lo,c_hi, &
                            IR,IR_lo,IR_hi, &
                            time,dt_react,do_update) bind(C, name="pc_postreact_state")

    use network           , only : nspec
    use meth_params_module, only : NVAR, URHO, UMX, UMZ, UEDEN, UEINT, UTEMP, &
                                   UFS
    use react_type_module
    use bl_constants_module, only : HALF
    use react_type_module

    implicit none

    integer          ::    lo(3),    hi(3)
    integer          :: uo_lo(3), uo_hi(3)
    integer          :: un_lo(3), un_hi(3)
    integer          :: as_lo(3), as_hi(3)
    integer          ::  rY_lo(3),  rY_hi(3)
    integer          ::  rE_lo(3),  rE_hi(3)
    integer          ::  m_lo(3),  m_hi(3)
    integer          ::  c_lo(3),  c_hi(3)
    integer          :: IR_lo(3), IR_hi(3)
    double precision :: uold(uo_lo(1):uo_hi(1),uo_lo(2):uo_hi(2),uo_lo(3):uo_hi(3),NVAR)
    double precision :: unew(un_lo(1):un_hi(1),un_lo(2):un_hi(2),un_lo(3):un_hi(3),NVAR)
    double precision :: asrc(as_lo(1):as_hi(1),as_lo(2):as_hi(2),as_lo(3):as_hi(3),NVAR)
    integer          :: mask(m_lo(1):m_hi(1),m_lo(2):m_hi(2),m_lo(3):m_hi(3))
    double precision :: rY(rY_lo(1):rY_hi(1),rY_lo(2):rY_hi(2),rY_lo(3):rY_hi(3),nspec+1)
    double precision :: rE(rE_lo(1):rE_hi(1),rE_lo(2):rE_hi(2),rE_lo(3):rE_hi(3),1)
    double precision :: cost(c_lo(1):c_hi(1),c_lo(2):c_hi(2),c_lo(3):c_hi(3))
    double precision :: IR(IR_lo(1):IR_hi(1),IR_lo(2):IR_hi(2),IR_lo(3):IR_hi(3),nspec+1)
    double precision :: time, dt_react
    integer          :: do_update

    integer          :: i, j, k
    double precision :: rho_e_K_new, rhoE_old, rhoE_new, rho_new, mom_new(3)


    do k = lo(3), hi(3)
       do j = lo(2), hi(2)
          do i = lo(1), hi(1)

             if (mask(i,j,k) .eq. 1) then

                rhoE_old    = uold(i,j,k,UEDEN)
                rho_new     = sum(rY(i,j,k,1:nspec))
                mom_new     = uold(i,j,k,UMX:UMZ) + dt_react*asrc(i,j,k,UMX:UMZ)
                rhoe_new    = rE(i,j,k,1) 
                rho_e_K_new = HALF * sum(mom_new**2) / rho_new
                rhoE_new    = rhoe_new + rho_e_K_new

                if (do_update .eq. 1) then

                   unew(i,j,k,URHO)            = rho_new
                   unew(i,j,k,UMX:UMZ)         = mom_new
                   unew(i,j,k,UEINT)           = rhoe_new
                   unew(i,j,k,UEDEN)           = rhoE_new
                   unew(i,j,k,UTEMP)           = rY(i,j,k,nspec+1)
                   unew(i,j,k,UFS:UFS+nspec-1) = rY(i,j,k,1:nspec)

                endif

                ! Add drhoY/dt to reactions MultiFab, but be
                ! careful because the reactions and state MFs may
                ! not have the same number of ghost cells.
                ! Also, be careful because the container for uold might be the same as that for unew
                if ( i .ge. IR_lo(1) .and. i .le. IR_hi(1) .and. &
                     j .ge. IR_lo(2) .and. j .le. IR_hi(2) .and. &
                     k .ge. IR_lo(3) .and. k .le. IR_hi(3) ) then

                   IR(i,j,k,1:nspec) = (rY(i,j,k,1:nspec) - uold(i,j,k,UFS:UFS+nspec-1)) / dt_react - asrc(i,j,k,UFS:UFS+nspec-1)
                   IR(i,j,k,nspec+1) = (        rhoE_new          -         rhoE_old        ) / dt_react - asrc(i,j,k,UEDEN          )
                endif

             end if
          end do
       enddo
    enddo

  end subroutine pc_postreact_state
#endif

end module reactions_module
