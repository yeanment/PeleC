! problem-specific Fortran derive routines go here
module problem_derive_module

  implicit none

  public

contains

  subroutine pc_dercp(cp,m_lo,m_hi,nm, &
                      dat,d_lo,d_hi,nc,lo,hi,domlo, &
                      domhi,delta,xlo,time,dt,bc,level,grid_no) &
                      bind(C, name="pc_dercp")
    !
    ! This routine will calculate cp
    !

    use meth_params_module, only: URHO, UEINT, UFS, UMX, UMY, UMZ, UEDEN, UTEMP
    use network, only: nspecies
    use eos_type_module
    use eos_module

    implicit none

    integer          :: lo(3), hi(3)
    integer          :: m_lo(3), m_hi(3), nm
    integer          :: d_lo(3), d_hi(3), nc
    integer          :: domlo(3), domhi(3)
    integer          :: bc(3,2,nc)
    double precision :: delta(3), xlo(3), time, dt
    double precision :: cp(m_lo(1):m_hi(1),m_lo(2):m_hi(2),m_lo(3):m_hi(3),nm)
    double precision :: dat(d_lo(1):d_hi(1),d_lo(2):d_hi(2),d_lo(3):d_hi(3),nc)
    integer          :: level, grid_no

    integer          :: i, j, k
    type(eos_t) :: eos_state

    call build(eos_state)

    do k = lo(3), hi(3)
       do j = lo(2), hi(2)
          do i = lo(1), hi(1)

             eos_state % rho = dat(i,j,k,URHO)
             eos_state % T = dat(i,j,k,UTEMP)
             eos_state % massfrac(1:nspecies) = dat(i,j,k,UFS:UFS+nspecies-1) / dat(i,j,k,URHO)
             call eos_rt(eos_state)
             call eos_cp(eos_state)

             cp(i,j,k,1) = eos_state % cp

          end do
       end do
    end do

    call destroy(eos_state)

  end subroutine pc_dercp

  subroutine pc_dercv(cv,m_lo,m_hi,nm, &
                      dat,d_lo,d_hi,nc,lo,hi,domlo, &
                      domhi,delta,xlo,time,dt,bc,level,grid_no) &
                      bind(C, name="pc_dercv")
    !
    ! This routine will calculate cv
    !

    use meth_params_module, only: URHO, UEINT, UFS, UMX, UMY, UMZ, UEDEN, UTEMP
    use network, only: nspecies
    use eos_type_module
    use eos_module

    implicit none

    integer          :: lo(3), hi(3)
    integer          :: m_lo(3), m_hi(3), nm
    integer          :: d_lo(3), d_hi(3), nc
    integer          :: domlo(3), domhi(3)
    integer          :: bc(3,2,nc)
    double precision :: delta(3), xlo(3), time, dt
    double precision :: cv(m_lo(1):m_hi(1),m_lo(2):m_hi(2),m_lo(3):m_hi(3),nm)
    double precision :: dat(d_lo(1):d_hi(1),d_lo(2):d_hi(2),d_lo(3):d_hi(3),nc)
    integer          :: level, grid_no

    integer          :: i, j, k
    type(eos_t) :: eos_state

    call build(eos_state)

    do k = lo(3), hi(3)
       do j = lo(2), hi(2)
          do i = lo(1), hi(1)

             eos_state % rho = dat(i,j,k,URHO)
             eos_state % T = dat(i,j,k,UTEMP)
             eos_state % massfrac(1:nspecies) = dat(i,j,k,UFS:UFS+nspecies-1) / dat(i,j,k,URHO)
             call eos_rt(eos_state)
             call eos_cv(eos_state)

             cv(i,j,k,1) = eos_state % cv

          end do
       end do
    end do

    call destroy(eos_state)

  end subroutine pc_dercv

  subroutine pc_derwbar(wbar,m_lo,m_hi,nm, &
                      dat,d_lo,d_hi,nc,lo,hi,domlo, &
                      domhi,delta,xlo,time,dt,bc,level,grid_no) &
                      bind(C, name="pc_derwbar")
    !
    ! This routine will calculate wbar
    !

    use meth_params_module, only: URHO, UEINT, UFS, UMX, UMY, UMZ, UEDEN, UTEMP
    use network, only: nspecies
    use eos_type_module
    use eos_module

    implicit none

    integer          :: lo(3), hi(3)
    integer          :: m_lo(3), m_hi(3), nm
    integer          :: d_lo(3), d_hi(3), nc
    integer          :: domlo(3), domhi(3)
    integer          :: bc(3,2,nc)
    double precision :: delta(3), xlo(3), time, dt
    double precision :: wbar(m_lo(1):m_hi(1),m_lo(2):m_hi(2),m_lo(3):m_hi(3),nm)
    double precision :: dat(d_lo(1):d_hi(1),d_lo(2):d_hi(2),d_lo(3):d_hi(3),nc)
    integer          :: level, grid_no

    integer          :: i, j, k
    type(eos_t) :: eos_state

    call build(eos_state)

    do k = lo(3), hi(3)
       do j = lo(2), hi(2)
          do i = lo(1), hi(1)

             eos_state % massfrac(1:nspecies) = dat(i,j,k,UFS:UFS+nspecies-1) / dat(i,j,k,URHO)
             call eos_wb(eos_state)

             wbar(i,j,k,1) = eos_state % wbar

          end do
       end do
    end do

    call destroy(eos_state)

  end subroutine pc_derwbar

  subroutine pc_dermu(mu,m_lo,m_hi,nm, &
                      dat,d_lo,d_hi,nc,lo,hi,domlo, &
                      domhi,delta,xlo,time,dt,bc,level,grid_no) &
                      bind(C, name="pc_dermu")
    !
    ! This routine will calculate the dynamic viscosity mu
    !

    use meth_params_module, only: URHO, UEINT, UFS, UMX, UMY, UMZ, UEDEN, UTEMP
    use network, only: nspecies
    use eos_type_module
    use eos_module
    use transport_module

    implicit none

    integer          :: lo(3), hi(3)
    integer          :: m_lo(3), m_hi(3), nm
    integer          :: d_lo(3), d_hi(3), nc
    integer          :: domlo(3), domhi(3)
    integer          :: bc(3,2,nc)
    double precision :: delta(3), xlo(3), time, dt
    double precision :: mu(m_lo(1):m_hi(1),m_lo(2):m_hi(2),m_lo(3):m_hi(3),nm)
    double precision :: dat(d_lo(1):d_hi(1),d_lo(2):d_hi(2),d_lo(3):d_hi(3),nc)
    integer          :: level, grid_no

    integer          :: i, j, k, n
    type(eos_t) :: eos_state

    integer :: np
    type (wtr_t) :: which_trans
    type (trv_t) :: coeff
    double precision,allocatable :: rho_inv(:), D(:)

    np = hi(1)-lo(1)+1
    call build(coeff,np)
    allocate(rho_inv(np))
    which_trans % wtr_get_mu = .true.

    do k = lo(3), hi(3)
       do j = lo(2), hi(2)

          rho_inv(:) = 1.0d0 / dat(lo(1):hi(1),j,k,URHO)
          do n=1,nspecies
             do i=1,np
                coeff%eos_state(i)%massfrac(n) = dat(lo(1)+i-1,j,k,UFS+n-1) * rho_inv(i)
             end do
          end do
          coeff%eos_state(1:np)%T   = dat(lo(1):hi(1),j,k,UTEMP)
          coeff%eos_state(1:np)%rho = dat(lo(1):hi(1),j,k,URHO) 

          call transport(which_trans, coeff)

          do i=1,np
             mu(lo(1)+i-1,j,k,1) = coeff%mu(i)
          end do
          
       end do
    end do

    deallocate(rho_inv)
    call destroy(coeff)

  end subroutine pc_dermu

  subroutine pc_derlam(lam,l_lo,l_hi,nl, &
                      dat,d_lo,d_hi,nc,lo,hi,domlo, &
                      domhi,delta,xlo,time,dt,bc,level,grid_no) &
                      bind(C, name="pc_derlam")
    !
    ! This routine will calculate the dynamic viscosity mu
    !

    use meth_params_module, only: URHO, UEINT, UFS, UMX, UMY, UMZ, UEDEN, UTEMP
    use network, only: nspecies
    use eos_type_module
    use eos_module
    use transport_module

    implicit none

    integer          :: lo(3), hi(3)
    integer          :: l_lo(3), l_hi(3), nl
    integer          :: d_lo(3), d_hi(3), nc
    integer          :: domlo(3), domhi(3)
    integer          :: bc(3,2,nc)
    double precision :: delta(3), xlo(3), time, dt
    double precision :: lam(l_lo(1):l_hi(1),l_lo(2):l_hi(2),l_lo(3):l_hi(3),nl)
    double precision :: dat(d_lo(1):d_hi(1),d_lo(2):d_hi(2),d_lo(3):d_hi(3),nc)
    integer          :: level, grid_no

    integer          :: i, j, k, n
    type(eos_t) :: eos_state

    integer :: np
    type (wtr_t) :: which_trans
    type (trv_t) :: coeff
    double precision,allocatable :: rho_inv(:), D(:)

    np = hi(1)-lo(1)+1
    call build(coeff,np)
    allocate(rho_inv(np))
    which_trans % wtr_get_lam = .true.

    do k = lo(3), hi(3)
       do j = lo(2), hi(2)

          rho_inv(:) = 1.0d0 / dat(lo(1):hi(1),j,k,URHO)
          do n=1,nspecies
             do i=1,np
                coeff%eos_state(i)%massfrac(n) = dat(lo(1)+i-1,j,k,UFS+n-1) * rho_inv(i)
             end do
          end do
          coeff%eos_state(1:np)%T   = dat(lo(1):hi(1),j,k,UTEMP)
          coeff%eos_state(1:np)%rho = dat(lo(1):hi(1),j,k,URHO) 

          call transport(which_trans, coeff)

          do i=1,np
             lam(lo(1)+i-1,j,k,1) = coeff%lam(i)
          end do
          
       end do
    end do

    deallocate(rho_inv)
    call destroy(coeff)

  end subroutine pc_derlam

end module problem_derive_module
