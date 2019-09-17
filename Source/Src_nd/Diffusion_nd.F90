module diffusion_module

  implicit none

  public

contains

  subroutine pc_diffextrap(lo, hi, vlo, vhi, dif, d_lo, d_hi, nc) &
       bind(C, name="pc_diffextrap")

    ! this routine extrapolates the diffusion term into the
    ! ghostcells

    implicit none

    integer :: lo(3), hi(3)
    integer :: vlo(3), vhi(3)
    integer :: d_lo(3), d_hi(3)
    integer :: nc
    double precision :: dif(d_lo(1):d_hi(1),d_lo(2):d_hi(2),d_lo(3):d_hi(3),nc)

    ! Local variables

    integer :: i, j, k, n

    do n=1,nc
       ! left side
       if (lo(1) .eq. vlo(1)) then
          do i = d_lo(1), lo(1)-1
             do k = lo(3), hi(3)
                do j = lo(2), hi(2)
                   dif(i,j,k,n) = dif(lo(1),j,k,n)
                end do
             end do
          end do
       endif

       ! right side
       if (hi(1) .eq. vhi(1)) then
          do i = hi(1)+1, d_hi(1)
             do k = lo(3), hi(3)
                do j = lo(2), hi(2)
                   dif(i,j,k,n) = dif(hi(1),j,k,n)
                end do
             end do
          end do
       endif

       ! bottom side
       if (lo(2) .eq. vlo(2)) then
          do j = d_lo(2), lo(2)-1
             do k = lo(3), hi(3)
                do i = lo(1), hi(1)
                   dif(i,j,k,n) = dif(i,lo(2),k,n)
                end do
             end do
          end do
       endif

       ! top side
       if (hi(2) .eq. vhi(2)) then
          do j = hi(2)+1, d_hi(2)
             do k = lo(3), hi(3)
                do i = lo(1), hi(1)
                   dif(i,j,k,n) = dif(i,hi(2),k,n)
                end do
             end do
          end do
       endif

       ! down side
       if (lo(3) .eq. vlo(3)) then
          do k = d_lo(3), lo(3)-1
             do j = lo(2), hi(2)
                do i = lo(1), hi(1)
                   dif(i,j,k,n) = dif(i,j,lo(3),n)
                end do
             end do
          end do
       endif

       ! up side
       if (hi(3) .eq. vhi(3)) then
          do k = hi(3)+1, d_hi(3)
             do j = lo(2), hi(2)
                do i = lo(1), hi(1)
                   dif(i,j,k,n) = dif(i,j,hi(3),n)
                end do
             end do
          end do
       endif

       ! k-edges
       if (lo(1) .eq. vlo(1) .and. lo(2) .eq. vlo(2)) then
          do j = d_lo(2), lo(2)-1
             do i = d_lo(1), lo(1)-1
                do k = lo(3), hi(3)
                   dif(i,j,k,n) = dif(lo(1),lo(2),k,n)
                end do
             end do
          end do
       endif

       if (lo(1) .eq. vlo(1) .and. hi(2) .eq. vhi(2)) then
          do j = hi(2)+1, d_hi(2)
             do i = d_lo(1), d_lo(1)-1
                do k = lo(3), hi(3)
                   dif(i,j,k,n) = dif(lo(1),hi(2),k,n)
                end do
             end do
          end do
       endif

       if (hi(1) .eq. vhi(1) .and. lo(2) .eq. vlo(2)) then
          do j = d_lo(2), lo(2)-1
             do i = hi(1)+1, d_hi(1)
                do k = lo(3), hi(3)
                   dif(i,j,k,n) = dif(hi(1),lo(2),k,n)
                end do
             end do
          end do
       endif

       if (hi(1) .eq. vhi(1) .and. hi(2) .eq. vhi(2)) then
          do j = hi(2)+1, d_hi(2)
             do i = hi(1)+1, d_hi(1)
                do k = lo(3), hi(3)
                   dif(i,j,k,n) = dif(hi(1),hi(2),k,n)
                end do
             end do
          end do
       endif

       ! j-edges
       if (lo(1) .eq. vlo(1) .and. lo(3) .eq. vlo(3)) then
          do k = d_lo(3), lo(3)-1
             do i = d_lo(1), lo(1)-1
                do j = lo(2), hi(2)
                   dif(i,j,k,n) = dif(lo(1),j,lo(3),n)
                end do
             end do
          end do
       endif

       if (lo(1) .eq. vlo(1) .and. hi(3) .eq. vhi(3)) then
          do k = hi(3)+1, d_hi(3)
             do i = d_lo(1), lo(1)-1
                do j = lo(2), hi(2)
                   dif(i,j,k,n) = dif(lo(1),j,hi(3),n)
                end do
             end do
          end do
       endif

       if (hi(1) .eq. vhi(1) .and. lo(3) .eq. vlo(3)) then
          do k = d_lo(3), lo(3)-1
             do i = hi(1)+1, d_hi(1)
                do j = lo(2), hi(2)
                   dif(i,j,k,n) = dif(hi(1),j,lo(3),n)
                end do
             end do
          end do
       endif

       if (hi(1) .eq. vhi(1) .and. hi(3) .eq. vhi(3)) then
          do k = hi(3)+1, d_hi(3)
             do i = hi(1)+1, d_hi(1)
                do j = lo(2), hi(2)
                   dif(i,j,k,n) = dif(hi(1),j,hi(3),n)
                end do
             end do
          end do
       endif

       ! i-edges
       if (lo(2) .eq. vlo(2) .and. lo(3) .eq. vlo(3)) then
          do k = d_lo(3), lo(3)-1
             do j = d_lo(2), lo(2)-1
                do i = lo(1), hi(1)
                   dif(i,j,k,n) = dif(i,lo(2),lo(3),n)
                end do
             end do
          end do
       endif

       if (lo(2) .eq. vlo(2) .and. hi(3) .eq. vhi(3)) then
          do k = hi(3)+1, d_hi(3)
             do j = d_lo(2), lo(2)-1
                do i = lo(1), hi(1)
                   dif(i,j,k,n) = dif(i,lo(2),hi(3),n)
                end do
             end do
          end do
       endif

       if (hi(2) .eq. vhi(2) .and. lo(3) .eq. vlo(3)) then
          do k = d_lo(3), lo(3)-1
             do j = hi(2)+1, d_hi(2)
                do i = lo(1), hi(1)
                   dif(i,j,k,n) = dif(i,hi(2),lo(3),n)
                end do
             end do
          end do
       endif

       if (hi(2) .eq. vhi(2) .and. hi(3) .eq. vhi(3)) then
          do k = hi(3)+1, d_hi(3)
             do j = hi(2)+1, d_hi(2)
                do i = lo(1), hi(1)
                   dif(i,j,k,n) = dif(i,hi(2),hi(3),n)
                end do
             end do
          end do
       endif

       ! corners
       if (lo(1) .eq. vlo(1) .and. lo(2) .eq. vlo(2) .and. lo(3) .eq. vlo(3)) then
          do k = d_lo(3), lo(3)-1
             do j = d_lo(2), lo(2)-1
                do i = d_lo(1), lo(1)-1
                   dif(i,j,k,n) = dif(lo(1),lo(2),lo(3),n)
                end do
             end do
          end do
       endif

       if (lo(1) .eq. vlo(1) .and. hi(2) .eq. vhi(2) .and. lo(3) .eq. vlo(3)) then
          do k = d_lo(3), lo(3)-1
             do j = hi(2)+1, d_hi(2)
                do i = d_lo(1), lo(1)-1
                   dif(i,j,k,n) = dif(lo(1),hi(2),lo(3),n)
                end do
             end do
          end do
       endif

       if (hi(1) .eq. vhi(1) .and. lo(2) .eq. vlo(2) .and. lo(3) .eq. vlo(3)) then
          do k = d_lo(3), lo(3)-1
             do j = d_lo(2), lo(2)-1
                do i = hi(1)+1, d_hi(1)
                   dif(i,j,k,n) = dif(hi(1),lo(2),lo(3),n)
                end do
             end do
          end do
       endif

       if (hi(1) .eq. vhi(1) .and. hi(2) .eq. vhi(2) .and. lo(3) .eq. vlo(3)) then
          do k = d_lo(3), lo(3)-1
             do j = hi(2)+1, d_hi(2)
                do i = hi(1)+1, d_hi(1)
                   dif(i,j,k,n) = dif(hi(1),hi(2),lo(3),n)
                end do
             end do
          end do
       endif

       if (lo(1) .eq. vlo(1) .and. lo(2) .eq. vlo(2) .and. hi(3) .eq. vhi(3)) then
          do k = hi(3)+1, d_hi(3)
             do j = d_lo(2), lo(2)-1
                do i = d_lo(1), lo(1)-1
                   dif(i,j,k,n) = dif(lo(1),lo(2),hi(3),n)
                end do
             end do
          end do
       endif

       if (lo(1) .eq. vlo(1) .and. hi(2) .eq. vhi(2) .and. hi(3) .eq. vhi(3)) then
          do k = d_hi(3), hi(3)-1
             do j = hi(2)+1, d_hi(2)
                do i = d_lo(1), lo(1)-1
                   dif(i,j,k,n) = dif(lo(1),hi(2),hi(3),n)
                end do
             end do
          end do
       endif

       if (hi(1) .eq. vhi(1) .and. lo(2) .eq. vlo(2) .and. hi(3) .eq. vhi(3)) then
          do k = d_hi(3), hi(3)-1
             do j = d_lo(2), lo(2)-1
                do i = hi(1)+1, d_hi(1)
                   dif(i,j,k,n) = dif(hi(1),lo(2),hi(3),n)
                end do
             end do
          end do
       endif

       if (hi(1) .eq. vhi(1) .and. hi(2) .eq. vhi(2) .and. hi(3) .eq. vhi(3)) then
          do k = d_hi(3), hi(3)-1
             do j = hi(2)+1, d_hi(2)
                do i = hi(1)+1, d_hi(1)
                   dif(i,j,k,n) = dif(hi(1),hi(2),hi(3),n)
                end do
             end do
          end do
       endif

    enddo

  end subroutine pc_diffextrap

  subroutine pc_move_transport_coeffs_to_ec(&
#ifdef PELEC_USE_ACC
       gpustream,&
#endif
       lo,hi,dlo,dhi, &
       cfab,c_lo,c_hi, &
       efab,e_lo,e_hi, dir, nc, do_harmonic) &
       bind(C, name="pc_move_transport_coeffs_to_ec")

    use prob_params_module, only : physbc_lo, physbc_hi

    implicit none

#ifdef PELEC_USE_ACC
    integer         , intent(in   ) :: gpustream
#endif
    integer         , intent(in   ) :: lo(3), hi(3)
    integer         , intent(in   ) :: dlo(3), dhi(3)
    integer         , intent(in   ) :: c_lo(3), c_hi(3)
    integer         , intent(in   ) :: e_lo(3), e_hi(3)
    integer         , intent(in   ) :: dir, nc, do_harmonic
    double precision, intent(in   ) :: cfab(c_lo(1):c_hi(1),c_lo(2):c_hi(2),c_lo(3):c_hi(3),nc)
    double precision, intent(inout) :: efab(e_lo(1):e_hi(1),e_lo(2):e_hi(2),e_lo(3):e_hi(3),nc)

    ! local variables
    integer          :: i, j, k, n, lo1, lo2, lo3, hi1, hi2, hi3

    lo1 = lo(1)
    lo2 = lo(2)
    lo3 = lo(3)
    hi1 = hi(1)
    hi2 = hi(2)
    hi3 = hi(3)

    if (do_harmonic .eq. 0) then
       if (dir .EQ. 0) then
#ifdef PELEC_USE_ACC
          !$acc parallel loop gang vector collapse(4) default(present) async(gpustream)
#endif
          do n = 1,nc
             do k = lo3, hi3
                do j = lo2, hi2
                   do i = lo1, hi1+1
                      efab(i,j,k,n) = 0.5d0*(cfab(i,j,k,n) + cfab(i-1,j,k,n))
                   end do
                end do
             end do
          end do
#ifdef PELEC_USE_ACC
          !$acc end parallel
#endif
       else if (dir .EQ. 1) then
#ifdef PELEC_USE_ACC
          !$acc parallel loop gang vector collapse(4) default(present) async(gpustream)
#endif
          do n = 1,nc
             do k = lo3, hi3
                do j = lo2, hi2+1
                   do i = lo1, hi1
                      efab(i,j,k,n) = 0.5d0*(cfab(i,j,k,n) + cfab(i,j-1,k,n))
                   end do
                end do
             end do
          end do
#ifdef PELEC_USE_ACC
          !$acc end parallel
#endif
       else if (dir .EQ. 2) then
#ifdef PELEC_USE_ACC
          !$acc parallel loop gang vector collapse(4) default(present) async(gpustream)
#endif
          do n = 1,nc
             do k = lo3, hi3+1
                do j = lo2, hi2
                   do i = lo(1), hi1
                      efab(i,j,k,n) = 0.5d0*(cfab(i,j,k,n) + cfab(i,j,k-1,n))
                   end do
                end do
             end do
          end do
#ifdef PELEC_USE_ACC
          !$acc end parallel
#endif
       end if
    else
       if (dir .EQ. 0) then
#ifdef PELEC_USE_ACC
          !$acc parallel loop gang vector collapse(4) default(present) async(gpustream)
#endif
          do n = 1,nc
             do k = lo3, hi3
                do j = lo2, hi2
                   do i = lo1, hi1+1
                      efab(i,j,k,n) = merge(2*(cfab(i,j,k,n)*cfab(i-1,j,k,n)) &
                                            /(cfab(i,j,k,n)+cfab(i-1,j,k,n)), &
                                            0.d0, &
                                            (cfab(i,j,k,n)*cfab(i-1,j,k,n)) .gt. 0.d0)
                   end do
                end do
             end do
          end do
#ifdef PELEC_USE_ACC
          !$acc end parallel
#endif
       else if (dir .EQ. 1) then
#ifdef PELEC_USE_ACC
          !$acc parallel loop gang vector collapse(4) default(present) async(gpustream)
#endif
          do n = 1,nc
             do k = lo3, hi3
                do j = lo2, hi2+1
                   do i = lo1, hi1
                      efab(i,j,k,n) = merge(2*(cfab(i,j,k,n)*cfab(i,j-1,k,n)) &
                                            /(cfab(i,j,k,n)+cfab(i,j-1,k,n)), &
                                            0.d0, &
                                            (cfab(i,j,k,n)*cfab(i,j-1,k,n)) .gt. 0.d0)
                   end do
                end do
             end do
          end do
#ifdef PELEC_USE_ACC
          !$acc end parallel
#endif
       else if (dir .EQ. 2) then
#ifdef PELEC_USE_ACC
          !$acc parallel loop gang vector collapse(4) default(present) async(gpustream)
#endif
          do n = 1,nc
             do k = lo3, hi3+1
                do j = lo2, hi2
                   do i = lo1, hi1
                      efab(i,j,k,n) = merge(2*(cfab(i,j,k,n)*cfab(i,j,k-1,n)) &
                                            /(cfab(i,j,k,n)+cfab(i,j,k-1,n)), &
                                            0.d0, &
                                            (cfab(i,j,k,n)*cfab(i,j,k-1,n)) .gt. 0.d0)
                   end do
                end do
             end do
          end do
#ifdef PELEC_USE_ACC
          !$acc end parallel
#endif
       end if
    end if

  end subroutine pc_move_transport_coeffs_to_ec

end module diffusion_module
