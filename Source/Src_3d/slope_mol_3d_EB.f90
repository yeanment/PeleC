module slope_module
  
  implicit none

  private

  public slopex, slopey, slopez

contains

      subroutine slopex(q,flatn,qd_lo1,qd_lo2,qd_lo3,qd_hi1,qd_hi2,qd_hi3, &
                        dqx,qt_lo1,qt_lo2,qt_lo3,qt_hi1,qt_hi2,qt_hi3, &
                        ilo1,ilo2,ilo3,ihi1,ihi2,ihi3,qvar,nqaux, &
                        domlo1,domlo2,domlo3,domhi1,domhi2,domhi3, &
                        qaux,qa_lo1,qa_lo2,qa_lo3,qa_hi1,qa_hi2,qa_hi3, &
                        flag,fglo1,fglo2,fglo3,fghi1,fghi2,fghi3)

      !use amrex_fort_module, only : amrex_real
      !use amrex_mempool_module, only : bl_allocate, bl_deallocate
      !use meth_params_module
      !use amrex_constants_module
      !use actual_network, only : nspec
      !use prob_params_module, only : physbc_lo, physbc_hi, Inflow

      !$acc routine(slopex) gang
      !$acc routine(get_neighbor_cells_int) seq
      !$acc routine(is_covered_cell) seq

      USE amrex_ebcellflag_module, ONLY: get_neighbor_cells_int, is_covered_cell
      USE meth_params_module, ONLY: qpres, qc, qrho, qu, qv, qw, qfs

      implicit none

      integer, parameter  :: nspec=9
      integer, intent(in) :: qd_lo1, qd_lo2, qd_lo3, qd_hi1, qd_hi2, qd_hi3
      integer, intent(in) :: qt_lo1, qt_lo2, qt_lo3, qt_hi1, qt_hi2, qt_hi3
      integer, intent(in) :: qa_lo1, qa_lo2, qa_lo3, qa_hi1, qa_hi2, qa_hi3
      integer, intent(in) :: ilo1, ilo2, ihi1, ihi2, ilo3, ihi3, qvar, nqaux
      integer, intent(in) :: domlo1, domlo2, domlo3, domhi1, domhi2, domhi3
      integer, intent(in) :: fglo1,fglo2,fglo3,fghi1,fghi2,fghi3
      integer, intent(in) :: flag(fglo1:fghi1,fglo2:fghi2,fglo3:fghi3)
      double precision, intent(in) :: q(qd_lo1:qd_hi1,qd_lo2:qd_hi2,qd_lo3:qd_hi3,qvar)
      double precision, intent(in) :: qaux(qa_lo1:qa_hi1,qa_lo2:qa_hi2,qa_lo3:qa_hi3,nqaux)
      double precision, intent(in) :: flatn(qd_lo1:qd_hi1,qd_lo2:qd_hi2,qd_lo3:qd_hi3)
      double precision, intent(out) :: dqx(qt_lo1:qt_hi1,qt_lo2:qt_hi2,qt_lo3:qt_hi3,qvar)

      integer i, j, k, n
      double precision :: dlft(1:qvar), drgt(1:qvar)
      double precision :: dlim, dcen
      integer :: nbr(-1:1,-1:1,-1:1)
      logical :: flagArrayL, flagArrayR

      !$acc loop gang vector collapse(3) private(nbr,dlft,drgt,flagarrayl,flagarrayr,n,dcen,dlim)
      do k = ilo3, ihi3
         do j = ilo2, ihi2
            do i = ilo1, ihi1
               call get_neighbor_cells_int(flag(i,j,k), nbr)
               flagArrayL = nbr(-1,0,0).eq.1 .and. .not. is_covered_cell(flag(i,j,k))
               flagArrayR = nbr(+1,0,0).eq.1 .and. .not. is_covered_cell(flag(i,j,k))
               dlft(1) = merge(0.5d0*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/qaux(i,j,k,QC) &
                    - 0.5d0*q(i,j,k,QRHO)*(q(i,j,k,QU)-q(i-1,j,k,QU)), 0.d0, flagArrayL)
               dlft(2) = merge(0.5d0*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/qaux(i,j,k,QC) &
                    + 0.5d0*q(i,j,k,QRHO)*(q(i,j,k,QU)-q(i-1,j,k,QU)), 0.d0, flagArrayL)
               dlft(3) = merge(q(i,j,k,QV)-q(i-1,j,k,QV), 0.d0, flagArrayL)
               dlft(4) = merge(q(i,j,k,QW)-q(i-1,j,k,QW), 0.d0, flagArrayL)
               drgt(1) = merge(0.5d0*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/qaux(i,j,k,QC) &
                    - 0.5d0*q(i,j,k,QRHO)*(q(i+1,j,k,QU)-q(i,j,k,QU)), 0.d0, flagArrayR)
               drgt(2) = merge(0.5d0*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/qaux(i,j,k,QC) &
                    + 0.5d0*q(i,j,k,QRHO)*(q(i+1,j,k,QU)-q(i,j,k,QU)), 0.d0, flagArrayR)
               drgt(3) = merge(q(i+1,j,k,QV)-q(i,j,k,QV), 0.d0, flagArrayR)
               drgt(4) = merge(q(i+1,j,k,QW)-q(i,j,k,QW), 0.d0, flagArrayR)
               do n = 1, nspec
                  dlft(4+n) = merge(q(i,j,k,QRHO)*q(i,j,k,QFS+n-1)-q(i-1,j,k,QRHO)*q(i-1,j,k,QFS+n-1) &
                               - q(i,j,k,QFS+n-1)*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/qaux(i,j,k,QC)**2, 0.d0, flagArrayL)
                  drgt(4+n) = merge(q(i+1,j,k,QRHO)*q(i+1,j,k,QFS+n-1)-q(i,j,k,QRHO)*q(i,j,k,QFS+n-1) &
                               - q(i,j,k,QFS+n-1)*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/qaux(i,j,k,QC)**2, 0.d0, flagArrayR)
               enddo
               do n = 1, qvar
                  dcen = 0.5d0 * (dlft(n)+drgt(n))
                  dlim = merge(2.d0 * min(abs(dlft(n)), abs(drgt(n))), 0.d0, dlft(n)*drgt(n) .ge. 0.d0)
                  dqx(i,j,k,n) = sign(1.d0, dcen)*min(dlim, abs(dcen))
               enddo
            enddo
         enddo
      enddo

    end subroutine slopex

    subroutine slopey(q,flatn,qd_lo1,qd_lo2,qd_lo3,qd_hi1,qd_hi2,qd_hi3, &
         dqy,qt_lo1,qt_lo2,qt_lo3,qt_hi1,qt_hi2,qt_hi3, &
         ilo1,ilo2,ilo3,ihi1,ihi2,ihi3,qvar,nqaux, &
         domlo1,domlo2,domlo3,domhi1,domhi2,domhi3, &
         qaux,qa_lo1,qa_lo2,qa_lo3,qa_hi1,qa_hi2,qa_hi3, &
         flag,fglo1,fglo2,fglo3,fghi1,fghi2,fghi3)

      !use amrex_fort_module, only : amrex_real
      !use amrex_mempool_module, only : bl_allocate, bl_deallocate
      !use meth_params_module
      !use amrex_constants_module
      !use actual_network, only : nspec
      !use prob_params_module, only : physbc_lo, physbc_hi, Inflow

      !$acc routine(slopey) gang
      !$acc routine(get_neighbor_cells_int) seq
      !$acc routine(is_covered_cell) seq

      USE amrex_ebcellflag_module, ONLY: get_neighbor_cells_int, is_covered_cell
      USE meth_params_module, ONLY: qpres, qc, qrho, qu, qv, qw, qfs

      implicit none

      integer, parameter  :: nspec=9
      integer, intent(in) :: qd_lo1, qd_lo2, qd_lo3, qd_hi1, qd_hi2, qd_hi3
      integer, intent(in) :: qt_lo1, qt_lo2, qt_lo3, qt_hi1, qt_hi2, qt_hi3
      integer, intent(in) :: qa_lo1, qa_lo2, qa_lo3, qa_hi1, qa_hi2, qa_hi3
      integer, intent(in) :: ilo1, ilo2, ihi1, ihi2, ilo3, ihi3, qvar, nqaux
      integer, intent(in) :: domlo1, domlo2, domlo3, domhi1, domhi2, domhi3
      integer, intent(in) :: fglo1,fglo2,fglo3,fghi1,fghi2,fghi3
      integer, intent(in) :: flag(fglo1:fghi1,fglo2:fghi2,fglo3:fghi3)
      double precision, intent(in) :: q(qd_lo1:qd_hi1,qd_lo2:qd_hi2,qd_lo3:qd_hi3,qvar)
      double precision, intent(in) :: qaux(qa_lo1:qa_hi1,qa_lo2:qa_hi2,qa_lo3:qa_hi3,nqaux)
      double precision, intent(in) :: flatn(qd_lo1:qd_hi1,qd_lo2:qd_hi2,qd_lo3:qd_hi3)
      double precision, intent(out) :: dqy(qt_lo1:qt_hi1,qt_lo2:qt_hi2,qt_lo3:qt_hi3,qvar)

      integer i, j, k, n
      double precision :: dlft(1:qvar), drgt(1:qvar)
      double precision :: dlim, dcen
      integer :: nbr(-1:1,-1:1,-1:1)
      logical :: flagArrayL, flagArrayR

      !$acc loop gang vector collapse(3) private(nbr,dlft,drgt,flagarrayl,flagarrayr,n,dcen,dlim)
      do k = ilo3, ihi3
         do j = ilo2, ihi2
            do i = ilo1, ihi1
               call get_neighbor_cells_int(flag(i,j,k), nbr)
               flagArrayL = nbr(0,-1,0).eq.1 .and. .not. is_covered_cell(flag(i,j,k))
               flagArrayR = nbr(0,+1,0).eq.1 .and. .not. is_covered_cell(flag(i,j,k))
               dlft(1) = merge(0.5d0*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/qaux(i,j,k,QC) &
                    - 0.5d0*q(i,j,k,QRHO)*(q(i,j,k,QV)-q(i,j-1,k,QV)), 0.d0, flagArrayL)
               dlft(2) = merge(0.5d0*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/qaux(i,j,k,QC) &
                    + 0.5d0*q(i,j,k,QRHO)*(q(i,j,k,QV)-q(i,j-1,k,QV)), 0.d0, flagArrayL)
               dlft(3) = merge(q(i,j,k,QU)-q(i,j-1,k,QU), 0.d0, flagArrayL)
               dlft(4) = merge(q(i,j,k,QW)-q(i,j-1,k,QW), 0.d0, flagArrayL)
               drgt(1) = merge(0.5d0*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/qaux(i,j,k,QC) &
                    - 0.5d0*q(i,j,k,QRHO)*(q(i,j+1,k,QV)-q(i,j,k,QV)), 0.d0, flagArrayR)
               drgt(2) = merge(0.5d0*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/qaux(i,j,k,QC) &
                    + 0.5d0*q(i,j,k,QRHO)*(q(i,j+1,k,QV)-q(i,j,k,QV)), 0.d0, flagArrayR)
               drgt(3) = merge(q(i,j+1,k,QU)-q(i,j,k,QU), 0.d0, flagArrayR)
               drgt(4) = merge(q(i,j+1,k,QW)-q(i,j,k,QW), 0.d0, flagArrayR)
               do n = 1, nspec
                  dlft(4+n) = merge(q(i,j,k,QRHO)*q(i,j,k,QFS+n-1)-q(i,j-1,k,QRHO)*q(i,j-1,k,QFS+n-1) &
                              - q(i,j,k,QFS+n-1)*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/qaux(i,j,k,QC)**2, 0.d0, flagArrayL)
                  drgt(4+n) = merge(q(i,j+1,k,QRHO)*q(i,j+1,k,QFS+n-1)-q(i,j,k,QRHO)*q(i,j,k,QFS+n-1) &
                              - q(i,j,k,QFS+n-1)*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/qaux(i,j,k,QC)**2, 0.d0, flagArrayR)
               enddo
               do n=1, qvar
                  dcen = 0.5d0 * (dlft(n)+drgt(n))
                  dlim = merge(2.d0 * min(abs(dlft(n)), abs(drgt(n))), 0.d0, dlft(n)*drgt(n) .ge. 0.d0)
                  dqy(i,j,k,n) = sign(1.d0, dcen)*min(dlim, abs(dcen))
               enddo
            enddo
         enddo
      enddo

    end subroutine slopey

    subroutine slopez(q,flatn,qd_lo1,qd_lo2,qd_lo3,qd_hi1,qd_hi2,qd_hi3, &
         dqz,qt_lo1,qt_lo2,qt_lo3,qt_hi1,qt_hi2,qt_hi3, &
         ilo1,ilo2,ilo3,ihi1,ihi2,ihi3,qvar,nqaux, &
         domlo1,domlo2,domlo3,domhi1,domhi2,domhi3, &
         qaux,qa_lo1,qa_lo2,qa_lo3,qa_hi1,qa_hi2,qa_hi3, &
         flag,fglo1,fglo2,fglo3,fghi1,fghi2,fghi3)

      !use amrex_fort_module, only : amrex_real
      !use amrex_mempool_module, only : bl_allocate, bl_deallocate
      !use meth_params_module
      !use amrex_constants_module
      !use actual_network, only : nspec
      !use prob_params_module, only : physbc_lo, physbc_hi, Inflow

      !$acc routine(slopez) gang
      !$acc routine(get_neighbor_cells_int) seq
      !$acc routine(is_covered_cell) seq

      USE amrex_ebcellflag_module, ONLY: get_neighbor_cells_int, is_covered_cell
      USE meth_params_module, ONLY: qpres, qc, qrho, qu, qv, qw, qfs

      implicit none

      integer, parameter  :: nspec=9
      integer, intent(in) :: qd_lo1, qd_lo2, qd_lo3, qd_hi1, qd_hi2, qd_hi3
      integer, intent(in) :: qt_lo1, qt_lo2, qt_lo3, qt_hi1, qt_hi2, qt_hi3
      integer, intent(in) :: qa_lo1, qa_lo2, qa_lo3, qa_hi1, qa_hi2, qa_hi3
      integer, intent(in) :: ilo1, ilo2, ihi1, ihi2, ilo3, ihi3, qvar, nqaux
      integer, intent(in) :: domlo1, domlo2, domlo3, domhi1, domhi2, domhi3
      integer, intent(in) :: fglo1,fglo2,fglo3,fghi1,fghi2,fghi3
      integer, intent(in) :: flag(fglo1:fghi1,fglo2:fghi2,fglo3:fghi3)
      double precision, intent(in) :: q(qd_lo1:qd_hi1,qd_lo2:qd_hi2,qd_lo3:qd_hi3,qvar)
      double precision, intent(in) :: qaux(qa_lo1:qa_hi1,qa_lo2:qa_hi2,qa_lo3:qa_hi3,nqaux)
      double precision, intent(in) :: flatn(qd_lo1:qd_hi1,qd_lo2:qd_hi2,qd_lo3:qd_hi3)
      double precision, intent(out) :: dqz(qt_lo1:qt_hi1,qt_lo2:qt_hi2,qt_lo3:qt_hi3,qvar)

      integer i, j, k, n
      double precision :: dlft(1:qvar), drgt(1:qvar)
      double precision :: dlim, dcen
      integer :: nbr(-1:1,-1:1,-1:1)
      logical :: flagArrayL, flagArrayR

      !$acc loop gang vector collapse(3) private(nbr,dlft,drgt,flagarrayl,flagarrayr,n,dcen,dlim)
      do k = ilo3, ihi3
         do j = ilo2, ihi2
            do i = ilo1, ihi1
               call get_neighbor_cells_int(flag(i,j,k), nbr)
               flagArrayL = nbr(0,0,-1).eq.1 .and. .not. is_covered_cell(flag(i,j,k))
               flagArrayR = nbr(0,0,+1).eq.1 .and. .not. is_covered_cell(flag(i,j,k))
               dlft(1) = merge(0.5d0*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/qaux(i,j,k,QC) &
                            - 0.5d0*q(i,j,k,QRHO)*(q(i,j,k,QW)-q(i,j,k-1,QW)), 0.d0, flagArrayL)
               dlft(2) = merge(0.5d0*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/qaux(i,j,k,QC) &
                    + 0.5d0*q(i,j,k,QRHO)*(q(i,j,k,QW)-q(i,j,k-1,QW)), 0.d0, flagArrayL)
               dlft(3) = merge(q(i,j,k,QU)-q(i,j,k-1,QU), 0.d0, flagArrayL)
               dlft(4) = merge(q(i,j,k,QV)-q(i,j,k-1,QV), 0.d0, flagArrayL)
               drgt(1) = merge(0.5d0*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/qaux(i,j,k,QC) &
                    - 0.5d0*q(i,j,k,QRHO)*(q(i,j,k+1,QW)-q(i,j,k,QW)), 0.d0, flagArrayR)
               drgt(2) = merge(0.5d0*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/qaux(i,j,k,QC) &
                    + 0.5d0*q(i,j,k,QRHO)*(q(i,j,k+1,QW)-q(i,j,k,QW)), 0.d0, flagArrayR)
               drgt(3) = merge(q(i,j,k+1,QU)-q(i,j,k,QU), 0.d0, flagArrayR)
               drgt(4) = merge(q(i,j,k+1,QV)-q(i,j,k,QV), 0.d0, flagArrayR)
               do n = 1, nspec
                  dlft(4+n) = merge(q(i,j,k,QRHO)*q(i,j,k,QFS+n-1)-q(i,j,k-1,QRHO)*q(i,j,k-1,QFS+n-1) &
                              - q(i,j,k,QFS+n-1)*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/qaux(i,j,k,QC)**2, 0.d0, flagArrayL)
                  drgt(4+n) = merge(q(i,j,k+1,QRHO)*q(i,j,k+1,QFS+n-1)-q(i,j,k,QRHO)*q(i,j,k,QFS+n-1) &
                              - q(i,j,k,QFS+n-1)*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/qaux(i,j,k,QC)**2, 0.d0, flagArrayR)
               enddo
               do n=1, qvar
                  dcen = 0.5d0 * (dlft(n)+drgt(n))
                  dlim = merge(2.d0 * min(abs(dlft(n)), abs(drgt(n))), 0.d0, dlft(n)*drgt(n) .ge. 0.d0)
                  dqz(i,j,k,n) = sign(1.d0, dcen)*min(dlim, abs(dcen))
               enddo
            enddo
         enddo
      enddo

    end subroutine slopez

end module slope_module
