module hyp_advection_module

  !use amrex_ebcellflag_module, only : get_neighbor_cells
  !use pelec_eb_stencil_types_module, only : eb_bndry_geom
  use riemann_util_module, only : riemann_md_singlepoint, riemann_md_vec
  use prob_params_module, only: dim

  implicit none
  private
  public pc_hyp_mol_flux
  contains

  !> Computes fluxes for hyperbolic conservative update.
  !> @brief
  !> Uses MOL formulation
  !! @param[inout] flux1  flux in X direction on X edges
  !> @param[in] q        (const)  input state, primitives
  !> @param[in] flatn    (const)  flattening parameter
  !> @param[in] src      (const)  source
  !> @param[in] nx       (const)  number of cells in X direction
  !> @param[in] ny       (const)  number of cells in Y direction
  !> @param[in] nz       (const)  number of cells in Z direction
  !> @param[in] dx       (const)  grid spacing in X direction
  !> @param[in] dy       (const)  grid spacing in Y direction
  !> @param[in] dz       (const)  grid spacing in Z direction
  !> @param[in] dt       (const)  time stepsize
  !> @param[inout] flux1    (modify) flux in X direction on X edges
  !> @param[inout] flux2    (modify) flux in Y direction on Y edges
  !> @param[inout] flux3    (modify) flux in Z direction on Z edges
    subroutine pc_hyp_mol_flux(gpustream,lo1,lo2,lo3,hi1,hi2,hi3, &
                     domlo, domhi, &
                     q, qd_lo, qd_hi, &
                     qaux, qa_lo, qa_hi, &
                     Ax,  Axlo,  Axhi,&
                     flux1, fd1_lo, fd1_hi, &
                     Ay,  Aylo,  Ayhi,&
                     flux2, fd2_lo, fd2_hi, &
                     Az,  Azlo,  Azhi,&
                     flux3, fd3_lo, fd3_hi, &
                     flatn, fltd_lo, fltd_hi, &
                     V, Vlo, Vhi, &
                     D, Dlo, Dhi,&
#ifdef PELEC_USE_EB
                     vfrac, vflo, vfhi, &
                     flag, fglo, fghi, &
                     ebg, Nebg, ebflux, nebflux, &
#endif
                     h) &
                     bind(C,name="pc_hyp_mol_flux")

    use meth_params_module, only : plm_iorder, QPRES, QRHO, QU, QV, QW, &
                                   QFS, QC, QCSML, NQAUX, nadv, &
                                   URHO, UMX, UMY, UMZ, UEDEN, UEINT, UFS, UTEMP, UFX, UFA, &
                                   small_dens, small_pres
    !use slope_module, only : slopex, slopey, slopez
    use actual_network, only : naux
    use eos_module, only : eos_rp_gpu
    use chemistry_module, only: Ru
    !                               eb_small_vfrac
    !use slope_module, only : slopex, slopey, slopez
    !use network, only : nspecies, naux
    !use eos_type_module
    !use eos_module, only : eos_t, eos_rp
    !use riemann_module, only: cmpflx, shock
    !use amrex_constants_module
    !use amrex_fort_module, only : amrex_real

    implicit none

    integer, parameter  :: nspec_2=9
    integer, parameter  :: nvar_2=16
    integer, parameter  :: qvar_2=17
    !   concept is to advance cells lo to hi
    !   need fluxes on the boundary
    !   if tile is eb need to expand by 2 cells in each directions
    !   would like to do this tile by tile
#ifdef PELEC_USE_EB
    integer, parameter  :: nextra = 3
#else
    integer, parameter  :: nextra = 0
#endif

    integer :: vis, vie, vic ! Loop bounds for vector blocking
    integer :: vi, vii ! Loop indicies for unrolled loops over

    integer, intent(in) :: gpustream
    integer, intent(in) ::      qd_lo(3),   qd_hi(3)
    integer, intent(in) ::      qa_lo(3),   qa_hi(3)
    integer, intent(in) :: lo1, lo2, lo3, hi1, hi2, hi3
    integer, intent(in) ::      domlo(3),   domhi(3)
    integer, intent(in) ::       Axlo(3),    Axhi(3)
    integer, intent(in) ::     fd1_lo(3),  fd1_hi(3)
    integer, intent(in) ::       Aylo(3),    Ayhi(3)
    integer, intent(in) ::     fd2_lo(3),  fd2_hi(3)
    integer, intent(in) ::       Azlo(3),    Azhi(3)
    integer, intent(in) ::     fd3_lo(3),  fd3_hi(3)
    integer, intent(in) ::    fltd_lo(3), fltd_hi(3)
    integer, intent(in) ::        Vlo(3),     Vhi(3)
    integer, intent(in) ::        Dlo(3),     Dhi(3)
    double precision, intent(in) :: h(3)

#ifdef PELEC_USE_EB
    integer, intent(in) ::  fglo(3),    fghi(3)
    integer, intent(in) ::  vflo(3),    vfhi(3)
    integer, intent(in) :: flag(fglo(1):fghi(1),fglo(2):fghi(2),fglo(3):fghi(3))
    double precision, intent(in) :: vfrac(vflo(1):vfhi(1),vflo(2):vfhi(2),vflo(3):vfhi(3))

    integer, intent(in) :: nebflux
    double precision, intent(inout) ::   ebflux(0:nebflux-1,1:nvar_2)
    integer,            intent(in   ) :: Nebg
    type(eb_bndry_geom),intent(in   ) :: ebg(0:Nebg-1)
    double precision :: eb_norm(3), full_area
#endif
    double precision, intent(in) ::     q(  qd_lo(1):  qd_hi(1),  qd_lo(2):  qd_hi(2),  qd_lo(3):  qd_hi(3),qvar_2)  !> State
    double precision, intent(in) ::  qaux(  qa_lo(1):  qa_hi(1),  qa_lo(2):  qa_hi(2),  qa_lo(3):  qa_hi(3),NQAUX) !> Auxiliary state
    double precision, intent(in) :: flatn(fltd_lo(1):fltd_hi(1),fltd_lo(2):fltd_hi(2),fltd_lo(3):fltd_hi(3))

    double precision, intent(in   ) ::    Ax(  Axlo(1):  Axhi(1),  Axlo(2):  Axhi(2),  Axlo(3):  Axhi(3))
    double precision, intent(inout) :: flux1(fd1_lo(1):fd1_hi(1),fd1_lo(2):fd1_hi(2),fd1_lo(3):fd1_hi(3),nvar_2)
    double precision, intent(in   ) ::    Ay(  Aylo(1):  Ayhi(1),  Aylo(2):  Ayhi(2),  Aylo(3):  Ayhi(3))
    double precision, intent(inout) :: flux2(fd2_lo(1):fd2_hi(1),fd2_lo(2):fd2_hi(2),fd2_lo(3):fd2_hi(3),nvar_2)
    double precision, intent(in   ) ::    Az(  Azlo(1):  Azhi(1),  Azlo(2):  Azhi(2),  Azlo(3):  Azhi(3))
    double precision, intent(inout) :: flux3(fd3_lo(1):fd3_hi(1),fd3_lo(2):fd3_hi(2),fd3_lo(3):fd3_hi(3),nvar_2)
    double precision, intent(inout) ::     V(   Vlo(1):   Vhi(1),   Vlo(2):   Vhi(2),   Vlo(3):   Vhi(3))
    double precision, intent(inout) ::     D(   Dlo(1):   Dhi(1),   Dlo(2):   Dhi(2),   Dlo(3):   Dhi(3),nvar_2)

    integer :: i, j, k, n, nsp, L, ivar
    integer :: qt_lo(3), qt_hi(3)
    integer :: ilo1, ilo2, ilo3, ihi1, ihi2, ihi3
    integer :: fglo1, fglo2, fglo3, fghi1, fghi2, fghi3
    integer :: domlo1, domlo2, domlo3, domhi1, domhi2, domhi3
    integer :: qd_lo1, qd_lo2, qd_lo3, qd_hi1, qd_hi2, qd_hi3
    integer :: qt_lo1, qt_lo2, qt_lo3, qt_hi1, qt_hi2, qt_hi3
    integer :: qa_lo1, qa_lo2, qa_lo3, qa_hi1, qa_hi2, qa_hi3

    ! Left and right state arrays (edge centered, cell centered)
    double precision :: dqx(lo1-nextra:hi1+nextra, lo2-nextra:hi2+nextra, lo3-nextra:hi3+nextra, 1:qvar_2)
    double precision :: dqy(lo1-nextra:hi1+nextra, lo2-nextra:hi2+nextra, lo3-nextra:hi3+nextra, 1:qvar_2)
    double precision :: dqz(lo1-nextra:hi1+nextra, lo2-nextra:hi2+nextra, lo3-nextra:hi3+nextra, 1:qvar_2)

    ! Other left and right state arrays
    double precision :: qtempl_x(1:5+nspec_2)
    double precision :: qtempl_y(1:5+nspec_2)
    double precision :: qtempl_z(1:5+nspec_2)
    double precision :: qtempr_x(1:5+nspec_2)
    double precision :: qtempr_y(1:5+nspec_2)
    double precision :: qtempr_z(1:5+nspec_2)
    double precision :: rhoe_l
    double precision :: rhoe_r
    double precision :: cspeed
    double precision :: gamc_l
    double precision :: gamc_r
    double precision :: cavg
    double precision :: csmall

    ! Scratch for neighborhood of cut cells
    integer :: nbr_x(-1:1,-1:1,-1:1)
    integer :: nbr_y(-1:1,-1:1,-1:1)
    integer :: nbr_z(-1:1,-1:1,-1:1)

    ! Riemann solve work arrays
    double precision:: u_gd, v_gd, w_gd, &
         p_gd, game_gd, re_gd, &
         r_gd, ustar
    double precision :: flux_tmp_x(nvar_2)
    double precision :: flux_tmp_y(nvar_2)
    double precision :: flux_tmp_z(nvar_2)
    integer, parameter :: idir = 1
    ! integer :: nextra
    integer, parameter :: coord_type = 0
    integer, parameter :: bc_test_val = 1

    double precision :: eos_state_rho
    double precision :: eos_state_p
    double precision :: eos_state_massfrac_x(nspec_2)
    double precision :: eos_state_massfrac_y(nspec_2)
    double precision :: eos_state_massfrac_z(nspec_2)
    double precision :: eos_state_gam1
    double precision :: eos_state_e
    double precision :: eos_state_cs

    integer, parameter :: R_RHO = 1
    integer, parameter :: R_UN  = 2
    integer, parameter :: R_UT1 = 3
    integer, parameter :: R_UT2 = 4
    integer, parameter :: R_P   = 5
    integer, parameter :: R_Y   = 6

    ! do L=1,3
    !    qt_lo(L) = lo(L) - nextra
    !    qt_hi(L) = hi(L) + nextra
    !    !if (qt_lo(L)-1 .lt. qd_lo(L) .or. qt_hi(L)+1 .gt. qd_hi(L)) then
    !    !   stop 1
    !    !endif
    ! enddo

    ilo1=lo1-nextra
    ilo2=lo2-nextra
    ilo3=lo3-nextra
    ihi1=hi1+nextra
    ihi2=hi2+nextra
    ihi3=hi3+nextra
    !fglo1=fglo(1)
    !fglo2=fglo(2)
    !fglo3=fglo(3)
    !fghi1=fghi(1)
    !fghi2=fghi(2)
    !fghi3=fghi(3)
    domlo1=domlo(1)
    domlo2=domlo(2)
    domlo3=domlo(3)
    domhi1=domhi(1)
    domhi2=domhi(2)
    domhi3=domhi(3)
    qd_lo1=qd_lo(1)
    qd_lo2=qd_lo(2)
    qd_lo3=qd_lo(3)
    qd_hi1=qd_hi(1)
    qd_hi2=qd_hi(2)
    qd_hi3=qd_hi(3)
    qt_lo1=lo1-nextra
    qt_lo2=lo2-nextra
    qt_lo3=lo3-nextra
    qt_hi1=hi1+nextra
    qt_hi2=hi2+nextra
    qt_hi3=hi3+nextra
    qa_lo1=qa_lo(1)
    qa_lo2=qa_lo(2)
    qa_lo3=qa_lo(3)
    qa_hi1=qa_hi(1)
    qa_hi2=qa_hi(2)
    qa_hi3=qa_hi(3)

    !$acc enter data create(dqx,dqy,dqz,qtempl_x,qtempl_y,qtempl_z,qtempr_x,qtempr_y,qtempr_z,eos_state_massfrac_x,eos_state_massfrac_y,eos_state_massfrac_z,flux_tmp_x,flux_tmp_y,flux_tmp_z,nbr_x,nbr_y,nbr_z) async(gpustream)

    !$acc parallel loop gang vector collapse(4) default(present) async(gpustream)
    do n = 1, qvar_2
       do k = qt_lo3, qt_hi3
          do j = qt_lo2, qt_hi2
             do i = qt_lo1, qt_hi1
                dqx(i,j,k,n) = 0.d0
             enddo
          enddo
       enddo
    enddo
    !$acc end parallel
    !if(plm_iorder.ne.1) then
    !   !$acc parallel default(present) async(gpustream)
    !   call slopex(q,flatn,qd_lo1,qd_lo2,qd_lo3,qd_hi1,qd_hi2,qd_hi3, &
    !              dqx,qt_lo1,qt_lo2,qt_lo3,qt_hi1,qt_hi2,qt_hi3, &
    !              ilo1,ilo2,ilo3, &
    !              ihi1,ihi2,ihi3,qvar_2,nqaux, &
    !              domlo1,domlo2,domlo3,domhi1,domhi2,domhi3, &
    !              qaux,qa_lo1,qa_lo2,qa_lo3,qa_hi1,qa_hi2,qa_hi3, &
    !              flag,fglo1,fglo2,fglo3,fghi1,fghi2,fghi3)
    !   !$acc end parallel
    !end if

    !$acc kernels default(present) async(gpustream)
    !$acc loop gang vector collapse(3) private(n,qtempl_x,qtempr_x,gamc_l,rhoe_l,rhoe_r,gamc_r,u_gd, v_gd, w_gd, p_gd, game_gd, re_gd, r_gd, ustar, eos_state_rho, eos_state_p, eos_state_massfrac_x, eos_state_e, eos_state_gam1, eos_state_cs, flux_tmp_x, csmall, cavg, vic, ivar)
    do k = ilo3, ihi3
       do j = ilo2, ihi2
          do i = ilo1+1, ihi1
             qtempl_x(R_UN) = q(i-1,j,k,QU) + 0.5d0 * ((dqx(i-1,j,k,2) - dqx(i-1,j,k,1)) / q(i-1,j,k,QRHO))
             qtempl_x(R_P) = q(i-1,j,k,QPRES) + 0.5d0 * (dqx(i-1,j,k,1) + dqx(i-1,j,k,2)) * qaux(i-1,j,k,QC)
             qtempl_x(R_UT1) = q(i-1,j,k,QV) + 0.5d0 * dqx(i-1,j,k,3)
             qtempl_x(R_UT2) = q(i-1,j,k,QW) + 0.5d0 * dqx(i-1,j,k,4)
             qtempl_x(R_RHO) = 0.d0
             do n = 1,nspec_2
                qtempl_x(R_Y-1+n) = q(i-1,j,k,QFS-1+n) * q(i-1,j,k,QRHO) + 0.5d0 * (dqx(i-1,j,k,4+n) &
                                  + q(i-1,j,k,QFS-1+n) * (dqx(i-1,j,k,1) + dqx(i-1,j,k,2)) / qaux(i-1,j,k,QC))
                qtempl_x(R_RHO) = qtempl_x(R_RHO) + qtempl_x(R_Y-1+n)
             enddo

             do n = 1,nspec_2
               qtempl_x(R_Y-1+n) = qtempl_x(R_Y-1+n) / qtempl_x(R_RHO)
             enddo

             qtempr_x(R_UN) = q(i,j,k,QU) - 0.5d0 * ((dqx(i,j,k,2) - dqx(i,j,k,1)) / q(i,j,k,QRHO))
             qtempr_x(R_P) = q(i,j,k,QPRES) - 0.5d0 * (dqx(i,j,k,1) + dqx(i,j,k,2)) * qaux(i,j,k,QC)
             qtempr_x(R_UT1) = q(i,j,k,QV) - 0.5d0 * dqx(i,j,k,3)
             qtempr_x(R_UT2) = q(i,j,k,QW) - 0.5d0 * dqx(i,j,k,4)
             qtempr_x(R_RHO) = 0.d0

             do n = 1,nspec_2
                qtempr_x(R_Y-1+n) = q(i,j,k,QFS-1+n) * q(i,j,k,QRHO) - 0.5d0 * &
                                    (dqx(i,j,k,4+n) + q(i,j,k,QFS-1+n) * &
                                    (dqx(i,j,k,1) + dqx(i,j,k,2)) / qaux(i,j,k,QC))
                qtempr_x(R_RHO) = qtempr_x(R_RHO) + qtempr_x(R_Y-1+n)
             enddo

             do n = 1,nspec_2
                qtempr_x(R_Y-1+n) = qtempr_x(R_Y-1+n)/qtempr_x(R_RHO)
             enddo

             cavg = 0.5d0 * (qaux(i,j,k,QC) + qaux(i-1,j,k,QC))
             csmall = min(qaux(i,j,k,QCSML), qaux(i-1,j,k,QCSML))

             eos_state_rho = qtempl_x(R_RHO)
             eos_state_p = qtempl_x(R_P)
             eos_state_massfrac_x = qtempl_x(R_Y:R_Y-1+nspec_2)
             call eos_rp_gpu(eos_state_rho, eos_state_p, eos_state_massfrac_x, eos_state_e, eos_state_gam1, eos_state_cs, nspec_2)
             rhoe_l = eos_state_rho * eos_state_e
             gamc_l = eos_state_gam1

             eos_state_rho = qtempr_x(R_RHO)
             eos_state_p = qtempr_x(R_P)
             eos_state_massfrac_x = qtempr_x(R_Y:R_Y-1+nspec_2)
             call eos_rp_gpu(eos_state_rho, eos_state_p, eos_state_massfrac_x, eos_state_e, eos_state_gam1, eos_state_cs, nspec_2)
             rhoe_r = eos_state_rho * eos_state_e
             gamc_r = eos_state_gam1

             call riemann_md_vec(qtempl_x(R_RHO), qtempl_x(R_UN), qtempl_x(R_UT1), qtempl_x(R_UT2), &
                                 qtempl_x(R_P), rhoe_l, qtempl_x(R_Y:R_Y-1+nspec_2), gamc_l, &
                                 qtempr_x(R_RHO), qtempr_x(R_UN), qtempr_x(R_UT1), qtempr_x(R_UT2), &
                                 qtempr_x(R_P), rhoe_r, qtempr_x(R_Y:R_Y-1+nspec_2), gamc_r,&
                                 u_gd, v_gd, w_gd, p_gd, game_gd, re_gd, r_gd, ustar, &
                                 eos_state_rho, eos_state_p, eos_state_massfrac_x, &
                                 eos_state_e, eos_state_gam1, eos_state_cs, nspec_2, &
                                 flux_tmp_x(URHO), flux_tmp_x(UMX), flux_tmp_x(UMY), flux_tmp_x(UMZ), &
                                 flux_tmp_x(UEDEN), flux_tmp_x(UEINT), bc_test_val, csmall, cavg, vic)

             do n = 0, nspec_2-1
                flux_tmp_x(UFS+n) = merge(flux_tmp_x(URHO)*qtempl_x(R_Y+n), flux_tmp_x(URHO)*qtempr_x(R_Y+n), ustar .ge. 0.d0)
                flux_tmp_x(UFS+n) = merge(flux_tmp_x(URHO)*0.5d0*(qtempl_x(R_Y+n) + qtempr_x(R_Y+n)), flux_tmp_x(UFS+n), ustar .eq. 0.d0)
             enddo

             flux_tmp_x(UTEMP) = 0.0
             do n = UFX, UFX+naux
                flux_tmp_x(n) = merge(0.d0, flux_tmp_x(n), naux .gt. 0)
             enddo
             do n = UFA, UFA+nadv
                flux_tmp_x(n) = merge(0.d0, flux_tmp_x(n), nadv .gt. 0)
             enddo

             do ivar = 1, nvar_2
                flux1(i,j,k,ivar) = flux1(i,j,k,ivar) + flux_tmp_x(ivar) * ax(i,j,k)
             enddo
          enddo
       enddo
    enddo
    !$acc end kernels

    !$acc parallel loop gang vector collapse(4) default(present) async(gpustream)
    do n = 1, qvar_2
       do k = qt_lo3, qt_hi3
          do j = qt_lo2, qt_hi2
             do i = qt_lo1, qt_hi1
                dqy(i,j,k,n) = 0.d0
             enddo
          enddo
       enddo
    enddo
    !$acc end parallel
    !if(plm_iorder.ne.1) then
    !   !$acc parallel default(present) async(gpustream)
    !   call slopey(q,flatn,qd_lo1,qd_lo2,qd_lo3,qd_hi1,qd_hi2,qd_hi3, &
    !              dqy,qt_lo1,qt_lo2,qt_lo3,qt_hi1,qt_hi2,qt_hi3, &
    !              ilo1,ilo2,ilo3, &
    !              ihi1,ihi2,ihi3,qvar_2,nqaux, &
    !              domlo1,domlo2,domlo3,domhi1,domhi2,domhi3, &
    !              qaux,qa_lo1,qa_lo2,qa_lo3,qa_hi1,qa_hi2,qa_hi3, &
    !              flag,fglo1,fglo2,fglo3,fghi1,fghi2,fghi3)
    !   !$acc end parallel
    !end if

    !$acc kernels default(present) async(gpustream)
    !$acc loop gang vector collapse(3) private(n,qtempl_y,qtempr_y,gamc_l,rhoe_l,rhoe_r,gamc_r,u_gd, v_gd, w_gd, p_gd, game_gd, re_gd, r_gd, ustar, eos_state_rho, eos_state_p, eos_state_massfrac_y, eos_state_e, eos_state_gam1, eos_state_cs, flux_tmp_y, csmall, cavg, vic, ivar)
    do k = ilo3, ihi3
       do j = ilo2+1, ihi2
          do i = ilo1, ihi1
             qtempl_y(R_UN) = q(i,j-1,k,QV) + 0.5d0 * ((dqy(i,j-1,k,2) - dqy(i,j-1,k,1)) / q(i,j-1,k,QRHO))
             qtempl_y(R_P) = q(i,j-1,k,QPRES) + 0.5d0 * (dqy(i,j-1,k,1) + dqy(i,j-1,k,2)) * qaux(i,j-1,k,QC)
             qtempl_y(R_UT1) = q(i,j-1,k,QU) + 0.5d0 * dqy(i,j-1,k,3)
             qtempl_y(R_UT2) = q(i,j-1,k,QW) + 0.5d0 * dqy(i,j-1,k,4)
             qtempl_y(R_RHO) = 0.d0
             do n = 1,nspec_2
               qtempl_y(R_Y-1+n) = q(i,j-1,k,QFS-1+n) * q(i,j-1,k,QRHO) + 0.5d0 * (dqy(i,j-1,k,4+n) &
                                 + q(i,j-1,k,QFS-1+n) * (dqy(i,j-1,k,1) + dqy(i,j-1,k,2)) / qaux(i,j-1,k,QC))
               qtempl_y(R_RHO) = qtempl_y(R_RHO) + qtempl_y(R_Y-1+n)
             enddo

             do n = 1,nspec_2
               qtempl_y(R_Y-1+n) = qtempl_y(R_Y-1+n)/qtempl_y(R_RHO)
             enddo

             qtempr_y(R_UN) = q(i,j,k,QV) - 0.5d0 * ((dqy(i,j,k,2) - dqy(i,j,k,1)) / q(i,j,k,QRHO))
             qtempr_y(R_P) = q(i,j,k,QPRES) - 0.5d0 * (dqy(i,j,k,1) + dqy(i,j,k,2)) * qaux(i,j,k,QC)
             qtempr_y(R_UT1) = q(i,j,k,QU) - 0.5d0 * dqy(i,j,k,3)
             qtempr_y(R_UT2) = q(i,j,k,QW) - 0.5d0 * dqy(i,j,k,4)
             qtempr_y(R_RHO) = 0.d0

             do n = 1,nspec_2
               qtempr_y(R_Y-1+n) = q(i,j,k,QFS-1+n) &
                                   * q(i,j,k,QRHO) - 0.5d0*(dqy(i,j,k,4+n) &
                                   + q(i,j,k,QFS-1+n) &
                                   * (dqy(i,j,k,1) + dqy(i,j,k,2)) &
                                   / qaux(i,j,k,QC))
               qtempr_y(R_RHO) = qtempr_y(R_RHO) + qtempr_y(R_Y-1+n)
             enddo

             do n = 1,nspec_2
               qtempr_y(R_Y-1+n) = qtempr_y(R_Y-1+n)/qtempr_y(R_RHO)
             enddo

             cavg = 0.5d0 * (qaux(i,j,k,QC) + qaux(i,j-1,k,QC))
             csmall = min(qaux(i,j,k,QCSML), qaux(i,j-1,k,QCSML))

             eos_state_rho = qtempl_y(R_RHO)
             eos_state_p = qtempl_y(R_P)
             eos_state_massfrac_y = qtempl_y(R_Y:R_Y-1+nspec_2)
             call eos_rp_gpu(eos_state_rho, eos_state_p, eos_state_massfrac_y, eos_state_e, eos_state_gam1, eos_state_cs, nspec_2)
             rhoe_l = eos_state_rho * eos_state_e
             gamc_l = eos_state_gam1

             eos_state_rho = qtempr_y(R_RHO)
             eos_state_p = qtempr_y(R_P)
             eos_state_massfrac_y = qtempr_y(R_Y:R_Y-1+nspec_2)
             call eos_rp_gpu(eos_state_rho, eos_state_p, eos_state_massfrac_y, eos_state_e, eos_state_gam1, eos_state_cs, nspec_2)
             rhoe_r = eos_state_rho * eos_state_e
             gamc_r = eos_state_gam1

             call riemann_md_vec( &
                  qtempl_y(R_RHO), qtempl_y(R_UN), qtempl_y(R_UT1), qtempl_y(R_UT2), qtempl_y(R_P), rhoe_l, qtempl_y(R_Y:R_Y-1+nspec_2), gamc_l,&
                  qtempr_y(R_RHO), qtempr_y(R_UN), qtempr_y(R_UT1), qtempr_y(R_UT2), qtempr_y(R_P), rhoe_r, qtempr_y(R_Y:R_Y-1+nspec_2), gamc_r,&
                  v_gd, u_gd, w_gd, p_gd, game_gd, re_gd, r_gd, ustar,&
                  eos_state_rho, eos_state_p, eos_state_massfrac_y, &
                  eos_state_e, eos_state_gam1, eos_state_cs, nspec_2,&
                  flux_tmp_y(URHO), flux_tmp_y(UMY), flux_tmp_y(UMX), flux_tmp_y(UMZ), flux_tmp_y(UEDEN), flux_tmp_y(UEINT), &
                  bc_test_val, csmall, cavg, vic)

             do n = 0, nspec_2-1
                flux_tmp_y(UFS+n) = merge(flux_tmp_y(URHO)*qtempl_y(R_Y+n), flux_tmp_y(URHO)*qtempr_y(R_Y+n), ustar .ge. 0.d0)
                flux_tmp_y(UFS+n) = merge(flux_tmp_y(URHO)*0.5d0*(qtempl_y(R_Y+n) + qtempr_y(R_Y+n)), flux_tmp_y(UFS+n), ustar .eq. 0.d0)
             enddo

             flux_tmp_y(UTEMP) = 0.0
             do n = UFX, UFX+naux
                flux_tmp_y(n) = merge(0.d0, flux_tmp_y(n), naux .gt. 0)
             enddo
             do n = UFA, UFA+nadv
                flux_tmp_y(n) = merge(0.d0, flux_tmp_y(n), nadv .gt. 0)
             enddo

             do ivar = 1, nvar_2
                flux2(i,j,k,ivar) = flux2(i,j,k,ivar) + flux_tmp_y(ivar) * ay(i,j,k)
             enddo
          enddo
       enddo
    enddo
    !$acc end kernels

    !$acc parallel loop gang vector collapse(4) default(present) async(gpustream)
    do n = 1, qvar_2
       do k = qt_lo3, qt_hi3
          do j = qt_lo2, qt_hi2
             do i = qt_lo1, qt_hi1
                dqz(i,j,k,n) = 0.d0
             enddo
          enddo
       enddo
    enddo
    !$acc end parallel
    !if(plm_iorder.ne.1) then
    !   !$acc parallel default(present) async(gpustream)
    !   call slopez(q,flatn,qd_lo1,qd_lo2,qd_lo3,qd_hi1,qd_hi2,qd_hi3, &
    !              dqz,qt_lo1,qt_lo2,qt_lo3,qt_hi1,qt_hi2,qt_hi3, &
    !              ilo1,ilo2,ilo3, &
    !              ihi1,ihi2,ihi3,qvar_2,nqaux, &
    !              domlo1,domlo2,domlo3,domhi1,domhi2,domhi3, &
    !              qaux,qa_lo1,qa_lo2,qa_lo3,qa_hi1,qa_hi2,qa_hi3, &
    !              flag,fglo1,fglo2,fglo3,fghi1,fghi2,fghi3)
    !   !$acc end parallel
    !end if

    !$acc kernels default(present) async(gpustream)
    !$acc loop gang vector collapse(3) private(n,qtempl_z,qtempr_z,gamc_l,rhoe_l,rhoe_r,gamc_r,u_gd, v_gd, w_gd, p_gd, game_gd, re_gd, r_gd, ustar, eos_state_rho, eos_state_p, eos_state_massfrac_z, eos_state_e, eos_state_gam1, eos_state_cs, flux_tmp_z, csmall, cavg, vic, ivar)
    do k = ilo3+1, ihi3
       do j = ilo2, ihi2
          do i = ilo1, ihi1
             qtempl_z(R_UN) = q(i,j,k-1,QW) + 0.5d0 * ((dqz(i,j,k-1,2) - dqz(i,j,k-1,1)) / q(i,j,k-1,QRHO))
             qtempl_z(R_P) = q(i,j,k-1,QPRES) + 0.5d0 * (dqz(i,j,k-1,1) + dqz(i,j,k-1,2)) * qaux(i,j,k-1,QC)
             qtempl_z(R_UT1) = q(i,j,k-1,QU) + 0.5d0 * dqz(i,j,k-1,3)
             qtempl_z(R_UT2) = q(i,j,k-1,QV) + 0.5d0 * dqz(i,j,k-1,4)
             qtempl_z(R_RHO) = 0.d0
             do n = 1,nspec_2
               qtempl_z(R_Y-1+n) = q(i,j,k-1,QFS-1+n) &
                                 * q(i,j,k-1,QRHO) + 0.5d0*(dqz(i,j,k-1,4+n) &
                                 + q(i,j,k-1,QFS-1+n) &
                                 * (dqz(i,j,k-1,1) + dqz(i,j,k-1,2)) &
                                 / qaux(i,j,k-1,QC))
               qtempl_z(R_RHO) = qtempl_z(R_RHO) + qtempl_z(R_Y-1+n)
             enddo

             do n = 1,nspec_2
               qtempl_z(R_Y-1+n) = qtempl_z(R_Y-1+n)/qtempl_z(R_RHO)
             enddo

             qtempr_z(R_UN) = q(i,j,k,QW) - 0.5d0 * ((dqz(i,j,k,2) - dqz(i,j,k,1)) / q(i,j,k,QRHO))
             qtempr_z(R_P) = q(i,j,k,QPRES) - 0.5d0 * (dqz(i,j,k,1) + dqz(i,j,k,2)) * qaux(i,j,k,QC)
             qtempr_z(R_UT1) = q(i,j,k,QU) - 0.5d0 * dqz(i,j,k,3)
             qtempr_z(R_UT2) = q(i,j,k,QV) - 0.5d0 * dqz(i,j,k,4)
             qtempr_z(R_RHO) = 0.d0

             do n = 1,nspec_2
                qtempr_z(R_Y-1+n) = q(i,j,k,QFS-1+n) &
                                  * q(i,j,k,QRHO) - 0.5d0*(dqz(i,j,k,4+n) &
                                  + q(i,j,k,QFS-1+n) &
                                  * (dqz(i,j,k,1) + dqz(i,j,k,2)) &
                                  / qaux(i,j,k,QC))
                qtempr_z(R_RHO) = qtempr_z(R_RHO) + qtempr_z(R_Y-1+n)
             enddo

             do n = 1,nspec_2
               qtempr_z(R_Y-1+n) = qtempr_z(R_Y-1+n)/qtempr_z(R_RHO)
             enddo

             cavg = 0.5d0 * (qaux(i,j,k,QC) + qaux(i,j,k-1,QC))
             csmall = min(qaux(i,j,k,QCSML), qaux(i,j,k-1,QCSML))

             eos_state_rho = qtempl_z(R_RHO)
             eos_state_p = qtempl_z(R_P)
             eos_state_massfrac_z = qtempl_z(R_Y:R_Y-1+nspec_2)
             call eos_rp_gpu(eos_state_rho, eos_state_p, eos_state_massfrac_z, eos_state_e, eos_state_gam1, eos_state_cs, nspec_2)
             rhoe_l = eos_state_rho * eos_state_e
             gamc_l = eos_state_gam1

             eos_state_rho = qtempr_z(R_RHO)
             eos_state_p = qtempr_z(R_P)
             eos_state_massfrac_z = qtempr_z(R_Y:R_Y-1+nspec_2)
             call eos_rp_gpu(eos_state_rho, eos_state_p, eos_state_massfrac_z, eos_state_e, eos_state_gam1, eos_state_cs, nspec_2)
             rhoe_r = eos_state_rho * eos_state_e
             gamc_r = eos_state_gam1

             call riemann_md_vec( &
                  qtempl_z(R_RHO), qtempl_z(R_UN), qtempl_z(R_UT1), qtempl_z(R_UT2), qtempl_z(R_P), rhoe_l, qtempl_z(R_Y:R_Y-1+nspec_2), gamc_l,&
                  qtempr_z(R_RHO), qtempr_z(R_UN), qtempr_z(R_UT1), qtempr_z(R_UT2), qtempr_z(R_P), rhoe_r, qtempr_z(R_Y:R_Y-1+nspec_2), gamc_r,&
                  w_gd, u_gd, v_gd, p_gd, game_gd, re_gd, r_gd, ustar,&
                  eos_state_rho, eos_state_p, eos_state_massfrac_z, &
                  eos_state_e, eos_state_gam1, eos_state_cs, nspec_2,&
                  flux_tmp_z(URHO), flux_tmp_z(UMZ), flux_tmp_z(UMX), flux_tmp_z(UMY), flux_tmp_z(UEDEN), flux_tmp_z(UEINT), &
                  bc_test_val, csmall, cavg, vic)

             do n = 0, nspec_2-1
                flux_tmp_z(UFS+n) = merge(flux_tmp_z(URHO)*qtempl_z(R_Y+n), flux_tmp_z(URHO)*qtempr_z(R_Y+n), ustar .ge. 0.d0)
                flux_tmp_z(UFS+n) = merge(flux_tmp_z(URHO)*0.5d0*(qtempl_z(R_Y+n) + qtempr_z(R_Y+n)), flux_tmp_z(UFS+n), ustar .eq. 0.d0)
             enddo

             flux_tmp_z(UTEMP) = 0.0
             do n = UFX, UFX+naux
                flux_tmp_z(n) = merge(0.d0, flux_tmp_z(n), naux .gt. 0)
             enddo
             do n = UFA, UFA+nadv
                flux_tmp_z(n) = merge(0.d0, flux_tmp_z(n), nadv .gt. 0)
             enddo

             do ivar = 1, nvar_2
                flux3(i,j,k,ivar) = flux3(i,j,k,ivar) + flux_tmp_z(ivar) * az(i,j,k)
             enddo
          enddo
       enddo
    enddo
    !$acc end kernels

    !$acc parallel loop gang vector collapse(4) default(present) async(gpustream)
    do ivar=1,nvar_2
       do k = ilo3+1, ihi3-1
          do j = ilo2+1, ihi2-1
             do i = ilo1+1, ihi1-1
                d(i,j,k,ivar) = - (flux1(i+1,j,k,ivar) - flux1(i,j,k,ivar) &
                                +  flux2(i,j+1,k,ivar) - flux2(i,j,k,ivar) &
                                +  flux3(i,j,k+1,ivar) - flux3(i,j,k,ivar)) / v(i,j,k)
             enddo
          enddo
       enddo
    enddo
    !$acc end parallel

    !$acc exit data delete(dqx,dqy,dqz,qtempl_x,qtempl_y,qtempl_z,qtempr_x,qtempr_y,qtempr_z,eos_state_massfrac_x,eos_state_massfrac_y,eos_state_massfrac_z,flux_tmp_x,flux_tmp_y,flux_tmp_z,nbr_x,nbr_y,nbr_z) async(gpustream)

    !do ivar=1,nvar_2
    !   do k = lo(3)-nextra+1, hi(3)+nextra-1
    !      do j = lo(2)-nextra+1, hi(2)+nextra-1
    !         do i = lo(1)-nextra+1, hi(1)+nextra-1
    !            print *, "FLUX3: ", flux3(i,j,k,ivar)
    !         enddo
    !      enddo
    !   enddo
    !enddo

  end subroutine pc_hyp_mol_flux
end module hyp_advection_module
