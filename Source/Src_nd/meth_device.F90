! The Fortran-specific parameters for the GPU side 

module meth_device_module 
  implicit none
  integer, parameter :: &
#ifdef NUM_ADV
                        NVAR     = 7 + 9 + 1 + NUM_ADV, &
                        QVAR     = 8 + 9 + 1 + NUM_ADV, &
#else
                        NVAR     = 7 + 9 + 1, &
                        QVAR     = 8 + 9 + 1, &
#endif
                        URHO     = 1, &
                        UMX      = 2, &
                        UMY      = 3, &
                        UMZ      = 4, &
                        UEDEN    = 5, &
                        UEINT    = 6, &
                        UTEMP    = 7, &
                        NQAUX    = 6, &
                        QGAMC    = 1, &
                        QC       = 2, &
                        QCSML    = 3, &
                        QDPDR    = 4, &
                        QDPDE    = 5, &
                        QRSPEC   = 6, &
                        QFS      = 9, &
                        QFX      = 18, &
                        npassive = 2
 
!  contains 
!  subroutine set_device_params()
!    implicit none 
!#ifdef NUM_ADV
!    NVAR = 7 + 9 + 1 + NUM_ADV !LiDryer Only! 
!    QVAR = 8 + 9 + 1 + NUM_ADV
!#else
!    NVAR = 7 + 9 + 1
!    QVAR = 8 + 9 + 1 + NUM_ADV 
!#endif 
!  
!    URHO   = 1
!    UMX    = 2
!    UMY    = 3
!    UMZ    = 4
!    UEDEN  = 5
!    UEINT  = 6
!    UTEMP  = 7
!
!    NQAUX  = 6 
!    QGAMC  = 1
!    QC     = 2
!    QCSML  = 3
!    QDPDR  = 4
!    QDPDE  = 5
!    QRSPEC = 6 
!    
!    npassive = 2 
!     
!  end subroutine set_device_params 

end module meth_device_module 
