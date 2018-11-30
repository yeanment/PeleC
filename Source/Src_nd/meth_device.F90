! The Fortran-specific parameters for the GPU side 

module meth_device_module 
  implicit none
  integer, parameter     :: NHYP    = 4
  integer, parameter     :: nb_nscbc_params = 4
  integer, parameter  :: NTHERM = 7, NVAR = 16, URHO = 1, UMX = 2, UMY = 3,&
  UMZ=4, UML=0, UMP=0, UEDEN=5, UTEMP=7, UFS=8, UFA=1, UFX=1, USHK=-1, &
  QTHERM=8, QVAR=17, NQAUX=6, QGAMC=1, QC=2, QCSML=3, QDPDR=4, QDPDE=5,&
  QRSPEC=6 , QFA = 1, QFX=1, nadv=0, NQ=17, npassive = 10, NGDNV=4, GDRHO=1,&
  GDU =2, GDV=3, GDW=4, GDPRES=5, GDGAME=6, nspec=9, naux=0
  integer, parameter :: QRHO=1, QU=2, QV=3, QW=4, QPRES=6, QREINT=7, QTEMP=8, QGAME=5
  integer, parameter :: QFS=9
end module meth_device_module 
