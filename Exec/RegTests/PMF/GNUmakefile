USE_EB = FALSE

PRECISION  = DOUBLE

PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 1
DIM        = 2

COMP	   = gcc

USE_MPI    = TRUE
USE_OMP    = FALSE

USE_REACT  = TRUE

USE_CUDA   =  FALSE

#######################
USE_DVODE=FALSE
ifeq ($(USE_DVODE), TRUE)
    DEFINES  += -DUSE_DVODE
endif

USE_SUNDIALS_3x4x=TRUE
CVODE_LIB_DIR = /home/amfelden/CODES/PeleLM/Source/ChemDriver/sundials-3.2.1/instdir/lib/ 

USE_FORTRAN_CVODE=TRUE
ifeq ($(USE_FORTRAN_CVODE), TRUE)
    DEFINES  += -DUSE_FORTRAN_CVODE
    USE_KLU = FALSE
    ifeq ($(USE_KLU), TRUE)
        DEFINES  += -DUSE_KLU
    endif
    USE_PYJAC = FALSE
    ifeq ($(USE_PYJAC), TRUE)
        DEFINES  += -DUSE_PYJAC
    endif
else
USE_KLU = FALSE
USE_PYJAC = FALSE
endif

USE_CUDA_CVODE = FALSE

USE_SDC_FORTRAN = FALSE
ifeq ($(USE_SDC_FORTRAN), TRUE)
    DEFINES  += -DUSE_SDC_F
endif

#######################

#HYP_TYPE   = MOL

# define the location of the PELE top directory
PELE_HOME  := ../../..

# This sets the EOS directory in $(PELE_PHYSICS_HOME)/Eos
#Eos_dir     := Soave-Redlich-Kwong
Eos_dir     := Fuego

# This sets the network directory in $(PELE_PHYSICS_HOME)/Reactions
Reactions_dir := Fuego

# This sets the model directory in $(PELE_PHYSICS_HOME)/Support/Fuego/Mechanism/Models
# Chemistry_Model := Davis
Chemistry_Model := LiDryer

# This sets the transport directory in $(PELE_PHYSICS_HOME)/Transport
Transport_dir := EGLib
#Transport_dir := Simple

Bpack   := ./Make.package
Blocs   := .

include ../../Make.PeleC
include ./Make.cvodeCinterface
