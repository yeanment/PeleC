PRECISION  = DOUBLE
PROFILE    = FALSE
USE_EB     = FALSE
DEBUG      = TRUE

DIM        = 2

COMP	   = gcc

USE_MPI    = TRUE
USE_OMP    = FALSE

USE_REACT  = TRUE

USE_SUNDIALS_3x4x=TRUE
USE_CUDA   =  FALSE
USE_CUDA_CVODE = FALSE
USE_SDC_FORTRAN = FALSE
ifeq ($(USE_SDC_FORTRAN), TRUE)
    DEFINES  += -DUSE_SDC_F
endif
USE_DVODE=TRUE

CVODE_LIB_DIR = /home/amfelden/CODES/PeleLM/Source/ChemDriver/sundials-3.2.1/instdir/lib/ 

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
#Chemistry_Model := LiDryer
#Chemistry_Model := drm19
#Chemistry_Model = ERC_nheptane
Chemistry_Model = dodecane_wang

# This sets the transport directory in $(PELE_PHYSICS_HOME)/Transport
Transport_dir := EGLib
#Transport_dir := Simple

Bpack   := ./Make.package
Blocs   := .

include ../Make.PeleC
