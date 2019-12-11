PRECISION  = DOUBLE
PROFILE    = FALSE
DEBUG      = FALSE

# Set number of dimension to use.

DIM        = 3

# Choose compilers

COMP	   = gnu

# Choose parallelization

USE_MPI    = TRUE
USE_OMP    = FALSE

# Set Embedded boundaries

USE_EB = FALSE

USE_REACT = FALSE

USE_SDC = TRUE

# This sets the EOS directory in $(PELE_PHYSICS_HOME)/Eos
# Options include GammaLaw, Peng-Robinson
# Eos_dir     := Soave-Redlich-Kwong
Eos_dir     := GammaLaw

# This sets the network directory in $(PELE_PHYSICS_HOME)/Reactions
Reactions_dir := Null_air

# This sets the model directory in $(PELE_PHYSICS_HOME)/Support/Fuego/Mechanism/Models                                                             
#Chemistry_Model := air

# This sets the transport directory in $(PELE_PHYSICS_HOME)/Transport
Transport_dir := Constant

Bpack   := ./Make.package
Blocs   := .

PELEC_HOME := ../../../
include ${PELEC_HOME}/Exec/Make.PeleC

ifeq ($(DEBUG), TRUE)
DEFINES += -DDEBUG
endif
