PRECISION  = DOUBLE
PROFILE    = FALSE
USE_EB     = FALSE
DEBUG      = FALSE

DIM        = 2

COMP	   = gcc

USE_MPI    = TRUE
USE_OMP    = FALSE
USE_REACT  = TRUE

# This sets the EOS directory in $(PELE_PHYSICS_HOME)/Eos
Eos_dir     := Fuego

# This sets the network directory in $(PELE_PHYSICS_HOME)/Reactions
Reactions_dir := Fuego

# This sets the model directory in $(PELE_PHYSICS_HOME)/Support/Fuego/Mechanism/Models
#Chemistry_Model := Davis
#Chemistry_Model := LuDME
Chemistry_Model := drm19

# This sets the transport directory in $(PELE_PHYSICS_HOME)/Transport
Transport_dir := EGLib
#Transport_dir := Simple

Bpack   := ./Make.package

# define the location of the PELE top directory
PELEC_HOME    := /home/bperry/PeleC/
include $(PELEC_HOME)/Exec/Make.PeleC

ifeq ($(DEBUG), TRUE)
DEFINES += -DDEBUG
endif
