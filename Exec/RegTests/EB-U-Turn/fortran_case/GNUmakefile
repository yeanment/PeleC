PRECISION  = DOUBLE
PROFILE    = FALSE
DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE
USE_REACT  = TRUE

USE_EB = TRUE

HYP_TYPE = MOL

# This sets the EOS directory in $(PELE_PHYSICS_HOME)/Eos
#Eos_dir     := GammaLaw
Eos_dir     := Fuego

# This sets the network directory in $(PELE_PHYSICS_HOME)/Reactions
#Reactions_dir := Null_air
Reactions_dir := Fuego

# This sets the model directory in $(PELE_PHYSICS_HOME)/Support/Fuego/Mechanism/Models
Chemistry_Model := LiDryer

# This sets the transport directory in $(PELE_PHYSICS_HOME)/Transport
#Transport_dir := Constant
Transport_dir := EGLib

Bpack   := ./Make.package
Blocs   := .

# define the location of the PELE top directory
include $(PELEC_HOME)/Exec/Make.PeleC
