# AMReX
DIM = 2
COMP = gnu
PRECISION = DOUBLE

# Profiling
PROFILE = FALSE
TINY_PROFILE = FALSE
COMM_PROFILE = FALSE
TRACE_PROFILE = FALSE
MEM_PROFILE = FALSE
USE_GPROF = FALSE

# Performance
USE_MPI = FALSE
USE_OMP = FALSE
USE_CUDA = FALSE
USE_HIP = FALSE
USE_DPCPP = FALSE

# Debugging
DEBUG = FALSE
FSANITIZER = FALSE
THREAD_SANITIZER = FALSE

# PeleC
USE_MASA = FALSE
Eos_Model := Fuego
Chemistry_Model := LuChenDMEAr2015
Transport_Model := Simple

# GNU Make
Bpack := ./Make.package
Blocs := .
PELEC_HOME := ../../..
SUNDIALS_HOME ?= $(PELEC_HOME)/Submodules/sundials

# Related with kinsol
INCLUDE_LOCATIONS += $(SUNDIALS_HOME)/include
LIBRARIES += -lsundials_kinsol -lsundials_nvecserial -lsundials_nvecmanyvector

include $(PELEC_HOME)/Exec/Make.PeleC
