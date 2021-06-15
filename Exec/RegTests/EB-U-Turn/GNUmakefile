# AMReX
DIM = 3
COMP = gcc
PRECISION = DOUBLE

# Profiling
PROFILE = FALSE
TINY_PROFILE = FALSE
COMM_PROFILE = FALSE
TRACE_PROFILE = FALSE
MEM_PROFILE = FALSE
USE_GPROF = FALSE

# Performance
USE_MPI = TRUE
USE_OMP = FALSE
USE_CUDA = TRUE
USE_HIP = FALSE
USE_DPCPP = FALSE

# Debugging
DEBUG = FALSE
FSANITIZER = FALSE
THREAD_SANITIZER = FALSE

# PeleC
USE_REACT = FALSE
USE_FORCING = FALSE
USE_EB = TRUE
Eos_dir := Fuego
#Transport_dir := EGLib
Transport_dir := Simple
Chemistry_Model := LiDryer

# GNU Make
Bpack := ./Make.package
Blocs := .
PELEC_HOME := ../../..
include $(PELEC_HOME)/Exec/Make.PeleC
