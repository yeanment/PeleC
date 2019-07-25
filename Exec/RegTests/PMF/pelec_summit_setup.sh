#!/bin/bash -l

# Example script for setting up PeleC cuda_acc branch and compiling on Summit

cmd() {
  echo "+ $@"
  eval "$@"
}

cmd "module unload xl && module load pgi/19.5 cuda/10.1.168"
cmd "git clone -b cuda_acc --recursive https://github.com/AMReX-Combustion/PeleC.git"
cmd "cd PeleC/Submodules/AMReX && git apply ../../Exec/RegTests/PMF/amrex_pelec_cuda_acc.patch && cd -"
cmd "cd PeleC/Exec/RegTests/PMF && COMP=pgi USE_CUDA=TRUE USE_ACC=TRUE USE_MPI=TRUE CUDA_ARCH=70 NVCC_HOST_COMP=pgi nice make -j16"

# Notes:
# Edit PeleC/Exec/RegTests/PMF/GNUmakefile to change USE_ACC=TRUE/FALSE and USE_CUDA=TRUE/FALSE or whatever else
# "make realclean" is the clean command
