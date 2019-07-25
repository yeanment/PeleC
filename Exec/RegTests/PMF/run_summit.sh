#!/bin/bash -l

#BSUB -P CMB138
#BSUB -J pelec_gpu
#BSUB -o pelec_gpu.o%J
#BSUB -W 00:10
#BSUB -nnodes 1
#BSUB -alloc_flags "smt1"
#BSUB -alloc_flags "gpumps"

module unload xl 

#Necessary for serial GPU
#module load pgi/18.10
#module load cuda/9.2.148

#Necessary for MPI with GPU
module load pgi/19.5
module load cuda/10.1.168

#7 MPI ranks per GPU with MPS
#jsrun -n 6 -a 7 -c 7 -g 1 -r 6 -l CPU-CPU -d packed -b packed:1 ./PeleC3d.pgi.MPI.ACC.CUDA.ex inputs-3d-acc pelec.do_gpu_react=1 pelec.chem_integrator=2

#1 MPI rank per GPU
#jsrun -n 6 -a 1 -c 1 -g 1 -r 6 -l CPU-CPU -d packed -b packed:1 ./PeleC3d.pgi.MPI.ACC.CUDA.ex inputs-3d-acc pelec.do_gpu_react=1 pelec.chem_integrator=2

#Serial GPU
#jsrun -n 1 -a 1 -c 1 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1 ./PeleC3d.pgi.ACC.CUDA.ex inputs-3d-acc pelec.do_gpu_react=1 pelec.chem_integrator=2
