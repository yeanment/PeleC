#!/bin/bash

#SBATCH --account=exact
#SBATCH --nodes=1
###SBATCH --partition=debug
#SBATCH --partition=short
###SBATCH --time=1:00:00
#SBATCH --time=4:00:00
#SBATCH -o job.out.%j
#SBATCH --error=job.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bruce.perry@nrel.gov

module purge
MODULES=modules-2019-05-08
module unuse ${MODULEPATH}
module use /nopt/nrel/ecom/hpacf/compilers/${MODULES}
module use /nopt/nrel/ecom/hpacf/utilities/${MODULES}
module use /nopt/nrel/ecom/hpacf/software/${MODULES}/gcc-7.4.0

module load gcc
module load git
module load mpich/3.3
module load python/3.7.3

ranks_per_node=36
mpi_ranks=$(expr $SLURM_JOB_NUM_NODES \* $ranks_per_node)
export OMP_NUM_THREADS=1  # Max hardware threads = 4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

echo "Job name       = $SLURM_JOB_NAME"
echo "Num. nodes     = $SLURM_JOB_NUM_NODES"
echo "Num. MPI Ranks = $mpi_ranks"
echo "Num. threads   = $OMP_NUM_THREADS"
echo "Working dir    = $PWD"

srun -n ${mpi_ranks} -c 1 --cpu_bind=cores ../../pelec_config/PeleC2d.gnu.MPI.ex inputs

echo "finished"
