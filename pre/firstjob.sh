#!/bin/bash


#SBATCH -N 1

#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J control

#SBATCH --mail-user=afu2479@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A m3571
#SBATCH -t 0:30:00

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spreads

#run the application: 
mpirun -n 62 python control.py |tee job.out
