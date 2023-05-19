#!/bin/sh

#SBATCH --nodes=2
#SBATCH -p short
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20
#SBATCH -t 00:05:00

#export OMP_NUM_THREADS=2  # In case you need OpenMP threads, remember this goes along the cpus-per-task option

# The following runs a script named program in the current working directory
for i in {1..24}; do
    srun -N 1 -n 1 ./program input${i} >& out${i} &
done

wait
