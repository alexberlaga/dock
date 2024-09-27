#!/bin/bash

# email on start, end, and abortion
#SBATCH -J AAAAA.an$1

#SBATCH -o out.out
#SBATCH -e error.out
#SBATCH --partition=gpu
#SBATCH --nodes=1            # SET NUM NODES 
#SBATCH --ntasks-per-node=1  # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --cpus-per-task=4    # SET NUM THREADS 
#SBATCH --account=pi-andrewferguson
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

# THIS EXAMPLE USES 1 GPU NODE - 1 MPI TASK - 4 THREADS PER TASK

# SET NUMBER OF MPI TASKS 
# SET NUMBER OF MD STEPS

#LOAD GROMACS MODULE 
module load cuda/11.2
source /scratch/midway3/berlaga/miniconda3/bin/activate sams
python run_annealing.py --top topol.top --struct solv.gro --nc 3 --fc $1


