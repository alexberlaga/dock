#!/bin/bash
# email on start, end, and abortion

#SBATCH --partition=gm4-pmext
#SBATCH --qos=gm4
#SBATCH --nodes=1            # SET NUM NODES 
#SBATCH --exclude=midway2-0647
#SBATCH --ntasks-per-node=1  # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --cpus-per-task=10    # SET NUM THREADS 
#SBATCH --account=pi-andrewferguson
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

gmx_mpi mdrun -deffnm funnel -v -plumed plumed.dat -cpi funnel.cpt -ntomp 1
echo non-Water | gmx_mpi trjconv -f funnel.xtc -o md_whole.xtc -s funnel.tpr -pbc nojump
echo non-Water | gmx_mpi trjconv -f npt.gro -o md_whole.gro -s funnel.tpr -pbc nojump
echo non-Water | gmx_mpi trjconv -f funnel.gro -o md_whole.gro -s funnel.tpr -pbc nojump
plumed driver --noatoms --plumed plumed_reweight.dat
gmx_mpi editconf -f md_whole.gro -o md_whole.pdb
