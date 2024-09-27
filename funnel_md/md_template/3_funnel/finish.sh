echo non-Water | gmx_mpi trjconv -f funnel.xtc -o md_whole.xtc -s funnel.tpr -pbc nojump
echo non-Water | gmx_mpi trjconv -f npt.gro -o md_whole.gro -s funnel.tpr -pbc nojump
echo non-Water | gmx_mpi trjconv -f funnel.gro -o md_whole.gro -s funnel.tpr -pbc nojump
plumed driver --noatoms --plumed plumed_reweight.dat
gmx_mpi editconf -f md_whole.gro -o md_whole.pdb
