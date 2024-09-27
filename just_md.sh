NAME=$1
PDB=$2
PROT=${PDB}.pdb
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#Make the structure

OUT_NAME=${PDB}_${NAME}


cd /project2/andrewferguson/berlaga/drugdiscovery/funnel_md
cp -r md_template $OUT_NAME
cp ../proteins/$PROT $OUT_NAME/a_insert/prot.pdb
cp ../proteins/$PROT $OUT_NAME/3_funnel/protein_ref.pdb
cd ../chai_results/$OUT_NAME
cp ../../ligands/$NAME.pdb ./ref_lig.pdb
val=$(find . -type f -name "pred.model*" | wc -l)
if [[ val == 0 ]];
then
	echo "Chai Failed"
	exit 1
fi
cp ../new_ligand.py .
python new_ligand.py
cp rank1.pdb /project2/andrewferguson/berlaga/drugdiscovery/funnel_md/$OUT_NAME/a_insert/ref_structure.pdb
cp protein_coords.pdb /project2/andrewferguson/berlaga/drugdiscovery/funnel_md/$OUT_NAME/a_insert/
#GET CORRECT DIHEDRALS
cd /project2/andrewferguson/berlaga/peptoids/structure_maker
python make_structure.py --seq $NAME --anglefile /project2/andrewferguson/berlaga/drugdiscovery/chai_results/$OUT_NAME/dihedrals.txt --file ${NAME}.pdb --tide --nobackup --cter NH2
if [[ -f ${NAME}.pdb ]];
then
	mv ${NAME}.pdb /project2/andrewferguson/berlaga/drugdiscovery/funnel_md/$OUT_NAME/a_insert/lig.pdb
else
	echo "structure generation #2 failed"
	exit 1
fi
cd /project2/andrewferguson/berlaga/drugdiscovery/funnel_md/$OUT_NAME/a_insert/

#REGULAR MD 
source /project2/andrewferguson/berlaga/new_gmx/bin/GMXRC
python superpose.py
gmx_mpi editconf -f lig.gro -o lig.pdb
sed -i "$(($(wc -l < prot.pdb)-1)),\$d" prot.pdb
sed -i "1,2d" lig.pdb
cat lig.pdb >> prot.pdb
python pdb2gmx.py
gmx_mpi insert-molecules -ci prot_lig.gro -o box.gro -nmol 1 -box 7.0 7.0 7.0
gmx_mpi editconf -f box.gro -o centered.gro -c
cd ..
cp a_insert/centered.gro b_em
cp a_insert/topol* b_em

cd b_em

gmx_mpi grompp -f em.mdp -c centered.gro -r centered.gro -p topol.top -o em.tpr -maxwarn 3

gmx_mpi mdrun -ntomp 1 -v -deffnm em
echo 0 | gmx_mpi trjconv -f em.gro -o em.gro -s em.tpr -pbc whole
cd ..

cp b_em/em.gro 1_solvate
cp b_em/topol* 1_solvate



cd 1_solvate

gmx_mpi solvate -cp em.gro -cs spc216.gro -o solv.gro -p topol.top
gmx_mpi grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr
echo SOL | gmx_mpi genion -s ions.tpr -o solv.gro -p topol.top -neutral

cd ..

cp 1_solvate/solv.gro 2_em
cp 1_solvate/topol* 2_em
cd 2_em

gmx_mpi grompp -f em.mdp -c solv.gro -p topol.top -o em.tpr -maxwarn 3

gmx_mpi mdrun -ntomp 20 -v -deffnm em
gmx_mpi grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 3
gmx_mpi mdrun -ntomp 10 -v -deffnm nvt
gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 3
gmx_mpi mdrun -ntomp 10 -v -deffnm npt
cp npt.gro topol* npt.cpt ../3_funnel
cd ../3_funnel
source /scratch/midway3/berlaga/miniconda3/bin/activate djax
python make_funnel.py
gmx_mpi grompp -f npt_run.mdp -c npt.gro -t npt.cpt -p topol.top -o funnel.tpr -maxwarn 2
gmx_mpi mdrun -deffnm funnel -v -plumed plumed.dat -ntomp 1
echo non-Water | gmx_mpi trjconv -f funnel.xtc -o md_whole.xtc -s funnel.tpr -pbc nojump
echo non-Water | gmx_mpi trjconv -f funnel.gro -o md_whole.gro -s funnel.tpr -pbc nojump
rm -f colvar_reweight.dat bias_reweight.dat
plumed driver --noatoms --plumed plumed_reweight.dat
gmx_mpi editconf -f md_whole.gro -o md_whole.pdb
