NAME=$1
PDB=$2
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#Make the structure

cd /project2/andrewferguson/berlaga/drugdiscovery/proteins
#Check for existence of protein PDB
PROT=${PDB}.pdb
if test -f "$PROT";
then
        echo $PROT
else
        if test -f "$PDB";
        then
                PROT=$PDB
                echo here2
        else
                echo hi
        fi
fi
OUT_NAME=${PROT::-4}_${NAME}

cd /project2/andrewferguson/berlaga/drugdiscovery/ligands

if test -f "${NAME}_smi.txt";
then
	echo "ligand already present"
else
	cd /project2/andrewferguson/berlaga/peptoids/structure_maker/
	rm molecule*.pdb
	python make_structure.py --seq $NAME --mini HELIX --file ${NAME}.pdb --nobackup --tide --cter NH2
	if [[ -f "${NAME}.pdb" ]]
	then
		mv ${NAME}.pdb /project2/andrewferguson/berlaga/drugdiscovery/ligands/${NAME}.pdb
		cd /project2/andrewferguson/berlaga/drugdiscovery/ligands
		obabel ${NAME}.pdb -O ${NAME}_noh.pdb -d
		python convert_to_smiles.py $NAME	
	else
		echo "structure generation didn't work"
		exit 1
	fi
fi
SMILES=$(cat ${NAME}_smi.txt)
#Run Chai
if ! [[ -f /project2/andrewferguson/berlaga/drugdiscovery/chai_results/$OUT_NAME/aggregate1.npy ]];
then
	cd /project2/andrewferguson/berlaga/drugdiscovery
	source /scratch/midway3/berlaga/miniconda3/bin/activate chai
	python run_chai.py ${PROT::-4} $NAME
else
	echo "Already Docked"
fi

