import numpy as np
import mdtraj as md
import subprocess
import sys
import os
import itertools

N_RUNS = 100

def _osremove(f):
    if os.path.isfile(f):
        os.remove(f)
        
def mdrun(deffnm, plumed=False, plumed_file='plumed.dat', np=1, nsteps=100000, checkpoint=False, checkpoint_file="md.cpt"):
    """
    Python wrapper for gmx mdrun -deffnm

    Parameters
    ----------
    deffnm : str
         File names for md run
    mpirun : bool
        Is this a multi-node run or not gmx (False) vs gmx_mpi (Default: True)
        number of processes (np)
    plumed : bool
        Turns plumed on/off
    np : int
        Number of processes to run mpirun on (Default 1 for non-mpi run)
    """
   
    
    if plumed:
        commands = ["gmx_mpi", "mdrun", "-deffnm", deffnm, "-ntomp", str(np),  "-nsteps", str(nsteps)]
    else:
        commands = ["gmx_mpi", "mdrun", "-deffnm", deffnm, "-ntomp", str(np), "-v", "-nsteps", str(nsteps)]

    if plumed:
        commands.extend(["-plumed", plumed_file])
    if checkpoint:
        commands.extend(["-cpi", checkpoint_file])
    subprocess.run(commands)
    return

protein_ref = md.load('protein_ref.pdb')
prot_n_res = protein_ref.atom_slice(protein_ref.top.select('protein')).top.n_residues
for i in range(N_RUNS):
    if i == 0:
        mdrun(f"md", np=os.environ.get("SLURM_CPUS_PER_TASK", 1), nsteps=500000, checkpoint=False)
    else:
        mdrun(f"md", np=os.environ.get("SLURM_CPUS_PER_TASK", 1), nsteps=500000, checkpoint=True, checkpoint_file=f"md.cpt")

    structure = md.load('md.gro')
    protein_idxs = structure.top.select(f"resid 0 to {prot_n_res-1}")
    ligand_idxs = structure.top.select("not protein and not water and not name NA and not name CL")
    dist_pairs = [p for p in itertools.product(protein_idxs, ligand_idxs)]
    min_dist = np.min(md.compute_distances(structure, dist_pairs))
    if min_dist > 0.75:
        break
   
    _osremove(f"md.gro")
