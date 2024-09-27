from pathlib import Path
import numpy as np
import torch
import subprocess
import sys
import os
from rdkit import Chem
from chai_lab.chai1 import run_inference

# We use fasta-like format for inputs.
# Every record may encode protein, ligand, RNA or DNA
#  see example below
cd = '/project2/andrewferguson/berlaga/drugdiscovery'
prot = sys.argv[1]
lig = sys.argv[2]
try:
    name = sys.argv[3]
except:
    name = sys.argv[2]

###PROCESS PROTEIN
if len(prot) < 8:
    if not prot.endswith('.pdb'):
        prot += '.pdb'
    
    fasta = prot[:-4] + '.fasta'
    if fasta not in os.listdir(os.path.join(cd, 'proteins')):
        if prot not in os.listdir(os.path.join(cd, 'proteins')):
            raise OSError(f'{prot} does not exist in the proteins directory')
        else:
            commands = ["obabel", os.path.join(cd, 'proteins', prot), "-O", os.path.join(cd, 'proteins', fasta)]
            subprocess.run(commands, stdout=subprocess.PIPE)
    with open(os.path.join(cd, 'proteins', fasta), 'r') as f:
        prot_fasta = f.readlines()[1]
else:
    prot_fasta = prot
###PROCESS LIGAND
if not f'{lig}_smi.txt' in os.listdir(os.path.join(cd, 'ligands')):
    try:
        Chem.MolFromSmiles(lig)
        smiles_str = lig
    except:
        raise ValueError('ligand is neither in the ligand directory nor a SMILES string')
else:
    with open(os.path.join(cd, 'ligands', f'{lig}_smi.txt'), 'r') as f:
        smiles_str = f.readlines()[0]
    
new_fasta = f"""
>protein|current-protein
{prot_fasta}
>ligand|current-ligand
{smiles_str}
""".strip()

fasta_path = Path("/tmp/prot_lig.fasta")
fasta_path.write_text(new_fasta)

if len(sys.argv[1]) < 8:
    protname = sys.argv[1]
else:
    try:
        protname = sys.argv[3]
    except:
        protname = f'newprot_{sys.argv[1][:4]}'

output_dir = Path(os.path.join(cd, 'chai_results', f'{protname}_{name}'))
output_pdb_paths = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # 'default' setup
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device=torch.device("cuda:0"),
    use_esm_embeddings=True,
)

# Load pTM, ipTM, pLDDTs and clash scores for sample 2
