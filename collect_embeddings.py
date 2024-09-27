from pathlib import Path
import numpy as np
import torch
import subprocess
import sys
import os
from rdkit import Chem
from get_embeddings import run_inference
import argparse
# We use fasta-like format for inputs.
# Every record may encode protein, ligand, RNA or DNA
#  see example below
cd = '/project2/andrewferguson/berlaga/drugdiscovery'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fn",
    type=int,
    default=0,
    metavar="fn",
    help="File Num"
)
parser.add_argument(
    "--single",
    action="store_true",
    help="Overwrite existing files."
)
parser.add_argument(
    "--prot",
    type=str,
    default=None,
    metavar="prot",
    help="Input protein"
)
parser.add_argument(
    "--lig",
    type=str,
    default=None,
    metavar="seq",
    help="Input Sequence"
)
parser.add_argument(
    "--toid",
    action="store_true",
    help="Overwrite existing files."
)
args = parser.parse_args()
prot = args.prot
###PROCESS PROTEIN
if not prot.endswith('.pdb'):
    prot += '.pdb'
fasta = prot[:-4] + '.fasta'
if fasta not in os.listdir(os.path.join(cd, 'proteins')):
    if prot not in os.listdir(os.path.join(cd, 'proteins')):
        raise OSError(f'{prot} does not exist in the proteins directory')
    else:
        commands = ["obabel", os.path.join(cd, 'proteins', prot), "-O", os.path.join(cd, 'proteins', fasta)]
        subprocess.run(commands, stdout=subprocess.PIPE)
        
if args.fn != 0:
    with open(f'/project2/andrewferguson/berlaga/peptoids/ligs{args.fn}.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        try:
            lig = line.strip()
            if args.toid:
                lig += '_t'
            if not f'{lig}_smi.txt' in os.listdir(os.path.join(cd, 'ligands')):
                try:
                    Chem.MolFromSmiles(lig)
                    smiles_str = lig
                except:
                    raise ValueError('ligand is neither in the ligand directory nor a SMILES string')
            else:
                with open(os.path.join(cd, 'ligands', f'{lig}_smi.txt'), 'r') as f:
                    smiles_str = f.readlines()[0]
            with open(os.path.join(cd, 'proteins', fasta), 'r') as f:
                prot_fasta = f.readlines()[1]
            fasta_path = Path("/tmp/prot_lig.fasta")
            new_fasta = f"""
            >protein|current-protein
            {prot_fasta}
            >ligand|current-ligand
            {smiles_str}
            """.strip()
            fasta_path.write_text(new_fasta)
            
            output_dir = Path(os.path.join(cd, 'chai_results', f'{prot[:-4]}_{lig}'))
            trunk_repr = run_inference(
                fasta_file=fasta_path,
                output_dir=output_dir,
                # 'default' setup
                num_trunk_recycles=3,
                num_diffn_timesteps=200,
                seed=42,
                device=torch.device("cuda:0"),
                use_esm_embeddings=True,
            )
            
            torch.save(trunk_repr, f'chai_results/{prot[:-4]}_{lig}/trunk_repr.pt')
            # torch.save(template_dist, f'chai_results/{prot[:-4]}_{lig}/template_dist.pt')
            print(lig)
        except:
            print(f'{line.strip()} failed')
# Load pTM, ipTM, pLDDTs and clash scores for sample 2


###PROCESS LIGAND

# else:
#     with open(os.path.join(cd, 'ligands', f'{lig}_smi.txt'), 'r') as f:
#         smiles_str = f.readlines()[0]




    
