import nglview as nv
import mdtraj as md
import numpy as np

ref_structure =  md.load('ref_structure.pdb')
lig = md.load_pdb('lig.pdb', standard_names=False)
non_h_indices = lig.top.select('not element H')
lig.superpose(ref_structure, atom_indices=non_h_indices, ref_atom_indices=np.arange(ref_structure.top.n_atoms))
lig.save_gro('lig.gro')

protein = md.load('prot.pdb')
protein = protein.atom_slice(protein.top.select('not name OXT'))
coords = md.load('protein_coords.pdb')
protein.xyz = coords.xyz
protein.save_pdb('prot.pdb')