import mdtraj as md
import numpy as np
import nglview as nv
import sys
import argparse

def get_dihedral(traj, dihedral, idx, toid=False):
    atom_angles_peptoid = [["CLP", "NL", "CA", "CLP"], ["NL", "CA", "CLP", "NL"], ["CAY", "CA", "CLP", "NL", "CA"]]
    atom_angles_peptide = [["C", "N", "CA", "C"], ["N", "CA", "C", "N"], ["CH3", "CA", "C", "N", "CA"]]
    dihedral_list = ["phi", "psi", "omega"]
    adjustment = [[-1, 0, 0, 0], [0, 0, 0, 1], [-1, -1, -1, 0, 0]]
    if toid:
        atom_angles = atom_angles_peptoid
    else:
        atom_angles = atom_angles_peptide
    k = dihedral_list.index(dihedral)
    at_angles = atom_angles[k]
    res_nums = np.array(adjustment[k]) + idx
    search_strings = [f"resid {rn} and name {ang}" for rn, ang in zip(res_nums, at_angles)]
    search_indices = traj.topology.select(" or ".join(search_strings))
    return md.compute_dihedrals(traj, [search_indices])

def get_all_dihedrals(traj, toid=False):
    dihedrals = np.zeros((traj.top.n_residues, 3))
    for i in range(1, traj.top.n_residues - 1):
        for j, d in enumerate(["phi", "psi", "omega"]):
            dihedrals[i, j] = get_dihedral(traj, d, i, toid)[0,0] * 180 / np.pi
    dihedrals[0] = dihedrals[1]
    dihedrals[traj.top.n_residues - 1] = dihedrals[traj.top.n_residues - 2]
    return dihedrals

parser = argparse.ArgumentParser()
parser.add_argument(
        "--toid",
        action="store_true",
        help="Make a peptoid, not a peptide"
    )    
args = parser.parse_args()
structure = md.load('pred.model_trunk_0_idx_0.pdb')
prot = structure.atom_slice(structure.top.select('chainid 0'))
lig = structure.atom_slice(structure.top.select('chainid 1'))
lig.save_pdb('rank1.pdb')
prot.save_pdb('protein_coords.pdb')

ref_lig = md.load('ref_lig.pdb')
non_h_indices = ref_lig.top.select("not element H")
table, bonds = lig.top.to_dataframe()
table2, bonds2 = ref_lig.atom_slice(non_h_indices).top.to_dataframe()
table[['name', 'resSeq']] = table2[['name', 'resSeq']]
# table['name'].iloc[-1] = 'CA'
lig.topology = md.Topology.from_dataframe(table, bonds)

dihedrals = get_all_dihedrals(lig, args.toid)
np.savetxt("dihedrals.txt", dihedrals)
