import mdtraj as md
import numpy as np
import itertools
import torch
import copy 
import json
import argparse

PARAMETERS = {}




def get_dihedral_idxs(traj, dihedral, idx, toid=False):
    atom_angles_peptoid = [["CLP", "NL", "CA", "CLP"], ["NL", "CA", "CLP", "NL"], ["CAY", "CA", "CLP", "NL", "CA"]]
    atom_angles_peptide = [["C", "N", "CA", "C"], ["N", "CA", "C", "N"]]
    dihedral_list = ["phi", "psi"]
    adjustment = [[-1, 0, 0, 0], [0, 0, 0, 1]]
    if toid:
        dihedral_list.append("omega")
        atom_angles = atom_angles_peptoid
        adjustment.append([-1, -1, -1, 0, 0])
    else:
        atom_angles = atom_angles_peptide
    k = dihedral_list.index(dihedral)
    at_angles = atom_angles[k]
    res_nums = np.array(adjustment[k]) + idx
    search_strings = [f"resid {rn} and name {ang}" for rn, ang in zip(res_nums, at_angles)]
    return traj.topology.select(" or ".join(search_strings)) + 1

def stringlist(l):
    return ",".join([str(s) for s in l])
def compute_all_distances(traj):
    idxs = np.arange(traj.top.n_atoms)
    grid = np.array(np.meshgrid(idxs, idxs)).T.reshape(-1, 2)
    pairs = grid[grid[:, 0] > grid[:, 1]]
    dists = md.compute_distances(traj, pairs)
    return dists
def loss_function(A, B, B_new):
    # Calculate vectors from A to all points
    
    vectors = B - A
    new_vector = B_new - A
    # Calculate similarities between new vector and all other vectors
    similarities = [torch.cosine_similarity(new_vector, v, dim=0) for v in vectors]
    return torch.mean(torch.stack(similarities))

def gradient_descent(A, B, learning_rate=0.3, num_iterations=100):
    
    B_new = torch.randn_like(A)  # Initialize B_new randomly
    B_new += A.detach()
    B_new.requires_grad = True
    optimizer = torch.optim.SGD([B_new], lr=learning_rate)
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = loss_function(A, B, B_new)
        loss.backward()
        optimizer.step()
    
    return B_new

def all_points_in_cone(p1, p2, r, q):
    """
    Determines if all points in q are inside a cone defined by points p1, p2, and radius r.
    
        Args:
        p1: Apex of the cone.
        p2: Center of the cone's base.
        r: Radius of the cone's base.
        q: Array of points to check.
        
        Returns:
        True if all points in q are inside the cone, False otherwise.
    """
    # Calculate cone axis vector
    axis = p1 - p2
    
    # Calculate vectors from p2 to q and p1
    v_pq = p1 - q
    # Calculate the angle between v_pq and axis
    cos_theta = np.dot(v_pq, axis) / (np.linalg.norm(v_pq, axis=1) * np.linalg.norm(axis))
    theta = np.arccos(cos_theta)
    # Calculate the cone's half angle
    cone_half_angle = np.arctan(r / np.linalg.norm(axis))
    
    # Check if point is within the cone's angle
    return np.all(theta < cone_half_angle)

def points_in_cylinder(p1, p2, r, points):
    """
    Determines which points in an array are inside a cylinder defined by points p1 and p2 with radius r.
    
        Args:
        p1: A numpy array representing the first endpoint of the cylinder.
        p2: A numpy array representing the second endpoint of the cylinder.
        r: The radius of the cylinder.
        points: A numpy array of points to check, where each row is a point.
        
        Returns:
        A numpy array of points that are inside the cylinder.
    """
    
    axis = p2 - p1
    axis_len_squared = np.dot(axis, axis)
    
    # Project all points onto the axis
    projections = np.dot(points - p1, axis) / axis_len_squared
    # Check if projected points are within the cylinder's end caps
    in_caps = np.logical_and(projections >= 0, projections <= 1)
    # Calculate distances from points to projected points
    proj_points = p1 + projections[:, np.newaxis] * axis
    distances = np.linalg.norm(points - proj_points, axis=1)
    # Check if distances are less than or equal to the radius
    in_radius = distances <= r
    # Return points that satisfy both conditions
    return np.where(in_radius)[0]

parser = argparse.ArgumentParser()
parser.add_argument(
        "--toid",
        action="store_true",
        help="Make a peptoid, not a peptide"
    )    
args = parser.parse_args()
#Load Structure
structure = md.load('npt.gro')
protein_ref = md.load('protein_ref.pdb')
prot_n_res = protein_ref.top.n_residues
protein = structure.atom_slice(structure.top.select(f'resid 0 to {prot_n_res-1}'))
#Get CA and heavy idxs
protein_ca_idxs = structure.top.select(f"resid 0 to {prot_n_res-1} and name CA")
PARAMETERS['protein_ca'] = stringlist(np.array(protein_ca_idxs) + 1)
protein_heavy_idxs = structure.top.select(f"resid 0 to {prot_n_res-1} and not element H")
n_prot_heavy = len(protein_heavy_idxs)

#Save starting CAs
protein_ca = structure.atom_slice(protein_ca_idxs)
protein_heavy = structure.atom_slice(protein_heavy_idxs)
protein_heavy.save_pdb('start.pdb')

#Get ligand idxs, COM
lig_idxs = structure.top.select(f"resid > {prot_n_res-1} and not water and not name NA and not name CL")
PARAMETERS['lig'] = stringlist(lig_idxs)
lig_heavy_idxs = structure.top.select(f"resid > {prot_n_res-1} and not element H and not water and not name NA and not name CL")
n_lig_heavy = len(lig_heavy_idxs)
PARAMETERS['lig_heavy'] = stringlist(lig_heavy_idxs)
ligand = structure.atom_slice(lig_idxs)
lig_com = np.round(md.compute_center_of_mass(ligand)[0], 3)
PARAMETERS['lig_com'] = stringlist(lig_com)

#Get closest distance
# Calculate pairwise distances between protein and ligand atoms
all_distances = md.compute_distances(structure, atom_pairs=list(itertools.product(protein_heavy_idxs, lig_heavy_idxs)))
min_index = np.unravel_index(all_distances.argmin(), (n_prot_heavy, n_lig_heavy))
prot_anchor = protein_heavy_idxs[min_index[0]] + 1
lig_anchor = lig_heavy_idxs[min_index[1]] + 1
PARAMETERS['prot_anchor'] = str(prot_anchor)
PARAMETERS['lig_anchor'] = str(lig_anchor)
MINS = min(max(-0.6, -1 * md.compute_rg(protein)[0] / 3), -1.5 * np.min(all_distances))

#Do gradient descent to find funnel axis pointing away from the protein
lc = torch.Tensor(lig_com)
lc.requires_grad = True
p = torch.Tensor(protein.xyz[0])
B_new = gradient_descent(lc, p)
bn = np.round(B_new.detach().numpy(), 3)
vec = (bn - lig_com) / np.linalg.norm(bn - lig_com)

#Get ligand size. The interaction distance is the ligand radius + epsilon
epsilon = 0.07
PARAMETERS['epsilon'] = str(epsilon)
lig_diameter = np.max(compute_all_distances(ligand))
PARAMETERS['lig_d'] = str(lig_diameter)
interaction_dist = lig_diameter / 2 + 3 * epsilon
PARAMETERS['interaction_dist'] = str(interaction_dist)
#Draw a cylinder from the ligand extending along the ligand axis, radius=interaction_distance. All atoms within this cylinder are interacting
interacting_idxs = points_in_cylinder(bn, lig_com + vec * MINS, lig_diameter / 2 + epsilon, protein.xyz[0])
interacting_protein = protein.atom_slice(interacting_idxs)

#Find the apex of the shortest cone that includes all interacting atoms inside it. 
zcc = copy.deepcopy(lig_com)
zcc += 0.5 * vec
for d in range(11, 36):
    zcc += 0.05 * vec
    cone_length = d * 0.05
    if all_points_in_cone(zcc, lig_com + vec * MINS, interaction_dist, protein.xyz[0, interacting_idxs]):
        break

if cone_length > 2:
    raise ValueError("Zcc too far away from the ligand center of mass. Funnel Construction Failed.")
PARAMETERS['zcc'] = stringlist(zcc)
PARAMETERS['cone_length'] = str(cone_length)
#cone angle is a function of zcc and the radius.



#Declare other MetaD parameters
RCYL = 0.4
MAXS = max(min(3.5, 2 * cone_length), 2.0)
alpha = (interaction_dist - RCYL) / cone_length
PARAMETERS['alpha'] = str(alpha)

RMSD_MAX = 0.1
KAPPA = 500000
BIASFACTOR = 20
NBIN = 500
HEIGHT = 3.0
SIGMA_lp = 0.35
SIGMA_ld = 0.05
SIGMA_d = 0.25
MINFS = 0.8 * MINS
MAXFS = 0.9 * MAXS
MAXFD = 1.1 * MAXFS
GRID_MIN = 0.0
GRID_MAX = 1.2 * MAXS

PARAMETERS['rcyl'] = str(RCYL)
PARAMETERS['maxs'] = str(MAXS)
PARAMETERS['mins'] = str(MINS)
PARAMETERS['rmsd_max'] = str(RMSD_MAX)
PARAMETERS['kappa'] = str(KAPPA)
PARAMETERS['biasfactor'] = str(BIASFACTOR)
PARAMETERS['nbin'] = str(NBIN)
PARAMETERS['height'] = str(HEIGHT)
PARAMETERS['sigma_d'] = str(SIGMA_d)
PARAMETERS['sigma_ld'] = str(SIGMA_ld)
PARAMETERS['sigma_lp'] = str(SIGMA_lp)
PARAMETERS['minfs'] = str(MINFS)
PARAMETERS['maxfs'] = str(MAXFS)
PARAMETERS['maxfd'] = str(MAXFD)
PARAMETERS['grid_min'] = str(GRID_MIN)
PARAMETERS['grid_max'] = str(GRID_MAX)
lig_diameter = round(lig_diameter, 3)
all_angles = []
angles = ["phi", "psi"]
if args.toid:
    angles.append("omega")
with open('plumed.dat', 'w') as f:
    f.write(f'WHOLEMOLECULES ENTITY0={stringlist(np.array(protein_ca_idxs) + 1)}\n')
    for angle in angles:
        for j in range(1, ligand.top.n_residues - 1):
            f.write(f"{angle}{j}: TORSION ATOMS={stringlist(get_dihedral_idxs(structure, angle, j + prot_n_res, args.toid))}\n")
            all_angles.append(f"{angle}{j}")
    n_ang = len(all_angles)
    f.write(f'd1: DISTANCE ATOMS={prot_anchor},{lig_anchor}\n')
    f.write(f'lig: COM ATOMS={stringlist(np.array(lig_heavy_idxs) + 1)}\n')
    f.write(f'fps: FUNNEL_PS LIGAND=lig REFERENCE=start.pdb ANCHOR={prot_anchor} POINTS={stringlist(lig_com)},{stringlist(bn)}\n')
    f.write('rmsd: RMSD REFERENCE=start.pdb TYPE=OPTIMAL\n')
    f.write(f'FUNNEL ARG=fps.lp,fps.ld ZCC={cone_length} ALPHA={round(alpha, 3)} RCYL={RCYL} MINS={round(MINS, 3)} MAXS={round(MAXS, 3)} KAPPA={KAPPA} NBINS={NBIN} NBINZ={NBIN} FILE=BIAS\n')
    sig_list = [str(SIGMA_lp)] + [str(SIGMA_ld)] + [str(SIGMA_d)]*(n_ang)
    npi_list = ['-pi']*n_ang
    pi_list = ['pi']*n_ang
    f.write(f'PBMETAD ARG=fps.lp,fps.ld,{stringlist(all_angles)} SIGMA={stringlist(sig_list)} HEIGHT={HEIGHT} PACE=500 TEMP=300 BIASFACTOR={BIASFACTOR} LABEL=metad GRID_MIN={round(MINS, 3)},0,{stringlist(npi_list)} GRID_MAX={round(MAXS, 3)},{round(lig_diameter, 3) + 2*epsilon},{stringlist(pi_list)}\n')
    f.write(f'LOWER_WALLS ARG=fps.lp AT={round(MINFS, 3)} KAPPA={10*KAPPA} EXP=2 OFFSET=0 LABEL=lwall\n')
    f.write(f'UPPER_WALLS ARG=rmsd AT={round(RMSD_MAX, 3)} KAPPA={10*KAPPA} EXP=2 OFFSET=0 LABEL=uwall-rmsd\n')
    f.write(f'UPPER_WALLS ARG=fps.lp AT={round(MAXFS, 3)} KAPPA={10*KAPPA} EXP=2 OFFSET=0 LABEL=uwall\n')
    f.write(f'UPPER_WALLS ARG=fps.ld AT={str(round(lig_diameter, 3))[:5]} KAPPA={10*KAPPA} EXP=2 OFFSET=0 LABEL=uwall-ld\n')
    f.write(f'UPPER_WALLS ARG=d1 AT={round(MAXFD, 3)} KAPPA={10*KAPPA} EXP=2 OFFSET=0 LABEL=distwall\n')   
    f.write('PRINT STRIDE=500 ARG=* FILE=COLVAR\n')

with open('plumed_reweight.dat', 'w') as f:
    f.write('RESTART\n')
    f.write('d1: READ FILE=COLVAR VALUES=d1 IGNORE_TIME IGNORE_FORCES EVERY=1\n')
    f.write('fps: READ FILE=COLVAR VALUES=fps.lp,fps.ld IGNORE_TIME IGNORE_FORCES EVERY=1\n')
    f.write('rmsd: READ FILE=COLVAR VALUES=rmsd IGNORE_TIME IGNORE_FORCES EVERY=1\n')
    for angle in angles:
        for j in range(1, ligand.top.n_residues - 1):
            f.write(f"{angle}{j}: READ FILE=COLVAR VALUES={angle}{j} IGNORE_TIME IGNORE_FORCES EVERY=1\n")
    f.write(f'FUNNEL ARG=fps.lp,fps.ld ZCC={cone_length} ALPHA={round(alpha, 3)} RCYL={RCYL} MINS={round(MINS, 3)} MAXS={round(MAXS, 3)} KAPPA={KAPPA} NBINS={NBIN} NBINZ={NBIN} FILE=bias_reweight.dat\n')
    
    f.write(f'PBMETAD ARG=fps.lp,fps.ld,{stringlist(all_angles)} SIGMA={stringlist(sig_list)} HEIGHT=0 PACE=500000 TEMP=300 BIASFACTOR={BIASFACTOR} LABEL=metad GRID_MIN={round(MINS, 3)},0,{stringlist(npi_list)} GRID_MAX={round(MAXS, 3)},{round(lig_diameter * 1.5, 3)},{stringlist(pi_list)}\n')
    f.write(f'LOWER_WALLS ARG=fps.lp AT={round(MINFS, 3)} KAPPA={10*KAPPA} EXP=2 OFFSET=0 LABEL=lwall\n')
    f.write(f'UPPER_WALLS ARG=rmsd AT={round(RMSD_MAX, 3)} KAPPA={10*KAPPA} EXP=2 OFFSET=0 LABEL=uwall-rmsd\n')
    f.write(f'UPPER_WALLS ARG=fps.lp AT={round(MAXFS, 3)} KAPPA={10*KAPPA} EXP=2 OFFSET=0 LABEL=uwall\n')
    f.write(f'UPPER_WALLS ARG=fps.ld AT={str(round(lig_diameter, 3))[:5]} KAPPA={10*KAPPA} EXP=2 OFFSET=0 LABEL=uwall-ld\n')
    f.write(f'UPPER_WALLS ARG=d1 AT={round(MAXFD, 3)} KAPPA={10*KAPPA} EXP=2 OFFSET=0 LABEL=distwall\n')   
    f.write(f'PRINT STRIDE=1 ARG=d1,fps.lp,fps.ld,{stringlist(all_angles)},rmsd,metad.bias FILE=colvar_reweight.dat\n')
    
with open("params.json", "w") as outfile: 
    json.dump(PARAMETERS, outfile)
