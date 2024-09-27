import sys, os
import copy
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import parmed as pmd
from openmmtools.forces import HarmonicRestraintBondForce
import mdtraj as md
import argparse
from parmed import gromacs
from sim_annealing_lib import run_annealing, resume_annealing

def restart(top_file,
                          struct_file,
                          steps=int(1e8),
                          num_copies=3,
                          first_copy=0,
                          config_save_prefix='',
            save_restart_freq=1e7):
    
    
    
    top = pmd.load_file(top_file)
    struc = pmd.load_file(struct_file)
    md_top = md.load(struct_file)
    helix_backbone_idxs = np.array([md_top.top.select('resid 0 to 13 and (name CA or name NL or name CLP)'), md_top.top.select('resid 14 to 27 and (name CA or name NL or name CLP)'), md_top.top.select('resid 28 to 41 and (name CA or name NL or name CLP)')])
    #Transfer unit cell information to topology object
    top.box = struc.box[:]
    platform = mm.Platform.getPlatformByName('CUDA')
    prop = {#'Threads': '1', #number of threads for CPU - all definitions must be strings (I think)
            'Precision': 'mixed', #for CUDA or OpenCL, select the precision (single, mixed, or double)
            'DeviceIndex': '0', #selects which GPUs to use - set this to zero if using CUDA_VISIBLE_DEVICES
            'DeterministicForces': 'True' #Makes sure forces with CUDA and PME are deterministic
            }
    systemRef = top.createSystem(nonbondedMethod=app.PME, #Uses PME for long-range electrostatics, simple cut-off for LJ
                                 nonbondedCutoff=10.0*u.angstroms, #Defines cut-off for non-bonded interactions
                                 rigidWater=True, #Use rigid water molecules
                                 constraints=app.HBonds, #Constrains all bonds involving hydrogens
                                 flexibleConstraints=False, #Whether to include energies for constrained DOFs
                                 removeCMMotion=True, #Whether or not to remove COM motion (don't want to if part of system frozen)
                                 )
    #Set up the integrator to use as a reference
    temperature = 320.0 * u.kelvin
    integratorRef = openmmtools.integrators.LangevinIntegrator(temperature,
                                                              collision_rate=collision_rate,
                                                               timestep=2.0 * u.femtoseconds,
                                                               splitting='V R O R V'
                                                              )
    # integratorRef = mm.LangevinMiddleIntegrator(temperature, #Temperature for Langevin
    #                                       1/u.picoseconds, #Friction coefficient
    #                                       2.0*u.femtoseconds, #Integration timestep
    #                                       )
    integratorRef.setConstraintTolerance(1.0E-08) 
    
    ee_save_prefix = config_save_prefix + str(first_copy) + '_anneal_prod'
    time_data = np.loadtxt(ee_save_prefix + "_out.txt") 
    anneal_data = np.loadtxt(str(first_copy) + "_annealing_output.txt")
    cur_traj = md.load(ee_save_prefix + ".xtc", top='solv.gro')
    
    start_step = int(time_data[-1, 0] % save_restart_freq)

    time_data = time_data[:start_step, :]
    anneal_data = anneal_data[:start_step, :]
    cur_traj = cur_traj[:start_step]
    

    np.savetxt(ee_save_prefix + "_out.txt", time_data)
    np.savetxt(str(first_copy) + "_annealing_output.txt", anneal_data)
    cur_traj.save_xtc(ee_save_prefix + ".xtc")

    restart_file = ee_save_prefix + "_restart.xml"
    _ = resume_annealing(restart_file,
                     start_step,
                     top,
                     systemRef,
                     integratorRef,
                     platform,
                     prop,
                     helix_backbone_idxs,
                     temperature_end=550*u.kelvin,
                     sim_steps=steps,
                     report_freq=int(1e4),
                     reporter_prefix=ee_save_prefix,
                     copy_prefix=str(first_copy),
                    )
                     
    
    for copy in range(first_copy + 1, first_copy + num_copies):
        ee_save_prefix = config_save_prefix + str(copy) + '_anneal_prod'
        _ = run_annealing(top, 
                    systemRef, 
                    integratorRef, 
                    platform, 
                    prop, 
                    helix_backbone_idxs,
                    temperature, 
                    temperature_end=550*u.kelvin,
                    state="_npt_equilState.xml",
                    sim_steps=steps,
                    report_freq=int(1e4),
                    reporter_prefix=ee_save_prefix,
                    copy_prefix=str(copy),
                    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="File Acceptor for MD sim"
    )
    parser.add_argument(
        "--top",
        type=str,
        default="topol.top",
        metavar="top",
        help="topology (.top file)"
    )
    parser.add_argument(
        "--struct",
        type=str,
        default="solv.gro",
        metavar="struct",
        help="structure (.gro file)"
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=3,
        metavar="num_copies",
        help="Number of simulations per gpu"
    )
    parser.add_argument(
        "--fc",
        type=int,
        default=0,
        metavar="first_copy",
        help="first iteration"
    )
    
    args = parser.parse_args()
    restart(args.top,
                          args.struct,
                          steps=int(1e8),
                          num_copies=args.nc,
                          first_copy=args.fc,
                          config_save_prefix='')
    
    traj = None
    for copy in range(args.fc, args.fc + args.nc):
        strcp = str(copy)
        traj = None
        for traj_frag in md.iterload(strcp + '_anneal_prod.xtc', top = 'solv.gro', chunk=10000):
            non_solvent_atoms = traj_frag.topology.select("not water")
            if traj is None:
                traj = traj_frag.atom_slice(non_solvent_atoms)
            else:
                traj = traj.join(traj_frag.atom_slice(non_solvent_atoms))
        traj.save_xtc(strcp + "_anneal_whole.xtc")
        traj[-1].save_pdb(strcp + "_anneal_whole.pdb")
