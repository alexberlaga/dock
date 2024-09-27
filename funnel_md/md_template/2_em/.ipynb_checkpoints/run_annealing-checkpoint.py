import sys, os
import copy
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import parmed as pmd
from openmmtools.forces import HarmonicRestraintBondForce
from openmmtools.integrators import LangevinIntegrator
import mdtraj as md
import argparse
from parmed import gromacs
from sim_annealing_lib import run_annealing
from peptoid_nve_lib import run_nvt, run_npt

gromacs.GROMACS_TOPDIR = os.environ.get("GMXLIB")


# runs an NVT equilibration, then NPT equilibration, then expanded ensemble
def run_workflow_anneal(top_file,
                          struct_file,
                          do_equilibration=True,
                          steps=int(2e8),
                          num_copies=3,
                          first_copy=0,
                          config_save_prefix='',
                          ): 
    md_top = md.load(struct_file)
    helix_backbone_idxs = np.array([md_top.top.select('resid 0 to 13 and name CA'), md_top.top.select('resid 14 to 27 and name CA'), md_top.top.select('resid 28 to 41 and name CA')])
    struc = pmd.load_file(struct_file)
    
    # print((mm.Vec3(vecs[0, 0], vecs[0, 1], vecs[0, 2]), mm.Vec3(vecs[1, 0], vecs[1,1], vecs[1,2]), mm.Vec3(vecs[2, 0], vecs[2, 1], vecs[2, 2])))
    top = pmd.load_file(top_file)
 
    #Transfer unit cell information to topology object
    top.box = struc.box[:]
    nonwater_hydrogens = md_top.top.select("element H and not water")
    print('box dimensions upon restart are', top.box)

    #Set up some global features to use in all simulations
    temperature = 320.0*u.kelvin

    #Define the platform (i.e. hardware and drivers) to use for running the simulation
    #This can be CUDA, OpenCL, CPU, or Reference 
    #CUDA is for NVIDIA GPUs
    #OpenCL is for CPUs or GPUs, but must be used for old CPUs (not SSE4.1 compatible)
    #CPU only allows single precision (CUDA and OpenCL allow single, mixed, or double)
    #Reference is a clear, stable reference for other code development and is very slow, using double precision by default
    platform = mm.Platform.getPlatformByName('CUDA')
    prop = {#'Threads': '1', #number of threads for CPU - all definitions must be strings (I think)
           'Precision': 'mixed', #for CUDA or OpenCL, select the precision (single, mixed, or double)
           'DeviceIndex': '0', #selects which GPUs to use - set this to zero if using CUDA_VISIBLE_DEVICES
           'DeterministicForces': 'False' #Makes sure forces with CUDA and PME are deterministic
            }
    plat2 = mm.Platform.getPlatformByName('CPU')
    prop2 = {'Threads': '8'}
    collision_rate = 1.0 / u.picosecond
    #Create the OpenMM system that can be used as a reference
    systemRef = top.createSystem(nonbondedMethod=app.PME, #Uses PME for long-range electrostatics, simple cut-off for LJ
                                 nonbondedCutoff=10.0*u.angstroms, #Defines cut-off for non-bonded interactions
                                 rigidWater=True, #Use rigid water molecules
                                 constraints=app.HBonds, #Constrains all bonds
                                 flexibleConstraints=False,
                                 removeCMMotion=True, #Whether or not to remove COM motion (don't want to if part of system frozen)
                                 hydrogenMass=1.0*u.amu
                                 )
    # forces = {systemRef.getForce(index).__class__.__name__: systemRef.getForce(index) for index in range(systemRef.getNumForces())}
    # forces['NonbondedForce'].setEwaldErrorTolerance(1.0E-05)
    # nonwater_hydrogens = md_top.top.select("element H and not water")
    # for i in range(systemRef.getForce(0).getNumBonds()):
    #     params = systemRef.getForce(0).getBondParameters(i)
    #     if params[0] in nonwater_hydrogens or params[1] in nonwater_hydrogens:
    #         systemRef.getForce(0).setBondParameters(i, params[0], params[1], params[2], 0 * u.kilojoules/(u.nanometer**2*u.mole))
    nvt_equil_integrator = integratorRef = mm.LangevinMiddleIntegrator(temperature,
                                                              # collision_rate=collision_rate,
                                                collision_rate,
                                                1.0 * u.femtoseconds
                                                               # timestep=4.0 * u.femtoseconds,
                                                              )
    npt_equil_integrator = integratorRef = mm.LangevinMiddleIntegrator(temperature,
                                                              # collision_rate=collision_rate,
                                                collision_rate,
                                                1.0 * u.femtoseconds
                                                               # timestep=4.0 * u.femtoseconds,
                                                              )
    #Set up the integrator to use as a reference
    integratorRef = mm.LangevinMiddleIntegrator(temperature,
                                                              # collision_rate=collision_rate,
                                                collision_rate,
                                                2.0 * u.femtoseconds
                                                               # timestep=4.0 * u.femtoseconds,
                                                              )
        
    integratorRef.setConstraintTolerance(1.0E-06)    
    # integratorRef = mm.LangevinMiddleIntegrator(temperature, #Temperature for Langevin
    #                                       1/u.picoseconds, #Friction coefficient
    #                                       2.0*u.femtoseconds, #Integration timestep
    #                                       )
  
   
    if do_equilibration:
        # NVT equilibration
        state_file_NVT, state_NVT = run_nvt(top, 
                                            systemRef, 
                                            nvt_equil_integrator, 
                                            platform, 
                                            prop, 
                                            temperature, 
                                            simSteps=int(5e4), # 100ps
                                            save_freq=int(1e2), # save every 2ps
                                            pos=struc.positions,
                                            set_box=None,
                                            save_prefix=config_save_prefix+'_nvt_equil',
                                            do_minimization=True,
                                            previous_barostat=False)
        
        # NPT equilibration
        npt_save_prefix = config_save_prefix+'_npt_equil'
        state_file_NPT, state_NPT = run_npt(top, 
                                            systemRef, 
                                            npt_equil_integrator, 
                                            platform, 
                                            prop, 
                                            temperature, 
                                            simSteps=int(5e4), # 100ps
                                            save_freq=int(1e3), # save every 2ps
                                            state=state_file_NVT,
                                            save_prefix=npt_save_prefix)
 
    # expanded-ensemble
    if do_equilibration:
        current_state = state_file_NPT
    else:
        current_state = '_npt_equilState.xml'

    for copy in range(first_copy, first_copy + num_copies):
        ee_save_prefix = config_save_prefix + str(copy) + '_anneal_prod'
        # _ = run_annealing(top, 
        #             systemRef, 
        #             integratorRef, 
        #             plat2, 
        #             prop2, 
        #             helix_backbone_idxs,
        #             temperature, 
        #             temperature_end=temperature,
        #             state=current_state,
        #             sim_steps=int(2.5e4),
        #             report_freq=int(1e4),
        #             reporter_prefix=ee_save_prefix +"_before",
        #             copy_prefix=str(copy),
        #             )
        _ = run_annealing(top, 
                    systemRef, 
                    integratorRef, 
                    platform, 
                    prop, 
                    helix_backbone_idxs,
                    temperature, 
                    temperature_end=550*u.kelvin,
                    # state=ee_save_prefix +"_beforeState.xml",
                    state=current_state,
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
    
    run_workflow_anneal(args.top,
                          args.struct,
                          do_equilibration=True,
                          steps=int(2e8),
                          num_copies=args.nc,
                          first_copy=args.fc,
                          config_save_prefix='')
    
    traj = None
    for copy in range(args.fc, args.fc + args.nc):
        traj = None
        strcp = str(copy)
        for traj_frag in md.iterload(strcp + '_anneal_prod.xtc', top = 'solv.gro', chunk=10000):
            non_solvent_atoms = traj_frag.topology.select("not water")
            if traj is None:
                traj = traj_frag.atom_slice(non_solvent_atoms)
            else:
                traj = traj.join(traj_frag.atom_slice(non_solvent_atoms))
        traj.save_xtc(strcp + "_anneal_whole.xtc")
        traj[-1].save_pdb(strcp + "_anneal_whole.pdb")
