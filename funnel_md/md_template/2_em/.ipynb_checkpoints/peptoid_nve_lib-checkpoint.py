#!/usr/bin/env python

from __future__ import division, print_function

import sys, os
import copy
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import parmed as pmd
from parmed.openmm.reporters import NetCDFReporter

def get_avg_box_l(fname_out):
    vol = np.loadtxt(fname_out)[:, -2]
    vol_mean = np.mean(vol[int(len(vol)/2):])
    return np.power(vol_mean, 1./3.)

def get_avg_energy(fname_out):
    total_energy = np.loadtxt(fname_out)[:, 4]
    return np.mean(total_energy[int(len(total_energy)/2):])

def run_nve(top,
            systemRef,
            integratorRef,
            platform,
            prop,
            simSteps=2000,
            e_to_set=None,
            pos=None,
            state=None,
            previous_barostat=False,
            save_dcd=False,
            save_prefix='nve',
            save_freq=4):

    #Copy the reference system
    system = copy.deepcopy(systemRef)

    # state will include a MonteCarloPressure so we need to include 
    # a MonteCarloBarostat here, but we'll set the frequency to 0 to turn it off
    # so the parameters don't matter here
    if previous_barostat:
        barostat = mm.MonteCarloBarostat(1.0*u.bar,
                                         300.0*u.kelvin, 
                                         100 #Time-steps between MC moves
                                         )
        system.addForce(barostat)
        barostat.setFrequency(0)


    integrator = mm.VerletIntegrator(2.0*u.femtoseconds)
    integrator.setConstraintTolerance(1.0E-08)

    #Create the simulation object for NVE simulation
    sim = app.Simulation(top.topology, system, integrator, platform, prop, state)

    if pos is not None:
        sim.context.setPositions(pos * u.angstroms)

    if e_to_set is not None:
        potential_energy = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        ke_to_set = e_to_set - potential_energy

        n_atoms = 0
        for res in top.residues:
            if res.name not in ['SOL']:
                for atom in res.atoms:
                    if system.getParticleMass(atom.idx) > 1e-10:
                        n_atoms += 1
                    else:
                        print('Found an atom with 0 mass. Excluding from DOF calculation.')
            else:
                n_atoms += 2
        print('n_atoms', n_atoms)
        temp_to_set = 2 * ke_to_set / 0.008314 / (3 * (n_atoms - 1))
        print('Setting velocities to temperature', temp_to_set)
        sim.context.setVelocitiesToTemperature(temp_to_set)
        
    #Set up the reporter to output energies, volume, etc.
    sim.reporters.append(app.StateDataReporter(save_prefix+'_out.txt', #Where to write - can be stdout or file name (default .csv, I prefer .txt)
                                               save_freq, #Number of steps between writes
                                               step=True, #Write step number
                                               time=True, #Write simulation time
                                               potentialEnergy=True, #Write potential energy
                                               kineticEnergy=True, #Write kinetic energy
                                               totalEnergy=True, #Write total energy
                                               temperature=True, #Write temperature
                                               volume=True, #Write volume
                                               density=False, #Write density
                                               speed=True, #Estimate of simulation speed
                                               separator='  ' #Default is comma, but can change if want (I like spaces)
                                               )
                         )

    if save_dcd:
        sim.reporters.append(app.dcdreporter.DCDReporter(save_prefix + '.dcd',
                                                         save_freq,
                                                         enforcePeriodicBox=False))
    else:
        #Set up reporter for printing coordinates (trajectory)
        sim.reporters.append(NetCDFReporter(save_prefix + '.nc', #File name to write trajectory to
                                            save_freq, #Number of steps between writes
                                            crds=True, #Write coordinates
                                            vels=False, #Write velocities
                                            frcs=False #Write forces
                                        )
                         )

    #Run NVT dynamics
    print("\nRunning NVE simulation...")
    sim.context.setTime(0.0)
    sim.step(simSteps)

    #Save simulation state if want to extend, etc.
    fname_xml = save_prefix + 'State.xml'

    sim.saveState(fname_xml)

    #Get the final positions and velocities
    return fname_xml, sim.context.getState(getPositions=True, 
                                           getVelocities=True, 
                                           getParameters=True, 
                                           enforcePeriodicBox=True)

def run_nvt(top,
            systemRef,
            integratorRef,
            platform,
            prop,
            temperature,
            simSteps=50000,
            save_freq=1000,
            state=None,
            pos=None,
            vels=None,
            set_box=None,
            do_minimization=False,
            save_dcd=False,
            save_prefix='nvt',
            previous_barostat=False,
            previous_barostat_type='isotropic',
            global_params={}):

    #Input a topology object, structure object, reference system, integrator, platform, platform properties, list of reporters,
    #and optionally state file, positions, velocities, and box size
    #If state is specified including positions and velocities and pos and vels are not None, the 
    #positions and velocities from the provided state will be overwritten
    #Does NVT and returns a simulation state object and state file that can be used to start other simulations

    #Copy the reference system and integrator objects
    system = copy.deepcopy(systemRef)
    integrator = copy.deepcopy(integratorRef)

    # state will include a MonteCarloPressure so we need to include 
    # a MonteCarloBarostat here, but we'll set the frequency to 0 to turn it off
    if previous_barostat:
        if previous_barostat_type == 'isotropic':
            barostat = mm.MonteCarloBarostat(1.0*u.bar,
                                             temperature, #Temperature should be SAME as for thermostat
                                             100 #Time-steps between MC moves
                                             )
        else:
            barostat = mm.MonteCarloAnisotropicBarostat([1.0*u.bar, 1.0*u.bar, 1.0*u.bar],
                                                        temperature, #Temperature should be SAME as for thermostat
                                                        False,
                                                        False,
                                                        True,
                                                        100 #Time-steps between MC moves
                                                        )
        system.addForce(barostat)
        barostat.setFrequency(0)

    #Create the simulation object for NVT simulation
    sim = app.Simulation(top.topology, system, integrator, platform, prop, state)

    #Set the particle positions
    if pos is not None:
        sim.context.setPositions(pos)

    if set_box is not None:
        sim.context.setPeriodicBoxVectors([set_box[0], 0, 0],
                                          [0, set_box[1], 0],
                                          [0, 0, set_box[2]])
        print('Box set to ', sim.context.getState().getPeriodicBoxVectors())
    else:
        print('Box is ', sim.context.getState().getPeriodicBoxVectors())

    #Apply constraints before starting the simulation
    sim.context.applyConstraints(1.0E-08)

    # set global parameters
    for param_name in global_params:
        print('setting parameter', param_name, 'to', global_params[param_name])
        sim.context.setParameter(param_name, global_params[param_name])

    #Minimize the energy
    if do_minimization:
        print("\nMinimizing energy...")
        # I think minimizeEnergy does too good of a job of approximating a global minimization
        # so that the peptoid conformation may change dramatically
        # (at least, just using minimizeEnergy worked to create the PES)
        # so using LocalEnergyMinimizer instead
        # sim.minimizeEnergy(tolerance=10.0*u.kilojoule/u.mole, #Energy threshold below which stops
        #                    maxIterations=1000 #Maximum number of iterations
        #                    )
        mm.openmm.LocalEnergyMinimizer.minimize(sim.context)

    #Initialize velocities if not specified
    if vels is not None:
        sim.context.setVelocities(vels)
    else:
        try:
            testvel = sim.context.getState(getVelocities=True).getVelocities()
            print("Velocities included in state, starting with 1st particle: %s"%str(testvel[0]))
            #If all the velocities are zero, then set them to the temperature
            if not np.any(testvel.value_in_unit(u.nanometer/u.picosecond)):
                print("Had velocities, but they were all zero, so setting based on temperature.")
                sim.context.setVelocitiesToTemperature(temperature)
        except:
            print("Could not find velocities, setting with temperature")
            sim.context.setVelocitiesToTemperature(temperature)

    #Set up the reporter to output energies, volume, etc.
    sim.reporters.append(app.StateDataReporter(save_prefix+'_out.txt', #Where to write - can be stdout or file name (default .csv, I prefer .txt)
                                               save_freq, #Number of steps between writes
                                               step=True, #Write step number
                                               time=True, #Write simulation time
                                               potentialEnergy=True, #Write potential energy
                                               kineticEnergy=True, #Write kinetic energy
                                               totalEnergy=True, #Write total energy
                                               temperature=True, #Write temperature
                                               volume=True, #Write volume
                                               density=False, #Write density
                                               speed=True, #Estimate of simulation speed
                                               separator='  ' #Default is comma, but can change if want (I like spaces)
                                               )
                         )

    #Set up reporter for printing coordinates (trajectory)
    if save_dcd:
        sim.reporters.append(app.dcdreporter.DCDReporter(save_prefix+'.dcd',
                                                         save_freq,
                                                         enforcePeriodicBox=False)
                         )
    else:
        sim.reporters.append(NetCDFReporter(save_prefix + '.nc', #File name to write trajectory to
                                            save_freq, #Number of steps between writes
                                            crds=True, #Write coordinates
                                            vels=False, #Write velocities
                                            frcs=False #Write forces
                                            )
                             )
        
    #Run NVT dynamics
    print("\nRunning NVT simulation...")
    sim.context.setTime(0.0)
    sim.step(simSteps)

    #Save simulation state if want to extend, etc.
    fname_xml = save_prefix+'State.xml'
    sim.saveState(fname_xml)

    #Get the final positions and velocities
    return fname_xml, sim.context.getState(getPositions=True, 
                                           getVelocities=True, 
                                           getParameters=True, 
                                           enforcePeriodicBox=True)

def run_npt(top,
            systemRef,
            integratorRef,
            platform,
            prop,
            temperature,
            simSteps=50000,
            save_freq=1000,
            state=None,
            pos=None,
            vels=None,
            save_prefix='npt',
            global_params={}):
    
    #Copy the reference system and integrator objects
    system = copy.deepcopy(systemRef)
    integrator = copy.deepcopy(integratorRef)

    #For NPT, add the barostat as a force
    system.addForce(mm.MonteCarloBarostat(1.0*u.bar,
                                          temperature, #Temperature should be SAME as for thermostat
                                          100 #Time-steps between MC moves
                                          )
                    )

    #Create new simulation object for NPT simulation
    sim = app.Simulation(top.topology, system, integrator, platform, prop, state)

    #Set the particle positions
    if pos is not None:
        sim.context.setPositions(pos)

    #Apply constraints before starting the simulation
    sim.context.applyConstraints(1.0E-08)

    # set global parameters
    for param_name in global_params:
        print('setting parameter', param_name, 'to', global_params[param_name])
        sim.context.setParameter(param_name, global_params[param_name])

    #Initialize velocities if not specified
    if vels is not None:
        sim.context.setVelocities(vels)
    else:
        try:
            testvel = sim.context.getState(getVelocities=True).getVelocities()
            print("Velocities included in state, starting with 1st particle: %s"%str(testvel[0]))
            #If all the velocities are zero, then set them to the temperature
            if not np.any(testvel.value_in_unit(u.nanometer/u.picosecond)):
                print("Had velocities, but they were all zero, so setting based on temperature.")
                sim.context.setVelocitiesToTemperature(temperature)
        except:
            print("Could not find velocities, setting with temperature")
            sim.context.setVelocitiesToTemperature(temperature)

    #Set up the reporter to output energies, volume, etc.
    sim.reporters.append(app.StateDataReporter(save_prefix+'_out.txt', #Where to write - can be stdout or file name (default .csv, I prefer .txt)
                                               save_freq, #Number of steps between writes
                                               step=True, #Write step number
                                               time=True, #Write simulation time
                                               potentialEnergy=True, #Write potential energy
                                               kineticEnergy=True, #Write kinetic energy
                                               totalEnergy=True, #Write total energy
                                               temperature=True, #Write temperature
                                               volume=True, #Write volume
                                               density=False, #Write density
                                               speed=True, #Estimate of simulation speed
                                               separator='  ' #Default is comma, but can change if want (I like spaces)
                                               )
                         )

    #Set up reporter for printing coordinates (trajectory)
    sim.reporters.append(NetCDFReporter(save_prefix+'.nc', #File name to write trajectory to
                                        save_freq, #Number of steps between writes
                                        crds=True, #Write coordinates
                                        vels=False, #Write velocities
                                        frcs=False #Write forces
                                        )
                         )
    
    #Run NPT dynamics
    print("\nRunning NPT simulation...")
    sim.context.setTime(0.0)
    sim.step(simSteps)
    
    #And save the final state of the simulation in case we want to extend it
    fname_xml = save_prefix+'State.xml'
    sim.saveState(fname_xml)
    
    #Get the final positions and velocities
    return fname_xml, sim.context.getState(getPositions=True, 
                                           getVelocities=True, 
                                           getParameters=True, 
                                           enforcePeriodicBox=True)

