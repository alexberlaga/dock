import sys
import copy
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import parmed as pmd
from openmm.app.xtcreporter import XTCReporter
from openmm.app.checkpointreporter import CheckpointReporter

import mdtraj as md
from pymbar import mbar
from cvpack import RadiusOfGyration, RadiusOfGyrationSq

                         
def run_annealing(top, systemRef, integratorRef, platform, prop, helix_backbone_idxs, temperature_start=300*u.kelvin, temperature_end=800*u.kelvin,
                 sim_steps=int(5e6), pos=None, vels=None, state=None, bias_or_temp='temp',
                 report_freq=int(1e5), # number of MD steps between writes to reporters (default is 200ps),
                 reporter_prefix='prod',
                 anneal_freq=int(5e5), # number of MD steps between temperature additions (default is 200ps)
                 save_restart_freq=int(1e6), # number of MD steps between saving a state file for restarts
                 copy_prefix="0"): 

    MAX_BLOB_COUNT = 20
    #Copy the reference system and integrator objects
    system = copy.deepcopy(systemRef)
    integrator = copy.deepcopy(integratorRef)
    rg0 = RadiusOfGyration(helix_backbone_idxs[0], pbc=True)
    rg0.setUnit(u.nanometers)
    rg1 = RadiusOfGyration(helix_backbone_idxs[1], pbc=True)
    rg1.setUnit(u.nanometers)
    rg2 = RadiusOfGyration(helix_backbone_idxs[2], pbc=True)
    rg2.setUnit(u.nanometers)
    
    if bias_or_temp == 'bias':
        rg0_bias = mm.CustomCVForce(f'0.5*k*(rg-1.5)^2')
        rg0_bias.addCollectiveVariable('rg', rg0)
        rg0_bias.addGlobalParameter('k', u.Quantity(0.0, u.kilojoules_per_mole / u.nanometer))
        rg0_bias.setForceGroup(8)
        system.addForce(rg0_bias)

        rg1_bias = mm.CustomCVForce(f'0.5*k*(rg-1.5)^2')
        rg1_bias.addCollectiveVariable('rg', rg1)
        rg1_bias.addGlobalParameter('k', u.Quantity(0.0, u.kilojoules_per_mole / u.nanometer))
        rg1_bias.setForceGroup(8)
        system.addForce(rg1_bias)

        rg2_bias = mm.CustomCVForce(f'0.5*k*(rg-1.5)^2')
        rg2_bias.addCollectiveVariable('rg', rg2)
        rg2_bias.addGlobalParameter('k', u.Quantity(0.0, u.kilojoules_per_mole / u.nanometer))
        rg2_bias.setForceGroup(8)
        system.addForce(rg2_bias)
           
    else:
        rg0.setUnusedForceGroup(0, system)
        system.addForce(rg0)
        rg1.setUnusedForceGroup(1, system)
        system.addForce(rg1)
        rg2.setUnusedForceGroup(2, system)
        system.addForce(rg2)
    
    
    
    #For NPT, add the barostat as a force
    pressure = 1.0 * u.bar
    system.addForce(mm.MonteCarloBarostat(pressure,
                                              temperature_start, # should be same as thermostat
                                              250 # time step between barostat MC moves
                                              )
                        )

    #Create new simulation object for NPT simulation
    sim = app.Simulation(top.topology, system, integrator, platform, prop, state)

    #Set the particle positions - should also figure out how to set box information!
    if pos is not None:
        sim.context.setPositions(pos)
        
    #Apply constraints before starting the simulation
    sim.context.applyConstraints(1.0E-08)

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
                sim.context.setVelocitiesToTemperature(temperature_start)
        except:
            print("Could not find velocities, setting with temperature")
            sim.context.setVelocitiesToTemperature(temperature_start)

    #Set up the reporter to output energies, volume, etc.
    sim.reporters.append(app.StateDataReporter(
        reporter_prefix+'_out.txt', #Where to write - can be stdout or file name (default .csv, I prefer .txt)
        report_freq, #Number of steps between writes
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

    sim.reporters.append(CheckpointReporter(reporter_prefix + '_check.chk',5*report_freq))
    #Set up reporter for printing coordinates (trajectory)
    sim.reporters.append(XTCReporter(
        reporter_prefix+'.xtc', #File name to write trajectory to
        report_freq
    )
                         )

    #Ready to perform expanded ensemble dynamics
    print("\nRunning production simulated annealing simulation...")
    sim.context.setTime(0.0)

    count_steps = 0

    # writeFreq is number of MC cycles to perform before writing information for MBAR
    # should be consistent with reporting frequency to make analysis easy
   

    
    
    # hold all alchemical info (even when weights are still being updated)
    
    annealingFile = open(copy_prefix + '_annealing_output.txt', 'w')
    if bias_or_temp == 'bias':
        annealingFile.write('#Step  Time (ps) Bias (kJ mol^-1 nm^-1) Rg (nm) \n')
    else:
        annealingFile.write('#Step  Time (ps) Temp (K) Rg (nm) \n')
    annealingFile.close()
    
    

    

    blob_count = 0
    temp_cur = temperature_start
    while count_steps < sim_steps:

        # generate new configuration with MD
        try:
            sim.step(report_freq)
        except:
            print("NaN during simulation due to OpenMM bug. Restarting from checkpoint")
            #To overcome openmm neighborlist bug when around PBC boundary 
            sim.loadCheckpoint(reporter_prefix+ '_check.chk')
            sim.step(report_freq)
            
        if count_steps % save_restart_freq == 0:
            sim.saveState(reporter_prefix+'_restart.xml')
            
        if bias_or_temp == 'bias':
            mean_rg_val = np.mean((rg0_bias.getCollectiveVariableValues(sim.context)[0], rg1_bias.getCollectiveVariableValues(sim.context)[0], rg2_bias.getCollectiveVariableValues(sim.context)[0]))
        else:
            mean_rg_val = np.mean((rg0.getValue(sim.context).value_in_unit(u.nanometers), rg1.getValue(sim.context).value_in_unit(u.nanometers), rg2.getValue(sim.context).value_in_unit(u.nanometers)))

        count_steps += report_freq
        
        annealingFile = open(copy_prefix + '_annealing_output.txt', 'a')
        if bias_or_temp == 'bias':
            annealingFile.write("%i  %10.1f  %10.2f  %10.5f \n"%(count_steps, 
                                                              sim.context.getState().getTime().value_in_unit(u.picosecond),
                                                              sim.context.getParameter('k'),
                                                             mean_rg_val
                                                              )
                                 )
        else:
            annealingFile.write("%i  %10.1f  %10.1f  %10.5f \n"%(count_steps, 
                                                              sim.context.getState().getTime().value_in_unit(u.picosecond),
                                                              integrator.getTemperature().value_in_unit(u.kelvin),
                                                             mean_rg_val
                                                              )
                                 )
        annealingFile.close()
        
        if count_steps % anneal_freq == 0:
            if bias_or_temp == 'bias':
                kval = sim.context.getParameter('k') * u.kilojoules_per_mole / u.nanometer
                kval += .05 * u.kilojoules_per_mole / u.nanometer
                sim.context.setParameter('k', kval)
            else:
                if temp_cur < temperature_end: 
                    temp_cur += 1.0 * u.kelvin
                sim.context.setParameter(mm.MonteCarloBarostat.Temperature(), temp_cur)
                integrator.setTemperature(temp_cur)


        if mean_rg_val < 0.8:
            blob_count += 1
        else:
            blob_count = 0

        if blob_count >= MAX_BLOB_COUNT:
            break
                        
                 

    #save the final state of the simulation
    sim.saveState(reporter_prefix+'State.xml')

    # return the final positions, velocities, and state weights
    return  sim.context.getState(getPositions=True, getVelocities=True, getParameters=True, enforcePeriodicBox=True)

def resume_annealing(restart_file, start_step, top, systemRef, integratorRef, platform, prop, helix_backbone_idxs, temperature_end=800*u.kelvin,
                 sim_steps=int(5e6), pos=None, vels=None,
                 report_freq=int(1e5), # number of MD steps between writes to reporters (default is 200ps),
                 reporter_prefix='prod',
                 anneal_freq=int(5e5), # number of MD steps between temperature additions (default is 200ps)
                 save_restart_freq=int(1e7), # number of MD steps between saving a state file for restarts
                 copy_prefix="0",
bias_or_temp="temp"):
    
    system = copy.deepcopy(systemRef)
    integrator = copy.deepcopy(integratorRef)

    rg0 = RadiusOfGyration(helix_backbone_idxs[0], pbc=True)
    rg0.setUnit(u.nanometers)
    rg1 = RadiusOfGyration(helix_backbone_idxs[1], pbc=True)
    rg1.setUnit(u.nanometers)
    rg2 = RadiusOfGyration(helix_backbone_idxs[2], pbc=True)
    rg2.setUnit(u.nanometers)
    
    if bias_or_temp == 'bias':
        rg0_bias = mm.CustomCVForce(f'0.5*k*(rg-1.5)^2')
        rg0_bias.addCollectiveVariable('rg', rg0)
        rg0_bias.addGlobalParameter('k', u.Quantity(0.0, u.kilojoules_per_mole / u.nanometer))
        rg0_bias.setForceGroup(8)
        system.addForce(rg0_bias)

        rg1_bias = mm.CustomCVForce(f'0.5*k*(rg-1.5)^2')
        rg1_bias.addCollectiveVariable('rg', rg1)
        rg1_bias.addGlobalParameter('k', u.Quantity(0.0, u.kilojules_per_mole / u.nanometer))
        rg1_bias.setForceGroup(8)
        system.addForce(rg1_bias)

        rg2_bias = mm.CustomCVForce(f'0.5*k*(rg-1.5)^2')
        rg2_bias.addCollectiveVariable('rg', rg2)
        rg2_bias.addGlobalParameter('k', u.Quantity(0.0, u.kilojules_per_mole / u.nanometer))
        rg2_bias.setForceGroup(8)
        system.addForce(rg2_bias)
           
    else:
        rg0.setUnusedForceGroup(0, system)
        system.addForce(rg0)
        rg1.setUnusedForceGroup(1, system)
        system.addForce(rg1)
        rg2.setUnusedForceGroup(2, system)
        system.addForce(rg2)
    #For NPT, add the barostat as a force
    pressure = 1.0 * u.bar
    system.addForce(mm.MonteCarloBarostat(pressure,
                                              300.0 * u.kelvin, # should be same as thermostat
                                              250 # time step between barostat MC moves
                                              )
                        )
    
    MAX_BLOB_COUNT = 20
    state = restart_file
    sim = app.Simulation(top.topology, system, integrator, platform, prop, state)
    #Set up the reporter to output energies, volume, etc.
    temp_cur = sim.context.getParameter('MonteCarloTemperature') * u.kelvin
    sim.context.setParameter(mm.MonteCarloBarostat.Temperature(), temp_cur)
    integrator.setTemperature(temp_cur)
    
    sim.reporters.append(app.StateDataReporter(
        reporter_prefix+'_out.txt', #Where to write - can be stdout or file name (default .csv, I prefer .txt)
        report_freq, #Number of steps between writes
        step=True, #Write step number
        time=True, #Write simulation time
        potentialEnergy=True, #Write potential energy
        kineticEnergy=True, #Write kinetic energy
        totalEnergy=True, #Write total energy
        temperature=True, #Write temperature
        volume=True, #Write volume
        density=False, #Write density
        speed=True, #Estimate of simulation speed
        separator='  ', #Default is comma, but can change if want (I like spaces)
        append=True
    )
                         )
    sim.reporters.append(CheckpointReporter(reporter_prefix + '_check.chk',5*report_freq))

    #Set up reporter for printing coordinates (trajectory)
    sim.reporters.append(XTCReporter(
        reporter_prefix+'.xtc', #File name to write trajectory to
        report_freq,
        append=True
    )
                         )
    sim.context.setTime(start_step * integrator.getStepSize())

    blob_count = 0
    count_steps = start_step
    
    while count_steps < sim_steps:

        # generate new configuration with MD
        try:
            sim.step(report_freq)
        except:
            print("NaN during simulation due to OpenMM bug. Restarting from checkpoint")
            #To overcome openmm neighborlist bug when around PBC boundary 
            sim.loadCheckpoint(reporter_prefix+ '_check.chk')
            sim.step(report_freq)
        
        if count_steps % save_restart_freq == 0:
            sim.saveState(reporter_prefix+'_restart.xml')
        
        if bias_or_temp == 'bias':
            mean_rg_val = np.mean((rg0_bias.getCollectiveVariableValues(sim.context)[0], rg1_bias.getCollectiveVariableValues(sim.context)[0], rg2_bias.getCollectiveVariableValues(sim.context)[0]))
        else:
            mean_rg_val = np.mean((rg0.getValue(sim.context).value_in_unit(u.nanometers), rg1.getValue(sim.context).value_in_unit(u.nanometers), rg2.getValue(sim.context).value_in_unit(u.nanometers)))
        
        count_steps += report_freq
        
        annealingFile = open(copy_prefix + '_annealing_output.txt', 'a')
        if bias_or_temp == 'bias':
            annealingFile.write("%i  %10.1f  %10.2f  %10.5f \n"%(count_steps, 
                                                              sim.context.getState().getTime().value_in_unit(u.picosecond),
                                                              sim.context.getParameter('k'),
                                                             mean_rg_val
                                                              )
                                 )
        else:
            annealingFile.write("%i  %10.1f  %10.1f  %10.5f \n"%(count_steps, 
                                                              sim.context.getState().getTime().value_in_unit(u.picosecond),
                                                              integrator.getTemperature().value_in_unit(u.kelvin),
                                                             mean_rg_val
                                                              )
                                 )
        annealingFile.close()

        if count_steps % anneal_freq == 0:
            if bias_or_temp == 'bias':
                kval = sim.context.getParameter('k')
                kval += .05 * u.kilojoules_per_mole / u.nanometer
                sim.context.setParameter('k', kval)
            else:
                if temp_cur < temperature_end: 
                    temp_cur += 1.0 * u.kelvin
                sim.context.setParameter(mm.MonteCarloBarostat.Temperature(), temp_cur)
                integrator.setTemperature(temp_cur)


        if mean_rg_val < 0.8:
            blob_count += 1
        else:
            blob_count = 0

        if blob_count >= MAX_BLOB_COUNT:
            break
                        
                 

    #save the final state of the simulation
    sim.saveState(reporter_prefix+'State.xml')

    # return the final positions, velocities, and state weights
    return  sim.context.getState(getPositions=True, getVelocities=True, getParameters=True, enforcePeriodicBox=True)
