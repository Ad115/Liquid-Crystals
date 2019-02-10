#ifndef SIMULATION_HEADER
#define SIMULATION_HEADER

/*
 * Molecular dynamics - Lennard Jones.
 * ===================================
 */

#include "particle_system.h"


/*  
    DECLARATIONS (INTERFACE)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void init_simple_positions(ParticleSystem *psystem, SimulationParameters _); /* 
    Initialize particle positions by assigning them on a simple cubic grid.
    */

void init_random_velocities(ParticleSystem *psystem, 
                            SimulationParameters parameters); /*
    * Initialize random velocities for the particles.
    * Scale apropriately to match the required temperature.
    */

void simulation_step(ParticleSystem *psystem, double time_step);



#endif