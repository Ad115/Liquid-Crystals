/*  
    PART II: DEFINITIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "random.h"
#include "particle_system.h"



// Slow, but useful for initialization.
void map_to_particles(ParticleSystem *psystem,
                      void(*particle_fn)(Particle *, int, void *),
                      void *some_parameter) { /*
    * Map the given function to each particle. 
    * The last parameter serves to pass additional information to the function.
    */
    int n = psystem->n_particles;
    Particle *particles = psystem->particles;

    for (int i=0; i<n; i++) {
        // Call function on particle
        (*particle_fn)(&particles[i], i, some_parameter);
    }
}



// Slow, but useful for initialization and non-recurrent measurements
double sum_all(ParticleSystem *psystem, 
               double (*measure_particle)(Particle *)) { /*
    * Measure the property over all particles and return the sum of the results.
    */
    int n = psystem->n_particles;
    Particle *particles = psystem->particles;

    double total = 0;
    for (int i=0; i<n; i++) {
        total += (*measure_particle)(&particles[i]);
    }

    return total;
}



ParticleSystem init_particle_system(
        SimulationParameters parameters,
        positions_init_fn *init_positions,
        velocities_init_fn *init_velocities) {

    int n = parameters.n_particles;
    Particle *particles = calloc(n, sizeof(*particles));

    double volume = n / parameters.numeric_density;
    double box_length = pow(volume, 1./DIMENSIONS);

    ParticleSystem psystem = {
        .n_particles = parameters.n_particles,
        .box_length = box_length,
        .cutoff_radius = parameters.cutoff_radius,
        .particles = particles
    };
    
    // Initialize the particle positions and velocities
    (*init_positions)(&psystem, parameters);
    (*init_velocities)(&psystem, parameters);    
    
    return psystem;
}



// ---
void print_particle_positions(Particle particles[], int n) {
    printf("[");
    for (int i=0; i<n; i++) {
        Particle p = particles[i];

        printf("[");
        for (int d=0; d<DIMENSIONS; d++) {
            printf("%f", p.position[d]);    

            if (d == DIMENSIONS-1) printf("]");
            else printf(", ");
        }

        if (i == n-1) printf("]");
            else printf(", ");
    }
    printf("\n");
}

void system_print_state(ParticleSystem *psystem) {
    print_particle_positions(psystem->particles, psystem->n_particles);
}
// ---