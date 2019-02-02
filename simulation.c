
/*  
    simulation: DEFINITIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <stddef.h>
#include <math.h>
#include "random.h"
#include "particle_system.h"
#include "simulation.h"

// ---
typedef struct PosParams {
    double L;
    int cube_length;
} PosParams;

void init_simple_position(Particle *p, int p_idx, void *pos_params) {
    // Fetch parameters
    PosParams parameters = *(PosParams*)pos_params;
    double L = parameters.L;
    int cube_length = parameters.cube_length;

    for (int D=0; D<DIMENSIONS; D++) {
        // Get position in a hypercube with volume = cube_length^DIMENSIONS.
        p->position[D] = ((int)( (p_idx / pow(cube_length, D)) )%cube_length);
        // Rescale to a box of volume = L^DIMENSIONS
        p->position[D] *= (L/cube_length);
        // Center the particles
        p->position[D] += (L/cube_length)/2;
    }
}

void init_simple_positions(ParticleSystem *psystem,
                           SimulationParameters _) {/* 
    Initialize particle positions by assigning them on a simple cubic grid.
    */
    // Setup the particle list
    int n = psystem->n_particles;
    double L = psystem->box_length;

    // The lowest integer such that cube_length^DIMENSIONS >= n. Think of a 
    // cube in DIMENSIONS with side cube_length where all particles are evenly 
    // spaced on a simple grid.
    int cube_length = (int) ceil(pow(n, 1./DIMENSIONS)); // Measured in particles

    PosParams parameters = {L, cube_length};

    map_to_particles(psystem, &init_simple_position, (void*)&parameters);
}
// ---



// ---
double particle_kinetic_energy(Particle *particle) {
    double energy=0;
    for (int D=0; D<DIMENSIONS;D++) {
        double vel = particle->velocity[D];
        energy += 1/2. * vel*vel;
    }
    return energy;
}

double measure_temperature(ParticleSystem *psystem) { /*
    * Return the temperature of the system as sum(1/2 v_i^2).
    */
    return sum_all(psystem, &particle_kinetic_energy);
}

void adjust_temperature(Particle *p, int _, void *adjustment_factor_ptr) {
    double adjustment_factor = *(double*)adjustment_factor_ptr;

    for (int D=0; D<DIMENSIONS; D++) {
        p->velocity[D] *= adjustment_factor;
    }
}

void set_random_velocity(Particle *p, int _, void *___) {
    for (int D=0; D<DIMENSIONS; D++) {
        p->velocity[D] = 2*random_uniform() - 1;
    }
}

void init_random_velocities(ParticleSystem *psystem, 
                            SimulationParameters parameters) { /*
    * Initialize random velocities for the particles.
    * Scale apropriately to match the required temperature.
    */
    double initial_temperature = parameters.initial_temperature;

    // Set initial random velocities
    map_to_particles(psystem, &set_random_velocity, NULL);

    // Adjust to correspond to the expected temperature
    double temperature = measure_temperature(psystem);
    double adjustment_factor = initial_temperature/temperature;
    map_to_particles(psystem, &adjust_temperature, (void*)&adjustment_factor);
}
// ---

// ---
double minimum_image(double r1, double r2, double box_length) { /*
    * Get the distance to the minimum image.
    */
    double half_length = box_length/2;
    double dr = r1 - r2;

    if (dr <= -half_length) {
        return (r1 + box_length-r2);

    } else if (dr <= half_length) {
        return dr;

    } else
        return -(r2 + box_length-r1);
}

typedef struct Distance {
    double diff_vector[DIMENSIONS];
    double distance_squared;

} Distance;

Distance particle_distance(Particle *p1, Particle *p2, double box_length) { /*
    * Calculate the distance using periodic boundaries.
    */
    double *r1 = p1->position;
    double *r2 = p2->position;

    Distance distance = {.distance_squared=0};
    for (int D=0; D<DIMENSIONS; D++) {
        double dx = minimum_image(r1[D], r2[D], box_length);
        distance.diff_vector[D] = dx;
        distance.distance_squared += dx*dx;
    }

    return distance;
}

typedef struct Force {
    double f[3];
} Force;

Force force_law(Particle *p1, 
                Particle *p2, 
                double box_length,
                double cutoff_radius) { /*
    * The force law: Lennard Johnes
    * =============================
    * See: http://www.pages.drexel.edu/~cfa22/msim/node26.html
    */
    Distance distance = particle_distance(p1, p2, box_length);
    double r2 = distance.distance_squared;
    double c2 = cutoff_radius * cutoff_radius;
    
    double f;
    if (r2 < c2) {
        double r6 = 1.0/(r2*r2*r2);
	    f = 48*(r6*r6-0.5*r6);

    } else { f = 0; }

    Force force;
    for (int D=0; D<DIMENSIONS; D++) {
        double dx = distance.diff_vector[D];
        double fx = dx * f/r2;
        force.f[D] = fx;
    }
    
    return force;
}

void update_forces(ParticleSystem *psystem) { /*
    * Calculate the forces between particles.
    */
    Particle *particles = psystem->particles;
    int n = psystem->n_particles;

    // Zero out all forces
    for (int i=0; i<n; i++) {
        for (int D=0; D<DIMENSIONS; D++) {
            particles[i].force[D] = 0;
        }
    }

    // All pairs of particles (O(n^2) algorithm)
    for (int i=0; i<n-1; i++) {
        for (int j=i+1; j<n; j++) {
            
            // Lennard Jones
            Force force = force_law(
                &particles[i], &particles[j], 
                psystem->box_length,
                psystem->cutoff_radius
            );

            // Newton's third law
            for (int D=0; D<DIMENSIONS; D++) {
                    particles[i].force[D] += force.f[D];
                    particles[j].force[D] += -force.f[D];
            }
        }
    }
}

double toroidal_wrap(double x, double box_length) {/*
    * Periodic boundary conditions.
    */
    x -= (x > box_length) * box_length;
    x += (x < 0) * box_length;

    return x;
}

void simulation_step(ParticleSystem *psystem, double time_step) { /*
    * Implementation of a velocity Vertlet integrator.
    * See: http://www.pages.drexel.edu/~cfa22/msim/node23.html#sec:nmni
    * 
    * This integrator gives a lower error O(dt^4) and more stability than
    * the standard forward integration (x(t+dt) += v*dt + 1/2 * f * dt^2)
    * by looking at more timesteps (t, t+dt) AND (t-dt), but in order to 
    * improve memory usage, the integration is done in two steps.
    */
    Particle *particles = psystem->particles;
    int n = psystem->n_particles;

    double dt = time_step;
    double dt2 = dt *dt;

    // --- First half step ---
    for (int i=0; i<n; i++) {
        Particle *p = &particles[i];

        for (int D=0; D<DIMENSIONS; D++) {

            // r(t + dt) = r(t) + v(t)*dt + 1/2*f(t)*dt^2
            double x = p->position[D];
            double dx = p->velocity[D]*dt + 1/2.*(p->force[D])*dt2;

            p->position[D] = toroidal_wrap(x+dx, psystem->box_length);

            // v(t + 1/2*dt) = v(t) + 1/2*f(t)*dt
            p->velocity[D] += 1/2.*(p->force[D])*dt;
        }
    }
    // r(t + dt)  -->  f(t + dt)
    update_forces(psystem);

    // --- Second half step ---
    for (int i=0; i<n; i++) {
        Particle *p = &particles[i];

        for (int D=0; D<DIMENSIONS; D++) {

            // v(t + dt) = v(t + 1/2*dt) + 1/2*f(t + dt)*dt
            p->velocity[D] += 1/2.*p->force[D]*dt;
        }
    }
}
// ---