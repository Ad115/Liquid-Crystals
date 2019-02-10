#ifndef PARTICLE_SYSTEM_HEADER
#define PARTICLE_SYSTEM_HEADER

/*  
    DECLARATIONS (INTERFACE)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#ifndef DIMENSIONS
    #define DIMENSIONS 3
#endif

typedef struct Particle Particle;
typedef struct ParticleSystem ParticleSystem;
typedef struct SimulationParameters SimulationParameters;

struct Particle {
    double position[DIMENSIONS];
    double velocity[DIMENSIONS];
    double force[DIMENSIONS];
};

struct ParticleSystem {
    int n_particles;
    double box_length;
    double cutoff_radius;
    Particle *particles;
};

struct SimulationParameters {
    int n_particles;
    double cutoff_radius;
    double numeric_density;
    double initial_temperature;
};

typedef void(positions_init_fn)(ParticleSystem * , SimulationParameters);
typedef void(velocities_init_fn)(ParticleSystem *, SimulationParameters);

ParticleSystem init_particle_system(
        SimulationParameters parameters,
        positions_init_fn *init_positions,
        velocities_init_fn *init_velocities); /*
    * Particle system constructor.
    */

// Slow, but useful for initialization.
void map_to_particles(ParticleSystem *psystem,
                      void(*particle_fn)(Particle *, int, void *),
                      void *some_parameter); /*
    * Map the given function to each particle. 
    * The last parameter serves to pass additional information to the function.
    */

// Slow, but useful for initialization and non-recurrent measures.
double sum_all(ParticleSystem *psystem, 
               double (*measure_particle)(Particle *)); /*
    * Measure the property over all particles and return the sum of the results.
    */

void system_print_state(ParticleSystem *psystem);

#endif