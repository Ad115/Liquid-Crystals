#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "simulation.h"
#include "particle_system.h"
#include "random.h"
#include "configuration.h"

int wait_server();
int init_server();

Config default_configuration = {
    .n_particles = 70,
    .sampling_frequency = 150,
    .numeric_density = 0.01, // no. particles / volume
    .time_step = 0.001,
    .cutoff_radius = 100,
    .initial_temperature = 5.0,    
    .random_seed = 23410981, // Random number generator seed
};


int main(int argc, char *argv[]) {

    Config config = parse_user_configuration(
        argc, argv, 
        default_configuration
    );
    
    init_random();
    
    ParticleSystem simulation = init_particle_system(
        (SimulationParameters) {
            .n_particles = config.n_particles,
            .numeric_density = config.numeric_density,
            .initial_temperature = config.initial_temperature,
            .cutoff_radius = config.cutoff_radius
        },
        &init_simple_positions,
        &init_random_velocities
    );

    // Initialize configuration in the server
    init_server(simulation.box_length, DIMENSIONS);
    
    int steps = 0;
    while (true) {
        simulation_step(&simulation, config.time_step);

        if(steps % config.sampling_frequency == 0) {
            system_print_state(&simulation);

            // Wait for the drawing loop to complete
            wait_server();
        }
        steps++;
    }
}

int wait_server() {
    double response; 
    return scanf("%lf", &response);
}

int init_server(double box_length, int dimensions) {
    wait_server();
    printf("{" // Send parameters in JSON
           " \"box_length\" : %lf,"
           " \"dimensions\" : %d"
           "}\n", box_length, dimensions);
    return 1;
}
