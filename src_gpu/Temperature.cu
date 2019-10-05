#ifndef THERMOSTAT_HEADER
#define THERMOSTAT_HEADER

#include "core/interfaces/Transformation.h"

class Temperature : Transformation {
public:

    double setpoint;

    Temperature(double setpoint)
     : setpoint(setpoint)
     {}

    template< typename ParticleSystem >
    static double measure(ParticleSystem& system) { /*
            * Measure the temperature of the system (the sum of the particle's 
            * kinetic energies).
            */
        using ParticleT = typename ParticleSystem::particle_type;
    
        double total_kinetic_energy = system.measure_particles( 
                [] __device__ (ParticleT p){ 
                    return p.kinetic_energy(); 
                } 
            );
        
        return 2./(3*system.n_particles) * total_kinetic_energy;
     }

    template< typename ParticleSystem>
    void operator()(ParticleSystem& system) { 
        
        double current_temperature = measure(system);
        double correction_factor = sqrt(setpoint / current_temperature);

        using particle_t = typename ParticleSystem::particle_type;

        system.map_to_particles(
            [correction_factor] __device__ (particle_t& p){
                p.velocity = correction_factor * p.velocity;
            }
        );
    }
};

#endif