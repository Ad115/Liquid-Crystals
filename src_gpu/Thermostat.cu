class Thermostat {
public:

    double setpoint;

    Thermostat(double setpoint)
     : setpoint(setpoint)
     {}

    template< typename ParticleSystem >
    static double measure(ParticleSystem& system) { /*
            * Measure the temperature of the system (the sum of the particle's 
            * kinetic energies).
            */
        using ParticleT = typename ParticleSystem::particle_type;
    
        return 
            system.template measure_particles<double>( 
                    [n=system.n_particles] __device__ (ParticleT p){ 
                        return 2./(3*n)*p.kinetic_energy(); 
                    } 
            );
     }

    template< typename ParticleSystem>
    void apply(ParticleSystem& system) { 
        
        double current_temperature = measure(system);
        double correction_factor = sqrt(setpoint / current_temperature);

        using ParticleT = typename ParticleSystem::particle_type;

        system.map_to_particles(
            [correction_factor] __device__ (ParticleT& p){
                p.velocity = correction_factor * p.velocity;
            }
        );
    }

    template< typename ParticleSystem>
    void operator()(ParticleSystem& system) { 
        
        (*this).apply(system);
    }
};