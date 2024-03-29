#pragma once

#include "pcuditas/gpu/gpu_array.cu"
#include "pcuditas/cpu/cpu_array.cu"



class Temperature {
public:

    double setpoint;

    Temperature(double setpoint)
     : setpoint(setpoint)
     {}

    
    template< typename ParticlesT >
    void operator()(ParticlesT &particles) { 
        Temperature::set(particles, setpoint);
    }

    // --- GPU
    template< typename ParticleT >
    static double measure(gpu_array<ParticleT> &particles) { /*
            * Measure the temperature of the system (the sum of the particle's 
            * kinetic energies).
            */

        using vector_t = typename ParticleT::vector_type;

        auto kinetic_energies 
            = particles.template transform<double>(
                []__device__ (ParticleT &p, int idx) {

                    auto vel = p.velocity;

                    return (double)(1/2. * (vel*vel));
            });
    
        auto total_kinetic_energy 
            = kinetic_energies.reduce( 
                [] __device__ (double a, double b){
                    return a+b;
                } 
            ).to_cpu();
        
        return 2./(3*particles.size) * total_kinetic_energy;
    }


    template< typename ParticleT >
    static void set(gpu_array<ParticleT> &particles, double value) {

        double current_temperature = measure(particles);
        double correction_factor = sqrt(value / current_temperature);

        particles.for_each(
            [correction_factor] 
            __device__ (ParticleT &p, int i) {
                p.velocity *= correction_factor;
            }
        );
    }


    // --- CPU
    template< typename ParticleT >
    static double measure(cpu_array<ParticleT> &particles) { /*
            * Measure the temperature of the system (the sum of the particle's 
            * kinetic energies).
            */

        using vector_t = typename ParticleT::vector_type;

        auto kinetic_energies 
            = particles.template transform<double>(
                [](ParticleT &p, int idx) {

                    auto vel = p.velocity;

                    return (double)(1/2. * (vel*vel));
            });
    
        auto total_kinetic_energy 
            = kinetic_energies.reduce( 
                [](double a, double b){
                    return a+b;
                } 
            );
        
        return 2./(3*particles.size) * total_kinetic_energy;
    }


    template< typename ParticleT >
    static void set(cpu_array<ParticleT> &particles, double value) {

        double current_temperature = measure(particles);
        double correction_factor = sqrt(value / current_temperature);

        particles.for_each(
            [correction_factor] 
             (ParticleT &p, int i) {
                p.velocity *= correction_factor;
            }
        );
    }
};