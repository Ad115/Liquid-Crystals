#ifndef PARTICLE_SYSTEM_INTERFACE
#define PARTICLE_SYSTEM_INTERFACE

#include <iostream>

template< typename ParticleT, typename ContainerT>
class ParticleSystem {
 protected:
    ParticleSystem() = default; // <-- Not a part of the interface

 public:
    unsigned int n_particles;
    
    using particle_type = ParticleT;
    using container_type = ContainerT;
    using vector_type = typename ParticleT::vector_type;
    static constexpr int dimensions = ParticleT::dimensions;

    ParticleSystem(unsigned int n, double numeric_density);

    template <typename MeasureFn>
    double measure_particles(MeasureFn measure_fn);

    template <typename ParticleFn>
    ParticleFn map_to_particles(ParticleFn particle_fn);

    template<typename TransformationT>
    TransformationT apply(TransformationT transformation);

    template<typename TransformationT>
    void simulation_init(TransformationT initial_conditions_fn);

    template<typename IntegratorT>
    void simulation_step(double dt, IntegratorT integrator);

    template<typename Particle, typename Container>
    friend std::ostream& operator<<(
        std::ostream& stream, 
        ParticleSystem<Particle, Container>& sys);

    void write_xyz(std::ostream& stream); /*
        * Output the positions of the particles in the XYZ format.
        * The format consists in a line with the number of particles,
        * then a comment line followed by the space-separated coordinates 
        * of each particle in different lines.
        * 
        * Example (for 3 particles in the xyz diagonal):
        * 
        *   10
        *   
        *   1.0 1.0 1.0
        *   1.5 1.5 1.5
        *   2.0 2.0 2.0
        */
};

template<typename ParticleT, typename ContainerT>
constexpr int ParticleSystem<ParticleT, ContainerT>::dimensions;

#endif