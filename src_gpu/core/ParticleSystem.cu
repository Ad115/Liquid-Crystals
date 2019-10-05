/* 
## Clase `ParticleSystem`

Esta clase está diseñada para manejarse desde el Host(CPU), lo importante es que 
contiene un `thrust::device_vector` de partículas, por lo que estas viven 
completamente en el GPU y de ahí se operan. A su vez, el `Container` forma parte 
de un `device_obj`, por lo que reside también completamente en el device.  El 
`kernel` es un integrador muy simple donde cada partícula tiene su propio hilo. 
Falta algo para calcular la fuerza, esto probablemente se podrá hacer con otro 
kernel. 
*/

#include <iostream>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include "device_obj.cu"
#include "Particle.cu"
#include "Container.cu"

#include "Transformations.cu"

template< 
    typename ParticleT=Particle<>, 
    typename ContainerT=EmptySpace<> 
>
class ParticleSystem
{

 public:
    thrust::device_vector< ParticleT > particles;
    device_obj< ContainerT > box;

    unsigned int n_particles;
    
    using particle_type = ParticleT;
    using container_type = ContainerT;
    using vector_type = typename ParticleT::vector_type;
    static constexpr int dimensions = ParticleT::dimensions;

    ParticleSystem(unsigned int n, double numeric_density) 
        : n_particles{n},
          particles{thrust::device_vector<ParticleT>(n)},
          box{pow(n/numeric_density, 1./dimensions)} {};

    template <typename MeasureFn>
    double measure_particles(MeasureFn measure_fn) {
        return 
            thrust::transform_reduce(
                    particles.begin(), particles.end(), 
                    measure_fn,
                    0.,
                    thrust::plus<double>{}
            );
    }

    template <typename ParticleFn>
    ParticleFn map_to_particles(ParticleFn particle_fn) {
        
        thrust::for_each(
            particles.begin(), particles.end(),
            particle_fn
        );

        return particle_fn;
    }

    template<typename TransformationT>
    TransformationT apply(TransformationT transformation) {
        transformation(*this);
        return transformation;
    }

    template<typename TransformationT>
    void simulation_init(TransformationT initial_conditions_fn) {
        (*this).apply(initial_conditions_fn);
    }

    template<typename IntegratorT>
    void simulation_step(double dt, IntegratorT integrator) {
        integrator.time_step = dt;
        this->apply(integrator);
    }

    friend std::ostream& operator<<(
        std::ostream& stream, 
        ParticleSystem<ParticleT, ContainerT>& sys) {
        stream 
            << "Container: \n\t"
            << sys.box.get() << "\n";
        
        thrust::host_vector<ParticleT> p(sys.particles);

        stream << "Particles: \n";
        for (int i=0; i<sys.n_particles; i++) {
            stream 
                << i << ":\t" 
                << p[i] << "\n";
        }

        return stream;
    }

    void write_xyz(std::ostream& stream) { /*
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
        
        thrust::host_vector<ParticleT> host_particles = particles;
        stream << n_particles << "\n";
        for (ParticleT p: host_particles) {
            stream << "\n";
            for (int D = 0; D < dimensions; D++)
                stream << p.position[D] << " ";
        }
        stream << std::endl;
        return;
    };
};

template<typename ParticleT, typename ContainerT>
constexpr int ParticleSystem<ParticleT, ContainerT>::dimensions;