#ifndef PARTICLE_SYSTEM_HEADER
#define PARTICLE_SYSTEM_HEADER

#include <vector>
#include <cmath>
#include <iostream>
#include <iterator>
#include <algorithm>
#include "Container.cu"
//#include "Simulation.h"


/*  
    Part I: DECLARATIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/* Local joke: Camel Case xD */
template< typename ParticleClass, typename ContainerClass >
class ParticleSystem { /*
    * The main class. Handles the particles vector and the container in which 
    * the simulation develops.
    */
    private:

        int n_particles_;

    public:
        using Container_t = ContainerClass;
        using Particle_t = ParticleClass;
        ContainerClass *_container;
        ParticleClass *particles;
        ParticleSystem( 
                int n_particles, // Number of particles to create
                double numeric_density // no. of particles / unit volume
        );

        ~ParticleSystem(){
            cudaFree(particles);
            cudaFree(_container);
        }
        __host__ void sample(){
            int i,j;
            for( i=0; i<n_particles(); ++i ){
                for( j=0; j<dimensions(); ++j )
                    particles[i].position[j]=2*(0.5 - random_uniform());
                    particles[i].force[j]=2*(0.5 - random_uniform());
                    particles[i].velocity[j]=2*(0.5 - random_uniform());
            }
        }
        __host__ __device__ ParticleClass *getParticlesPtr(){ return particles; };
        __host__ __device__ unsigned dimensions() const; /*
        * Getter for the dimensionality of the system.
        */
        __host__ __device__ unsigned n_particles() const; /*
        * The number of particles in the system.
        */
        __host__ __device__ const ContainerClass& container() const; /*
        * The space in which the particles interact.
        */

        __host__ __device__ ContainerClass& container();

        
};



/*  
    Part II: IMPLEMENTATION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

template< typename ParticleClass, typename ContainerClass >
ParticleSystem<ParticleClass, ContainerClass>::ParticleSystem( 
                int n_particles, // Number of particles to create
                double numeric_density // Initial numeric density (particles/volume)
    )
     : n_particles_(n_particles) {

        #ifdef CUDA_ENABLED
            //Reserve memory and start a new instance of the object in the GPU
            int blocks=(n_particles/128)+1;
            cudaMallocManaged(&particles, sizeof(ParticleClass) * n_particles );

            //Initialize the object in the device
            initParticlesDevice<<<blocks,128>>>( particles, n_particles );

            //Initialize the container in the device
            cudaMallocManaged( &_container, sizeof(*_container) );
            init_device_container<<<1,1>>>( _container, dimensions() );
            
        #endif
    }

template < typename ParticleClass, typename ContainerClass >
__host__ __device__ unsigned ParticleSystem<ParticleClass, ContainerClass>::dimensions() const { /*
        * Getter for the dimensionality of the system.
        */
        return container().dimensions();
}


template < typename ParticleClass, typename ContainerClass >
__host__ __device__ unsigned ParticleSystem<ParticleClass, ContainerClass>::n_particles() const { /*
        * The number of particles in the system.
        */
        return n_particles_;
}

template < typename ParticleClass, typename ContainerClass >
__host__ __device__ const ContainerClass& ParticleSystem<ParticleClass, ContainerClass>::container() const { /*
        * The space in which the particles interact.
        */
        return *_container;
}

template < typename ParticleClass, typename ContainerClass >
__host__ __device__ ContainerClass& ParticleSystem<ParticleClass, ContainerClass>::container() { /*
        * The space in which the particles interact.
        */
        return *_container;
}

template < typename ParticleClass >
__global__ void printParticleSystemDevice( ParticleClass *particlesPtr )
{
    //printf("%f\n", particlesPtr);
    printf("%f %f %f\n", particlesPtr[blockIdx.x].force[0], 
                         particlesPtr[blockIdx.x].velocity[1], 
                         particlesPtr[blockIdx.x].position[2]  
    );
}

#endif

