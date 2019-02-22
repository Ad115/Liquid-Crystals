#include <iostream>
#include <fstream>
#include <numeric>

#define CUDA_ENABLED 1
#include "src/Vector.cu"
#include "src/ParticleSystem.cu"
#include "src/Particle.cu"


int main( int argc, char **argv )
{
    int n_particles = 200;
    int dimensions = 3; 
    double numeric_density = .01;

    ParticleSystem<LennardJones, PeriodicBoundaryBox<3>>
        system(n_particles, numeric_density);
        
    //system.sample();


    int i,j;
    for( i=0; i<n_particles; ++i ){
        (system.getParticlesPtr()+i)->position=Vector<>::random_unit();
        (system.getParticlesPtr()+i)->force=Vector<>::random_unit();
        (system.getParticlesPtr()+i)->velocity=Vector<>::random_unit();
        printf("Setting value @ %p\n", (system.getParticlesPtr()+i));
    }

    cudaDeviceSynchronize();

    for (int i=0; i<system.n_particles(); i++) {
        for(int j=0; j<3; j++) {
            //system.getParticlesPtr()[i].velocity[j] = 1;
            printf("%f ", system.getParticlesPtr()[i].position[j]);
            printf("%f ", system.getParticlesPtr()[i].velocity[j]);
            printf("%f ", system.getParticlesPtr()[i].force[j]);
        }
        printf("%p ", (system.getParticlesPtr()+i));
        printf("\n");
    }
    printf("------------------------------------\n");
    printParticleSystemDevice<LennardJones><<<n_particles,1>>>( system.getParticlesPtr() );

    //cudaFree(container_ptr);
    return 0;
}