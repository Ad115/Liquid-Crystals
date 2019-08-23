/* 
### La simulación del sistema de partículas de Lennard Jones.

Modelo de esferas que interaccionan con el potencial de Lennard Jones.
Las partículas viven e interaccionan en el GPU. 

Para compilar y ejecutar:
```
nvcc main_gpu.cu -std=c++11 -arch=sm_75
./a.out
```
*/

#include "src_gpu/ParticleSystem.cu"
#include "src_gpu/Particle.cu"
#include "src_gpu/Vector.cu"

#include <fstream>

int main(void)
{
  int n_particles = 200;
  double numeric_density = 0.1;

  std::ofstream outputf("output.xyz");

  ParticleSystem<LennardJones<>, PeriodicBoundaryBox<>> system(n_particles, numeric_density);
  system.simulation_init();
  //system.print();

  int simulation_steps = 15000;    // understand it as "frames", how many steps in time
  double time_step = 0.00001;
  double sample_period = 0.00005;
  
   double t = 0;
   for (int i=0; i<simulation_steps; i++) {
	   
	   if(t > sample_period) {
            system.write_xyz(outputf);
            t = 0;
        }

        system.simulation_step(time_step);
        t += time_step;

        printf("%d \n", i);
    }
  
 //system.print();
}