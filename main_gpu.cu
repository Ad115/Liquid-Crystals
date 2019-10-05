/* 
### La simulación del sistema de partículas de Lennard Jones.

Modelo de esferas que interaccionan con el potencial de Lennard Jones.
Las partículas viven e interaccionan en el GPU. 

Para compilar y ejecutar:
```
nvcc main_gpu.cu -std=c++11 -arch=sm_75 --expt-extended-lambda
./a.out
```
*/

#include "src_gpu/core/ParticleSystem.cu"
#include "src_gpu/PeriodicBoundaryBox.cu"
#include "src_gpu/LennardJones.cu"
#include "src_gpu/core/Vector.cu"
#include "src_gpu/Thermostat.cu" 

#include <fstream>

using LJSystem = ParticleSystem<LennardJones<>, PeriodicBoundaryBox<>>;

int main(void)
{
  int n_particles = 200;
  double numeric_density = 0.5;
  Thermostat thermostat{5e4};

  std::ofstream outputf("output.xyz");

  LJSystem system(n_particles, numeric_density);
  system.simulation_init();

	printf("Initial system temperature: %lf\n", thermostat.measure(system));
  system.apply(thermostat);
	printf("Corrected system temperature: %lf\n", thermostat.measure(system));

  //system.print();

  int simulation_steps = 15000;    // understand it as "frames", how many steps in time
  double time_step = 0.0000001;
  double sample_period = 0.0000005;
  
   double t = 0;
   for (int i=0; i<simulation_steps; i++) {
	   
	   if(t > sample_period) {
            system.write_xyz(outputf);
            t = 0;
        }

		    system.simulation_step(time_step);

        thermostat.setpoint += 5e-2;
		    //system.apply(thermostat);

        t += time_step;

        printf("%d : temperature %lf\n", i, thermostat.measure(system));
    }
  
 //system.print();
}