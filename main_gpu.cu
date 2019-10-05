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

#include "src_gpu/core/GPUParticleSystem.cu"
#include "src_gpu/PeriodicBoundaryBox.cu"
#include "src_gpu/LennardJones.cu"
#include "src_gpu/InitialConditions.cu"
#include "src_gpu/VelocityVertletIntegrator.cu"
#include "src_gpu/Temperature.cu" 

#include <fstream>
#include <iostream>

using LJSystem = GPUParticleSystem<
  LennardJones<>,
  PeriodicBoundaryBox<>
>;

int main(void)
{
  int n_particles = 200;
  double numeric_density = 0.5;
  Temperature thermostat{5e2};

  std::ofstream outputf("output.xyz");

  LJSystem system(n_particles, numeric_density);
  system.simulation_init(initial_conditions{});

	//std::cout << "Initial system temperature: " << Temperature::measure(system) << std::endl;
  //system.apply(thermostat);
  //std::cout << "Corrected system temperature: " << Temperature::measure(system) << std::endl;


  //std::cout << system;

  int simulation_steps = 15000;    // understand it as "frames", how many steps in time
  double time_step = 1e-10;
  double sample_period = 1e-10;

  auto integration_method = VelocityVertlet{};
  
   double t = 0;
   for (int i=0; i<simulation_steps; i++) {
	   
	   if(t > sample_period) {
            system.write_xyz(outputf);
            std::cout << i << " temperature " << Temperature::measure(system) << std::endl;
            t = 0;
        }

        system.simulation_step(time_step, integration_method);

        thermostat.setpoint += 5e-2;
        //system.apply(thermostat);

        t += time_step;
    }
  
  //std::cout << system;
}