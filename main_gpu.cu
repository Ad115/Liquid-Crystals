/* 
### El primer `main`

Lo siguiente es para probar la funcionalidad del `ParticleSystem`, pero también 
sirve como primer prototipo de la simulación completa, ya que ya contamos con 
los elementos básicos para ella.

Para compilar y ejecutar:
```
nvcc test_ParticleSystem.cu -o test_ParticleSystem
./test_ParticleSystem
```

Salida esperada:
```
Container: 
	Container = {side_lengths:[10.00, 10.00, 10.00]}
Particles: 
	{"position": [0.50, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.50, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.50, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.50, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.50, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.50, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.50, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.50, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.50, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Container: 
	Container = {side_lengths:[10.00, 10.00, 10.00]}
Particles: 
	{"position": [1.00, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.00, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.00, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.00, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.00, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.00, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.00, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.00, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.00, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
```
*/

#include "src_gpu/ParticleSystem.cu"
#include "src_gpu/Particle.cu"
#include "src_gpu/Vector.cu"

#include <fstream>

int main(void)
{
  int n_particles = 100;
  double numeric_density = 0.05;

  std::ofstream outputf("output.xyz");

  ParticleSystem<LennardJones<>, PeriodicBoundaryBox<>> system(n_particles, numeric_density);
  system.simulation_init();
  system.print();

  int simulation_steps = 10000;    // understand it as "frames", how many steps in time
  double time_step = 0.0001;
  double sample_period = 0.0007;
  
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
  
 system.print();
}