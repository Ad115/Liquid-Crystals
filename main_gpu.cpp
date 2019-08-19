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

#include "src/ParticleSystem.cu"
#include "src/Particle.cu"
#include "src/Vector.cu"

#include <fstream>

int main(void)
{
  int n_particles = 100;
  double numeric_density = 0.1;
  int steps = 500;    // understand it as "frames", how many steps in time
  double dt = 0.0003;

  std::ofstream outputf("output.xyz");

  ParticleSystem<LennardJones<>, PeriodicBoundaryBox<>> sys(n_particles, numeric_density);
  sys.simulation_init();
  sys.print();
  
  // Particle data lives in GPU now so we call the kernel on them few times!
  // This is great! As we don't have to be retrieving and re-sending, Thrust
  // functionality shines in this step. Great framework.
  for (int i=0; i<steps; i++) {
	sys.simulation_step(dt);

	sys.write_xyz(outputf);
	printf("%d \n", i);
  }
  
 sys.print();
}