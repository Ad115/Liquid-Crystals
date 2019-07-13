#include "src/ParticleSystem.cu"

int main(void)
{
  int n_particles = 10;
  int steps = 3;    // understand it as "frames", how many steps in time

  ParticleSystem<> sys(n_particles);
  
  // Particle data lives in GPU now so we call the kernel on them few times!
  // This is great! As we don't have to be retrieving and re-sending, Thrust
  // functionality shines in this step. Great framework.
  for (int i=0; i<steps; i++) {
    sys.simulation_step();
    sys.print();
  }
}