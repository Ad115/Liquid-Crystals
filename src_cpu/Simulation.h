#ifndef SIMULATION_HEADER
#define SIMULATION_HEADER

class NoType { public: void operator()(void){}; };

template< typename ParticleSystem, 
          typename StepModifier=NoType, 
          typename Sampler=NoType  >
class Simulation {
    unsigned simulation_steps;
    double time_step;
    double sample_period;
    
    ParticleSystem &system;
    StepModifier step_modifier;
    Sampler sampler;
    

    public:
        Simulation(ParticleSystem& system, 
                   unsigned steps,
                   double dt)
         : system(system),
           simulation_steps(steps),
           time_step(dt)
         {}

        Simulation(ParticleSystem& system, 
                   unsigned steps,
                   double dt,
                   double sampleT,
                   StepModifier step_fn,
                   Sampler sample_fn)
         : system(system),
           simulation_steps(steps),
           time_step(dt),
           sample_period(sampleT),
           sampler(sample_fn),
           step_modifier(step_fn)
         {}

        template< typename Step >
        auto at_each_step(Step step_fn) {
            return Simulation<ParticleSystem, Step, Sampler>(
                        system,
                        simulation_steps, time_step, sample_period,
                        step_fn,
                        sampler
                   );
        }

        template< typename Sample >
        auto take_samples(double sampling_period, Sample sample_fn) {
            return Simulation<ParticleSystem, StepModifier, Sample>(
                        system,
                        simulation_steps, time_step, sampling_period,
                        step_modifier,
                        sample_fn
                   );
        }

        void run() { /*
            * Simulation loop: 
            *   -> sample (if sufficient time has elapsed)
            *   -> step
            *   -> callback for step corrections.
            *   REPEAT.
            */

            double t = 0;
            for (int step=0; step<simulation_steps; step++) {

                if(t > sample_period) {
                    sampler(step);
                    t = 0;
                }

                system.simulation_step(time_step);
                t += time_step;

                step_modifier(step);
            }
        }
};


#endif