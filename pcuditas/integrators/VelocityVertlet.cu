#pragma once

#include "pcuditas/gpu/gpu_array.cu"
#include "pcuditas/gpu/gpu_object.cu"
#include "pcuditas/environments/EmptySpace.cu"
#include "force_calculation.cu"

#include <curand.h>
#include <curand_kernel.h>
#include <memory>


class VelocityVertlet { /*
    * The simplest integrator (Runge-Kutta):
    *   1. update f
    *   2. x -> x + v dt;
    *   3. v -> v + f dt;
    */
    gpu_object<EmptySpace> default_environment;
public:

    VelocityVertlet() = default;

    template <class ParticleT, class EnvironmentT>
    void integration_step(
            gpu_array<ParticleT> &particles, 
            gpu_object<EnvironmentT> &environment,
            double dt = 0.001) { /*
        * Implementation of a velocity Vertlet integrator.
        * See: http://www.pages.drexel.edu/~cfa22/msim/node23.html#sec:nmni
        * 
        * This integrator gives a lower error O(dt^4) and more stability than
        * the standard forward integration (x(t+dt) += v*dt + 1/2 * f * dt^2)
        * by looking at more timesteps (t, t+dt) AND (t-dt), but in order to 
        * improve memory usage, the integration is done in two steps.
        */

        // r(t + dt) = r(t) + v(t)*dt + 1/2*f(t)*dt^2
        update_positions(particles, environment, dt);
        // v(t + 1/2*dt) = v(t) + 1/2*f(t)*dt
        update_velocities(particles, dt);

        // r(t + dt)  -->  f(t + dt)
        update_forces(particles, environment);

        // v(t + dt) = v(t + 1/2*dt) + 1/2*f(t + dt)*dt
        update_velocities(particles, dt);
    }

    template <class ParticleT, class EnvironmentT>
    void update_positions(
            gpu_array<ParticleT> &particles, 
            gpu_object<EnvironmentT> &env,
            double dt = 0.001) {

        
        particles.for_each(
            [dt, box_ptr=env.gpu_pointer()] 
            __device__ (ParticleT &p, int i) {

                // r(t + dt) = r(t) + v(t)*dt + 1/2*f(t)*dt^2
                auto new_pos = p.position + p.velocity*dt + 0.5*p.force*dt*dt;

                p.position = box_ptr->apply_boundary_conditions(new_pos); 
            }
        );
    }

    template <class ParticleT>
    void update_velocities(gpu_array<ParticleT> &particles, double dt = 0.001) {

        particles.for_each(
            [dt] __device__ (ParticleT &p, int i) {

                // v(t + dt) = v(t) + 1/2*f(t + dt)*dt
                p.velocity = p.velocity + 0.5*p.force*dt;
            }
        );
    }
    
    template <class ParticleT, class EnvironmentT>
    void update_forces(
                gpu_array<ParticleT> &particles, 
                gpu_object<EnvironmentT> &env) {
        update_forces_shared(particles, env);
    }

    template<typename SystemT>
    void operator()(SystemT& system) {
        integration_step(system);
    }

    template <class ParticleT>
    void operator()(
            gpu_array<ParticleT> &particles,
            double dt = 0.001) {

        integration_step(particles, default_environment, dt);
    }

    template <class ParticleT, class EnvironmentT>
    void operator()(
            gpu_array<ParticleT> &particles, 
            gpu_object<EnvironmentT> &env,
            double dt = 0.001) {

        integration_step(particles, env, dt);
    }
};

