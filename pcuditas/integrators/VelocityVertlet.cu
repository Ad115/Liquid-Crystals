#pragma once

#include "pcuditas/gpu/gpu_array.cu"
#include "pcuditas/gpu/gpu_object.cu"
#include "pcuditas/environments/EmptySpace.cu"
#include "pcuditas/interactions/LennardJones.cu"
#include "pcuditas/integrators/force_calculation/shared2.cu"

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

    template <class ParticleT, class EnvironmentT, class InteractionT>
    void integration_step(
            gpu_array<ParticleT> &particles, 
            gpu_object<EnvironmentT> &environment,
            InteractionT interaction,
            double dt = 0.001,
            int n_blocks = 1024,
            int threads_per_block = 32) { /*
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
        update_forces(particles, environment, interaction, n_blocks, threads_per_block);

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
    
    template <class ParticleT, class EnvironmentT, class InteractionT>
    void update_forces(
                gpu_array<ParticleT> &particles, 
                gpu_object<EnvironmentT> &env,
                InteractionT &interaction,
                int n_blocks,
                int threads_per_block) {

        using vector_t = typename ParticleT::vector_type;

        auto update_force = 
            [particles_gpu=particles.gpu_pointer()]
            __device__
            (ParticleT &p, vector_t &force, int idx) {
                p.force = force;
        };

        
        update_forces_shared2(
            particles, interaction, vector_t::zero(), update_force,
            n_blocks, threads_per_block
        );
    }

    template <class ParticleT, class InteractionT>
    void operator()(
            gpu_array<ParticleT> &particles,
            InteractionT &interaction,
            double dt = 0.001,
            int n_blocks = 1024,
            int threads_per_block = 32) {

        integration_step(particles, default_environment, interaction, dt, n_blocks, threads_per_block);
    }

    template <class ParticleT, class EnvironmentT, class InteractionT>
    void operator()(
            gpu_array<ParticleT> &particles, 
            gpu_object<EnvironmentT> &env,
            InteractionT &interaction,
            double dt = 0.001,
            int n_blocks = 1024,
            int threads_per_block = 32) {

        integration_step(particles, env, interaction, dt, n_blocks, threads_per_block);
    }
};


