#pragma once

#include "pcuditas/gpu/gpu_array.cu"
#include "pcuditas/gpu/gpu_object.cu"
#include "pcuditas/environments/EmptySpace.cu"
#include "force_calculation/shared2.cu"

#include <curand.h>
#include <curand_kernel.h>
#include <memory>


class SimpleIntegrator { /*
    * The simplest integrator (Runge-Kutta):
    *   1. update f
    *   2. x -> x + v dt;
    *   3. v -> v + f dt;
    */
    gpu_object<EmptySpace> default_environment;
public:

    SimpleIntegrator() = default;

    template <class ParticleT, class EnvironmentT>
    void operator()(
            gpu_array<ParticleT> &particles, 
            gpu_object<EnvironmentT> &env,
            double dt = 0.001) {
        
        this->move(particles, dt);

        this->apply_boundary_conditions(particles, env);
    }

    template <class ParticleT, class EnvironmentT, class InteractionT>
    void operator()(
            gpu_array<ParticleT> &particles, 
            gpu_object<EnvironmentT> &env,
            InteractionT &interaction,
            double dt = 0.001) {

        this->update_forces(particles, interaction);
        
        this->move(particles, dt);

        this->apply_boundary_conditions(particles, env);
    }


    template <class ParticleT, class EnvironmentT>
    void apply_boundary_conditions(
                gpu_array<ParticleT> &particles, 
                gpu_object<EnvironmentT> &env) {

        particles.for_each(
            [env_ptr=env.gpu_pointer()] 
            __device__ 
            (ParticleT &p, size_t idx) {
                p.position = env_ptr->apply_boundary_conditions(p.position);
            }
        );
    }

    template <class ParticleT, class InteractionT>
    void update_forces(
                gpu_array<ParticleT> &particles, 
                InteractionT &interaction) {

        using vector_t = typename ParticleT::vector_type;

        auto update_force = 
            [particles_gpu=particles.gpu_pointer()]
            __device__
            (ParticleT &p, vector_t &force, int idx) {
                p.force = force;
        };

        
        update_forces_shared2(
            particles, interaction, vector_t::zero(), update_force
        );
    }

    template <class ParticleT>
    void move(
            gpu_array<ParticleT> &particles,
            double dt = 0.01) {

        particles.for_each(
            [dt] 
            __device__ (ParticleT &p, size_t i) {
                // x -> x + v dt;
                p.position += p.velocity * dt;

                // v -> v + f dt;
                p.velocity += 1/2. * p.force * dt * dt;
            }
        );
    }
};


/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/

#ifdef __TESTING__

#include "tests/doctest.h"
#include <typeinfo>   // operator typeid
#include "pcuditas/initial_conditions/random.cu"
#include "pcuditas/particles/SimpleParticle.cu"
#include "pcuditas/vectors/EuclideanVector.cu"

TEST_SUITE("Simple Integrator specification") {

    SCENARIO("No forces") {
        GIVEN("A GPU array of particles with arbitrary positions and velocities") {

            using vector_t = EuclideanVector<3>;
            using particle_t = Particle<vector_t>;
            double L = 100.;
            double V = 10.;

            auto particles = gpu_array<particle_t>(100);
            set_random_positions(particles, L);
            set_random_velocities(particles, V);

            auto snapshot = particles.copy();

            WHEN("A Simple Integrator is used to move them") {
                auto integrator = SimpleIntegrator{};
                int steps = 1000;
                for (int i=0; i<steps; i++) {
                    integrator.move(particles);
                }

                THEN("Each particle has moved along the direction of their velocity") {

                    auto are_colineal = [] (vector_t v1, vector_t v2) {
                        return (
                            (v1 * v2) / (v1.magnitude() * v2.magnitude() )
                            == doctest::Approx(1.)
                        );
                    };
                    
                    snapshot.to_cpu();
                    particles.to_cpu();
                    for (int i=0; i<particles.size; i++) {
                        auto initial_velocity = snapshot[i].velocity;
                        auto final_velocity = particles[i].velocity;

                        // Velocity didn't change
                        CHECK(initial_velocity == final_velocity);

                        auto displacement = particles[i].position - snapshot[i].position;

                        CHECK(are_colineal(displacement, initial_velocity));
                    }
                }
            }
        }
    }
}

#endif