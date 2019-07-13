#ifndef PARTICLE_HEADER
#define PARTICLE_HEADER
#include "Vector.cu"

template< typename VectorT=Vector<> >
class Particle {
    public:
        VectorT position;
        VectorT velocity;
        VectorT force;

        __host__ __device__
        unsigned dimensions() const { return position.dimensions(); }

        __host__ __device__
        void set_position(const VectorT& new_position) { position = new_position; }

        __host__ __device__
        void set_velocity(const VectorT& new_velocity) { velocity = new_velocity; }

        __host__ __device__
        void set_force(const VectorT& new_force) { force = new_force; }

        __host__ __device__
        double kinetic_energy() {
            Vector<>& v = velocity;
            return 1/2. * (v*v);
        }
};

template<typename VectorT>
__host__ __device__
void print_particle( Particle<VectorT> *P) {
    printf("{");
        printf("\"position\": ");
        print_vector(&P->position);
        printf(", ");

        printf("\"velocity\": ");
        print_vector(&P->velocity);
        printf(", ");

        printf("\"force\": ");
        print_vector(&P->force);
    printf("}");
}

template<typename VectorT>
__host__ __device__
void init_particle(Particle<VectorT> *P) {
        new (P) Particle<VectorT>{};
}

#endif