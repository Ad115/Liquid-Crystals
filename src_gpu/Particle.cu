/* 
## Clase `Particle`

Al igual que la clase `Vector`, la clase `Particle` es una clase que puede ser 
instanciada tanto en el Host como en el Device. Contiene vectores de posición, 
velocidad y fuerza. 
*/ 

#ifndef PARTICLE_HEADER
#define PARTICLE_HEADER
#include "Vector.cu"

template< typename VectorT=Vector<> >
class Particle {
    public:
        
        using vector_type = VectorT;
        static constexpr int dimensions = VectorT::dimensions;
        
        VectorT position;
        VectorT velocity;
        VectorT force;

        __host__ __device__
        double kinetic_energy() {
            auto& v = velocity;
            return 1/2. * (v*v);
        }
};

template<class VectorT>
constexpr int Particle<VectorT>::dimensions;

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

template<typename ParticleT>
__host__ __device__
void init_particle(ParticleT *P) {
        new (P) ParticleT{};
}


#endif