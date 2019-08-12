/* 
## Clase `Container`

El espacio donde se mueven las partículas. Contiene la información sobre las 
condiciones de frontera y la distancia entre partículas. Está diseñada para 
vivir en el Device. 
*/

#ifndef CONTAINER_HEADER
#define CONTAINER_HEADER

#include "Vector.cu"


template < typename VectorT >
class Container {
    private:

    public:
        using vector_type = VectorT;
        static constexpr int dimensions = VectorT::dimensions;

        __host__ __device__ 
        VectorT apply_boundary_conditions(const VectorT& position) const;

        __host__ __device__
        VectorT distance_vector(const VectorT& p1, const VectorT& p2) const;
};

template< typename VectorT=Vector<> >
class PeriodicBoundaryBox : public Container<VectorT> {

    double side_length;

    public: 

    __host__ __device__ PeriodicBoundaryBox( double L )
            : side_length(L) {}

    __host__ __device__ 
    VectorT apply_boundary_conditions(const VectorT& position) const {
        VectorT new_pos(position);
        for (int i=0; i<new_pos.dimensions;i++){
            new_pos[i] -= (new_pos[i] > side_length) * side_length;
            new_pos[i] += (new_pos[i] < 0) * side_length;
        }
        return new_pos;
    }

    __host__ __device__ 
    VectorT box_size() { 
        VectorT side_lengths;
        for (int i=0; i<dimensions; i++)
            side_lengths[i] = side_length;

        return side_lengths;
    }

    __host__ __device__ 
    VectorT distance_vector(const VectorT& r1, const VectorT& r2) const { /*
        * Get the distance to the minimum image.
        */
        double half_length = side_length/2;
        VectorT dr = r2 - r1;

        for(int D=0; D<dimensions; D++) {

            if (dr[D] <= -half_length) {
                dr[D] += side_length;

            } else if (dr[D] > +half_length) {
                dr[D] -= side_length;
            }
        }
        return dr;
    }
};

template<typename ContainerT> 
__global__ 
void init_container_kernel(ContainerT *ptr, double side_length) {
    new (ptr) ContainerT(side_length);
}

template<typename ContainerT>
__host__ __device__
void print_container(ContainerT *ptr) {
    printf("Container = {side_lengths:");
    auto sides = (*ptr).box_size();
    print_vector( &sides );
    printf("}");
}

template<typename ContainerT>
__global__ 
void print_container_kernel(ContainerT *ptr) {
    print_container(ptr);
    printf("\n");
}


#endif
