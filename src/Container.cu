#ifndef CONTAINER_HEADER
#define CONTAINER_HEADER

#include "Vector.cu"


template < int Size=3 >
class Container {
    private:
        double _side_lengths[Size];
        //std::vector<double> _side_lengths;

    public:
        __host__ __device__ Container( double side_length ) {
            for (int i=0; i<Size; i++) {
                _side_lengths[i] = side_length;
            }

        }

        template <typename... T>
        __host__ __device__ Container(T ...entries) : _side_lengths{entries...} {}

        __host__ __device__ unsigned int dimensions() const { return Size; }
        __host__ __device__ double side_length() const { return _side_lengths[0]; }
        // friend std::ostream& operator<<(std::ostream&, const Container&);

        __host__ __device__ Vector<> apply_boundary_conditions(const Vector<>& position) const;
        __host__ __device__ Vector<> distance_vector(const Vector<>& p1, const Vector<>& p2) const;

        template <int S>
        friend __global__ void print_device_container(Container<S> *ptr);
};

template <int Size>
class PeriodicBoundaryBox : public Container<> {
    public: 

    __host__ __device__ PeriodicBoundaryBox( double side_length )
            : Container<Size>(side_length)
              {}

    __host__ __device__ Vector<> apply_boundary_conditions(const Vector<>& position) const {
        Vector<> new_pos(position);
        for (int i=0; i<new_pos.dimensions();i++){
            new_pos[i] -= (new_pos[i] > side_length()) * side_length();
            new_pos[i] += (new_pos[i] < 0) * side_length();
        }
        return new_pos;
    }

    __host__ __device__ Vector<> distance_vector(const Vector<>& r1, const Vector<>& r2) const { /*
        * Get the distance to the minimum image.
        */
        double half_length = side_length()/2;
        Vector<> dr = r1 - r2;

        for(int D=0; D<dimensions(); D++) {

            if (dr[D] <= -half_length) {
                dr[D] += side_length();

            } else if (dr[D] > +half_length) {
                dr[D] -= side_length();
            }
        }
        return dr;
    }
};

template <int Size> 
__global__ void init_device_container(Container<Size> *ptr, double side_length) {
    new (ptr) Container<Size>(side_length);
}

template <int Size> 
__global__ void print_device_container(Container<Size> *ptr) {
    for (int i=0; i<Size; i++) {
        printf("%f ", (*ptr)._side_lengths[i]);
    }
    printf("\n");
}

/*
std::ostream& operator<<(std::ostream& stream, 
                         const Container& box) {
    stream << "{";
    stream << "\"side_lengths\": [";

    auto sides = box.side_lengths();
    std::copy(std::begin(sides), std::end(sides)-1, 
              std::ostream_iterator<double>(stream, ", "));

    stream << sides.back() 
           << "]";

    stream << "}";
    return stream;
}
*/

#endif
