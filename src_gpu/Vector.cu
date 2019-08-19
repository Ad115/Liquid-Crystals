/* 
## Clase `Vector`
Esta clase representa en realidad un array (por default su tamaño es 3 y de 
tipo double).

Las anotaciones "`__host__ __device__` " permiten que este vector pueda existir 
en el CPU(host) o en el GPU(device). 

Esta clase quedó básicamente como ya la teníamos, con cambios menores. 
*/

#ifndef VECTOR_HEADER
#define VECTOR_HEADER

#include <iostream>
#include <initializer_list>

template<int Size=3, typename Type=double>
class Vector {
    private:
        //In stack
        Type vector[Size];

    public:

    using value_type = Type;    
    static constexpr int dimensions = Size;

    template <typename... T>
    __host__ __device__ 
    Vector(T ...entries) : vector{entries...} {}

    __host__ __device__ 
    double& operator[](int index) { return vector[index]; }

    __host__ __device__ 
    double operator[](int index) const { return vector[index]; }

    __host__ __device__ 
    Vector<Size, Type> operator+=(const Vector<Size, Type>& other){
        for (int i=0; i<Size; i++) 
            (*this)[i] += other[i];
        return *this;
    }

    __host__ __device__ 
    Vector<Size, Type> operator-=(const Vector<Size, Type>& other){
        for (int i=0; i<Size; i++) 
            (*this)[i] -= other[i];
        return *this;
    }

    __host__ __device__ 
    Vector<Size, Type> operator+(const Vector<Size, Type>& other) const { /*
        * Component-wise addition:
        * (v1 + v2)[i] == v1[i] + v2[i]
        */            
        Vector<Size,Type> result{*this};
        result += other;

        return result;
    }

    __host__ __device__ 
    Vector<Size, Type> operator-(const Vector<Size, Type>& other) const { /*
        * Component-wise substraction:
        * (v1 - v2)[i] == v1[i] - v2[i]
        */
        Vector<Size, Type> result(*this);
        result -= other;

        return result;
    }

    __host__ __device__ 
    double operator*(const Vector<Size, Type>& other) const { /*
        * Dot product: 
        * (v1 * v2) = v1[1]*v2[1] + v1[2]*v2[2] + ...
        */
        double result = 0;
        for (int i=0; i<Size; i++) 
            result += (*this)[i] * other[i];

        return result;
    }


    friend std::ostream& operator<<(
        std::ostream& stream, 
        const Vector<Size, Type>& vector) {

        stream << '[';
        for (int i=0; i < Size-1; i++)
            stream << vector[i] << ", ";
                    
        stream << vector[Size-1] << ']';
        return stream;
    }
};

template <int Size, typename T>
__host__ __device__ 
Vector<Size, T> operator*(double r, const Vector<Size, T>& v) {
    Vector<Size, T> result(v);
    for (int i=0; i<Size; i++) {
        result[i] *= r;
    }
    return result;
}


template <int Size, typename T>
__host__ __device__ 
Vector<Size, T> operator*(const Vector<Size, T>& v, double r) {
    Vector<Size, T> result(v);
    for (int i=0; i<Size; i++) {
        result[i] *= r;
    }
    return result;
}


template <int Size, typename T>
__host__ __device__ 
Vector<Size, T> operator+(double c, const Vector<Size, T>& v) {
    Vector<Size, T> result(v);
    for (int i=0; i<Size; i++) {
        result[i] += c;
    }
    return result;
}

template <int Size, typename T>
__host__ __device__ 
void print_vector( Vector<Size,T> *ptr) {
    printf("[");
    for (int i=0; i<(Size-1); i++) {
        printf("%.2f, ", (*ptr)[i]);
    }
    printf("%.2f]", (*ptr)[Size-1]);
}

template <int Size, typename T>
__global__ 
void print_vector_kernel( Vector<Size,T> *ptr) {
    print_vector(ptr);
    printf("\n");
}

template <int Size, typename T>
__host__ __device__ 
void init_vector( Vector<Size,T> *ptr) {
    new (ptr) Vector<Size,T>();
}

/* Init the vector object in the device */
template <int Size, typename T>
__global__ 
void init_vector_kernel( Vector<Size,T> *ptr) {
    init_vector(ptr);
}

template <int Size, typename T>
void cuda_alloc_vector( Vector<Size,T> **vector_ptr) {
    cudaMallocManaged(vector_ptr, sizeof(*vector_ptr));
    init_vector_kernel<<<1,1>>>(*vector_ptr);
}

#endif