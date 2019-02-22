#ifndef VECTOR_HEADER
#define VECTOR_HEADER

#include <vector>
#include <iostream>
#include <iterator>
#include <initializer_list>
#include "Random.hpp"

#ifdef CUDA_ENABLED

template< int Size=3, typename Type=double>
class Vector {
    private:
        //In stack
        Type vector[Size];                

    public:
        template <typename... T>
        __host__ __device__ Vector(T ...entries) : vector{entries...} {}

        __host__ __device__ unsigned dimensions() const { return Size; }

        __host__ __device__ double& operator[](int index) { return vector[index]; }
        __host__ __device__ double operator[](int index) const { return vector[index]; }

        static Vector<Size, Type> random_unit() { /*
            * Generate a new random unit vector
            */
            Vector<Size, Type> result{};

            for (int D=0; D<Size; D++)
                result[D] = 2*(0.5 - random_uniform());

            return result;
        }

        __host__ __device__ Vector<Size, Type> operator+(
                const Vector<Size, Type>& other) const { /*
            * Component-wise addition:
            * (v1 + v2)[i] == v1[i] + v2[i]
            */            
            Vector<Size,Type> result{*this};
            for (int i=0; i<Size; i++) 
                result[i] += other[i];

            return result;
        }

        __host__ __device__ Vector<Size, Type> operator-(
                const Vector<Size, Type>& other) const { /*
            * Component-wise substraction:
            * (v1 - v2)[i] == v1[i] - v2[i]
            */
            Vector<Size, Type> result(*this);
            for (int i=0; i<Size; i++) 
                result[i] -= other[i];

            return result;
        }

        __host__ __device__ double operator*(
                const Vector<Size, Type>& other) const { /*
            * Dot product: 
            * (v1 * v2) = v1[1]*v2[1] + v1[2]*v2[2] + ...
            */
            double result = 0;
            for (int i=0; i<Size; i++) 
                result += (*this)[i] * other[i];

            return result;
        }

        __host__ __device__ Vector<Size, Type> operator+=(const Vector<Size, Type>& other){
                for (int i=0; i<Size; i++) 
                    (*this)[i] += other[i];
            return *this;
        }

        __host__ __device__ Vector<Size, Type> operator-=(const Vector<Size, Type>& other){
                for (int i=0; i<Size; i++) 
                    (*this)[i] -= other[i];
            return *this;
        }


        //friend std::ostream& operator<<(std::ostream& stream, const Vector<Size, Type>& vector);
};
/*
__host__ std::ostream& operator<<(std::ostream& stream, const Vector& v) {
    stream << '[';
    std::copy(std::begin(v.vector), std::end(v.vector)-1, 
              std::ostream_iterator<double>(stream, ", "));
    stream << v.vector.back() << ']';
    return stream;
}
*/
template <int Size, typename T>
__host__ __device__ Vector<Size, T> operator*(double r, const Vector<Size, T>& v) {
    Vector<Size, T> result(v);
    for (int i=0; i<Size; i++) {
        result[i] *= r;
    }
    return result;
}
template <int Size, typename T>
__host__ __device__ void printVector( Vector<Size,T> *ptr) {
    for (int i=0; i<Size; i++) {
        printf("%f ", (*ptr)[i]);
    }
    printf("\n");
}

template <int Size, typename T>
__global__ void printVectorDevice( Vector<Size,T> *ptr) {
    for (int i=0; i<Size; i++) {
        printf("%f ", (*ptr)[i]);
    }
    printf("\n");
}

/* Init the vector object in the device */
template <int Size, typename T>
__global__ void initVector( Vector<Size,T> *ptr) {
    new (ptr) Vector<Size,T>();

    for (int i=0; i<Size; i++) {
        (*ptr)[i] = 5.*i;
    }
}


/* If CUDA is NOT enabled, then create a chafa sequential Vector */
#else
template< int Size=3, typename Type=double>
class Vector {
    private:
        //In stack
        Type vector[Size];                    

    public:
        
        __host__ __device__ unsigned dimensions() const { return Size; }

         double& operator[](int index) { return vector[index]; }
         double operator[](int index) const { return vector[index]; }

        static Vector random_unit(unsigned int dimensions) { /*
            * Generate a new random unit vector
            */
            Vector result(dimensions);

            for (int D=0; D<dimensions; D++)
                result[D] = 2*(0.5 - random_uniform());

            return result;
        }

        Vector operator+(const Vector& other) const { /*
            * Component-wise addition:
            * (v1 + v2)[i] == v1[i] + v2[i]
            */            
            Vector result(*this);
            for (int i=0; i<result.dimensions(); i++) 
                result[i] += other[i];

            return result;
        }

        Vector operator-(const Vector& other) const { /*
            * Component-wise substraction:
            * (v1 - v2)[i] == v1[i] - v2[i]
            */
            Vector result(*this);
            for (int i=0; i<result.dimensions(); i++) 
                result[i] -= other[i];

            return result;
        }

        double operator*(const Vector& other) const { /*
            * Dot product: 
            * (v1 * v2) = v1[1]*v2[1] + v1[2]*v2[2] + ...
            */
            double result = 0;
            for (int i=0; i<other.dimensions(); i++) 
                result += (*this)[i] * other[i];

            return result;
        }

	Vector operator+=(const Vector& other){
            for (int i=0; i<other.dimensions(); i++) 
                (*this)[i] += other[i];
	    return *this;
	}

	Vector operator-=(const Vector& other){
            for (int i=0; i<other.dimensions(); i++) 
                (*this)[i] -= other[i];
	    return *this;
	}


        //friend std::ostream& operator<<(std::ostream& stream, const Vector& vector);
};
/*
std::ostream& operator<<(std::ostream& stream, const Vector& v) {
    stream << '[';
    std::copy(std::begin(v.vector), std::end(v.vector)-1, 
              std::ostream_iterator<double>(stream, ", "));
    stream << v.vector.back() << ']';
    return stream;
}
*/

template < int Size, typename T>
Vector<Size,T> operator*(double r, const Vector<Size,T>& v) {
    Vector<Size,T> result(v);
    for (int i=0; i<v.dimensions(); i++) {
        result[i] *= r;
    }
    return result;
}

#endif




#endif
