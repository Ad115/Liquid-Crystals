/* 
## Clase `Vector`
Esta clase representa en realidad un array (por default su tamaño es 3 y de 
tipo double).

Las anotaciones "`__host__ __device__` " permiten que este vector pueda existir 
en el CPU(host) o en el GPU(device). 
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
    Type& operator[](int index) { return vector[index]; }

    __host__ __device__ 
    Type operator[](int index) const { return vector[index]; }

    __host__ __device__ 
    bool operator==(const Vector<Size, Type>& other) const {
        for (int i=0; i<Size; i++) 
            if ((*this)[i] != other[i]) return false;
        return true;
    }

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
    Vector<Size, Type> operator*=(const double& r){
        for (int i=0; i<Size; i++) 
            (*this)[i] *= r;
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
    Vector<Size, Type> operator-() const { /*
        * Minus operator:
        * (-v)[i] == -v[i]
        */
        Vector<Size, Type> result;
        result -= *this;

        return result;
    }

    __host__ __device__ 
    Type operator*(const Vector<Size, Type>& other) const { /*
        * Dot product: 
        * (v1 * v2) = v1[1]*v2[1] + v1[2]*v2[2] + ...
        */
        Type result = 0;
        for (int i=0; i<Size; i++) 
            result += (*this)[i] * other[i];

        return result;
    }

    __host__ __device__ 
    static Vector<Size, Type> null() { return Vector<Size, Type>{}; }

    __host__ __device__ 
    bool is_null(){
        auto magnitude_sqrd = (*this) * (*this);
        return magnitude_sqrd == (Type) 0.;
    }

    __host__ __device__ 
    bool is_null(Type epsilon){
        auto magnitude_sqrd = (*this) * (*this);
        return magnitude_sqrd <= epsilon;
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

template<int Size, typename T>
constexpr int Vector<Size, T>::dimensions;

template <int Size, typename T>
__host__ __device__ 
Vector<Size, T> operator*(double r, const Vector<Size, T>& v) { /*
    * Multiplication by scalar (from the right):
    *   (a * v)[i] = a * v[i]
    */
    Vector<Size, T> result(v);
    for (int i=0; i<Size; i++) {
        result[i] *= r;
    }
    return result;
}


template <int Size, typename T>
__host__ __device__ 
Vector<Size, T> operator*(const Vector<Size, T>& v, double r) { /*
    * Multiplication by scalar (from the left).
    */
    return r * v; // <- Delegate to multiplication by scalar from the right
}

template <int Size, typename T>
__host__ __device__ 
Vector<Size, T> operator/(const Vector<Size, T>& v, double r) { /*
    * Division by scalar.
    */
    Vector<Size, T> result(v);
    for (int i=0; i<Size; i++) {
        result[i] /= r;
    }
    return result;
}


template <int Size, typename T>
__host__ __device__ 
Vector<Size, T> operator+(double c, const Vector<Size, T>& v) { /*
    * Sum with scalar from the right (broadcasting):
    *   (a + v)[i] = a + v[i]
    * This behavior is similar to numpy arrays. Useful if one wants a vector 
    * filled with a specific value:
    *       auto all_ones = (1 + Vector<>{});
    */
    Vector<Size, T> result(v);
    for (int i=0; i<Size; i++) {
        result[i] += c;
    }
    return result;
}

template <int Size, typename T>
__host__ __device__ 
Vector<Size, T> operator+(const Vector<Size, T>& v, double c) { /*
    * Sum with scalar from the left (broadcasting).
    */
    return (c+v); // <- Delegate to sum with scalar from the right
}



/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/

#ifdef __TESTING__

#include "doctest.h"
#include <typeinfo>   // operator typeid

SCENARIO("Vector specification") {

    GIVEN("A vector") {

        Vector<3, double> v {1., 3., 5.};
        REQUIRE(v.dimensions == 3);
    
        THEN("It has elements that can be accessed randomly") {
    
            CHECK(v[1] == 3.);
            CHECK(v[0] == 1.);
            CHECK(v[2] == 5.);
    
            AND_THEN("It's elements can be modified independently") {
    
                v[1] += 2;
                CHECK(v[1] == 5.);
    
                v[2] = 0.3;
                CHECK(v[2] == 0.3);
            }
        }
    
        THEN("Another vector can be initialized from it") {
            auto v2 = v;
            CHECK(v2 == Vector<>{1., 3., 5.});
        }

        THEN("It can be inverted") {
            auto minus_v = -v;
            CHECK(minus_v[0] == -v[0]);
            CHECK(minus_v[1] == -v[1]);
            CHECK(minus_v[2] == -v[2]);
        }

        THEN("It is unchanged by adding or substracting a null vector") {
            auto null_vector = Vector<>::null();
            CHECK(v + null_vector == v);
            CHECK(v - null_vector == v);
        }

        THEN("Dot product with itself gives the magnitude squared") {
            auto magnitude_sqrd = (
                v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
            );

            CHECK(v*v == magnitude_sqrd);
        }

        THEN("It can be multiplied by a scalar from both sides") {
            CHECK(2*v == v + v);
            CHECK(v*2 == v + v);
        }
    }
    
    GIVEN("Two vectors") {
        Vector<> u {1., 3., 5.};
        Vector<> v {4., 2., 0.};
    
        THEN("They can be added component-wise") {
            CHECK((u + v) == Vector<>{5., 5., 5.});
    
            Vector<> sum;
            sum += u;
            sum += v;
            CHECK(sum == Vector<>{5., 5., 5.});
        }
    
        THEN("They can be substracted component-wise") {
            CHECK((u - v) == Vector<>{-3., 1., 5.});
    
            Vector<> diff = u;
            diff -= v;
            CHECK(diff == Vector<>{-3., 1., 5.});
        }

        THEN("We can form the dot product between them") {
            auto dot_product = (
                u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
            );
            CHECK(u*v == dot_product);
        }

        THEN("We can form a linear combination of them") {
            auto linear_combination = Vector<>{
                2*u[0]+3*v[0], 2*u[1]+3*v[1], 2*u[2]+3*v[2]
            };
            CHECK((2*u + 3*v) == linear_combination);
        }
    }

    SUBCASE("There can be vectors of different sizes and types") {

        GIVEN("A vector with seven integers") {
            auto vector_7_int = Vector<7, int>{1,2,3,4,5,6,7};

            THEN("The instance values must be correct") {
                CHECK( vector_7_int.dimensions == 7 );
                CHECK( vector_7_int[0] == 1 );
                CHECK( vector_7_int[3] == 4 );
                CHECK( vector_7_int[6] == 7 );
            }

            THEN("The class values must be correct") {
                
                // vector_7_int_t <- class of vector_7_int
                using vector_7_int_t = decltype(vector_7_int);

                CHECK( vector_7_int_t::dimensions == 7 );

                // vector_7_int_value_t <- class of each item in vector_7_int
                using vector_7_int_value_t = typename vector_7_int_t::value_type;

                CHECK( typeid(vector_7_int_value_t) == typeid(int) );
            }
        }
    }
} // <- Vector spec
#endif
#endif