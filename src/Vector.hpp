#ifndef VECTOR_HEADER
#define VECTOR_HEADER

#include <vector>
#include <iostream>
#include <iterator>
#include "Random.hpp"

class Vector {
    private:
        std::vector<double> vector;                            

    public:
        explicit Vector(int dimensions) 
            : vector(dimensions) {}

        unsigned dimensions() const { return vector.size(); }

        double& operator[](int index) { return vector[index]; }
        double operator[](int index) const { return vector[index]; }

        static Vector random_unit(unsigned int dimensions) { /*
            * Generate a new random unit vector
            */
            Vector result(dimensions);

            for (int D=0; D<dimensions; D++)
                result[D] = random_uniform();

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

        friend std::ostream& operator<<(std::ostream& stream, const Vector& vector);
};

std::ostream& operator<<(std::ostream& stream, const Vector& v) {
    stream << '[';
    std::copy(std::begin(v.vector), std::end(v.vector)-1, 
              std::ostream_iterator<double>(stream, ", "));
    stream << v.vector.back() << ']';
    return stream;
}

#endif