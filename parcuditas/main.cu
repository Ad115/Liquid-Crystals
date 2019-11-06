#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#define DIMENSIONS 3
#include <iostream>
using namespace std;



/*
    This class will have the values of each dimension, it can be
    used for velocity, position, density, etc.

    The overloaded operators are supposed to work with cuda kernel.
*/
using data_t = double;
class Vector {
    private:
        data_t values[DIMENSIONS]; //X Y Z

    public:
        // Constructores 
        __host__ __device__ Vector(){}
        __host__ __device__ Vector(data_t X, data_t Y, data_t Z)
            : values{X,Y,Z} {
                
            }

        __host__ __device__ data_t X() {
            return values[0];
        }
        __host__ __device__ data_t Y() {
            return values[1];
        }
        __host__ __device__ data_t Z() {
            return values[2];
        }

        __host__ __device__ void set_X( data_t new_value ) {
            values[0]=new_value;
        }
        __host__ __device__ void set_Y( data_t new_value ) {
            values[1]=new_value;
        }
        __host__ __device__ void set_Z( data_t new_value ) {
            values[2]=new_value;
        }
        
        __host__ __device__ void set( size_t position, data_t new_value ) {
            values[position]=new_value;
        }
        __host__ __device__ data_t operator[](size_t position){
            return values[position];
        }

        __host__ __device__ data_t operator[]( size_t position) const {
            return values[position];
        }
        
        __host__ __device__ data_t get(size_t position){
            return values[position];
        }


        __host__ __device__ size_t size() const{
            return DIMENSIONS;
        }
        
        __host__ __device__ Vector friend operator-(Vector v, Vector w);
        __host__ __device__ Vector friend operator*(data_t c, Vector v);
        __host__ __device__ void operator=(Vector v) {
            for (size_t i=0; i<DIMENSIONS; i++){
                values[i] = v.get(i);
            }
        }

        friend std::ostream& operator<<(std::ostream& stream, const Vector& vector);
};

std::ostream& operator<<(std::ostream& stream, const Vector& v) {
    stream << '[';
    for(int i=0; i<v.size()-1; i++) {
        stream << v[i] << ", ";
    }
    stream << v[v.size()-1] << ']';
    return stream;
}

__device__ Vector operator*(double c, Vector v){
    Vector result;
    for (int i=0; i<DIMENSIONS; i++){
        result.set(i, c * v.get(i) );
    }

    return result;
}

__device__ Vector operator-(Vector v, Vector w){
    Vector outVector;
    for (int i=0;i<DIMENSIONS;i++){
       outVector.set(i,v.values[i] - w.values[i]);
    }
    return outVector;
}
  


/*
    This class will have a particle with its own properties like position,
    velocity, etc. All memory will be reserved in the GPU and all the lambda
    functions.
    
*/
class Particle{
    public:
        Vector velocity;
        Vector position;
        Vector force;
        __host__ __device__ Particle(){};

};

thrust::device_vector<Particle> applyFunctionParticle( thrust::device_vector<Particle> vect )
{
    thrust::device_vector<Particle> dev_out( vect.size() );
    
    auto ff = [=]  __device__ (thrust::tuple<Particle, int> pair)
    {
        Particle p = thrust::get<0>(pair);
        int i = thrust::get<1>(pair);
        p.velocity=Vector(1,2,3);
        p.position=Vector(4,5,6);
        p.force=Vector(7,8,9);

        return p;
    };

    thrust::transform(vect.begin(), vect.end(), dev_out.begin(), PositionInitializer());

    return dev_out;
}
thrust::device_vector<Particle> applyFunctionParticleIter( thrust::device_vector<Particle> vect )
{
    int i=0;
    // second way
    for(thrust::device_vector<Particle>::iterator iter = vect.begin(); iter != vect.end(); iter++)
    {
        //Particle ptr=thrust::raw_pointer_cast(*iter);
        //ptr->velocity.set_X=i;
        //((Particle)(*iter)).velocity.set_X(i);
        //++i;
        /*(static_cast<Particle>(*iter)).velocity.set_Y(i);
        ++i;
        (<Particle>(*iter)).velocity.set_Z(i);
        ++i;*/
    }

    return vect;
}
/*

*/

void printParticles(thrust::device_vector<Particle> vecs_device){
    thrust::host_vector<Particle> host = vecs_device;
    for(int i=0; i<host.size(); ++i ){
        cout << host[i].velocity << " " 
             << host[i].position << " " 
             << host[i].force << endl;
    }
    return;
}


int main(void)
{
    // initialize all ten integers of a device_vector to 1
    thrust::device_vector<Particle> D(5),D2;
    //D2=applyFunctionParticle(D);
    D2=applyFunctionParticleIter(D);
    printParticles(D2);

    return 0;
}