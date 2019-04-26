#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>
using namespace std;
// https://stackoverflow.com/questions/34637490/thrusttransform-custom-function
typedef struct particula
{
    double velocity[3];
    double position[3];
    double density[3];
} Particula;


thrust::device_vector<Particula> applyFunction( thrust::device_vector<Particula> vect )
{
    thrust::device_vector<Particula> dev_out( vect.size() );
    auto ff = [=]  __device__ (Particula x)
    {
        Particula z;
        z.velocity[0]=0;
        z.velocity[1]=1;
        z.velocity[2]=2;

        z.position[0]=0;
        z.position[1]=1;
        z.position[2]=2;

        z.density[0]=0;
        z.density[1]=1;
        z.density[2]=2;

        return z;
    };
    thrust::transform(vect.begin(), vect.end(), dev_out.begin(),ff);
    return dev_out;
}

void printParticles(thrust::device_vector<Particula> dev)
{
    //Copiar del device al host
    thrust::host_vector<Particula> host;

    host=dev;
    int j=0;
    for( int i=0; i<host.size(); ++i )
    {
        cout << "Velocity: ";
        for( j=0; j<3; ++j )
            cout << " " << host[i].velocity[j];
        cout << endl;
        cout << "Position: ";
        for( j=0; j<3; ++j )
            cout << " " << host[i].position[j];
        cout << endl;
        cout << "Density: ";
        for( j=0; j<3; ++j )
            cout << " " << host[i].density[j];
        cout << endl;
    }
}
int main(void)
{
    // initialize all ten integers of a device_vector to 1
    thrust::device_vector<Particula> D(10);
    thrust::device_vector<Particula> D2;
    D2=applyFunction(D);
    printParticles(D2);
    /*
    // set the first seven elements of a vector to 9
    thrust::fill(D.begin(), D.begin() + 7, 9);

    // initialize a host_vector with the first five elements of D
    thrust::host_vector<int> H(D.begin(), D.begin() + 5);

    // set the elements of H to 0, 1, 2, 3, ...
    thrust::sequence(H.begin(), H.end());

    // copy all of H back to the beginning of D
    thrust::copy(H.begin(), H.end(), D.begin());

    // print D
    for(int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;
*/
    return 0;
}