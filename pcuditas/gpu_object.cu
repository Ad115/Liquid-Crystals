/* 
## Clase `gpu_object`

Una clase que emula el comportamiento de un *smart pointer* para objetos en el 
GPU. La contraparte del `gpu_array` pero para un sólo objeto.

La clase abstrae la alocación y liberación de memoria además de las operaciones 
de copia entre Host y Device. Esta es una abstracción del Host, por lo que no 
se puede utilizar en un kernel. 
*/

#include "pcuditas/gpu/macros.cu"

template<typename T>
__global__
void _init_object_kernel(T *device_pointer) {
         
  new (device_pointer) T();
}


template< typename T >
class gpu_object {
  
    T *_gpu_pointer;
    T *_cpu_pointer;
    
    public:

    using value_t = T;
    
    gpu_object() {
        // <-- Allocate and initialize on GPU
        CUDA_CALL(cudaMalloc(&_gpu_pointer, sizeof(T)));      

        _init_object_kernel<<<1,1>>>(_gpu_pointer);


        // <-- Allocate and initialize on CPU
        _cpu_pointer = (T *) malloc(sizeof(T));
        new (&_cpu_pointer) T();
    }
    
    T *gpu_pointer() const {
        return _gpu_pointer;
    }

    void to_cpu() {
        CUDA_CALL(cudaMemcpy(
            _cpu_pointer, _gpu_pointer, 
            sizeof(T), 
            cudaMemcpyDeviceToHost
        ));
    }

    T *cpu_pointer() const {
        to_cpu();
        return _cpu_pointer;
    }

    T operator*() {
        to_cpu();
        return *_cpu_pointer;
    }
    
    ~gpu_object() {
        free(_cpu_pointer);
        CUDA_CALL(cudaFree(_gpu_pointer));
    }
};

/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/

#ifdef __TESTING__

#include "doctest.h"
#include <typeinfo>   // operator typeid

TEST_SUITE("GPU object specification") {

    SCENARIO("GPU object initialization") {

        GIVEN("A type of object") {
            using element_t = int;

            THEN("A GPU object can be initialized without failure") {

                auto gpu_obj = gpu_object<element_t>{};

                using gpu_obj_element_t = decltype(gpu_obj)::element_t;
                CHECK(typeid(gpu_obj_element_t) == typeid(element_t));
            }
        }
    }
}
#endif