#pragma once

/* 
## Clase `gpu_array`

Un *smart pointer* para arreglos de objetos en el GPU. La contraparte del 
`gpu_object` pero para arreglos.

La clase abstrae la alocación y liberación de memoria además de las operaciones 
de copia entre Host y Device. Esta es una abstracción del Host, por lo que no 
se puede utilizar en un kernel. 
*/

#include "pcuditas/gpu/macros.cu"


template<typename T>
__global__
void _init_array_kernel(T *gpu_array, size_t n) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n; 
        i += blockDim.x * gridDim.x) 
    {
        new (&gpu_array[i]) T();
    }
}

template<typename T, typename Transformation>
__global__
void _transform_kernel(T *gpu_array, size_t n, Transformation fn) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n; 
        i += blockDim.x * gridDim.x) 
    {
        gpu_array[i] = fn(gpu_array[i], i);
    }
}

template<typename T, typename Transformation>
__global__
void _for_each_kernel(T *gpu_array, size_t n, Transformation fn) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n; 
        i += blockDim.x * gridDim.x) 
    {
        fn(gpu_array[i], i);
    }
}



template< typename T >
class gpu_array {
  
    T *_gpu_pointer;
    T *_cpu_pointer;
    
    public:

    size_t size;
    using element_t = T;
    
    gpu_array(size_t n): size(n) {
        // <-- Allocate and initialize on GPU
        CUDA_CALL(cudaMalloc(&_gpu_pointer, n * sizeof(T)));      

        _init_array_kernel<<<128,32>>>(_gpu_pointer, n);


        // <-- Allocate and initialize on CPU
        _cpu_pointer = (T *) malloc(n * sizeof(T));

        for (int i=0; i<n; i++) {
            new (&_cpu_pointer[i]) T();
        }
    }
    
    T *gpu_pointer() const {
        return _gpu_pointer;
    }
    
    void to_cpu() {
        CUDA_CALL(cudaMemcpy(
            _cpu_pointer, _gpu_pointer, 
            size*sizeof(T), 
            cudaMemcpyDeviceToHost
        ));
    }

    T *cpu_pointer() const {
        to_cpu();
        return _cpu_pointer;
    }

    T operator[](size_t idx) {
        return _cpu_pointer[idx];
    }

    template <class TransformationT>
    void transform(TransformationT gpu_fn){
        _transform_kernel<<<128,32>>>(_gpu_pointer, size, gpu_fn);
    }

    template <class FunctionT>
    void for_each(FunctionT gpu_fn){
        _for_each_kernel<<<128,32>>>(_gpu_pointer, size, gpu_fn);
    }

    // Iterator protocol
    T* begin() { return _cpu_pointer; }
    T* end() { return _cpu_pointer + size; }
    
    ~gpu_array() {
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
#include <assert.h>

TEST_SUITE("GPU Array specification") {

    SCENARIO("GPU Array initialization") {

        GIVEN("A size and the type of the elements") {

            int size = 10;
            using element_t = int;

            THEN("A GPU array can be initialized without failure") {

                auto array = gpu_array<element_t>(size);

                using array_element_t = decltype(array)::element_t;
                CHECK(typeid(array_element_t) == typeid(element_t));
                CHECK(array.size == size);
            }
        }
    }

    SCENARIO("GPU Array transformation") {
        GIVEN("A GPU array") {
            int size = 10;
            using element_t = int;

            auto array = gpu_array<element_t>(size);

            WHEN("A transformation kernel is applied on it") {
                array.transform(
                    [] __device__ (element_t current_val, int idx) {
                        return element_t(idx);
                });

                THEN("The values on GPU are changed accordingly") {

                    array.for_each( // <-- check on GPU
                        [] __device__ (element_t current_val, int idx){
                            assert(current_val == idx);
                    });

                    array.to_cpu(); // <-- check on CPU
                    for(int i=0; i<array.size; i++){
                        CHECK(array[i] == element_t(i));
                    }
                }
            }

            SUBCASE("Another syntax for a transformation") {
                array.for_each(
                    [] __device__ (element_t &el, int idx) {
                        el = element_t(idx * idx);
                });
                
                array.to_cpu();
                for(int i=0; i<array.size; i++){
                    CHECK(array[i] == element_t(i*i));
                }
            }

        }
    }
}
#endif