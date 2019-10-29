#pragma once
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
void _init_empty_object_kernel(T *device_pointer) {
  new (device_pointer) T();
}

template<typename T>
__global__
void _init_from_object_kernel(T *device_pointer, T value) {
  (*device_pointer) = value;
}

template<typename T, typename TransformationT>
__global__
void _transform_object_kernel(T *device_pointer, TransformationT transform) {
  *device_pointer = transform(*device_pointer);
}

template<typename T, typename CallableT>
__global__
void _object_call_on_gpu_kernel(T *device_pointer, CallableT gpu_fn) {
    gpu_fn(*device_pointer);
}


template< typename T >
class gpu_object {
  
    T *_gpu_pointer;
    T *_cpu_pointer;
    
    public:

    using value_t = T;
    
    gpu_object(T object) {
        // <-- Allocate and initialize on GPU
        CUDA_CALL(cudaMalloc(&_gpu_pointer, sizeof(T)));      

        _init_from_object_kernel<<<1,1>>>(_gpu_pointer, object);


        // <-- Allocate and initialize on CPU
        _cpu_pointer = (T *) malloc(sizeof(T));
        (*_cpu_pointer) = object;
    }

    gpu_object() {
        // <-- Allocate and initialize on GPU
        CUDA_CALL(cudaMalloc(&_gpu_pointer, sizeof(T)));      

        _init_empty_object_kernel<<<1,1>>>(_gpu_pointer);


        // <-- Allocate and initialize on CPU
        _cpu_pointer = (T *) malloc(sizeof(T));
        new (&_cpu_pointer) T();
    }
    
    T *gpu_pointer() const {
        return _gpu_pointer;
    }

    T cpu_object() const {
        return *_cpu_pointer;
    }

    T to_cpu() {
        CUDA_CALL(cudaMemcpy(
            _cpu_pointer, _gpu_pointer, 
            sizeof(T), 
            cudaMemcpyDeviceToHost
        ));

        return cpu_object();
    }

    T *cpu_pointer() const {
        to_cpu();
        return _cpu_pointer;
    }

    template <class TransformationT>
    void transform(TransformationT gpu_fn){
        _transform_object_kernel<<<1,1>>>(_gpu_pointer, gpu_fn);
    }

    template <class FunctionT>
    void call_on_gpu(FunctionT gpu_fn){
        _object_call_on_gpu_kernel<<<1,1>>>(_gpu_pointer, gpu_fn);
    }
    
    ~gpu_object() {
        free(_cpu_pointer);
        CUDA_CALL(cudaFree(_gpu_pointer));
    }
};

template <typename T>
gpu_object<T> gpu_object_from(T object) {
    return gpu_object<T>{object};
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
#include <assert.h>

// <- Test structure
template<int N>
struct simple_array {
    static constexpr int size = N;
    int values[N];
};

template<int N>
constexpr int simple_array<N>::size;


TEST_SUITE("GPU object specification") {

    SCENARIO("GPU object initialization") {

        GIVEN("A type of object") {
            using element_t = int;

            THEN("An empty GPU object can be initialized without failure") {

                auto gpu_obj = gpu_object<element_t>{};

                // Check on GPU
                gpu_obj.call_on_gpu([] __device__ (element_t &obj) {
                    assert(obj == element_t{});
                });

                // Check on CPU
                using gpu_obj_element_t = decltype(gpu_obj)::value_t;
                CHECK(typeid(gpu_obj_element_t) == typeid(element_t));
            }
        }

        GIVEN("A CPU object") {
            using element_t = simple_array<3>;
            element_t cpu = {0,1,2};

            WHEN("A GPU object is initialized to mirror it") {
                auto gpu = gpu_object_from(cpu);

                THEN("The object is represented correctly") {

                    // <-- Check on GPU
                    gpu.call_on_gpu([] __device__ (element_t &obj) {

                            assert(obj.size == element_t::size);

                            for(int i=0; i<3; i++)
                                assert(obj.values[i] == i);
                    });

                    // <-- Check on CPU

                    using gpu_element_t = decltype(gpu)::value_t;
                    CHECK(typeid(gpu_element_t) == typeid(element_t));
                    
                    auto cpu_ = gpu.to_cpu();
                    for(int i=0; i<3; i++) {
                        CHECK(cpu_.values[i] == i);
                    }
                }
            }
        }
    }

    SCENARIO("GPU object transformation") {
        using element_t = simple_array<3>;
        element_t cpu = {0,1,2};

        GIVEN("A GPU object") {
            auto gpu = gpu_object_from(cpu);

            WHEN("A transformation is used to change the value on GPU") {
                gpu.transform([] __device__ (element_t obj) {
                    obj.values[2] = 0;
                    return obj;
                });

                THEN("The changes persist on GPU") {
                    gpu.call_on_gpu([] __device__ (element_t obj) {
                        assert(obj.values[0] == 0);
                        assert(obj.values[1] == 1);
                        assert(obj.values[2] == 0);
                    });
                }

                AND_THEN("The changes persist on CPU") {
                    auto cpu_ = gpu.to_cpu();
                    CHECK(cpu_.values[0] == 0);
                    CHECK(cpu_.values[1] == 1);
                    CHECK(cpu_.values[2] == 0);
                }
            }

            SUBCASE("Alternative syntax for GPU object transformation") {
                gpu.call_on_gpu([] __device__ (element_t &obj) {
                    obj.values[2] = 0;
                });

                // check on GPU
                gpu.call_on_gpu([] __device__ (element_t obj) {
                    assert(obj.values[0] == 0);
                    assert(obj.values[1] == 1);
                    assert(obj.values[2] == 0);
                });

                // check on CPU
                auto cpu_ = gpu.to_cpu();
                CHECK(cpu_.values[0] == 0);
                CHECK(cpu_.values[1] == 1);
                CHECK(cpu_.values[2] == 0);
            }
        }

        
    }
}
#endif