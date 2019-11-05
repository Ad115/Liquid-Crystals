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
#include "pcuditas/gpu/gpu_object.cu"


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

template<typename T, typename Reduction>
__global__
void _reduce_kernel(T *gpu_array, size_t n, T *out, Reduction fn) { /*
    Log-reduction based from the one in the book "The CUDA Handbook" by 
    Nicholas Wilt.
    */

    extern __shared__ T partials[];

    const int tid = threadIdx.x;

    auto reduced = T{};
    for (int i = blockIdx.x * blockDim.x + tid; 
         i < n; 
         i += blockDim.x * gridDim.x) {

        reduced = fn(reduced, gpu_array[i]);
    }
    partials[tid] = reduced;
    __syncthreads();


    for (int active_threads = blockDim.x / 2;
         active_threads > 0;
         active_threads /= 2) {
        
        auto is_active_thread = tid < active_threads;
        if (is_active_thread) {
            partials[tid] = fn(partials[tid], partials[tid + active_threads]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = partials[0];
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
    void transform(
            TransformationT gpu_fn,
            int n_blocks = 1024, 
            int n_threads = 32 ){
        _transform_kernel<<<n_blocks, n_threads>>>(_gpu_pointer, size, gpu_fn);
    }

    template <class FunctionT>
    void for_each(
            FunctionT gpu_fn,
            int n_blocks = 1024, 
            int n_threads = 32 ) {
        _for_each_kernel<<<n_blocks, n_threads>>>(_gpu_pointer, size, gpu_fn);
    }

    template <class ReductionT>
    gpu_object<T> reduce(
            ReductionT reduce_fn, 
            int n_blocks = 128, 
            int threads_per_block = 32 /* <-- Must be a power of 2! */ ) {

        unsigned int shared_memory_size = threads_per_block * sizeof(T);

        auto thread_partials = gpu_array<T>(threads_per_block);
        _reduce_kernel<<<n_blocks, threads_per_block, shared_memory_size>>>(
            _gpu_pointer, size, thread_partials.gpu_pointer(), reduce_fn
        );

        auto out = gpu_object<T>();
        _reduce_kernel<<<1, threads_per_block, shared_memory_size>>>(
           thread_partials.gpu_pointer(), size, out.gpu_pointer(), reduce_fn
        );

        return out;
    }

    // Iterator protocol
    T* begin() { return _cpu_pointer; }
    T* end() { return _cpu_pointer + size; }
    
    gpu_array<T> copy() {
        auto copied = gpu_array<T>(this->size);

        // Copy in GPU
        copied.for_each(
            [old_one=this->gpu_pointer()]
            __device__ (T &new_el, int i) {
                new_el = old_one[i];
        });

        // Copy in CPU
        for(int i=0; i<size; i++) {
            copied[i] = (*this)[i];
        }

        return copied;
    }

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


                // Check on gpu
                array.for_each(
                    [] __device__ (element_t val, int idx){
                        assert(val == element_t(idx*idx));
                });

                // Check on cpu
                array.to_cpu();
                for(int i=0; i<array.size; i++){
                    CHECK(array[i] == element_t(i*i));
                }
            }

        }
    }

    SCENARIO("GPU Array reduction") {
        GIVEN("A GPU array with arbitrary elements") {
            int size = 1000;
            using element_t = int;

            auto array = gpu_array<element_t>(size);
            array.for_each(
                [] __device__ (element_t &el, int i) {
                    el = i+1;
            });

            WHEN("A reducion operation is applied on it") {
                auto sum_gpu = array.reduce(
                    [] __device__ (element_t reduced, element_t el) {
                        return reduced + el;
                }).to_cpu();

                THEN("The reduction on CPU yields the same result") {

                    array.to_cpu(); // <-- check on CPU

                    auto sum_cpu = array[0];
                    for(int i=1; i<array.size; i++){
                        sum_cpu += array[i];
                    }

                    CHECK(sum_cpu == sum_gpu);
                }
            }

        }
    }
}
#endif