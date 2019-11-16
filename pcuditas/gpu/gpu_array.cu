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
#include "pcuditas/gpu/kernels.cu"
#include "pcuditas/gpu/gpu_object.cu"
#include <assert.h>



template< typename T >
class gpu_array {
  
    T *_gpu_pointer;
    T *_cpu_pointer;
    
    public:

    size_t size;
    using element_t = T;
    
    gpu_array(size_t n): size(n) {
        assert(size > 0);

        // <-- Allocate and initialize on GPU
        CUDA_CALL(cudaMalloc(&_gpu_pointer, n * sizeof(T)));      

        init_array_kernel<<<128,32>>>(_gpu_pointer, n);


        // <-- Allocate and initialize on CPU
        _cpu_pointer = (T *) malloc(n * sizeof(T));

        for (int i=0; i<n; i++) {
            new (&_cpu_pointer[i]) T();
        }
    }

    template <class InitializerT>
    gpu_array(size_t n, InitializerT init_fn)
            : gpu_array(n) { /*

        Instantiate with a function to initialize each value.

        Example:
            // Initialize to {0, 1, 2, 3, 4...9}
            auto array = gpu_array<int>(10, 
                []__device__ (int &el, int i) {
                    el = i;
            });
        */

        // Apply the initialization function to each element
        (*this).for_each(init_fn);
    }
    
    T *gpu_pointer() const {
        return _gpu_pointer;
    }
    
    T *to_cpu() { 
        CUDA_CALL(cudaMemcpy(
            _cpu_pointer, _gpu_pointer, 
            size*sizeof(T), 
            cudaMemcpyDeviceToHost
        ));

        return _cpu_pointer;
    }

    T *cpu_pointer() const {
        return _cpu_pointer;
    }

    T operator[](size_t idx) {
        return _cpu_pointer[idx];
    }

    template <class TransformedT, class TransformationT>
    gpu_array<TransformedT> transform(
            TransformationT gpu_fn,
            int n_blocks = 1024, 
            int n_threads = 32 ) { /*

        Create a new array with the transformed elements.
        
        Example:

            // Create gpu_array "array" with the numbers 0 to 9
            auto array = gpu_array<int>(10, 
                []__device__ (int &el, int i) {
                    el = i;
            });
            
            
            // Transform to pairs of the number and it's squares
            auto squares = array.transform<int2>(
                [] __device__ (int2 el, int idx) {
                    return make_int2(el, el*el);
            });
    */

        auto transformed = gpu_array<TransformedT>{this->size};

        transform_kernel<<<n_blocks, n_threads>>>(
            _gpu_pointer, size, 
            transformed.gpu_pointer(),
            gpu_fn
        );

        return transformed;
    }

    template <class FunctionT>
    gpu_array<T>& for_each(
            FunctionT gpu_fn,
            int n_blocks = 1024, 
            int n_threads = 32 ) {/*

        Apply the function in-place for each element of the array.
        
        Example:

            // Create gpu_array "array" with the numbers 0 to 9
            auto array = gpu_array<int>(10, 
                []__device__ (int &el, int i) {
                    el = i;
            });
            
            
            // Make a linear transformation
            int a = 12;
            int b = 34;

            array.for_each(
                [a,b] __device__ (int2 &el, int idx) {
                    el = a*el + b;
            });
        */

        for_each_kernel<<<n_blocks, n_threads>>>(_gpu_pointer, size, gpu_fn);
        return *this;
    }

    template <class ReductionT>
    gpu_object<T> reduce(
            ReductionT reduce_fn, 
            int n_blocks = 128, 
            int threads_per_block = 32 /* <-- Must be a power of 2! */ ) { /*
        
        Perform a reduction of the elements of the array using the provided function.

        Example:

            // Create gpu_array "array" with the numbers 0 to 9
            auto array = gpu_array<int>(10, 
                []__device__ (int &el, int i) {
                    el = i;
            });
            
            
            // Multipliy the elements in GPU
            gpu_object<int> product 
                = array.reduce(
                    []__device__ (int a, int b) {
                        return a * b
                });
        */

        unsigned int shared_memory_size = threads_per_block * sizeof(T);

        auto block_partials = gpu_array<T>(n_blocks);
        reduce_2step_kernel<<<n_blocks, threads_per_block, shared_memory_size>>>(
            _gpu_pointer, size, block_partials.gpu_pointer(), reduce_fn
        );

        auto out = gpu_object<T>();
        reduce_2step_kernel<<<1, threads_per_block, shared_memory_size>>>(
            block_partials.gpu_pointer(), n_blocks, out.gpu_pointer(), reduce_fn
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

#include "tests/doctest.h"
#include <typeinfo>   // operator typeid
#include <assert.h>

template<class T>
struct pair {
    T first; T second;
};

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

    SCENARIO("GPU Array for_each") {
        GIVEN("A GPU array") {
            int size = 10;
            using element_t = int;

            auto array = gpu_array<element_t>(size);

            WHEN("It's elements are modified with for_each") {
                array.for_each(
                    [] __device__ (element_t &el, int idx) {
                        el = idx * idx;
                });

                THEN("The values on GPU are changed accordingly") {

                    array.for_each( // <-- check on GPU
                        [] __device__ (element_t current_val, int idx){
                            assert(current_val == idx*idx);
                    });

                    array.to_cpu(); // <-- check on CPU
                    for(int i=0; i<array.size; i++){
                        CHECK(array[i] == i*i);
                    }
                }
            }
        }
    }

    SCENARIO("GPU Array transformation") {
        GIVEN("A GPU array") {
            int size = 10;
            using element_t = int;

            // Initialize to {0, 1, 2, 3, 4...9}
            auto array = gpu_array<element_t>(size, 
                []__device__ (element_t &el, int i) {
                    el = i;
            });

            WHEN("A new array is obtained as a transformation of it") {

                auto squares = array.transform<pair<element_t>>(
                    [] __device__ (element_t &el, int idx) {
                        return pair<element_t>{el, el*el};
                });

                THEN("The values on GPU are changed accordingly") {

                    squares.for_each( // <-- check on GPU
                        [] __device__ (pair<element_t> p, int idx) {
                            assert(p.first == idx);
                            assert(p.second == idx*idx);
                    });

                    squares.to_cpu(); // <-- check on CPU
                    for(int i=0; i<squares.size; i++){
                        CHECK(squares[i].first == i);
                        CHECK(squares[i].second == i*i);
                    }
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

        SUBCASE("Reduction on a very large array") {
            auto n = 500000;
            
            // Initialize to {0, 1, 2, 3, 4...n}
            auto nums = gpu_array<int>(n, 
                [] __device__ (int &el, int idx) {
                    el = idx;
            });

            auto addition = 
                [] __device__ (int a, int b) {
                    return a + b;
            };

            auto sum = nums.reduce(addition).to_cpu();

            CHECK(sum == n*(n-1)/2);
        }
    }
}
#endif