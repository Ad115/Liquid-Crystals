#pragma once

#include <assert.h>

template< typename T >
class cpu_array {
  
    T *array;
    
    public:

    size_t size;
    using element_t = T;
    
    cpu_array(size_t n): size(n) {
        assert(n > 0);

        // <-- Allocate and initialize on CPU
        array = (T *) malloc(n * sizeof(T));

        for (int i=0; i<n; i++) {
            new (&array[i]) T();
        }
    }

    template <class InitializerT>
    cpu_array(size_t n, InitializerT init_fn)
            : cpu_array(n) { /*

        Instantiate with a function to initialize each value.

        Example:
            // Initialize to {0, 1, 2, 3, 4...9}
            auto array = cpu_array<int>(10, 
                [](int &el, int i) {
                    el = i;
            });
        */

        // Apply the initialization function to each element
        (*this).for_each(init_fn);
    }
    
    T& operator[](size_t idx) {
        return array[idx];
    }

    template <class TransformedT, class TransformationT>
    cpu_array<TransformedT> transform(
            TransformationT cpu_fn) { /*

        Create a new array with the transformed elements.
        
        Example:

            // Create cpu_array "array" with the numbers 0 to 9
            auto array = cpu_array<int>(10, 
                [](int &el, int i) {
                    el = i;
            });
            
            
            // Transform to pairs of the number and it's squares
            auto squares = array.transform<int2>(
                [](int2 el, int idx) {
                    return make_int2(el, el*el);
            });
    */

        auto transformed = cpu_array<TransformedT>{this->size};

        for (int i=0; i < this->size; i++) {
            transformed[i] = cpu_fn(array[i], i);
        }

        return transformed;
    }

    template <class FunctionT>
    cpu_array<T>& for_each(
            FunctionT cpu_fn) {/*

        Apply the function in-place for each element of the array.
        
        Example:

            // Create cpu_array "array" with the numbers 0 to 9
            auto array = cpu_array<int>(10, 
                [](int &el, int i) {
                    el = i;
            });
            
            
            // Make a linear transformation
            int a = 12;
            int b = 34;

            array.for_each(
                [a,b](int2 &el, int idx) {
                    el = a*el + b;
            });
        */

        for(int i=0; i < this->size; i++) {
            cpu_fn(array[i], i);
        }

        return *this;
    }

    template <class ReductionT>
    T reduce(
            ReductionT reduce_fn) { /*
        
        Perform a reduction of the elements of the array using the provided function.

        Example:

            // Create cpu_array "array" with the numbers 0 to 9
            auto array = cpu_array<int>(10, 
                [](int &el, int i) {
                    el = i;
            });
            
            
            // Multipliy the elements in GPU
            cpu_object<int> product 
                = array.reduce(
                    [](int a, int b) {
                        return a * b
                });
        */

        auto reduced = array[0];
        for(int i=1; i < this->size; i++) {
            reduced = reduce_fn(reduced, array[i]);
        }

        return reduced;
    }

    // Iterator protocol
    T* begin() { return array; }
    T* end() { return array + size; }
    
    cpu_array<T> copy() {
        auto copied = cpu_array<T>(this->size);

        for(int i=0; i<size; i++) {
            copied[i] = array[i];
        }

        return copied;
    }

    ~cpu_array() {
        free(array);
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

// template<class T>
// struct pair {
//     T first; T second;
// };

TEST_SUITE("CPU Array specification") {

    SCENARIO("CPU Array initialization") {

        GIVEN("A size and the type of the elements") {

            int size = 10;
            using element_t = int;

            THEN("A GPU array can be initialized without failure") {

                auto array = cpu_array<element_t>(size);

                using array_element_t = decltype(array)::element_t;
                CHECK(typeid(array_element_t) == typeid(element_t));
                CHECK(array.size == size);
            }
        }
    }

    SCENARIO("CPU Array for_each") {
        GIVEN("A CPU array") {
            int size = 10;
            using element_t = int;

            auto array = cpu_array<element_t>(size);

            WHEN("It's elements are modified with for_each") {
                array.for_each(
                    [] (element_t &el, int idx) {
                        el = idx * idx;
                });

                THEN("The values are persistent") {

                    for(int i=0; i<array.size; i++){
                        CHECK(array[i] == i*i);
                    }
                }
            }
        }
    }

    SCENARIO("CPU Array transformation") {
        GIVEN("A CPU array") {
            int size = 10;
            using element_t = int;

            // Initialize to {0, 1, 2, 3, 4...9}
            auto array = cpu_array<element_t>(size, 
                [](element_t &el, int i) {
                    el = i;
            });

            WHEN("A new array is obtained as a transformation of it") {

                auto squares = array.transform<pair<element_t>>(
                    [] (element_t &el, int idx) {
                        return pair<element_t>{el, el*el};
                });

                THEN("The values persist accordingly") {

                    for(int i=0; i<squares.size; i++){
                        CHECK(squares[i].first == i);
                        CHECK(squares[i].second == i*i);
                    }
                }
            }
        }
    }

    SCENARIO("CPU Array reduction") {
        GIVEN("A CPU array with arbitrary elements") {
            int size = 1000;
            using element_t = int;

            // Initialize to {0, 1, 2, 3, 4...n}
            auto nums = cpu_array<int>(size, 
                []  (int &el, int idx) {
                    el = idx;
            });

            WHEN("A reducion operation is applied on it") {
                auto addition = 
                    []  (int a, int b) {
                        return a + b;
                };
              
                auto sum = nums.reduce(addition);

                THEN("The reduction yields the expected value") {

                    CHECK(sum == size*(size-1)/2);
                }
            }

        }

    }
}
#endif