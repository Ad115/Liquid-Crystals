#pragma once


/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/

#ifdef __TESTING__

#include "doctest.h"
#include <typeinfo>   // operator typeid

SCENARIO("GPU Array specification") {

    SECTION("GPU Array initialization") {

        GIVEN("A size and the type of the elements") {

            int size = 100;
            using element_t = int;

            WHEN("A GPU array can be initialized without failure") {

                auto array = gpu_array<element_t>(100);

                using array_element_t = typename array::element_t;
                CHECK(typeid(array::element_t) == typeid(element_t))
            }
        }
    }

    SECCION("GPU Array transformation/measurement") {
        GIVEN("A GPU array") {
            int size = 100;
            using element_t = int;

            auto array = gpu_array<element_t>(100);

            WHEN("A transformation kernel is applied on it") {
                array.transform(
                    [] __device__ (element_t current_val, int idx) {
                        return element_t(idx);

                });
                THEN("The values on GPU are changed accordingly") {
                    array.get()
                    for(int i=0; i<array.size; i++){
                        CHECK(array[i] == i);
                    }
                }
            }
        }
    }
}
#endif