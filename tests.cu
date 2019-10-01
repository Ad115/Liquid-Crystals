/* run tests with:
   $ make test

   The tests are given in the format of "Executable documentation" as described 
   in Kevlin Henney's talk "Structure and Interpretation of Test Cases" 
   (https://youtu.be/tWn8RA_DEic) and they are writen using the doctest 
   framework (https://github.com/onqtam/doctest). 
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "src_gpu/Vector.cu"
#include <typeinfo>   // operator typeid


SCENARIO("Vector specification") {

    GIVEN("A vector") {

        Vector<3, double> v {1., 3., 5.};
        REQUIRE(v.dimensions == 3);
    
        THEN("It has elements that can be accessed randomly") {
    
            CHECK(v[1] == 3.);
            CHECK(v[0] == 1.);
            CHECK(v[2] == 5.);
    
            AND_THEN("It's elements can be modified independently") {
    
                v[1] += 2;
                CHECK(v[1] == 5.);
    
                v[2] = 0.3;
                CHECK(v[2] == 0.3);
            }
        }
    
        THEN("Another vector can be initialized from it") {
            auto v2 = v;
            CHECK(v2 == Vector<>{1., 3., 5.});
        }

        THEN("It is unchanged by adding or substracting a null vector") {
            auto null_vector = Vector<>::null();
            CHECK(v + null_vector == v);
            CHECK(v - null_vector == v);
        }

        THEN("Dot product with itself gives the magnitude squared") {
            auto magnitude_sqrd = (
                v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
            );

            CHECK(v*v == magnitude_sqrd);
        }

        THEN("It can be multiplied by a scalar from both sides") {
            CHECK(2*v == v + v);
            CHECK(v*2 == v + v);
        }
    }
    
    GIVEN("Two vectors") {
        Vector<> u {1., 3., 5.};
        Vector<> v {4., 2., 0.};
    
        THEN("They can be added component-wise") {
            CHECK((u + v) == Vector<>{5., 5., 5.});
    
            Vector<> sum;
            sum += u;
            sum += v;
            CHECK(sum == Vector<>{5., 5., 5.});
        }
    
        THEN("They can be substracted component-wise") {
            CHECK((u - v) == Vector<>{-3., 1., 5.});
    
            Vector<> diff = u;
            diff -= v;
            CHECK(diff == Vector<>{-3., 1., 5.});
        }

        THEN("We can form the dot product between them") {
            auto dot_product = (
                u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
            );
            CHECK(u*v == dot_product);
        }

        THEN("We can form a linear combination of them") {
            auto linear_combination = Vector<>{
                2*u[0]+3*v[0], 2*u[1]+3*v[1], 2*u[2]+3*v[2]
            };
            CHECK((2*u + 3*v) == linear_combination);
        }
    }

    SUBCASE("There can be vectors of different sizes and types") {

        GIVEN("A vector with seven integers") {
            auto vector_7_int = Vector<7, int>{1,2,3,4,5,6,7};

            THEN("The instance values must be correct") {
                CHECK( vector_7_int.dimensions == 7 );
                CHECK( vector_7_int[0] == 1 );
                CHECK( vector_7_int[3] == 4 );
                CHECK( vector_7_int[6] == 7 );
            }

            THEN("The class values must be correct") {
                
                // vector_7_int_t <- class of vector_7_int
                using vector_7_int_t = decltype(vector_7_int);

                CHECK( vector_7_int_t::dimensions == 7 );

                // vector_7_int_value_t <- class of each item in vector_7_int
                using vector_7_int_value_t = typename vector_7_int_t::value_type;

                // Class of the contained values must be int
                CHECK( typeid(vector_7_int_value_t) == typeid(int) );
            }
        }
    }
} // <- Vector
