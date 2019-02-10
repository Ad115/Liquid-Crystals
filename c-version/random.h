#ifndef RANDOM_HEADER
#define RANDOM_HEADER

#include <stdbool.h>

/*  
    DECLARATIONS (INTERFACE)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void init_random_with_seed(unsigned long int seed); /*
    * Initialize the c random number generator with the given seed.
    */

void init_random(); /*
    * Initialize the c random number generator with time as the seed.
    * Useful for when one does not need repeatability.
    */

int constrain(int i, int min, int max); /**
    Constrain the input to the range [min, max]
    */

int random_integer(int a, int b);  /**
    Return a random integer in the range [a,b].
    */

int random_index(int n); /**
    Return a random integer in the range [0,n).
    */

double random_uniform(); /**
    A random number between 0 and 1.
    */

bool random_test(double probability); /**
    Return a boolean for a test with the given probability of acceptance.
    */

#endif