/*  
    random: DEFINITIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <stdlib.h>
#include <time.h>
#include "random.h"

void init_random_with_seed(unsigned long int seed) { /*
    * Initialize the c random number generator with the given seed.
    */
    srand(seed);
}

void init_random(){ /*
    * Initialize the c random number generator with time as the seed.
    * Useful for when one does not need repeatability.
    */
    init_random_with_seed(time(0));
}

int constrain(int i, int min, int max) { /**
    Constrain the input to the range [min, max]
    */
    i = (i > min) ? i : min;
    i = (i < max) ? i : max;
    return i;
}

int random_integer(int a, int b) { /**
    Return a random integer in the range [a,b].
    */
    int r =  a + (rand() % (b-a + 1));
    return constrain(r, a, b);
}

int random_index(int n) { /**
    Return a random integer in the range [0,n).
    */
    int r = random_integer(0, n-1);
    return constrain(r, 0, n-1);
}

double random_uniform() { /**
    A random number between 0 and 1.
    */
    return (double)rand() / (double)RAND_MAX; 
} 

bool random_test(double probability) { /**
    Return a boolean for a test with the given probability of acceptance.
    */
    double selector = random_uniform();
    return (selector < probability);
}