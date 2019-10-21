/* run tests with:
   $ make test

   The tests are found in each source file and given in the format of 
   "Executable documentation" as described in Kevlin Henney's talk "Structure 
   and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic) and they are 
   writen using the doctest framework (https://github.com/onqtam/doctest).
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#define __TESTING__

// --- Files to test
#include "pcuditas/gpu_array.cu"
#include "pcuditas/vectors/EuclideanVector.cu"
#include "pcuditas/particles/SimpleParticle.cu"
#include "pcuditas/transform_measure/Spatial.cu"