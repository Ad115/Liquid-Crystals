/* run tests with:
   $ make test

   The tests are found in each source file and given in the format of 
   "Executable documentation" as described in Kevlin Henney's talk "Structure 
   and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic) and they are 
   writen using the doctest framework (https://github.com/onqtam/doctest).
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#define __TEST__

// --- Files to test
#include "src_gpu/Vector.cu"
#include "src_gpu/Container.cu"