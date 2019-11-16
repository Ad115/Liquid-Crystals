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
#include "pcuditas/gpu/gpu_array.cu"
#include "pcuditas/gpu/gpu_object.cu"
#include "pcuditas/vectors/EuclideanVector.cu"
#include "pcuditas/particles/SimpleParticle.cu"
#include "pcuditas/initial_conditions/move_to_origin.cu"
#include "pcuditas/integrators/RandomWalk.cu"
#include "pcuditas/integrators/SimpleIntegrator.cu"
#include "pcuditas/integrators/force_calculation/shared2.cu"
#include "pcuditas/interactions/LennardJones.cu"
#include "pcuditas/environments/EmptySpace.cu"
#include "pcuditas/environments/PeriodicBoundaryBox.cu"
#include "pcuditas/input_output/XYZformat.cu"
