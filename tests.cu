#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "src_gpu/Vector.cu"

TEST_CASE("testing compilation") {
    CHECK(1 == 1);
}