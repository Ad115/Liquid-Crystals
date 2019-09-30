#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "src_gpu/Vector.cu"

TEST_CASE("A vector") {

    Vector<3, double> v {1., 3., 5.};
    REQUIRE(v.size == 3);

    SUBCASE("Has elements that can be accessed randomly") {

        REQUIRE(v[1] == 3.);
        REQUIRE(v[0] == 1.);
        REQUIRE(v[2] == 5.);

        SUBCASE("And modified independently") {

            v[1] += 2;
            REQUIRE(v[1] == 5.);

            v[2] = 0.3;
            REQUIRE(v[2] == 0.3);
        }
    }

    SUBCASE("Can be initialized from another one") {
        auto v2 = v;

        REQUIRE(v2[1] == 3.);
        REQUIRE(v2[0] == 1.);
        REQUIRE(v2[2] == 5.);

        REQUIRE(v2 == Vector<>{1., 3., 5.});
    }
}

TEST_CASE("Two vectors") {
    Vector<> u {1., 3., 5.};
    Vector<> v {4., 2., 0.};

    SUBCASE("Can be added component-wise") {
        auto sum = u + v;
        REQUIRE(sum == Vector<>{5., 5., 5.});

        Vector<> sum_;
        sum_ += u;
        sum_ += v;
        REQUIRE(sum_ == Vector<>{5., 5., 5.});
    }

    SUBCASE("Can be substracted component-wise") {
        auto sum = u - v;
        REQUIRE(sum == Vector<>{-3., 1., 5.});

        Vector<> sum_ = u;
        sum_ -= v;
        REQUIRE(sum_ == Vector<>{-3., 1., 5.});
    }
}