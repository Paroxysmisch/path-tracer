#include <catch2/catch.hpp>
#include "util.cuh"
#include "check_cuda_errors.h"
#include <cmath>

__global__ void immutable_addition_test(pathtracer::vec3 a, 
                                        pathtracer::vec3 b, 
                                        pathtracer::vec3* res) {
    *res = a + b;
}

TEST_CASE("vec3 immutable operations", "[util]") {
    pathtracer::vec3 a{0, 1, 2};
    pathtracer::vec3 b{4, 3, 2};

    SECTION("vec3 equality simple") {
        pathtracer::vec3 c{0, 1, 2};

        bool res1 = (a == c);
        bool res2 = (b == c);

        REQUIRE(res1 == true);
        REQUIRE(res2 == false);
    }

    SECTION("vec3 equality high precision") {
        pathtracer::vec3 _a{-10.0000, 1.2345, 2.6789};
        pathtracer::vec3 _b{-10.0000, 1.2345, 2.6789};
        pathtracer::vec3 _c{-10.0000, 1.2346, 2.6789};

        bool res1 = (_a == _b);
        bool res2 = (_a == _c);

        REQUIRE(res1 == true);
        REQUIRE(res2 == false);
    }

    SECTION("Addition") {
        pathtracer::vec3* res;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&res), sizeof(pathtracer::vec3)) );

        immutable_addition_test<<<1,1>>>(a, b, res);

        checkCudaErrors( cudaDeviceSynchronize() );

        REQUIRE(pathtracer::f_equal(res->x, 4));
        REQUIRE(pathtracer::f_equal(res->y, 4));
        REQUIRE(pathtracer::f_equal(res->z, 4));

        cudaFree(res);
    }

    SECTION("Subtraction") {
        pathtracer::vec3 expected{-4, -2, 0};

        bool res = ((a - b) == expected);

        REQUIRE(res == true);
    }

    SECTION("Negation") {
        pathtracer::vec3 _a{-10.0000, 1.2345, 2.6789};

        pathtracer::vec3 expected{10.0000, -1.2345, -2.6789};

        bool res = ((-_a) == expected);

        REQUIRE(res == true);
    }

    SECTION("Scalar multiplication") {
        pathtracer::vec3 _a{-10.0000, 1.2345, 2.6789};

        pathtracer::vec3 expected{25.0000, -3.08625, -6.69725};

        bool res = ((_a * -2.5) == expected);

        REQUIRE(res == true);
    }

    SECTION("Hadamard product") {
        pathtracer::vec3 expected{0, 3, 4};

        bool res = ((a & b) == expected);

        REQUIRE(res == true);
    }

    SECTION("Scalar division") {
        pathtracer::vec3 _a{-10.0000, 1.2345, 2.6789};

        pathtracer::vec3 expected{4, -0.4938, -1.07156};

        bool res = ((_a / -2.5) == expected);

        REQUIRE(res == true);
    }

    SECTION("Magnitude") {
        pathtracer::vec3 _a{-10.0000, 1.2345, -2.6789};

        float expected1 = 10.4259529761;
        // expected2 is considered equal due to limited precision of floats
        float expected2 = 10.4259529762;
        float expected3 = 10.4259539761;

        bool res1 = _a.mag() == expected1;
        bool res2 = _a.mag() == expected2;
        bool res3 = _a.mag() == expected3;

        REQUIRE(res1 == true);
        REQUIRE(res2 == true);
        REQUIRE(res3 == false);
    }

    SECTION("Dot product") {
        pathtracer::vec3 _a{-10.0000, 1.2345, -2.6789};

        float expected1 = -4.1233;
        float expected2 = -4.1234;

        bool res1 = a * _a == expected1;
        bool res2 = a * _a == expected2;

        REQUIRE(res1 == true);
        REQUIRE(res2 == false);
    }

    SECTION("Cross product") {
        pathtracer::vec3 _a{0.5, -1, 2};
        pathtracer::vec3 expected{-8, 7, 5.5};
        
        bool res = ((_a ^ b) == expected);

        REQUIRE(res == true);
    }
}

TEST_CASE("vec3 mutable operations", "[util]") {
    pathtracer::vec3 a{-10.0000, 1.2345, -2.6789};
    pathtracer::vec3 b{0.12345, 6.78901, -2.345};

    SECTION("Assignment") {
        pathtracer::vec3 expected{0, 0, 0};

        a = {0, 0, 0};

        bool res = (a == expected);

        REQUIRE(res == true);
    }

    SECTION("Addition") {
        pathtracer::vec3 expected{-9.87655, 8.02351, -5.0239};

        bool res = ((a += b) == expected);

        REQUIRE(res == true);
    }

    SECTION("Subtraction") {
        pathtracer::vec3 expected{-10.12345, -5.55451, -0.3339};

        bool res = ((a -= b) == expected);

        REQUIRE(res == true);
    }

    SECTION("Scalar multiplication") {
        pathtracer::vec3 _a{-10.0000, 1.2345, 2.6789};

        pathtracer::vec3 expected{25.0000, -3.08625, -6.69725};

        bool res = ((_a *= -2.5) == expected);
        bool x = pathtracer::f_equal(_a.x, 25.0000);
        bool y = pathtracer::f_equal(_a.y, -3.08625);
        bool z = pathtracer::f_equal(_a.z, -6.69725);

        REQUIRE(res == true);
        REQUIRE(x == true);
        REQUIRE(y == true);
        REQUIRE(z == true);
    }

    SECTION("Hadamard product") {
        pathtracer::vec3 expected{-1.2345, 8.381032845, 6.2820205};

        bool res = ((a &= b) == expected);

        REQUIRE(res == true);
    }

    SECTION("Scalar division") {
        pathtracer::vec3 _a{-10.0000, 1.2345, 2.6789};

        pathtracer::vec3 expected{4, -0.4938, -1.07156};

        bool res = ((_a /= -2.5) == expected);
        bool x = pathtracer::f_equal(_a.x, 4);
        bool y = pathtracer::f_equal(_a.y, -0.4938);
        bool z = pathtracer::f_equal(_a.z, -1.07156);

        REQUIRE(res == true);
        REQUIRE(x == true);
        REQUIRE(y == true);
        REQUIRE(z == true);
    }

    SECTION("Normalization") {
        pathtracer::vec3 expected{-0.959144, 0.1184064, -0.256945};

        bool res = (a.normalize() == expected);

        REQUIRE(res == true);
    }

    SECTION("Cross product") {
        pathtracer::vec3 expected{15.29217, -23.78071, -68.04249};

        bool res = ((a ^= b) == expected);

        REQUIRE(res == true);
    }
}
