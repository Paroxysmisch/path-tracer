#include <catch2/catch.hpp>
#include "util.cuh"
#include "check_cuda_errors.h"

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
}

// TEST_CASE("vec3 mutable operations", "[util]") {
    
// }