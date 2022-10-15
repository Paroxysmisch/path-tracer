#include <catch2/catch.hpp>
#include <iostream>
#include "util.cuh"
#include "bvh.cuh"
#include "check_cuda_errors.h"

TEST_CASE("BVH", "[acceleron_datastructures]") {
    SECTION("Convert range") {
        pathtracer::vec3 v1{-1.f, -1.f, -1.f};
        pathtracer::vec3 expected1{0.f, 0.f, 0.f};

        pathtracer::vec3 v2{1.f, 1.f, 1.f};
        pathtracer::vec3 expected2{1.f, 1.f, 1.f};

        pathtracer::vec3 v3{0.f, 0.f, 0.f};
        pathtracer::vec3 expected3{0.5f, 0.5f, 0.5f};

        auto test = pathtracer::convert_range(v1);

        REQUIRE((pathtracer::convert_range(v1) == expected1) == true);
        REQUIRE((pathtracer::convert_range(v2) == expected2) == true);
        REQUIRE((pathtracer::convert_range(v3) == expected3) == true);
    }
}