#include <catch2/catch.hpp>
#include <iostream>
#include "util.cuh"
#include "check_cuda_errors.h"

TEST_CASE("Quaternion", "[util]") {
    pathtracer::vec3 v1{0, 1, 0};
    pathtracer::vec3 v2{1,0,0};

    SECTION("2D 90deg ACW rotation") {
        pathtracer::quaternion q = pathtracer::quaternion::get_rotation_between(v1, v2);
        pathtracer::vec3 expected{1,0,0};

        bool res = (pathtracer::quaternion::rotate_vector_by_quaternion(v1, q) == expected);

        REQUIRE(res == true);
    }

    SECTION("3D rotation") {
        pathtracer::vec3 v3{-1.f, -1.f, -1.f};
        pathtracer::vec3 v4{-1.f, -1.f, 0.f};
        pathtracer::vec3 expected{-sqrtf(2.f / 3.f),
                                  -sqrtf(2.f / 3.f),
                                   sqrtf(2.f / 3.f)};

        pathtracer::quaternion q = pathtracer::quaternion::get_rotation_between(v3, v4);

        pathtracer::vec3 res_vec = pathtracer::quaternion::rotate_vector_by_quaternion(v4, q);

        bool res = (res_vec == expected);

        REQUIRE(res == true);
    }
}