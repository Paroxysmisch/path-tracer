#include <catch2/catch.hpp>
#include <iostream>
#include "util.cuh"
#include "check_cuda_errors.h"

TEST_CASE("Quaternion", "[util]") {
    pathtracer::vec3 v1{0, 1, 0};
    pathtracer::vec3 v2{1,0,0};

    SECTION("2D 90deg ACW Rotation") {
        pathtracer::quaternion q = pathtracer::quaternion::get_rotation_between(v1, v2);
        pathtracer::vec3 expected{1,0,0};

        std::cout << q.w << " "
                  << q.ijk.x << " "
                  << q.ijk.y << " "
                  << q.ijk.z << " " << std::endl;

        bool res = (pathtracer::quaternion::rotate_vector_by_quaternion(v1, q) == expected);

        REQUIRE(res == true);
    }
}