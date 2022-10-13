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

    SECTION("General 3D rotation") {
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

    SECTION("Rotate to z-axis") {
        pathtracer::vec3 expected{0.f, 0.f, 1.f};

        pathtracer::vec3 v3{-1.f, -1.f, -1.f};
        v3.normalize();
        
        pathtracer::quaternion q1 = pathtracer::quaternion::get_rotation_to_z_axis(v3);

        pathtracer::vec3 res_vec = pathtracer::quaternion::rotate_vector_by_quaternion(v3, q1);

        REQUIRE((res_vec == expected) == true);

        pathtracer::quaternion q1_prime = pathtracer::quaternion::get_rotation_from_z_axis(v3);

        res_vec = pathtracer::quaternion::rotate_vector_by_quaternion(res_vec, q1_prime);

        REQUIRE((res_vec == v3) == true);

        bool equality_w = pathtracer::quaternion::get_inverse_rotation(q1).w == q1_prime.w;

        bool equality = (equality_w && pathtracer::quaternion::get_inverse_rotation(q1).ijk == q1_prime.ijk);

        REQUIRE(equality == true);

        pathtracer::vec3 v4{-0.764f, 3.851f, 172.10002f};
        v4.normalize();

        pathtracer::quaternion q2 = pathtracer::quaternion::get_rotation_to_z_axis(v4);

        res_vec = pathtracer::quaternion::rotate_vector_by_quaternion(v4, q2);

        REQUIRE((res_vec == expected) == true);

        pathtracer::quaternion q2_prime = pathtracer::quaternion::get_rotation_from_z_axis(v4);

        res_vec = pathtracer::quaternion::rotate_vector_by_quaternion(expected, q2_prime);

        REQUIRE((res_vec == v4) == true);

        equality_w = pathtracer::quaternion::get_inverse_rotation(q2).w == q2_prime.w;

        equality = (equality_w && pathtracer::quaternion::get_inverse_rotation(q2).ijk == q2_prime.ijk);

        REQUIRE(equality == true);
    }
}