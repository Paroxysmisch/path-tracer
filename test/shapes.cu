#include "check_cuda_errors.h"
#include "ray.cuh"
#include "shapes.cuh"
#include "util.cuh"
#include <catch2/catch.hpp>

TEST_CASE("Shape intersections", "[shapes, ray]") {
    SECTION("Ray with sphere") {
        pathtracer::ray ray{{-2.f, 0.f, 0.f}, {1.f, 0.f, 0.f}};
        pathtracer::intersection intersection_buffer[2];
        int object_index{0};

        pathtracer::sphere sphere{pathtracer::mat4::get_identity()};

        int num_intersections = sphere.find_intersections(ray, intersection_buffer, object_index);

        REQUIRE(pathtracer::f_equal(intersection_buffer[0].t_value, 1.f) == true);
        REQUIRE(pathtracer::f_equal(intersection_buffer[1].t_value, 3.f) == true);
        REQUIRE((intersection_buffer[0].object_index == 0) == true);
        REQUIRE((intersection_buffer[1].object_index == 0) == true);
        REQUIRE((num_intersections == 2) == true);
    }
}