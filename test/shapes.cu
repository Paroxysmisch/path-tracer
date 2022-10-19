#include "check_cuda_errors.h"
#include "constants.h"
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

TEST_CASE("Shape utils") {
    SECTION("Sphere transformations") {
        pathtracer::sphere sphere1{pathtracer::mat4::get_identity()};

        REQUIRE((sphere1.transformation_to_world == pathtracer::mat4::get_identity()));
        REQUIRE((sphere1.transformation_to_object == pathtracer::mat4::get_identity()));

        pathtracer::sphere sphere2{pathtracer::mat4::get_rotation_z(pathtracer::pi / 2.f)};
        
        REQUIRE((sphere2.transformation_to_world == pathtracer::mat4{0.f, -1.f, 0.f, 0.f,
                                                                     1.f, 0.f, 0.f, 0.f,
                                                                     0.f, 0.f, 1.f, 0.f,
                                                                     0.f, 0.f, 0.f, 1.f}));

        bool success_flag;
        REQUIRE((sphere2.transformation_to_object == pathtracer::mat4{0.f, -1.f, 0.f, 0.f,
                                                                     1.f, 0.f, 0.f, 0.f,
                                                                     0.f, 0.f, 1.f, 0.f,
                                                                     0.f, 0.f, 0.f, 1.f}.inverse(success_flag)));
    }
    
    SECTION("Closest positive intersection") {
        pathtracer::intersection intersection_buffer[5] {
            {-2.5f, 1},
            {-0.023f, 0},
            {0.f, 4},
            {0.23f, 5},
            {0.229f, 3}
        };

        pathtracer::intersection* res = pathtracer::get_closest_positive_intersection(intersection_buffer, 5);

        REQUIRE((pathtracer::f_equal(res->t_value, 0.229f)) == true);
    }
}