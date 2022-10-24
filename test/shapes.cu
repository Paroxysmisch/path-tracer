#include "check_cuda_errors.h"
#include "constants.h"
#include "ray.cuh"
#include "shapes.cuh"
#include "util.cuh"
#include <catch2/catch.hpp>

TEST_CASE("Shape intersections", "[shapes, ray]") {
    SECTION("Ray with sphere") {
        pathtracer::ray ray1{{-2.f, 0.f, 0.f}, {1.f, 0.f, 0.f}};
        pathtracer::intersection intersection_buffer[2];
        int object_index{0};

        pathtracer::sphere sphere1{pathtracer::mat4::get_identity()};

        int num_intersections = sphere1.find_intersections(ray1, intersection_buffer, object_index);

        REQUIRE(pathtracer::f_equal(intersection_buffer[0].t_value, 1.f) == true);
        REQUIRE(pathtracer::f_equal(intersection_buffer[1].t_value, 3.f) == true);
        REQUIRE((intersection_buffer[0].object_index == 0) == true);
        REQUIRE((intersection_buffer[1].object_index == 0) == true);
        REQUIRE((num_intersections == 2) == true);

        pathtracer::ray ray2{{-0.f, 0.f, 0.f}, {0.f, 0.f, 1.f}};
        pathtracer::sphere sphere2{pathtracer::mat4::get_translation(0.f, 0.f, 2.f)};
        
        num_intersections = sphere2.find_intersections(ray2, intersection_buffer, object_index);
        REQUIRE(pathtracer::f_equal(intersection_buffer[0].t_value, 1.f));
        REQUIRE(pathtracer::f_equal(intersection_buffer[1].t_value, 3.f));
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
            {0.229f, 5},
            {0.30f, 3}
        };

        pathtracer::intersection* res = pathtracer::get_closest_positive_intersection(intersection_buffer, 5);

        REQUIRE((pathtracer::f_equal(res->t_value, 0.229f)) == true);
    }

    SECTION("Local normal at") {
        pathtracer::sphere sphere{pathtracer::mat4::get_scaling(0.5f, 1.f, 1.f)};

        pathtracer::vec3 expected{0.577350f, 0.577350f, 0.577350f};

        REQUIRE((sphere.local_normal_at(expected) == expected) == true);
    }

    SECTION("World normal at") {
        pathtracer::sphere sphere{pathtracer::mat4::get_scaling(1.f, 0.5f, 1.f) * pathtracer::mat4::get_rotation_z(pathtracer::pi / 5)};

        pathtracer::vec3 world_surface_point{0.f, 0.707106f, -0.707106f};

        pathtracer::vec3 expected{0.f, 0.97014f, -0.24254f};

        auto temp = sphere.world_normal_at(world_surface_point);

        REQUIRE((sphere.world_normal_at(world_surface_point) == expected) == true);
    }
}