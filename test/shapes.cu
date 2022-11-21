#include "check_cuda_errors.h"
#include "constants.h"
#include "ray.cuh"
#include "shapes.cuh"
#include "util.cuh"
#include "camera.cuh"
#include "world.cuh"
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
    SECTION("Ray with triangle") {
        pathtracer::intersection intersection_buffer[1];
        int object_index{0};

        pathtracer::ray ray1{{0.f, -1.f, 2.f}, {0.f, 1.f, 0.f}};
        pathtracer::triangle triangle1({0.f, 1.f, 0.f}, {-1.f, 0.f, 0.f}, {1.f, 0.f, 0.f});
        int num_intersections1 = triangle1.find_intersections(ray1, intersection_buffer, object_index);
        REQUIRE((num_intersections1 == 0) == true);

        pathtracer::ray ray2{{1.f, 1.f, -2.f}, {0.f, 0.f, 1.f}};
        int num_intersections2 = triangle1.find_intersections(ray2, intersection_buffer, object_index);
        REQUIRE((num_intersections2 == 0) == true);

        pathtracer::ray ray3{{-1.f, 1.f, -2.f}, {0.f, 0.f, 1.f}};
        int num_intersections3 = triangle1.find_intersections(ray3, intersection_buffer, object_index);
        REQUIRE((num_intersections3 == 0) == true);

        pathtracer::ray ray4{{0.f, -1.f, 2.f}, {0.f, 0.f, 1.f}};
        int num_intersections4 = triangle1.find_intersections(ray4, intersection_buffer, object_index);
        REQUIRE((num_intersections4 == 0) == true);

        pathtracer::ray ray5{{0.f, 0.5f, -2.f}, {0.f, 0.f, 1.f}};
        int num_intersections5 = triangle1.find_intersections(ray5, intersection_buffer, object_index);
        REQUIRE((num_intersections5 == 1) == true);
        REQUIRE((pathtracer::f_equal(intersection_buffer[0].t_value, 2.f)) == true);
        REQUIRE((intersection_buffer[0].object_index == 0) == true);
    }
    SECTION("Ray with triangle normal interpolation") {
        pathtracer::intersection intersection_buffer[1];
        int object_index{0};

        pathtracer::ray ray1{{-0.2f, 0.3f, -2.f}, {0.f, 0.f, 1.f}};
        pathtracer::triangle triangle1({0.f, 1.f, 0.f}, {-1.f, 0.f, 0.f}, {1.f, 0.f, 0.f});
        int num_intersections1 = triangle1.find_intersections(ray1, intersection_buffer, object_index);
        REQUIRE((num_intersections1 == 1) == true);
        REQUIRE(pathtracer::f_equal(intersection_buffer[0].u, 0.45f));
        REQUIRE(pathtracer::f_equal(intersection_buffer[0].v, 0.25f));
    }
    SECTION("Triangle scene") {
        constexpr int canvas_pixels = 1000;
        pathtracer::canvas c{canvas_pixels, canvas_pixels};

        pathtracer::camera camera(1000, 1000, pathtracer::pi / 2.f, {0.f, 0.f, -10.f}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f});

        pathtracer::object obj1{pathtracer::TRIANGLE,
             pathtracer::sphere(pathtracer::mat4::get_identity()),
             pathtracer::MICROFACET,
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)};
        obj1.shape_d.triangle = pathtracer::triangle({-1.f, -1.f, -1.f}, {-1.f, 1.f, -1.f}, {0.5f, 0.f, -1.f});
        obj1.mat_d.microfacet = pathtracer::microfacet{{0.95f, 0.25f, 0.5f}, {0.f, 0.f, 0.f}, 0.75f, 0.2f, 0.f, 1.f};

        pathtracer::world w1{{&obj1}, 1, 1};

        for (int i{0}; i < canvas_pixels; ++i) {
            for (int j{0}; j < canvas_pixels; ++j) {
                pathtracer::ray ray = camera.gen_ray_for_pixel(i, j);

                int collision_buffer[3];
                pathtracer::intersection intersection_buffer[3];
                bool success_flag = false;

                w1.intersect_world(ray, success_flag, collision_buffer, intersection_buffer);

                if (success_flag)
                    c.write_pixel(i, j, {1.f, 1.f, 1.f});
            }
        }

        c.export_as_PPM("Triangle shadow.ppm");
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

    SECTION("World normal at sphere") {
        pathtracer::sphere sphere{pathtracer::mat4::get_scaling(1.f, 0.5f, 1.f) * pathtracer::mat4::get_rotation_z(pathtracer::pi / 5)};

        pathtracer::vec3 world_surface_point{0.f, 0.707106f, -0.707106f};

        pathtracer::vec3 expected{0.f, 0.97014f, -0.24254f};

        auto temp = sphere.world_normal_at(world_surface_point);

        REQUIRE((sphere.world_normal_at(world_surface_point) == expected) == true);
    }

    SECTION("World normal at triangle") {
        pathtracer::triangle triangle1({0.f, 1.f, 0.f}, {-1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {-1.f, 0.f, 0.f}, {1.f, 0.f, 0.f});

        pathtracer::vector calculated_normal = triangle1.world_normal_at({0.f, 0.f, 0.f}, 0.45f, 0.25f);

        REQUIRE((calculated_normal == pathtracer::vec3(-0.5547f, 0.83205f, 0)) == true);
    }
}