#include <catch2/catch.hpp>
#include <iostream>
#include "ray.cuh"
#include "check_cuda_errors.h"

TEST_CASE("Ray intersection", "[ray, acceleron_datastructures]") {
    SECTION("With BVH bounding box") {
        pathtracer::vec3 lower = {-0.5f, -0.5f, -0.5f};
        pathtracer::vec3 upper = {0.5f, 0.5f, 0.5f};
        pathtracer::bvh_node* leaf = pathtracer::bvh_node::gen_leaf_node(0, lower, upper);

        pathtracer::ray ray_x1{{-1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}};
        pathtracer::ray ray_y1{{0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}};
        pathtracer::ray ray_z1{{0.f, 0.f, 1.f}, {0.f, 0.f, -1.f}};

        REQUIRE(ray_x1.check_bvh_node_intersection(leaf) == true);
        REQUIRE(ray_y1.check_bvh_node_intersection(leaf) == true);
        REQUIRE(ray_z1.check_bvh_node_intersection(leaf) == true);
// 
        // Backward intersections, where t-value is negative,
        // should be rejected
        pathtracer::ray ray_x2{{-1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}};
        pathtracer::ray ray_y2{{0.f, -1.f, 0.f}, {0.f, -1.f, 0.f}};
        pathtracer::ray ray_z2{{0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}};

        REQUIRE(ray_x2.check_bvh_node_intersection(leaf) == false);
        REQUIRE(ray_y2.check_bvh_node_intersection(leaf) == false);
        REQUIRE(ray_z2.check_bvh_node_intersection(leaf) == false);

        pathtracer::ray ray3{{0.f, 0.f, 0.f}, {0.5f, 0.4f, 0.5f}};

        REQUIRE(ray3.check_bvh_node_intersection(leaf) == true);

        pathtracer::ray ray4{{0.f, -1.f, 0.f}, {0.f, 1.f, 0.8f}};

        REQUIRE(ray4.check_bvh_node_intersection(leaf) == true);

        pathtracer::ray ray5{{0.f, 0.f, 1.f}, {1.f, 0.f, 0.f}};

        REQUIRE(ray5.check_bvh_node_intersection(leaf) == false);

        pathtracer::ray ray6{{0.f, 0.f, 1.f}, {0.5f, 0.5f, -0.4f}};

        REQUIRE(ray6.check_bvh_node_intersection(leaf) == false);

    }
}