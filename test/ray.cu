#include <catch2/catch.hpp>
#include <iostream>
#include "ray.cuh"
#include "check_cuda_errors.h"
#include "bvh.cuh"
#include "shapes.cuh"

TEST_CASE("Ray intersection", "[ray, acceleron_datastructures]") {
    SECTION("With BVH bounding box") {
        pathtracer::vec3 lower = {-0.5f, -0.5f, -0.5f};
        pathtracer::vec3 upper = {0.5f, 0.5f, 0.5f};
        pathtracer::bvh_arena arena1{1};
        pathtracer::bvh_node* leaf = pathtracer::bvh_node::gen_leaf_node(0, lower, upper, &arena1);

        pathtracer::ray ray_x1{{-1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}};
        pathtracer::ray ray_y1{{0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}};
        pathtracer::ray ray_z1{{0.f, 0.f, 1.f}, {0.f, 0.f, -1.f}};

        REQUIRE(ray_x1.check_bvh_node_intersection(leaf) == true);
        REQUIRE(ray_y1.check_bvh_node_intersection(leaf) == true);
        REQUIRE(ray_z1.check_bvh_node_intersection(leaf) == true);

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

        pathtracer::vec3 lower2 = {-0.9f, -0.9f, -0.9f};
        pathtracer::vec3 upper2 = {0.9f, 0.9f, 0.9f};
        pathtracer::bvh_arena arena2{1};
        pathtracer::bvh_node* leaf2 = pathtracer::bvh_node::gen_leaf_node(0, lower2, upper2, &arena2);

        pathtracer::ray ray7{{0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}};

        REQUIRE(ray7.check_bvh_node_intersection(leaf2) == true);

        arena1.free_arena();
        arena2.free_arena();
    }
}

TEST_CASE("BVH traversal", "[ray, acceleron_datastructures]") {
    SECTION("Find intersections") {
        unsigned int sorted_morton_codes[] {0b111000u, 0b111010u, 0b111011, 0b111100u, 0b111111u};
        int sorted_object_indices[] {0, 1, 2, 3, 4};
        pathtracer::vec3 temp_dimensions[] {
            {-1.f, -1.f, -1.f}, {-0.6f, -0.6f, -0.6f},
            {-0.4f, -0.4f, -0.4f}, {1.f, 1.f, 1.f},
            {0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f},
            {0.4f, 0.4f, 0.4f}, {1.f, 1.f, 1.f},
            {-0.9f, -0.9f, -0.9f}, {0.9f, 0.9f, 0.9f}
        };
        pathtracer::bvh_arena arena{5};

        pathtracer::bvh_node* root = pathtracer::bvh_node::gen_hierarchy(sorted_morton_codes, sorted_object_indices, temp_dimensions, 0, 4, &arena);

        pathtracer::ray ray1{{0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}};

        int collision_buffer[10];

        for (size_t i{0}; i < 10; ++i) collision_buffer[i] = -1;

        pathtracer::vec3 root_lower = {-1.f, -1.f, -1.f};
        pathtracer::vec3 root_upper = {1.f, 1.f, 1.f};
        pathtracer::vec3 root_right_lower = {-0.9f, -0.9f, -0.9f};
        pathtracer::vec3 root_right_upper = {1.f, 1.f, 1.f};

        REQUIRE((root->left->left->object_index == 0) == true);
        REQUIRE((root->left->right->left->object_index == 1) == true);
        REQUIRE((root->left->right->right->object_index == 2) == true);
        REQUIRE((root->right->left->object_index == 3) == true);
        REQUIRE((root->right->right->object_index == 4) == true);
        REQUIRE((root->lower == root_lower) == true);
        REQUIRE((root->upper == root_upper) == true);
        REQUIRE((root->right->lower == root_right_lower) == true);
        REQUIRE((root->right->upper == root_right_upper) == true);

        int num_intersections = ray1.find_intersections(root, collision_buffer);

        REQUIRE((num_intersections == 2) == true);
        REQUIRE((collision_buffer[0] == 1) == true);
        REQUIRE((collision_buffer[1] == 4) == true);

        arena.free_arena();
    }
}

TEST_CASE("Ray utils", "[ray]") {
    pathtracer::ray ray1{{0.1f, 0.2f, 0.3f}, {-0.4f, -0.5f, 0.6f}};
    SECTION("Shoot distance") {
        pathtracer::point expected{-0.1f, -0.05f, 0.6f};

        REQUIRE((ray1.shoot_distance(0.5f) == expected) == true);
    }
}
