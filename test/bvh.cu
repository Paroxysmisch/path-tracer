#include <catch2/catch.hpp>
#include <iostream>
#include "util.cuh"
#include "bvh.cuh"
#include "check_cuda_errors.h"

TEST_CASE("BVH utilities", "[acceleron_datastructures]") {
    pathtracer::vec3 v{-0.6f, -0.4f, 0.5f};

    SECTION("Convert range") {
        pathtracer::vec3 v1{-1.f, -1.f, -1.f};
        pathtracer::vec3 expected1{0.f, 0.f, 0.f};

        pathtracer::vec3 v2{1.f, 1.f, 1.f};
        pathtracer::vec3 expected2{1.f, 1.f, 1.f};

        pathtracer::vec3 v3{0.f, 0.f, 0.f};
        pathtracer::vec3 expected3{0.5f, 0.5f, 0.5f};

        pathtracer::vec3 expected{0.2f, 0.3f, 0.75f};

        REQUIRE((pathtracer::convert_range(v1) == expected1) == true);
        REQUIRE((pathtracer::convert_range(v2) == expected2) == true);
        REQUIRE((pathtracer::convert_range(v3) == expected3) == true);
        REQUIRE((pathtracer::convert_range(v) == expected) == true);
    }

    SECTION("Expands bits to Morton") {
        unsigned int i1 = 129;
        unsigned int i2 = 526;
        unsigned int i3 = 1023;

        unsigned int expected1 = 0b1000000000000000000001u;
        unsigned int expected2 = 0b1000000000000000001001001000u;
        unsigned int expected3 = 0b1001001001001001001001001001u;

        REQUIRE((pathtracer::expand_bits_morton(i1) == expected1) == true);
        REQUIRE((pathtracer::expand_bits_morton(i2) == expected2) == true);
        REQUIRE((pathtracer::expand_bits_morton(i3) == expected3) == true);
    }

    SECTION("Point to Morton") {
        unsigned int x = (unsigned int) 204.8f;
        unsigned int y = (unsigned int) 307.2f;
        unsigned int z = (unsigned int) 768.f;

        unsigned int expected = pathtracer::expand_bits_morton(x) * 4 + 
                                pathtracer::expand_bits_morton(y) * 2 +
                                pathtracer::expand_bits_morton(z);

        REQUIRE((pathtracer::point_to_morton(v) == expected) == true);
    }

    SECTION("Find split") {
        unsigned int sorted_morton_codes[] {0b111000u, 0b111010u, 0b111011, 0b111100u, 0b111111u};

        REQUIRE((pathtracer::find_split(sorted_morton_codes, 0, 4) == 2) == true);
    }
}

TEST_CASE("BVH", "[acceleron_datastructures]") {
    SECTION("Generate leaf node") {
        pathtracer::bvh_node* leaf = pathtracer::bvh_node::gen_leaf_node(0);

        REQUIRE((leaf->is_leaf()) == true);

        REQUIRE((leaf->left == nullptr) == true);
        REQUIRE((leaf->right == nullptr) == true);

        REQUIRE((leaf->object_index == 0) == true);
    }

    SECTION("Generate internal node") {
        pathtracer::bvh_node* leaf_left = pathtracer::bvh_node::gen_leaf_node(0);
        pathtracer::bvh_node* leaf_right = pathtracer::bvh_node::gen_leaf_node(1);

        pathtracer::bvh_node* root = pathtracer::bvh_node::gen_internal_node(leaf_left, leaf_right);

        REQUIRE((root->is_leaf()) == false);

        REQUIRE((root->left == leaf_left) == true);
        REQUIRE((root->right == leaf_right) == true);

        REQUIRE(((root->left)->object_index == 0) == true);
        REQUIRE(((root->right)->object_index == 1) == true);
    }

    SECTION("Generate hierarchy") {
        unsigned int sorted_morton_codes[] {0b111000u, 0b111010u, 0b111011, 0b111100u, 0b111111u};
        int sorted_object_indices[] {0, 1, 2, 3, 4};

        pathtracer::bvh_node* res = pathtracer::bvh_node::gen_hierarchy(sorted_morton_codes, sorted_object_indices, 0, 4);

        REQUIRE((res->left->left->object_index == 0) == true);
        REQUIRE((res->left->right->left->object_index == 1) == true);
        REQUIRE((res->left->right->right->object_index == 2) == true);
        REQUIRE((res->right->left->object_index == 3) == true);
        REQUIRE((res->right->right->object_index == 4) == true);
    }
}