#include <catch2/catch.hpp>
#include "bvh.cuh"
#include "shapes.cuh"
#include "scene.cuh"
#include "util.cuh"

TEST_CASE("BVH construction from objects", "[scene, acceleron_datastructures, shapes]") {
    SECTION("Singular object") {
        pathtracer::object objects[1] = {{pathtracer::SPHERE, pathtracer::sphere(pathtracer::mat4::get_identity())}};

        pathtracer::morton_and_index out_buffer[1];

        pathtracer::gen_sorted_morton_codes_and_indices(objects, 1, out_buffer);

        pathtracer::bvh_arena arena{1};

        pathtracer::bvh_node* root = pathtracer::_gen_bvh(out_buffer, objects, 0, 0, &arena);

        REQUIRE((root->object_index == 0) == true);
    }

    SECTION("Multiple objects render GPU") {
        constexpr int num_objects = 2;
        pathtracer::object objects[num_objects] = {{pathtracer::SPHERE, pathtracer::sphere(pathtracer::mat4::get_identity())},
                                                   {pathtracer::SPHERE, pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 0.f, 0.5f))}};

        pathtracer::bvh_arena arena{num_objects};

        pathtracer::bvh_node* root = pathtracer::gen_bvh(objects, num_objects, &arena);
        
        REQUIRE((root->lower == pathtracer::vec3(-1.f, -1.f, -1.f)));
        REQUIRE((root->upper == pathtracer::vec3(1.f, 1.f, 1.5f)));
    }
}