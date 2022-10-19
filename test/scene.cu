#include <cuda.h>
#include <catch2/catch.hpp>
#include "check_cuda_errors.h"
#include "bvh.cuh"
#include "shapes.cuh"
#include "scene.cuh"
#include "util.cuh"

__global__ void shadow_test(pathtracer::canvas<1000, 1000> c, pathtracer::object* objects, pathtracer::bvh_node* root) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_original = j;
    int num_threads_i = blockDim.y * gridDim.y;
    int num_threads_j = blockDim.x * gridDim.x;

    pathtracer::vec3 color_black{0.f, 0.f, 0.f};
    pathtracer::vec3 color_red{0.75f, 0.25f, 0.f};
    pathtracer::vec3 color_green{0.25f, 0.75f, 0.f};

    pathtracer::vec3 ray_origin{0, 0, -5.f};
    
    float wall_z = 10.f;

    float wall_size = 7.f;

    float pixel_size = wall_size / 1000;

    float half = wall_size / 2;

    while (i < 1000) {
        float world_y = half - pixel_size * i;
        while (j < 1000) {
            c.write_pixel(i, j, color_black);

            float world_x = -half + pixel_size * j;

            pathtracer::vec3 pos{world_x, world_y, wall_z};

            pathtracer::ray ray{ray_origin, pos};

            int collision_buffer[2];

            int possible_intersections = ray.find_intersections(root, collision_buffer);

            if (possible_intersections == 2) {
                c.write_pixel(i, j, color_red);
            } else if (possible_intersections == 1) {
                c.write_pixel(i, j, color_green);
            }
            
            j += num_threads_j;
        }
        i += num_threads_i;
        j = j_original;
    }
}

TEST_CASE("BVH construction from objects", "[scene, acceleron_datastructures, shapes, ray]") {
    SECTION("Singular object") {
        pathtracer::object objects[1] = {{pathtracer::SPHERE, pathtracer::sphere(pathtracer::mat4::get_identity())}};

        pathtracer::morton_and_index out_buffer[1];

        pathtracer::gen_sorted_morton_codes_and_indices(objects, 1, out_buffer);

        pathtracer::bvh_arena arena{1};

        pathtracer::bvh_node* root = pathtracer::_gen_bvh(out_buffer, objects, 0, 0, &arena);

        REQUIRE((root->object_index == 0) == true);
    }

    SECTION("Multiple objects render GPU") {
        constexpr int canvas_pixels = 1000;
        pathtracer::canvas<canvas_pixels, canvas_pixels> c{};

        constexpr int num_objects = 2;

        pathtracer::object* objects;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&objects), num_objects * sizeof(pathtracer::object)) );

        objects[0].shape_t = pathtracer::SPHERE;
        objects[0].data = pathtracer::sphere(pathtracer::mat4::get_identity());

        objects[1].shape_t = pathtracer::SPHERE;
        objects[1].data = pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 0.5f, 0.f));

        // pathtracer::object objects[num_objects] = {{pathtracer::SPHERE, pathtracer::sphere(pathtracer::mat4::get_identity())},
                                                //    {pathtracer::SPHERE, pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 0.f, 0.5f))}};

        pathtracer::bvh_arena arena{num_objects};

        pathtracer::bvh_node* root = pathtracer::gen_bvh(objects, num_objects, &arena);
        
        REQUIRE((root->lower == pathtracer::vec3(-1.f, -1.f, -1.f)));
        REQUIRE((root->upper == pathtracer::vec3(1.f, 1.5f, 1.f)));

        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        shadow_test<<<blocks, threads>>>(c, objects, root);

        checkCudaErrors( cudaDeviceSynchronize() );

        c.export_as_PPM("Shadow_Test_Advanced_GPU.ppm");
    }
}