#include <catch2/catch.hpp>
#include <cstdlib>
#include <new>
#include "check_cuda_errors.h"
#include "world.cuh"
#include "shapes.cuh"
#include "phong.cuh"

__global__ void world_phong_test(pathtracer::canvas<1000, 1000> c, pathtracer::world world, pathtracer::object* light) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_original = j;
    int num_threads_i = blockDim.y * gridDim.y;
    int num_threads_j = blockDim.x * gridDim.x;

    pathtracer::vec3 color_black{0.f, 0.f, 0.f};
    pathtracer::vec3 color_red{0.75f, 0.25f, 0.f};
    pathtracer::vec3 color_green{0.25f, 0.75f, 0.f};

    pathtracer::vec3 ray_origin{0, 0, -5.f};
    
    float wall_z = 0.f;

    float wall_size = 7.f;

    float pixel_size = wall_size / 1000;

    float half = wall_size / 2;

    int collision_buffer_offset = world.num_objects * (i * num_threads_j + j);
    int* collision_buffer = (world.collision_buffer + collision_buffer_offset);

    int intersection_buffer_offset = 2 * world.num_objects * (i * num_threads_j + j);

    pathtracer::intersection* intersection_buffer = (world.intersection_buffer + intersection_buffer_offset);

    while (i < 1000) {
        float world_y = half - pixel_size * i;
        while (j < 1000) {
            c.write_pixel(i, j, color_black);

            float world_x = -half + pixel_size * j;

            pathtracer::vec3 pos{world_x, world_y, wall_z};

            pathtracer::ray ray{ray_origin, (pos - ray_origin).normalize()};

            bool success_flag{false};

            pathtracer::computations comp = world.intersect_world(ray, success_flag, collision_buffer, intersection_buffer);

            if (success_flag) {
                pathtracer::vec3 color = pathtracer::phong_lighting(&world.objects[comp.intersection.object_index], light, &comp.surface_point, &comp.eye_vector, &comp.surface_normal);
                c.write_pixel(i, j, color);
            }
            
            j += num_threads_j;
        }
        i += num_threads_i;
        j = j_original;
    }
}

TEST_CASE("Full world renders") {
    SECTION("Phong") {
        constexpr int canvas_pixels = 1000;
        pathtracer::canvas<canvas_pixels, canvas_pixels> c{};

        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        pathtracer::world w({
            {pathtracer::SPHERE, 
             pathtracer::sphere(pathtracer::mat4::get_translation(-2.f, 0.f, -2.f)),
             pathtracer::PHONG,
             pathtracer::phong({0.25f, 0.25f, 0.95f}, 0.1f, 0.9f, 0.9f, 200)},
            {pathtracer::SPHERE, 
             pathtracer::sphere(pathtracer::mat4::get_translation(-1.f, -1.f, 0.f)),
             pathtracer::PHONG,
             pathtracer::phong({0.35f, 0.25f, 0.75f}, 0.1f, 0.9f, 0.9f, 200)},
            {pathtracer::SPHERE, 
             pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 0.f, -1.f)),
             pathtracer::PHONG,
             pathtracer::phong({0.75f, 0.25f, 0.5f}, 0.1f, 0.9f, 0.9f, 100)},
            {pathtracer::SPHERE, 
             pathtracer::sphere(pathtracer::mat4::get_translation(1.f, 1.f, 2.f)),
             pathtracer::PHONG,
             pathtracer::phong({0.75f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)},
            {pathtracer::SPHERE, 
             pathtracer::sphere(pathtracer::mat4::get_translation(2.f, 0.f, 1.f)),
             pathtracer::PHONG,
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)}
        }, blocks, threads);

        pathtracer::object* light;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&light), sizeof(pathtracer::object)) );

        light[0].shape_t = pathtracer::SPHERE;
        light[0].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(-10.f, 0.f, -10.f));
        light[0].mat_t = pathtracer::LIGHT;
        light[0].mat_d.light = pathtracer::light({1.f, 1.f, 1.f});

        world_phong_test<<<blocks, threads>>>(c, w, light);

        checkCudaErrors( cudaDeviceSynchronize() );

        c.export_as_PPM("World_Phong_Test_GPU.ppm");
    }
}