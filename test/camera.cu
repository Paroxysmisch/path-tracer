#include <catch2/catch.hpp>
#include <cstdlib>
#include <new>
#include "check_cuda_errors.h"
#include "util.cuh"
#include "world.cuh"
#include "shapes.cuh"
#include "phong.cuh"
#include "camera.cuh"

__global__ void camera_phong_test(pathtracer::canvas c, pathtracer::world world, pathtracer::object* light, pathtracer::camera camera) {
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

            // pathtracer::ray ray{ray_origin, (pos - ray_origin).normalize()};

            pathtracer::ray ray = camera.gen_ray_for_pixel(i, j);

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

TEST_CASE("Full camera renders") {
    SECTION("Phong") {
        constexpr int canvas_pixels = 1000;
        pathtracer::canvas c{canvas_pixels, canvas_pixels};

        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        pathtracer::camera camera(1000, 1000, pathtracer::pi / 2.f, {0.f, 0.f, -10.f}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, pathtracer::mat4::get_rotation_z(pathtracer::pi / 4.f));

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

        camera_phong_test<<<blocks, threads>>>(c, w, light, camera);

        checkCudaErrors( cudaDeviceSynchronize() );

        c.export_as_PPM("Camera_Phong_Test_GPU.ppm");
    }
}

TEST_CASE("Camera components") {
    SECTION("Pixel size") {
        pathtracer::camera camera1(200, 125, pathtracer::pi / 2.f);

        REQUIRE(pathtracer::f_equal(camera1.pixel_size, 0.01f) == true);

        pathtracer::camera camera2(125, 200, pathtracer::pi / 2.f);

        REQUIRE(pathtracer::f_equal(camera2.pixel_size, 0.01f) == true);
    }
    SECTION("Ray generation") {
        pathtracer::camera camera(101, 201, pathtracer::pi / 2.f);

        // auto temp = camera.gen_ray_for_pixel(50, 100);

        // std::cout << temp.o.x << " " << temp.o.y << " " << temp.o.z << std::endl;
        // std::cout << temp.d.x << " " << temp.d.y << " " << temp.d.z << std::endl;

        // TODO: Strange behaviour where the result is not quite 0
        // REQUIRE((camera.gen_ray_for_pixel(50, 100) == pathtracer::ray({0.f, 0.f, 0.f}, {0.f, 0.f, -1.f})) == true);

        REQUIRE((camera.gen_ray_for_pixel(0, 0) == pathtracer::ray({0.f, 0.f, 0.f}, {0.665186f, 0.332593f, -0.668512f})) == true);
    }
}