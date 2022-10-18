#include <cuda.h>
#include <catch2/catch.hpp>
#include "check_cuda_errors.h"
#include "util.cuh"
#include "constants.h"
#include "shapes.cuh"

TEST_CASE("Canvas", "[util]") {
    pathtracer::canvas<pathtracer::height, pathtracer::width> c{};

    SECTION("Byte conversion") {
        float n = 0.25;

        bool res = (pathtracer::to_byte(n) == 63);

        REQUIRE(res == true);
    }

    SECTION("Setting pixel values and PPM output") {
        pathtracer::vec3 color{0.75f, 0.25f, 0.f};
        pathtracer::vec3 expected{0.75f, 0.25f, 0.f};

        for (size_t i{0}; i < pathtracer::height; i += 2) {
            for (size_t j{0}; j < pathtracer::width; ++j) {
                c.write_pixel(i, j, color);
            }
        }

        bool res = (c.m_data[(pathtracer::height / 2) * pathtracer::width + (pathtracer::width / 2)] == expected);

        c.export_as_PPM("PPM_Test.ppm");

        REQUIRE(res == true);
    }
}

__global__ void canvas_test(pathtracer::canvas<pathtracer::height, pathtracer::width> c) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_original = j;
    int num_threads_i = blockDim.y * gridDim.y;
    int num_threads_j = blockDim.x * gridDim.x;

    pathtracer::vec3 color{0.75f, 0.25f, 0.f};

    while (i < pathtracer::height) {
        while (j < pathtracer::width) {
            if (i % 2 == 0) {
                c.write_pixel(i, j, color);
            }
            j += num_threads_j;
        }
        i += num_threads_i;
        j = j_original;
    }
}

TEST_CASE("Canvas on GPU", "[util]") {
    pathtracer::canvas<pathtracer::height, pathtracer::width> c{};
    pathtracer::vec3 expected{0.75f, 0.25f, 0.f};

    dim3 blocks(16, 16);
    dim3 threads(16, 16);
    canvas_test<<<blocks, threads>>>(c);

    checkCudaErrors( cudaDeviceSynchronize() );

    bool res = (c.m_data[(pathtracer::height / 2) * pathtracer::width + (pathtracer::width / 2)] == expected);

    c.export_as_PPM("PPM_Test_GPU.ppm");

    REQUIRE(res == true);
}

__global__ void shadow_test(pathtracer::canvas<100, 100> c) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_original = j;
    int num_threads_i = blockDim.y * gridDim.y;
    int num_threads_j = blockDim.x * gridDim.x;

    pathtracer::vec3 color_black{0.f, 0.f, 0.f};
    pathtracer::vec3 color_red{0.75f, 0.25f, 0.f};

    pathtracer::vec3 ray_origin{0, 0, -5.f};
    
    float wall_z = 10.f;

    float wall_size = 7.f;

    float pixel_size = wall_size / 100;

    float half = wall_size / 2;

    pathtracer::sphere sphere{pathtracer::mat4::get_identity()};

    while (i < 100) {
        float world_y = half - pixel_size * i;
        while (j < 100) {
            c.write_pixel(i, j, color_black);
            float world_x = -half + pixel_size * j;

            pathtracer::vec3 pos{world_x, world_y, wall_z};

            pathtracer::ray ray{ray_origin, pos};
            pathtracer::intersection intersection_buffer[2];
            int object_index{0};

            int num_intersections = sphere.find_intersections(ray, intersection_buffer, object_index);

            if (num_intersections == 2) {
                c.write_pixel(i, j, color_red);
            }
            
            j += num_threads_j;
        }
        i += num_threads_i;
        j = j_original;
    }
}

TEST_CASE("Shadow of sphere", "[util]") {
    constexpr int canvas_pixels = 100;
    pathtracer::canvas<canvas_pixels, canvas_pixels> c{};
    pathtracer::vec3 expected{0.75f, 0.25f, 0.f};

    dim3 blocks(16, 16);
    dim3 threads(16, 16);
    shadow_test<<<blocks, threads>>>(c);

    checkCudaErrors( cudaDeviceSynchronize() );

    bool res = (c.m_data[(100 / 2) * 100+ (100 / 2)] == expected);

    c.export_as_PPM("Shadow_Test_GPU.ppm");

    REQUIRE(res == true);
}