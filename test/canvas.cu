#include <cuda.h>
#include <catch2/catch.hpp>
#include "check_cuda_errors.h"
#include "util.cuh"
#include "constants.h"

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