#include <catch2/catch.hpp>
#include <cstdlib>
#include <new>
#include <curand_kernel.h>
#include "brdf.cuh"
#include "check_cuda_errors.h"
#include "constants.h"
#include "util.cuh"
#include "world.cuh"
#include "shapes.cuh"
#include "phong.cuh"
#include "camera.cuh"

// __device__ pathtracer::vec3 trace_path(pathtracer::ray ray, int depth, pathtracer::world world, int* collision_buffer, pathtracer::intersection* intersection_buffer, curandState* state) {
//     if (depth <= 0) return {0.f, 0.f, 0.f};

//     bool success_flag{false};

//     pathtracer::computations comp = world.intersect_world(ray, success_flag, collision_buffer, intersection_buffer);

//     if (!success_flag) return {0.f, 0.f, 0.f};

//     pathtracer::object& object = world.objects[comp.intersection.object_index];

//     if (object.mat_t == pathtracer::LIGHT) {
//         return {1.f, 1.f, 1.f};
//     }

//     float u = curand_uniform(state);
//     float v = curand_uniform(state);
//     float pdf;

//     pathtracer::point new_direction = pathtracer::cosine_sample_hemisphere(u, v, pdf);

//     pathtracer::quaternion q_to_world = pathtracer::quaternion::get_rotation_from_z_axis(comp.surface_normal.normalize());

//     new_direction = pathtracer::quaternion::rotate_vector_by_quaternion(new_direction, q_to_world) + pathtracer::vec3(0.01f, 0.01f, 0.01f);

//     pathtracer::ray new_ray{comp.surface_point, new_direction.normalize()};

//     float cos_theta = new_ray.d * comp.surface_normal;
//     pathtracer::vec3 BRDF = object.mat_d.phong.color * object.mat_d.phong.diffuse / pathtracer::pi;

//     pathtracer::vec3 incoming = trace_path(new_ray, depth - 1, world, collision_buffer, intersection_buffer, state);

//     return (BRDF & incoming) * (cos_theta / pdf);
// }

__global__ void constant_brdf_test(pathtracer::canvas<1000, 1000> c, pathtracer::world world, pathtracer::camera camera, curandState* d_states) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_original = j;
    int num_threads_i = blockDim.y * gridDim.y;
    int num_threads_j = blockDim.x * gridDim.x;

    curandState* state = &d_states[i * num_threads_j + j];
    curand_init(1234, i * num_threads_j + j, 0, state);

    pathtracer::vec3 color_black{0.f, 0.f, 0.f};

    int collision_buffer_offset = world.num_objects * (i * num_threads_j + j);
    int* collision_buffer = (world.collision_buffer + collision_buffer_offset);

    int intersection_buffer_offset = 2 * world.num_objects * (i * num_threads_j + j);

    pathtracer::intersection* intersection_buffer = (world.intersection_buffer + intersection_buffer_offset);

    constexpr int max_depth = 10;
    constexpr int num_samples = 1000;

    while (i < 1000) {
        while (j < 1000) {
            pathtracer::vec3 color{0.f, 0.f, 0.f};

            for (int k{0}; k < num_samples; ++k) {
                pathtracer::ray ray = camera.gen_ray_for_pixel(i, j);
                bool success_flag{false};
                pathtracer::vec3 multiplier{1.f, 1.f, 1.f};
                multiplier *= pathtracer::one_over_pi;

                for (int l{0}; l < max_depth; ++l) {
                    pathtracer::computations comp = world.intersect_world(ray, success_flag, collision_buffer, intersection_buffer);

                    if (!success_flag) {
                        multiplier &= {0.f, 0.f, 0.f};
                        break;
                    }

                    pathtracer::object& object = world.objects[comp.intersection.object_index];

                    float u = curand_uniform(state);
                    float v = curand_uniform(state);
                    float pdf;

                    pathtracer::point new_direction = pathtracer::cosine_sample_hemisphere(u, v, pdf);

                    pathtracer::quaternion q_to_world = pathtracer::quaternion::get_rotation_from_z_axis(comp.surface_normal.normalize());

                    new_direction = pathtracer::quaternion::rotate_vector_by_quaternion(new_direction, q_to_world);

                    ray = pathtracer::ray(comp.surface_point + (comp.surface_normal * 0.01f), new_direction.normalize());

                    float cos_theta = ray.d * comp.surface_normal;

                    if (object.mat_t == pathtracer::LIGHT) {
                        multiplier &= {500.f, 500.f, 500.f};
                        break;
                    }

                    // sample_color += BRDF_weighting & (object.mat_d.phong.color * object.mat_d.phong.ambient) * (cos_theta / pdf) * pathtracer::one_over_pi;

                    color += multiplier & (object.mat_d.phong.color * object.mat_d.phong.ambient) * cos_theta;

                    multiplier &= (object.mat_d.phong.color * object.mat_d.phong.diffuse) * (cos_theta / pdf) * pathtracer::one_over_pi;
                }

                color += multiplier;
            }

            color /= num_samples;
                
            c.write_pixel(i, j, color);
            
            j += num_threads_j;
        }
        i += num_threads_i;
        j = j_original;
    }
}

TEST_CASE("Full brdf renders") {
    SECTION("Constant") {
        constexpr int canvas_pixels = 1000;
        pathtracer::canvas<canvas_pixels, canvas_pixels> c{};

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
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)},
            {pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(-10.f, 0.f, -10.f)),
             pathtracer::LIGHT,
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)},
            {pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(10.f, 0.f, -10.f)),
             pathtracer::LIGHT,
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)}
        }, blocks, threads);

        curandState* d_states;

        checkCudaErrors( cudaMalloc(reinterpret_cast<void**>(&d_states), blocks.y * blocks.x * threads.y * threads.x * sizeof(curandState)) );

        constant_brdf_test<<<blocks, threads>>>(c, w, camera, d_states);

        checkCudaErrors( cudaDeviceSynchronize() );

        c.export_as_PPM("Constant_BRDF_Test_GPU.ppm");
    }
}