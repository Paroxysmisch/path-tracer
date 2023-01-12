#include <catch2/catch.hpp>
#include <cstdlib>
#include <new>
#include <curand_kernel.h>
#include <vector>
#include <string>
#include "brdf.cuh"
#include "check_cuda_errors.h"
#include "constants.h"
#include "util.cuh"
#include "world.cuh"
#include "shapes.cuh"
#include "phong.cuh"
#include "camera.cuh"
#include "denoise.cuh"

__global__ void mesh_constant_brdf_test(pathtracer::canvas c, pathtracer::world world, pathtracer::camera camera, curandState* d_states) {
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
                // floats a and b for anti-aliasing
                float a = curand_uniform(state);
                float b = curand_uniform(state);
                pathtracer::ray ray = camera.gen_ray_for_pixel(i, j, a, b);
                bool success_flag{false};
                pathtracer::vec3 multiplier{1.f, 1.f, 1.f};
                multiplier *= pathtracer::one_over_pi;
                float current_refractive_index = 1.f;

                for (int l{0}; l < max_depth; ++l) {
                    pathtracer::computations comp = world.intersect_world(ray, success_flag, collision_buffer, intersection_buffer);

                    if (!success_flag) {
                        multiplier &= {0.f, 0.f, 0.f};
                        break;
                    }

                    pathtracer::object& object = world.objects[comp.intersection.object_index];
                    pathtracer::microfacet material_copy = object.mat_d.microfacet;

                    // If the object intersected is a triangle and uses textures,
                    // we manually calculate its diffuse material color
                    if (object.shape_t == pathtracer::TRIANGLE && object.shape_d.triangle.texture_idx > -1) {
                        pathtracer::vec3 interpolated_texture_coordinate = ((object.shape_d.triangle.tex2 * comp.intersection.u) + (object.shape_d.triangle.tex3 * comp.intersection.v) + (object.shape_d.triangle.tex1 * (1.f - comp.intersection.u - comp.intersection.v))) / 3;
                        float* texture = world.textures[object.shape_d.triangle.texture_idx];
                        int h = static_cast<int>(interpolated_texture_coordinate.x * 618);
                        int w = static_cast<int>(interpolated_texture_coordinate.y * 1100);
                        int offset = h * 1100 * 4 + w * 4;
                        pathtracer::vector diffuse_color = {texture[offset + 0], texture[offset + 1], texture[offset + 2]};
                        material_copy.color = diffuse_color;
                    }

                    float u = curand_uniform(state);
                    float v = curand_uniform(state);
                    float t = curand_uniform(state);

                    if (object.mat_t == pathtracer::LIGHT) {
                        multiplier &= {1000.f, 1000.f, 1000.f};
                        break;
                    }

                    pathtracer::vector out_ray_direction;

                    pathtracer::vector out_sample_weight;

                    // bool eval_successful = pathtracer::eval_brdf(u, v, t, current_refractive_index, comp.surface_normal, comp.eye_vector, out_ray_direction, out_sample_weight, current_refractive_index, object.mat_d.microfacet);
                    bool eval_successful = pathtracer::eval_brdf(u, v, t, current_refractive_index, comp.surface_normal, comp.eye_vector, out_ray_direction, out_sample_weight, current_refractive_index, material_copy);

                    if (!eval_successful) {
                        multiplier &= {0.f, 0.f, 0.f};
                        break;
                    };

                    multiplier &= out_sample_weight;

                    if ((0.f < t && t <= object.mat_d.microfacet.transmissiveness)) {
                        ray = pathtracer::ray(comp.surface_point + ((-comp.surface_normal) * 0.01f), out_ray_direction);
                    } else {
                        ray = pathtracer::ray(comp.surface_point + (comp.surface_normal * 0.01f), out_ray_direction);
                    }
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

TEST_CASE("Full mesh brdf renders") {
    SECTION("Constant") {
        constexpr int canvas_pixels = 1000;
        pathtracer::canvas c{canvas_pixels, canvas_pixels};

        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        pathtracer::camera camera(1000, 1000, pathtracer::pi / 2.f, {0.f, 0.f, -10.f}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, pathtracer::mat4::get_rotation_z(pathtracer::pi / 4.f));

        pathtracer::object obj0{pathtracer::SPHERE, 
             pathtracer::sphere(pathtracer::mat4::get_translation(-2.f, 0.f, -2.f)),
             pathtracer::MICROFACET,
             pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
        obj0.mat_d.microfacet = pathtracer::microfacet{{0.25f, 0.25f, 0.95f}, {0.f, 0.f, 0.f}, 0.95f, 1.5f, 0.f, 1.f};

        pathtracer::object obj1{pathtracer::SPHERE, 
             pathtracer::sphere(pathtracer::mat4::get_translation(-1.f, -1.f, 5.f)),
             pathtracer::MICROFACET,
             pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
        obj1.mat_d.microfacet = pathtracer::microfacet{{0.35f, 0.25f, 0.75f}, {0.f, 0.f, 0.f}, 0.75f, 0.2f, 0.f, 4.f};

        pathtracer::object obj3{pathtracer::SPHERE, 
             pathtracer::sphere(pathtracer::mat4::get_translation(1.f, 1.f, 2.f)),
             pathtracer::MICROFACET,
             pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
        obj3.mat_d.microfacet = pathtracer::microfacet{{0.75f, 0.25f, 0.5f}, {0.f, 0.f, 0.f}, 0.75f, 0.2f, 0.f, 1.f};

        pathtracer::object obj4{pathtracer::SPHERE, 
             pathtracer::sphere(pathtracer::mat4::get_translation(2.f, 0.f, 1.f)),
             pathtracer::MICROFACET,
             pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
        obj4.mat_d.microfacet = pathtracer::microfacet{{0.95f, 0.25f, 0.5f}, {0.f, 0.f, 0.f}, 0.75f, 0.2f, 0.f, 1.f};

        pathtracer::object obj5{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(-10.f, 0.f, -10.f)),
             pathtracer::LIGHT,
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)};

        pathtracer::object obj6{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(-10.f, 0.f, -10.f)),
             pathtracer::LIGHT,
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)};

        pathtracer::object obj7{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(10.f, 0.f, -10.f)),
             pathtracer::LIGHT,
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)};

        pathtracer::object obj8{pathtracer::SPHERE, 
             pathtracer::sphere(pathtracer::mat4::get_translation(2.5f, 3.f, 2.f)),
             pathtracer::MICROFACET,
             pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
        obj8.mat_d.microfacet = pathtracer::microfacet{{0.95f, 0.95f, 0.f}, {500.f, 500.f, 0.f}, 1.f, 0.01f, 0.f, 1.f};

        pathtracer::object obj9{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(10.f, 10.f, -10.f)),
             pathtracer::LIGHT,
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)};


        pathtracer::world w({
            &obj0, &obj1, &obj3, &obj4, &obj5, &obj6, &obj8
        }, {"teapot_full.obj"}, {pathtracer::mat4::get_translation(0.f, 0.f, -5.f) * pathtracer::mat4::get_scaling(0.1f, 0.1f, 0.1f)}, {"cursed.exr"}, blocks, threads);

        // pathtracer::world w({
        //  &obj1, &obj6, &obj7
        // }, {"teapot.obj"}, {pathtracer::mat4::get_identity()}, blocks, threads);


        curandState* d_states;

        checkCudaErrors( cudaMalloc(reinterpret_cast<void**>(&d_states), blocks.y * blocks.x * threads.y * threads.x * sizeof(curandState)) );

        mesh_constant_brdf_test<<<blocks, threads>>>(c, w, camera, d_states);

        checkCudaErrors( cudaDeviceSynchronize() );

        c.export_as_PPM("Mesh_Test_GPU.ppm");
        c.export_as_EXR("Mesh_Test_GPU.exr");
        pathtracer::denoise(canvas_pixels, canvas_pixels, "Mesh_Test_GPU.exr", w, camera, "Mesh_Test_GPU_denoised.exr");
    }
}