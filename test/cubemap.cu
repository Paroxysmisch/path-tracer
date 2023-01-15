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

__global__ void cubemap_test(pathtracer::canvas c, pathtracer::world world, pathtracer::camera camera, curandState* d_states) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int j_original = j;
    int num_threads_i = blockDim.y * gridDim.y;
    int num_threads_j = blockDim.x * gridDim.x;

    curandState* state = &d_states[i * num_threads_j + j];
    curand_init(1234, i * num_threads_j + j, 0, state);

    pathtracer::vec3 color_black{0.f, 0.f, 0.f};

    int collision_buffer_offset;
    if (pathtracer::collision_buffer_limit_enable) {
        collision_buffer_offset = min(world.num_objects, pathtracer::collision_buffer_limit) * (i * num_threads_j + j);
    } else {
        collision_buffer_offset = world.num_objects * (i * num_threads_j + j);
    }
    int* collision_buffer = (world.collision_buffer + collision_buffer_offset);

    int intersection_buffer_offset;
    if (pathtracer::intersection_buffer_limit_enable) {
        intersection_buffer_offset = 2 * min(world.num_objects, pathtracer::intersection_buffer_limit) * (i * num_threads_j + j);
    } else {
        intersection_buffer_offset = 2 * world.num_objects * (i * num_threads_j + j);
    }
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
                        // int num_intersections = xy_posz_top.find_intersections(ray, triangle_buffer, 0);
                        // if (num_intersections > 0) {
                        //     multiplier &= {1.f, 0.f, 0.f};
                        // } else{
                        float t_xy_posz = (1 - ray.o.z) * ray.d_inv.z;
                        pathtracer::point intersection_point = ray.shoot_distance(t_xy_posz);
                        if (t_xy_posz > 0 && -1.f <= intersection_point.x && intersection_point.x <= 1.f && -1.f <= intersection_point.y && intersection_point.y <= 1.f) {
                            float tex_u = (intersection_point.x + 1) / 2.f;
                            float tex_v = (intersection_point.y + 1) / 2.f;
                            float* texture = world.textures[0];
                            int w = static_cast<int>(fmod(tex_u - pathtracer::epsilon, 1.f) * world.texture_datas[0].width);
                            int h = static_cast<int>(fmod(tex_v - pathtracer::epsilon, 1.f) * world.texture_datas[0].height);
                            int offset = h * world.texture_datas[0].width * 4 + w * 4;
                            multiplier &= {texture[offset + 0], texture[offset + 1], texture[offset + 2]};
                        } else {
                            multiplier &= {0.25f, 0.25f, 0.25f};
                        }
                        // }
                        break;
                    }

                    pathtracer::object& object = world.objects[comp.intersection.object_index];
                    pathtracer::microfacet material_copy = object.mat_d.microfacet;

                    // If the object intersected is a triangle and uses textures,
                    // we manually calculate its diffuse material color
                    if (object.shape_t == pathtracer::TRIANGLE && object.shape_d.triangle.texture_idx > -1) {
                        pathtracer::vec3 interpolated_texture_coordinate = ((object.shape_d.triangle.tex2 * comp.intersection.u) + (object.shape_d.triangle.tex3 * comp.intersection.v) + (object.shape_d.triangle.tex1 * (1.f - comp.intersection.u - comp.intersection.v)));
                        float* texture = world.textures[object.shape_d.triangle.texture_idx];
                        int w = static_cast<int>(fmod(interpolated_texture_coordinate.x - pathtracer::epsilon, 1.f) * world.texture_datas[object.shape_d.triangle.texture_idx].width);
                        int h = static_cast<int>(fmod(interpolated_texture_coordinate.y - pathtracer::epsilon, 1.f) * world.texture_datas[object.shape_d.triangle.texture_idx].height);
                        int offset = h * world.texture_datas[object.shape_d.triangle.texture_idx].width * 4 + w * 4;
                        pathtracer::vector diffuse_color = {texture[offset + 0], texture[offset + 1], texture[offset + 2]};
                        material_copy.color = diffuse_color;
                    }

                    float u = curand_uniform(state);
                    float v = curand_uniform(state);
                    float t = curand_uniform(state);

                    if (object.mat_t == pathtracer::LIGHT) {
                        multiplier &= {100.f, 100.f, 100.f};
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

TEST_CASE("Cubemap renders") {
    SECTION("Constant") {
        constexpr int canvas_pixels = 1000;
        pathtracer::canvas c{canvas_pixels, canvas_pixels};

        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        pathtracer::camera camera(1000, 1000, pathtracer::pi / 2.f, {0.f, 0.f, -0.85f}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, pathtracer::mat4::get_identity());


        pathtracer::object obj0{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(-0.5f, 0.f, -0.8f) * pathtracer::mat4::get_scaling(0.1f, 0.1f, 0.1f)),
             pathtracer::LIGHT,
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)};

        pathtracer::object obj1{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(-0.5f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.25f, 0.25f, 0.25f)),
             pathtracer::MICROFACET,
             pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
        obj1.mat_d.microfacet = pathtracer::microfacet{{0.35f, 0.25f, 0.75f}, {0.f, 0.f, 0.f}, 0.9f, 0.2f, 0.f, 1.f};

        pathtracer::object obj2{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(0.5f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.25f, 0.25f, 0.25f)),
             pathtracer::MICROFACET,
             pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
        obj2.mat_d.microfacet = pathtracer::microfacet{{0.65f, 0.85f, 0.15f}, {0.f, 0.f, 0.f}, 0.1f, 0.8f, 0.f, 1.f, 0.01f};


        // pathtracer::world w({
        //     &obj0
        // }, {"teapot_full.obj", "xy_wall.obj"}, {pathtracer::mat4::get_scaling(0.01f, 0.01f, 0.01f),  pathtracer::mat4::get_translation(0.f, 0.f, 1.f)}, {{"cursed.exr", 618, 1100}, {"landscape.exr", 1705, 2729}}, blocks, threads);

        pathtracer::world w({
            &obj0, &obj1, &obj2
        }, {"teapot_full.obj"}, {pathtracer::mat4::get_scaling(0.01f, 0.01f, 0.01f)}, {{"cursed.exr", 618, 1100}}, blocks, threads);


        // pathtracer::world w({
        //  &obj1, &obj6, &obj7
        // }, {"teapot.obj"}, {pathtracer::mat4::get_identity()}, blocks, threads);


        curandState* d_states;

        checkCudaErrors( cudaMalloc(reinterpret_cast<void**>(&d_states), blocks.y * blocks.x * threads.y * threads.x * sizeof(curandState)) );

        cubemap_test<<<blocks, threads>>>(c, w, camera, d_states);

        checkCudaErrors( cudaDeviceSynchronize() );

        c.export_as_PPM("Cubemap_Test_GPU.ppm");
        c.export_as_EXR("Cubemap_Test_GPU.exr");
        pathtracer::denoise(canvas_pixels, canvas_pixels, "Cubemap_Test_GPU.exr", w, camera, "Cubemap_Test_GPU_denoised.exr");
    }
}