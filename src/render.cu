#include "denoise.cuh"
#include "render.cuh"
#include "util.cuh"

namespace pathtracer {

    __global__ void render_kernel(canvas c, world world, camera camera, curandState* d_states, int num_samples, bool enable_adaptive_sampling, const float adaptive_sampling_variance_threshold) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        const int j_original = j;
        int num_threads_i = blockDim.y * gridDim.y;
        int num_threads_j = blockDim.x * gridDim.x;

        // extern __shared__ float s[];
        // float* x = s;
        // float* x_2 = &x[blockDim.x * blockDim.y];
        const int adaptive_sampling_rate = 50;
        const int threads_per_block = blockDim.y * blockDim.x;

        curandState* state = &d_states[i * num_threads_j + j];
        curand_init(1234, i * num_threads_j + j, 0, state);

        vec3 color_black{0.f, 0.f, 0.f};

        int collision_buffer_offset;
        if (collision_buffer_limit_enable) {
            collision_buffer_offset = min(world.num_objects, collision_buffer_limit) * (i * num_threads_j + j);
        } else {
            collision_buffer_offset = world.num_objects * (i * num_threads_j + j);
        }
        int* collision_buffer = (world.collision_buffer + collision_buffer_offset);

        int intersection_buffer_offset;
        if (intersection_buffer_limit_enable) {
            intersection_buffer_offset = 2 * min(world.num_objects, intersection_buffer_limit) * (i * num_threads_j + j);
        } else {
            intersection_buffer_offset = 2 * world.num_objects * (i * num_threads_j + j);
        }
        intersection* intersection_buffer = (world.intersection_buffer + intersection_buffer_offset);

        constexpr int max_depth = 10;

        float refraction_idx_buffer[max_depth + 1];
        refraction_idx_buffer[0] = 1.f;
        int refraction_idx_buffr_ptr = 0;

        while (i < c.m_height) {
            while (j < c.m_width) {
                vec3_d color{0.f, 0.f, 0.f};
                int num_samples_taken{-1};
                vec3_d color_2{0.f, 0.f, 0.f};

                for (int k{0}; k < num_samples; ++k) {
                    // floats a and b for anti-aliasing
                    float a = curand_uniform(state);
                    float b = curand_uniform(state);
                    ray ray = camera.gen_ray_for_pixel(i, j, a, b);
                    bool success_flag{false};
                    vec3 multiplier{1.f, 1.f, 1.f};
                    float current_refractive_index = 1.f;

                    for (int l{0}; l < max_depth; ++l) {
                        computations comp = world.intersect_world(ray, success_flag, collision_buffer, intersection_buffer);

                        if (!success_flag) {
                            float a = ray.d * ray.d;
                            float b = 2 * (ray.d * ray.o);
                            float c = (ray.o * ray.o) - 1.f;

                            float discriminant = (b * b) - (4 * a * c);

                            if (world.environment_map != nullptr && discriminant >= 0 && ray.o.mag_2() <= 1) {
                                point intersection_point = ray.shoot_distance((-b + sqrtf(discriminant)) / (2 * a));
                                intersection_point = vec3(0.f, 0.f, 0.f) - intersection_point;
                                float tex_u = 0.5f + atan2f(intersection_point.z, intersection_point.x) * 0.5f * one_over_pi;
                                float tex_v = 0.5f + asinf(intersection_point.y) * one_over_pi;
                                int w = static_cast<int>(fmod(tex_u - epsilon, 1.f) * world.environment_map_width);
                                int h = static_cast<int>(fmod(tex_v - epsilon, 1.f) * world.environment_map_height);
                                int offset = h * world.environment_map_width * 4 + w * 4;
                                multiplier &= {world.environment_map[offset + 0], world.environment_map[offset + 1], world.environment_map[offset + 2]};
                            } else {
                                multiplier &= {0.f, 0.f, 0.f};
                            }
                            break;
                        }

                        object& object = world.objects[comp.intersection.object_index];
                        microfacet material_copy = object.mat_d.microfacet;

                        // If the object intersected is a triangle and uses textures,
                        // we manually calculate its diffuse material color
                        if (object.shape_t == TRIANGLE && object.shape_d.triangle.texture_idx > -1) {
                            vec3 interpolated_texture_coordinate = ((object.shape_d.triangle.tex2 * comp.intersection.u) + (object.shape_d.triangle.tex3 * comp.intersection.v) + (object.shape_d.triangle.tex1 * (1.f - comp.intersection.u - comp.intersection.v)));
                            float* texture = world.textures[object.shape_d.triangle.texture_idx];
                            int w = static_cast<int>(fmod(interpolated_texture_coordinate.x - epsilon, 1.f) * world.texture_datas[object.shape_d.triangle.texture_idx].width);
                            int h = static_cast<int>(fmod(interpolated_texture_coordinate.y - epsilon, 1.f) * world.texture_datas[object.shape_d.triangle.texture_idx].height);
                            int offset = h * world.texture_datas[object.shape_d.triangle.texture_idx].width * 4 + w * 4;
                            vector diffuse_color = {texture[offset + 0], texture[offset + 1], texture[offset + 2]};
                            material_copy.color = diffuse_color;
                        }

                        float u = curand_uniform(state);
                        float v = curand_uniform(state);
                        float t = curand_uniform(state);

                        if (object.mat_t == LIGHT) {
                            multiplier &= object.mat_d.light.color * 16.f;
                            break;
                        }

                        vector out_ray_direction;

                        vector out_sample_weight;

                        vector tangent;
                        vector bitangent;

                        if (object.shape_t == TRIANGLE) {
                            tangent = object.shape_d.triangle.tan1;
                            bitangent = object.shape_d.triangle.tan2;
                        } else {
                            tangent = object.shape_d.sphere.world_tangent_at(comp.surface_point);
                            bitangent = (tangent ^ comp.surface_normal).normalize();
                        }

                        bool eval_successful = eval_brdf_anisotropic(u, v, t, comp.surface_normal, comp.eye_vector, out_ray_direction, out_sample_weight, material_copy, tangent, bitangent, comp.intersection.t_value, comp.is_inside, refraction_idx_buffer, refraction_idx_buffr_ptr);

                        if (!eval_successful) {
                            multiplier &= {0.f, 0.f, 0.f};
                            break;
                        };

                        multiplier &= out_sample_weight;

                        // Russian Roulette
                        float p = maxf(multiplier.x, maxf(multiplier.y, multiplier.z));
                        float c = curand_uniform(state);
                        if (c > p) {
                            break;
                        }

                        // Add energy lost due to Russian Roulette
                        multiplier /= p;


                        if ((0.f < t && t <= object.mat_d.microfacet.transmissiveness)) {
                            ray = pathtracer::ray(comp.surface_point + ((-comp.surface_normal) * 0.01f), out_ray_direction);
                        } else {
                            ray = pathtracer::ray(comp.surface_point + (comp.surface_normal * 0.01f), out_ray_direction);
                        }
                    }

                    // Radiance clamping
                    float clamp_value = 16.f;
                    multiplier.x = minf(multiplier.x, clamp_value);
                    multiplier.y = minf(multiplier.y, clamp_value);
                    multiplier.z = minf(multiplier.z, clamp_value);

                    color += multiplier;
                    color_2 += (multiplier & multiplier);

                    if (enable_adaptive_sampling && (k > 0) && (k % adaptive_sampling_rate == 0)) {
                        vec3 mu = color / k;
                        vec3 variance = (color_2 - ((color & color) / k)) * (1.f / (k - 1));
                        variance.x = sqrtf(fabsf(variance.x));
                        variance.y = sqrtf(fabsf(variance.y));
                        variance.z = sqrtf(fabsf(variance.z));

                        vec3 convergence = variance * (1.96f / sqrtf(k));

                        if (convergence.x < adaptive_sampling_variance_threshold * mu.x &&
                            convergence.y < adaptive_sampling_variance_threshold * mu.y &&
                            convergence.z < adaptive_sampling_variance_threshold * mu.z ) {
                            num_samples_taken = k;
                            break;
                        }
                    }
                }

                if (num_samples_taken == -1) {
                    color /= num_samples;
                } else {
                    color /= num_samples_taken;
                }
                    
                c.write_pixel(i, j, {static_cast<float>(color.x), static_cast<float>(color.y), static_cast<float>(color.z)});
                
                j += num_threads_j;
            }
            i += num_threads_i;
            j = j_original;
        }    
    }

    __host__ void render(camera& camera, world& world, std::string filename, int num_samples, bool enable_adaptive_sampling, const float adaptive_sampling_variance_threshold) {
        canvas canvas{camera.height, camera.width};

        dim3 blocks = world.blocks;
        dim3 threads = world.threads;

        curandState* d_states;

        checkCudaErrors( cudaMalloc(reinterpret_cast<void**>(&d_states), blocks.y * blocks.x * threads.y * threads.x * sizeof(curandState)) );

        render_kernel<<<blocks, threads>>>(canvas, world, camera, d_states, num_samples, enable_adaptive_sampling, adaptive_sampling_variance_threshold);

        checkCudaErrors( cudaDeviceSynchronize() );

        canvas.export_as_PPM(filename + ".ppm");
        canvas.export_as_EXR(filename + ".exr");
        denoise(camera.height, camera.width, filename + ".exr", world, camera, filename + "_denoised.exr");
    }

}
