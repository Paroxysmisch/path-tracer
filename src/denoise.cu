#include <cmath>
#include <cstddef>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include "denoise.cuh"
#include "util.cuh"

namespace pathtracer {

    __global__ void get_albedo_point(canvas c, world world, camera camera) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        const int j_original = j;
        int num_threads_i = blockDim.y * gridDim.y;
        int num_threads_j = blockDim.x * gridDim.x;

        pathtracer::vec3 color_black{0.f, 0.f, 0.f};

        int collision_buffer_offset = world.num_objects * (i * num_threads_j + j);
        int* collision_buffer = (world.collision_buffer + collision_buffer_offset);

        int intersection_buffer_offset = 2 * world.num_objects * (i * num_threads_j + j);

        pathtracer::intersection* intersection_buffer = (world.intersection_buffer + intersection_buffer_offset);

        while (i < c.m_height) {
            while (j < c.m_width) {
                pathtracer::ray ray = camera.gen_ray_for_pixel(i, j);
                bool success_flag{false};

                pathtracer::computations comp = world.intersect_world(ray, success_flag, collision_buffer, intersection_buffer);

                if (!success_flag) {
                    c.write_pixel(i, j, color_black);
                } else {
                    pathtracer::object& object = world.objects[comp.intersection.object_index];
                    switch (object.mat_t) {
                        case MICROFACET:
                            c.write_pixel(i, j, object.mat_d.microfacet.color + object.mat_d.microfacet.emission);
                            break;
                        case PHONG:
                            c.write_pixel(i, j, object.mat_d.phong.color);
                            break;
                        case LIGHT:
                            c.write_pixel(i, j, object.mat_d.light.color);
                            break;
                    }
                }

                j += num_threads_j;
            }
            i += num_threads_i;
            j = j_original;
        }
    }

    __global__ void get_normal_point(canvas c, world world, camera camera) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        const int j_original = j;
        int num_threads_i = blockDim.y * gridDim.y;
        int num_threads_j = blockDim.x * gridDim.x;

        pathtracer::vec3 no_normal{0.f, 0.f, 0.f};

        int collision_buffer_offset = world.num_objects * (i * num_threads_j + j);
        int* collision_buffer = (world.collision_buffer + collision_buffer_offset);

        int intersection_buffer_offset = 2 * world.num_objects * (i * num_threads_j + j);

        pathtracer::intersection* intersection_buffer = (world.intersection_buffer + intersection_buffer_offset);

        while (i < c.m_height) {
            while (j < c.m_width) {
                vec3 camera_space_normal;

                pathtracer::ray ray = camera.gen_ray_for_pixel(i, j);
                bool success_flag{false};

                pathtracer::computations comp = world.intersect_world(ray, success_flag, collision_buffer, intersection_buffer);

                if (!success_flag) {
                    c.write_pixel(i, j, no_normal);
                } else {
                    pathtracer::object& object = world.objects[comp.intersection.object_index];
                    
                    switch (object.shape_t) {
                        case SPHERE:
                            vec3 world_normal = object.shape_d.sphere.world_normal_at(comp.surface_point);
                            camera_space_normal = (camera.inverse_transform).transpose().transform_vector(world_normal);
                            break;
                    }

                    // A normal can be negative, so we write the absolute value of the normal
                    camera_space_normal.x = fabsf(camera_space_normal.x);
                    camera_space_normal.y = fabsf(camera_space_normal.y);
                    camera_space_normal.z = fabsf(camera_space_normal.z);

                    c.write_pixel(i, j, camera_space_normal);
                }

                j += num_threads_j;
            }
            i += num_threads_i;
            j = j_original;
        }
    }

    bool get_albedo_exr(int height, int width, const world& world, const camera& camera, const std::string &filename) {
        canvas c{height, width};

        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        get_albedo_point<<<blocks, threads>>>(c, world, camera);

        checkCudaErrors( cudaDeviceSynchronize() );

        return c.export_as_EXR(filename);
    }

    bool get_normal_map_exr(int height, int width, const world& world, const camera& camera, const std::string &filename) {
        canvas c{height, width};

        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        get_normal_point<<<blocks, threads>>>(c, world, camera);

        checkCudaErrors( cudaDeviceSynchronize() );

        return c.export_as_EXR(filename);
    }

    pid_t call_optixDenoiser(const char* argv[]) {
        char* _argv[9];
        memcpy(_argv, argv, 9 * sizeof(char *));
        pid_t pid = fork();
        if (pid == 0) {
            execvp(_argv[0], _argv);
            perror("Could not execve optixDenoiser! Are you sure \"optixDenoiser\" is in your PATH?");
            exit(1);
        } else {
            return pid;
        }
    }

    void denoise(int height, int width,  const std::string &in_filename, const world& world, const camera& camera, const std::string &out_filename) {
        std::string normals = in_filename.substr(0, in_filename.size() - 4) + "_normal_map.exr";
        std::string albedo = in_filename.substr(0, in_filename.size() - 4) + "_albedo.exr";

        get_normal_map_exr(height, width, world, camera, normals);
        get_albedo_exr(height, width, world, camera, albedo);

        const char* argv[] = { "optixDenoiser", "-n", normals.c_str(), "-a", albedo.c_str(), "-o", out_filename.c_str(), in_filename.c_str(), nullptr };
        pid_t pid = call_optixDenoiser(argv);
        std::cout << "Denoiser (optixDenoiser) running on pid: " << pid << std::endl;
        wait(nullptr);
    }

}
