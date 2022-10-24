#include <cuda.h>
#include <catch2/catch.hpp>
#include "check_cuda_errors.h"
#include "bvh.cuh"
#include "shapes.cuh"
#include "scene.cuh"
#include "util.cuh"
#include "phong.cuh"

__global__ void bounding_box_test(pathtracer::canvas<1000, 1000> c, pathtracer::object* objects, pathtracer::bvh_node* root) {
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

__global__ void multiple_shadow_test(pathtracer::canvas<1000, 1000> c, pathtracer::object* objects, pathtracer::bvh_node* root) {
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

            int collision_buffer[5];

            int possible_intersections = ray.find_intersections(root, collision_buffer);

            pathtracer::intersection intersection_buffer[2 * 5];

            pathtracer::intersection* intersection_buffer_ptr = intersection_buffer;
            int intersection_buffer_size{0};

            for (int k{0}; k < possible_intersections; ++k) {
                int object_index = collision_buffer[k];

                int num_intersections = objects[object_index].shape_d.sphere.find_intersections(ray, intersection_buffer_ptr, object_index);

                intersection_buffer_ptr += num_intersections;
                intersection_buffer_size += num_intersections;
            }

            pathtracer::intersection* closest_positive_intersection = pathtracer::get_closest_positive_intersection(intersection_buffer, intersection_buffer_size);

            if (closest_positive_intersection != nullptr) {
                c.write_pixel(i, j, {0.1f * closest_positive_intersection->object_index, 0.5f, 0.2f * closest_positive_intersection->object_index});
            }
            
            j += num_threads_j;
        }
        i += num_threads_i;
        j = j_original;
    }
}

__global__ void phong_test(pathtracer::canvas<1000, 1000> c, pathtracer::object* objects, pathtracer::bvh_node* root, pathtracer::object* light) {
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

    while (i < 1000) {
        float world_y = half - pixel_size * i;
        while (j < 1000) {
            c.write_pixel(i, j, color_black);

            float world_x = -half + pixel_size * j;

            pathtracer::vec3 pos{world_x, world_y, wall_z};

            pathtracer::ray ray{ray_origin, (pos - ray_origin).normalize()};

            int collision_buffer[5];

            int possible_intersections = ray.find_intersections(root, collision_buffer);

            pathtracer::intersection intersection_buffer[2 * 5];

            pathtracer::intersection* intersection_buffer_ptr = intersection_buffer;
            int intersection_buffer_size{0};

            for (int k{0}; k < possible_intersections; ++k) {
                int object_index = collision_buffer[k];

                int num_intersections = objects[object_index].shape_d.sphere.find_intersections(ray, intersection_buffer_ptr, object_index);

                intersection_buffer_ptr += num_intersections;
                intersection_buffer_size += num_intersections;
            }

            pathtracer::intersection* closest_positive_intersection = pathtracer::get_closest_positive_intersection(intersection_buffer, intersection_buffer_size);

            if (closest_positive_intersection != nullptr) {
                pathtracer::point world_point = ray.shoot_distance(closest_positive_intersection->t_value);
                pathtracer::vector normal_vector = objects[closest_positive_intersection->object_index].shape_d.sphere.world_normal_at(world_point);
                pathtracer::vector eye_vector = (ray_origin - world_point).normalize();
                pathtracer::vec3 color = pathtracer::phong_lighting(&objects[closest_positive_intersection->object_index], light, &world_point, &eye_vector, &normal_vector);
                c.write_pixel(i, j, color);
            }
            
            j += num_threads_j;
        }
        i += num_threads_i;
        j = j_original;
    }
}

TEST_CASE("BVH construction from objects", "[scene, acceleron_datastructures, shapes, ray]") {
    SECTION("Singular object") {
        pathtracer::object objects[1] = {{pathtracer::SPHERE, pathtracer::sphere(pathtracer::mat4::get_identity()), pathtracer::PHONG, pathtracer::phong({0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f)}};

        pathtracer::morton_and_index out_buffer[1];

        pathtracer::gen_sorted_morton_codes_and_indices(objects, 1, out_buffer);

        pathtracer::bvh_arena arena{1};

        pathtracer::bvh_node* root = pathtracer::_gen_bvh(out_buffer, objects, 0, 0, &arena);

        REQUIRE((root->object_index == 0) == true);
    }

    SECTION("Multiple objects bounding box render GPU") {
        constexpr int canvas_pixels = 1000;
        pathtracer::canvas<canvas_pixels, canvas_pixels> c{};

        constexpr int num_objects = 2;

        pathtracer::object* objects;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&objects), num_objects * sizeof(pathtracer::object)) );

        objects[0].shape_t = pathtracer::SPHERE;
        objects[0].shape_d = pathtracer::sphere(pathtracer::mat4::get_identity());

        objects[1].shape_t = pathtracer::SPHERE;
        objects[1].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 0.5f, 0.f));

        pathtracer::bvh_arena arena{num_objects};

        pathtracer::bvh_node* root = pathtracer::gen_bvh(objects, num_objects, &arena);
        
        REQUIRE((root->lower == pathtracer::vec3(-1.f, -1.f, -1.f)));
        REQUIRE((root->upper == pathtracer::vec3(1.f, 1.5f, 1.f)));

        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        bounding_box_test<<<blocks, threads>>>(c, objects, root);

        checkCudaErrors( cudaDeviceSynchronize() );

        c.export_as_PPM("Bounding_Box_Test_GPU.ppm");
    }

    SECTION("Multiple objects shadow render GPU") {
        constexpr int canvas_pixels = 1000;
        pathtracer::canvas<canvas_pixels, canvas_pixels> c{};

        constexpr int num_objects = 5;

        pathtracer::object* objects;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&objects), num_objects * sizeof(pathtracer::object)) );

        objects[0].shape_t = pathtracer::SPHERE;
        objects[0].shape_d = pathtracer::sphere(pathtracer::mat4::get_identity());

        objects[1].shape_t = pathtracer::SPHERE;
        objects[1].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 0.5f, 0.f));

        objects[2].shape_t = pathtracer::SPHERE;
        objects[2].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(0.f, -0.5f, 0.f));

        objects[3].shape_t = pathtracer::SPHERE;
        objects[3].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(0.5f, 0.5f, 0.f));

        objects[4].shape_t = pathtracer::SPHERE;
        objects[4].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(-0.5f, 0.5f, 0.f));

        pathtracer::bvh_arena arena{num_objects};

        pathtracer::bvh_node* root = pathtracer::gen_bvh(objects, num_objects, &arena);
        
        REQUIRE((root->lower == pathtracer::vec3(-1.5f, -1.5f, -1.f)));
        REQUIRE((root->upper == pathtracer::vec3(1.5f, 1.5f, 1.f)));

        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        multiple_shadow_test<<<blocks, threads>>>(c, objects, root);

        checkCudaErrors( cudaDeviceSynchronize() );

        c.export_as_PPM("Multiple_Shadow_Test_GPU.ppm");

        REQUIRE((root->left->object_index == 4) == true);
        REQUIRE((root->right->left->object_index == 2) == true);
        REQUIRE((root->right->right->right->object_index == 3) == true);
        REQUIRE((root->right->right->left->left->object_index == 0) == true);
        REQUIRE((root->right->right->left->right->object_index == 1) == true);
    }
}

TEST_CASE("Full renders") {
    SECTION("Phong") {
        constexpr int canvas_pixels = 1000;
        pathtracer::canvas<canvas_pixels, canvas_pixels> c{};

        constexpr int num_objects = 5;

        pathtracer::object* objects;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&objects), num_objects * sizeof(pathtracer::object)) );

        objects[0].shape_t = pathtracer::SPHERE;
        objects[0].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(-2.f, 0.f, -2.f));
        objects[0].mat_t = pathtracer::PHONG;
        objects[0].mat_d.phong = pathtracer::phong({0.25f, 0.25f, 0.95f}, 0.1f, 0.9f, 0.9f, 200);

        objects[1].shape_t = pathtracer::SPHERE;
        objects[1].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(-1.f, -1.f, 0.f));
        objects[1].mat_t = pathtracer::PHONG;
        objects[1].mat_d.phong = pathtracer::phong({0.35f, 0.25f, 0.75f}, 0.1f, 0.9f, 0.9f, 200);

        objects[2].shape_t = pathtracer::SPHERE;
        objects[2].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 0.f, -1.f));
        objects[2].mat_t = pathtracer::PHONG;
        objects[2].mat_d.phong = pathtracer::phong({0.75f, 0.25f, 0.5f}, 0.1f, 0.9f, 0.9f, 100);

        objects[3].shape_t = pathtracer::SPHERE;
        objects[3].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(1.f, 1.f, 2.f));
        objects[3].mat_t = pathtracer::PHONG;
        objects[3].mat_d.phong = pathtracer::phong({0.75f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10);

        objects[4].shape_t = pathtracer::SPHERE;
        objects[4].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(2.f, 0.f, 1.f));
        objects[4].mat_t = pathtracer::PHONG;
        objects[4].mat_d.phong = pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10);

        pathtracer::bvh_arena arena{num_objects};

        pathtracer::bvh_node* root = pathtracer::gen_bvh(objects, num_objects, &arena);

        pathtracer::object* light;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&light), sizeof(pathtracer::object)) );

        light[0].shape_t = pathtracer::SPHERE;
        light[0].shape_d = pathtracer::sphere(pathtracer::mat4::get_translation(-10.f, 0.f, -10.f));
        light[0].mat_t = pathtracer::LIGHT;
        light[0].mat_d.light = pathtracer::light({1.f, 1.f, 1.f});

        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        phong_test<<<blocks, threads>>>(c, objects, root, light);

        checkCudaErrors( cudaDeviceSynchronize() );

        c.export_as_PPM("Phong_Test_GPU.ppm");
    }
}