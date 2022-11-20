#pragma once
#include "bvh.cuh"
#include "shapes.cuh"
#include <initializer_list>
#include <vector>

namespace pathtracer {

    struct computations {
        intersection intersection;
        point surface_point;
        vector eye_vector;
        vector surface_normal;
        bool is_inside;
    };

    struct world {
        int num_objects;
        pathtracer::object* objects;
        pathtracer::bvh_arena* arena;
        pathtracer::bvh_node* bvh_root;
        int* collision_buffer;
        pathtracer::intersection* intersection_buffer;

        world(const std::initializer_list<object> l, dim3 blocks, dim3 threads);

        world(const std::initializer_list<object*> l, dim3 blocks, dim3 threads);

        world(const std::vector<object*> l, const std::vector<std::string> obj_filenames, const std::vector<mat4> obj_to_world_transformations, dim3 blocks, dim3 threads);

        __host__ __device__ computations prepare_computations(const intersection& intersection, const ray& r);

        __host__ __device__ computations intersect_world(const ray& r, bool& success_flag, int* collision_buffer, pathtracer::intersection* intersection_buffer);
    };

}