#pragma once
#include "ray.cuh"
#include "util.cuh"

namespace pathtracer {
    struct intersection {
        float t_value;
        int object_index;
    };

    __host__ __device__ intersection* get_closest_positive_intersection(intersection* intersection_buffer, int size);

    struct shape {
        vec3 lower;
        vec3 upper;
        mat4 transformation_to_world;
        mat4 transformation_to_object;

        __host__ __device__ shape(vec3 lower, vec3 upper, mat4 transformation_to_world);

        __host__ __device__ virtual int find_intersections(const ray& r, intersection* intersection_buffer, int object_index) = 0;
    };

    struct sphere : shape {
        __host__ __device__ sphere(const mat4& transformation_to_world);

        __host__ __device__ virtual int find_intersections(const ray& r, intersection* intersection_buffer, int object_index) override;
    };

    enum shape_type {
        SPHERE
    };

    union shape_data {
        sphere sphere;
    };

    struct object {
        shape_type shape_t;
        shape_data data;
    };
}