#pragma once

#include "util.cuh"
#include "bvh.cuh"

namespace pathtracer {

    struct ray {
        point o;
        vector d;
        vec3 d_inv;

        __host__ __device__ ray(const point& o, const vector& d);

        __host__ __device__ bool operator==(const ray& other) const;

        __host__ __device__ bool check_bvh_node_intersection(bvh_node* b) const;

        __host__ __device__ int find_intersections(bvh_node* root, int* collision_buffer) const;

        __host__ __device__ point shoot_distance(float t) const;
    };

}
