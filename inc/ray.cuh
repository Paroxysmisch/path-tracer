#pragma once

#include "util.cuh"
#include "bvh.cuh"

namespace pathtracer {

    struct ray {
        const point o;
        const vector d;
        const vec3 d_inv;

        __host__ __device__ ray(const point& o, const vector& d);

        __host__ __device__ bool check_bvh_node_intersection(bvh_node* root);
    };

}
