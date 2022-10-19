#pragma once

#include "check_cuda_errors.h"
#include "util.cuh"

namespace pathtracer {

    struct bvh_arena;

    struct bvh_node {
        bvh_node* left;
        bvh_node* right;
        int object_index;
        vec3 lower;
        vec3 upper;
        int count;

        __host__ __device__ bool is_leaf();

        static bvh_node* gen_leaf_node(int object_index, const vec3& lower, const vec3& upper, bvh_arena* arena);

        static bvh_node* gen_internal_node(bvh_node* left, bvh_node* right, bvh_arena* arena);

        static bvh_node* gen_hierarchy(unsigned int* sorted_morton_codes,
                                       int* sorted_object_indices,
                                       vec3* temp_dimensions,
                                       int first,
                                       int last,
                                       bvh_arena* arena);
    };

    // Convert a point with coordinates in range [-1,1]
    // to [0,1]
    vec3 convert_range(const vec3& v);

    // Inserts two 0s between each bit, converting
    // a 10-bit integer to a 30-bit one
    unsigned int expand_bits_morton(unsigned int v);

    // For a point with coordinates within [-1, -1], 
    // transfrom to coordinates within [0,1] and 
    // generate its Morton code
    unsigned int point_to_morton(const vec3& v);

    int find_split(const unsigned int* sorted_morton_codes, int first, int last);

    struct bvh_arena {
        int max_num_elems;
        int current;
        bvh_node* data;

        bvh_arena(int num_objects);

        bool bvh_arena_malloc(bvh_node** b);

        void free_arena();
    };

}
