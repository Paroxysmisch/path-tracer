#include "check_cuda_errors.h"
#include "util.cuh"

namespace pathtracer {
    struct bvh_node {
        bvh_node* left;
        bvh_node* right;
        int object_index;
        vec3 lower;
        vec3 upper;

        bool is_leaf();

        static bvh_node* gen_leaf_node(int object_index, const vec3& lower, const vec3& upper);

        static bvh_node* gen_internal_node(bvh_node* left, bvh_node* right);

        static bvh_node* gen_hierarchy(unsigned int* sorted_morton_codes,
                                       int* sorted_object_indices,
                                       int first,
                                       int last);
    };

    // Convert a point with coordinates in range [-1,-1]
    // to [0,1]
    vec3 convert_range(vec3& v);

    // Inserts two 0s between each bit, converting
    // a 10-bit integer to a 30-bit one
    unsigned int expand_bits_morton(unsigned int v);

    // For a point with coordinates within [-1, -1], 
    // transfrom to coordinates within [0,1] and 
    // generate its Morton code
    unsigned int point_to_morton(vec3& v);

    int find_split(unsigned int* sorted_morton_codes, int first, int last);
}
