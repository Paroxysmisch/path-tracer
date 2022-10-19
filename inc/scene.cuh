#pragma once

#include "bvh.cuh"
#include "shapes.cuh"

namespace pathtracer {

    int compare(const void* a, const void* b);

    struct morton_and_index {
        unsigned int morton_code;
        int object_index;
    };

    void gen_sorted_morton_codes_and_indices(const object* objects, int num_objects, morton_and_index* out_buffer);

    int find_split(const morton_and_index* sorted_morton_and_index, int first, int last);

    bvh_node* _gen_bvh(const morton_and_index* sorted_morton_and_index,
                      const object* objects,
                      int first,
                      int last,
                      bvh_arena* arena);

    bvh_node* gen_bvh(const object* objects, int num_objects, bvh_arena* arena);

}
