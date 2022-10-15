#include <cmath>
#include "bvh.cuh"

namespace pathtracer {

    bool bvh_node::is_leaf() {
        return !left && !right;
    }

    bvh_node* bvh_node::gen_leaf_node(int object_index) {
        bvh_node* result;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&result), 
                                           sizeof(bvh_node)) );
        result->left = nullptr;
        result->right = nullptr;

        return result;
    }

    bvh_node* bvh_node::gen_internal_node(bvh_node* left, bvh_node* right) {
        bvh_node* result;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&result), 
                                           sizeof(bvh_node)) );
        result->left = left;
        result->right = right;

        return result;
    }

    bvh_node* bvh_node::gen_hierarchy(unsigned int* sorted_morton_codes,
                                      int* sorted_object_indices,
                                      int first,
                                      int last) {
        // Create a leaf node, if a single object
        if (first == last)
            return gen_leaf_node(sorted_object_indices[first]);

        int split = find_split(sorted_morton_codes, first, last);

        bvh_node* left = bvh_node::gen_hierarchy(sorted_morton_codes, sorted_object_indices, first, split);
        bvh_node* right = bvh_node::gen_hierarchy(sorted_morton_codes, sorted_object_indices, split + 1, last);

        return bvh_node::gen_internal_node(left, right);
    }

    vec3 convert_range(vec3& v) {
        return vec3{(v.x + 1.f) / 2.f,
                    (v.y + 1.f) / 2.f,
                    (v.z + 1.f) / 2.f};
    }

    unsigned int expand_bits_morton(unsigned int i) {
        i = (i * 0x00010001u) & 0xFF0000FFu;
        i = (i * 0x00000101u) & 0x0F00F00Fu;
        i = (i * 0x00000011u) & 0xC30C30C3u;
        i = (i * 0x00000005u) & 0x49249249u;
        return i;
    }

    unsigned int point_to_morton(vec3& v) {
        vec3 _v = convert_range(v);
        // Ensure the x, y and z components are representable by 10 bits
        float x = fminf(fmaxf(_v.x * 1024.0f, 0.0f), 1023.0f);
        float y = fminf(fmaxf(_v.y * 1024.0f, 0.0f), 1023.0f);
        float z = fminf(fmaxf(_v.z * 1024.0f, 0.0f), 1023.0f);
        unsigned int _x = expand_bits_morton((unsigned int)x);
        unsigned int _y = expand_bits_morton((unsigned int)y);
        unsigned int _z = expand_bits_morton((unsigned int)z);
        return _x * 4 + _y * 2 + _z;
    }

    int find_split(unsigned int* sorted_morton_codes, int first, int last) {
        unsigned int first_code = sorted_morton_codes[first];
        unsigned int last_code = sorted_morton_codes[last];

        if (first_code == last_code)
            // Return value in the middle if the Morton codes
            // at either end are the same
            return first + last >> 1;

        // Find number of highest bits which are the same
        // using a compiler builtin
        int common_prefix_length = __builtin_clz(first_code ^ last_code);

        // Use binary search to find where the next bit differs
        int split_point = first;
        int step = last - first;

        do {
            step = (step + 1) >> 1;
            int new_split_point = split_point + step;

            if (new_split_point < last) {
                unsigned int split_code = sorted_morton_codes[new_split_point];
                int split_prefix = __builtin_clz(first_code ^ split_code);
                if (split_prefix > common_prefix_length) {
                    split_point = new_split_point;
                }
            }
        } while (step > 1);

        return split_point;
    }

}
