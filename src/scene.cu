#include <cstdlib>
#include "bvh.cuh"
#include "scene.cuh"

namespace pathtracer {

    int compare(const void* a, const void* b) {
        morton_and_index _a = *((morton_and_index*) a);
        morton_and_index _b = *((morton_and_index*) b);

        if (_a.morton_code < _b.morton_code) return -1;
        else if (_a.morton_code > _b.morton_code) return 1;
        else return 0;
    }

    void gen_sorted_morton_codes_and_indices(const object* objects, int num_objects, morton_and_index* out_buffer) {
        for (int i{0}; i < num_objects; ++i) {
            switch (objects[i].shape_t) {
                case SPHERE:
                    out_buffer[i] = {point_to_morton(objects[i].shape_d.sphere.transformation_to_world.transform_point({0.f, 0.f, 0.f})),
                                     i};
                    break;
                case TRIANGLE:
                    // TODO:
                    // Need to take average here
                    out_buffer[i] = {point_to_morton(objects[i].shape_d.triangle.p1),
                                     i};
                    break;
                }
        }

        qsort(out_buffer, num_objects, sizeof(morton_and_index), compare);
    }

    int find_split(const morton_and_index* sorted_morton_and_index, int first, int last) {
        unsigned int first_code = sorted_morton_and_index[first].morton_code;
        unsigned int last_code = sorted_morton_and_index[last].morton_code;

        if (first_code == last_code)
            // Return value in the middle if the Morton codes
            // at either end are the same
            return (first + last) >> 1;

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
                unsigned int split_code = sorted_morton_and_index[new_split_point].morton_code;
                int split_prefix = __builtin_clz(first_code ^ split_code);
                if (split_prefix > common_prefix_length) {
                    split_point = new_split_point;
                }
            }
        } while (step > 1);

        return split_point;
    }
    
    bvh_node* _gen_bvh(const morton_and_index* sorted_morton_and_index,
                      const object* objects,
                      int first,
                      int last,
                      bvh_arena* arena) {
        // Create a leaf node, if a single object
        if (first == last) {
            int obj_index = sorted_morton_and_index[first].object_index;
            switch (objects[obj_index].shape_t) {
                case SPHERE:
                    return bvh_node::gen_leaf_node(obj_index, objects[obj_index].shape_d.sphere.lower, objects[obj_index].shape_d.sphere.upper, arena);
                case TRIANGLE:
                    return bvh_node::gen_leaf_node(obj_index, objects[obj_index].shape_d.triangle.lower, objects[obj_index].shape_d.triangle.upper, arena);
                }
        }   

        int split = find_split(sorted_morton_and_index, first, last);

        bvh_node* left = _gen_bvh(sorted_morton_and_index, objects, first, split, arena);
        bvh_node* right = _gen_bvh(sorted_morton_and_index, objects, split + 1, last, arena);

        return bvh_node::gen_internal_node(left, right, arena);
    }

    bvh_node* gen_bvh(const object* objects, int num_objects, bvh_arena* arena) {
        pathtracer::morton_and_index out_buffer[num_objects];

        pathtracer::gen_sorted_morton_codes_and_indices(objects, num_objects, out_buffer);

        return pathtracer::_gen_bvh(out_buffer, objects, 0, num_objects - 1, arena);
    }

}