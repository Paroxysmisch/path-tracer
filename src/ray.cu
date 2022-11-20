#include <cmath>
#include "ray.cuh"
#include <iostream>
#include <math.h>

namespace pathtracer {

    __host__ __device__ ray::ray(const point& origin, const vector& direction): 
        o(origin), d(direction), d_inv({1.f / direction.x, 1.f / direction.y, 1.f / direction.z}) {}

    __host__ __device__ bool ray::operator==(const ray &other) const {
        return (o == other.o) && (d == other.d) && (d_inv == other.d_inv);
    }

    __host__ __device__ bool ray::check_bvh_node_intersection(bvh_node* b) const {
        // https://tavianator.com/2011/ray_box.html#:~:text=The%20fastest%20method%20for%20performing,remains%2C%20it%20intersected%20the%20box
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        // https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525

        // Special case when ray origin is within the bounding box,
        // intersection is guaranteed (even with a 0 vector for efficiency)
        if ((b->lower.x <= o.x) && (o.x <= b->upper.x) &&
            (b->lower.y <= o.y) && (o.y <= b->upper.y) &&
            (b->lower.z <= o.z) && (o.z <= b->upper.z)) return true;

        vec3 t0 = (b->lower - o) & d_inv;
        vec3 t1 = (b->upper - o) & d_inv;
        vec3 tmin{fminf(t0.x, t1.x),
                  fminf(t0.y, t1.y),
                  fminf(t0.z, t1.z)};

        vec3 tmax{fmaxf(t0.x, t1.x),
                  fmaxf(t0.y, t1.y),
                  fmaxf(t0.z, t1.z)};
        
        return (tmin.max_component() <= tmax.min_component()) && tmin.max_component() >= 0.f;
    }

    __host__ __device__ int ray::find_intersections(bvh_node* root, int* collision_buffer) const {
        bvh_node* stack[64];
        bvh_node** stack_ptr = stack;
        *stack_ptr = nullptr;
        ++stack_ptr;

        int num_collisions{0};

        bvh_node* node = root;

        if (root->is_leaf() && check_bvh_node_intersection(root)) {
            collision_buffer[num_collisions] = root->object_index;
            ++num_collisions;
            return num_collisions;
        }

        if (root->is_leaf() && !check_bvh_node_intersection(root)) {
            return num_collisions;
        }

        do {
            // std::cout << stack_ptr << " currently pointing at" << *stack_ptr << std::endl;

            // if (node->left->is_leaf()) {
            //     std::cout << "left: " << node->left->object_index << std::endl;
            // }

            // if (node->right->is_leaf()) {
            //     std::cout << "right: " << node->right->object_index << std::endl;
            // }

            bvh_node* child_l = node->left;
            bvh_node* child_r = node->right;
            bool overlap_l = check_bvh_node_intersection(child_l);
            bool overlap_r = check_bvh_node_intersection(child_r);

            if (overlap_l && child_l->is_leaf()) {
                collision_buffer[num_collisions] = child_l->object_index;
                ++num_collisions;
                // std::cout << "Collision left! " << child_l->object_index << std::endl;
            }

            if (overlap_r && child_r->is_leaf()) {
                collision_buffer[num_collisions] = child_r->object_index;
                ++num_collisions;
                // std::cout << "Collision right! " << child_r->object_index << std::endl;
            }

            bool traverse_l = (overlap_l && !(child_l->is_leaf()));
            bool traverse_r = (overlap_r && !(child_r->is_leaf()));

            // std::cout << "traverse_l: " << traverse_l << ", traverse_r: " << traverse_r << std::endl;

            if (!traverse_l && !traverse_r) {
                node = *--stack_ptr;
            } else {
                node = (traverse_l) ? child_l : child_r;
                if (traverse_l && traverse_r) {
                    *stack_ptr++ = child_r;
                    // std::cout << "Put right" << std::endl;
                }
            }
        } while (node != nullptr);

        return num_collisions;
    }

    __host__ __device__ point ray::shoot_distance(float t) const {
        return o + (d * t);
    }

}
