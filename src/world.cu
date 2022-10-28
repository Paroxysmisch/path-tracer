#include "check_cuda_errors.h"
#include "scene.cuh"
#include "shapes.cuh"
#include "world.cuh"
#include "bvh.cuh"

namespace pathtracer {

    world::world(const std::initializer_list<const object> l, dim3 blocks, dim3 threads):
        num_objects(l.size()), arena(new bvh_arena(l.size())) {
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&objects), num_objects * sizeof(pathtracer::object)) );
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&collision_buffer), blocks.x * blocks.y * threads.x * threads.y * 2 * num_objects * sizeof(int)) );
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&intersection_buffer), blocks.x * blocks.y * threads.x * threads.y * 2 * num_objects * sizeof(intersection)) );

        int current_object{0};
        for (auto it = l.begin(); it < l.end(); ++it) {
            objects[current_object].shape_t = it->shape_t;
            switch (it->shape_t) {
                case SPHERE:
                    objects[current_object].shape_d.sphere = it->shape_d.sphere;
                    break;
            }
            objects[current_object].mat_t = it->mat_t;
            objects[current_object].mat_d = it->mat_d;
            ++current_object;
        }

        bvh_root = pathtracer::gen_bvh(objects, num_objects, arena);
    }

    __host__ __device__ computations world::prepare_computations(const intersection& intersection, const ray& r) {
        point surface_point = r.shoot_distance(intersection.t_value);
        vector surface_normal;
        object& intersected_object = objects[intersection.object_index];
        switch (intersected_object.shape_t) {
            case SPHERE:
                surface_normal = intersected_object.shape_d.sphere.world_normal_at(surface_point);
                break;
        }
        vector eye = (-r.d).normalize();
        bool inside = false;
        if (surface_normal * eye < 0) {
            inside = true;
            surface_normal = -surface_normal;
        }
        return {
            {intersection.t_value, intersection.object_index},
            surface_point,
            eye,
            surface_normal,
            inside
        };
    }

    __host__ __device__ computations world::intersect_world(const ray& r, bool& success_flag, int* collision_buffer, pathtracer::intersection* intersection_buffer) {
        int possible_intersections = r.find_intersections(bvh_root, collision_buffer);

        pathtracer::intersection* intersection_buffer_ptr = intersection_buffer;
        int intersection_buffer_size{0};

        for (int k{0}; k < possible_intersections; ++k) {
                int object_index = collision_buffer[k];

                int num_intersections;

                switch (objects[object_index].shape_t) {
                    case SPHERE:
                        num_intersections = objects[object_index].shape_d.sphere.find_intersections(r, intersection_buffer_ptr, object_index);
                        break;
                }

                intersection_buffer_ptr += num_intersections;
                intersection_buffer_size += num_intersections;
        }

        pathtracer::intersection* closest_positive_intersection = pathtracer::get_closest_positive_intersection(intersection_buffer, intersection_buffer_size);

        if (closest_positive_intersection != nullptr) {
            success_flag = true;
            return prepare_computations(*closest_positive_intersection, r);
        } else {
            success_flag = false;
            return {};
        }
    }

}