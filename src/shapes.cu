#include <cmath>
#include "shapes.cuh"

namespace pathtracer {

    __host__ __device__ intersection* get_closest_positive_intersection(intersection* intersection_buffer, int size) {
        intersection* result = nullptr;
        float t_value = INFINITY;

        for (int i{0}; i < size; ++i) {
            if (intersection_buffer[i].t_value > 0.f && 
                intersection_buffer[i].t_value < t_value) {
                result = &intersection_buffer[i];
            }
        }

        return result;
    }

    __host__ __device__ shape::shape(vec3 lower, vec3 upper, mat4 transformation_to_world): lower(lower), upper(upper), transformation_to_world(transformation_to_world) {
        bool success_flag;
        mat4 transformation_to_object_copy = mat4{transformation_to_world};
        transformation_to_object = transformation_to_object_copy.inverse(success_flag);
    }

    __host__ __device__ sphere::sphere(const mat4& transformation_to_world): shape(transformation_to_world.transform_point({-1.f, -1.f, -1.f}),
                                                      transformation_to_world.transform_point({1.f, 1.f, 1.f}), transformation_to_world) {}

    // __host__ __device__ sphere& sphere::operator=(const sphere& other) {
    //     lower = other.lower;
    //     upper = other.upper;
    //     transformation_to_world = other.transformation_to_world;
    //     transformation_to_object = other.transformation_to_object;

    //     return *this;
    // }

    __host__ __device__ int sphere::find_intersections(const ray& r, intersection* intersection_buffer, int object_index) {
        ray _r{transformation_to_object.transform_point(r.o),
               transformation_to_object.transform_vector(r.d)};

        float a = _r.d * _r.d;
        float b = 2 * (_r.d * _r.o);
        float c = (_r.o * _r.o) - 1.f;

        float discriminant = (b * b) - (4 * a * c);

        if (discriminant < 0) return 0;

        intersection_buffer[0] = intersection{(-b - sqrtf(discriminant)) / (2 * a), object_index};
        intersection_buffer[1] = intersection{(-b + sqrtf(discriminant)) / (2 * a), object_index};

        return 2;
    }

    __host__ __device__ shape_data& shape_data::operator=(const struct sphere& other) {
        sphere = other;

        return *this;
    }

}