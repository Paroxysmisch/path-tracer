#include <cmath>
#include "shapes.cuh"

namespace pathtracer {

    __host__ __device__ intersection* get_closest_positive_intersection(intersection* intersection_buffer, int size) {
        if (size <= 0) return nullptr;

        intersection* result = &intersection_buffer[0];

        for (int i{1}; i < size; ++i) {
            if (intersection_buffer[i].t_value >= 0.f && 
                intersection_buffer[i].t_value < result->t_value) {
                result = &intersection_buffer[i];
            }
        }

        if (result->t_value >= 0.f) 
            return result;
        else
            return nullptr;
    }

    __host__ __device__ shape::shape(vec3 lower, vec3 upper, mat4 transformation_to_world): lower(lower), upper(upper), transformation_to_world(transformation_to_world) {
        bool success_flag;
        mat4 transformation_to_object_copy = mat4{transformation_to_world};
        transformation_to_object = transformation_to_object_copy.inverse(success_flag);
    }

    __host__ __device__ sphere::sphere(const mat4& transformation_to_world): shape(transformation_to_world.transform_point({-1.f, -1.f, -1.f}),
                                                      transformation_to_world.transform_point({1.f, 1.f, 1.f}), transformation_to_world) {}

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

}