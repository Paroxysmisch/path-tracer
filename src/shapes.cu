#include <cmath>
#include "constants.h"
#include "shapes.cuh"
#include "util.cuh"

namespace pathtracer {

    __host__ __device__ intersection* get_closest_positive_intersection(intersection* intersection_buffer, int size) {
        intersection* result = nullptr;
        float res = INFINITY;

        for (int i{0}; i < size; ++i) {
            if (intersection_buffer[i].t_value > 0.f && 
                intersection_buffer[i].t_value < res) {
                res = intersection_buffer[i].t_value;
                result = &intersection_buffer[i];
            }
        }

        return result;
    }

    __host__ __device__ shape::shape(vec3 lower, vec3 upper, mat4 transformation_to_world): lower(lower), upper(upper), transformation_to_world(transformation_to_world) {
        bool success_flag;
        mat4 transformation_to_world_copy = mat4{transformation_to_world};
        transformation_to_object = transformation_to_world_copy.inverse(success_flag);
        inverse_transpose = transformation_to_object.transpose();
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

    __host__ __device__ vec3 sphere::local_normal_at(const point& local_surface_point, float u, float v) const {
        return local_surface_point;
    }

    __host__ __device__ vec3 sphere::world_normal_at(const point& world_surface_point, float u, float v) const {
        const point local_surface_point = transformation_to_object.transform_point(world_surface_point);
        vector transformed_normal = inverse_transpose.transform_vector(local_surface_point);
        return transformed_normal.normalize();
    }

    __host__ __device__ vec3 sphere::world_tangent_at(const point& world_surface_point) const {
        // vector centre_to_surface = transformation_to_object.transform_point(world_surface_point);
        // if (centre_to_surface == vec3(0.f, 1.f, 0.f)) {
        //     return {1.f, 0.f, 0.f};
        // }
        // return (vec3{0.f, 1.f, 0.f} ^ (world_surface_point - transformation_to_world.transform_point(vec3(0.f, 0.f, 0.f)))).normalize();
        float theta = atan2f(world_surface_point.y, world_surface_point.x);
        float cos_phi = (world_surface_point.z / world_surface_point.mag());

        return {-sinf(theta), 0.f, cos_phi};
    }

    __host__ __device__ point triangle_get_lower(const point& p1, const point& p2, const point& p3) {
        return {
            minf(p1.x, minf(p2.x, p3.x)),
            minf(p1.y, minf(p2.y, p3.y)),
            minf(p1.z, minf(p2.z, p3.z))
        };
    }

    __host__ __device__ point triangle_get_upper(const point& p1, const point& p2, const point& p3) {
        return {
            maxf(p1.x, maxf(p2.x, p3.x)),
            maxf(p1.y, maxf(p2.y, p3.y)),
            maxf(p1.z, maxf(p2.z, p3.z))
        };
    }

    __host__ __device__ triangle::triangle(const point& p1, const point& p2, const point& p3) :
        shape(triangle_get_lower(p1, p2, p3), triangle_get_upper(p1, p2, p3), mat4::get_identity()),
        p1{p1},
        p2{p2},
        p3{p3},
        e1{p2 - p1},
        e2{p3 - p1},
        n1{e2 ^ e1},
        n2{e2 ^ e1},
        n3{e2 ^ e1} {}

    __host__ __device__ triangle::triangle(const point& p1, const point& p2, const point& p3, const vector& n1, const vector& n2, const vector& n3, const vector& tex1, const vector& tex2, const vector& tex3, int texture_index) :
        shape(triangle_get_lower(p1, p2, p3), triangle_get_upper(p1, p2, p3), mat4::get_identity()),
        p1{p1},
        p2{p2},
        p3{p3},
        e1{p2 - p1},
        e2{p3 - p1},
        n1{n1},
        n2{n2},
        n3{n3},
        tex1{tex1},
        tex2{tex2},
        tex3{tex3},
        texture_idx{texture_index} {
            float x1 = p2.x - p1.x;
            float x2 = p3.x - p1.x;
            float y1 = p2.y - p1.y;
            float y2 = p3.y - p1.y;
            float z1 = p2.z - p1.z;
            float z2 = p3.z - p1.z;

            float s1 = tex2.x - tex1.x;
            float s2 = tex3.x - tex1.x;
            float t1 = tex2.y - tex1.y;
            float t2 = tex3.y - tex1.y;

            float r = 1.f / (s1 * t2 - s2 * t1);

            tan1 = vec3((t2 * x1 - t1 * x2) * r,
                    (t2 * y1 - t1 * y2) * r,
                    (t2 * z1 - t1 * z2) * r).normalize();
            tan2 = vec3((s1 * x2 - s2 * x1) * r,
                    (s1 * y2 - s2 * y1) * r,
                    (s1 * z2 - s2 * z1) * r).normalize();
        }

    __host__ __device__ int triangle::find_intersections(const ray& r, intersection* intersection_buffer, int object_index) {
        vector dir_cross_e2 = r.d ^ e2;
        float det = e1 * dir_cross_e2;
        if (fabsf(det) < epsilon) return 0;

        float f = 1.f / det;
        vector p1_to_origin = r.o - p1;
        float u = f * (p1_to_origin * dir_cross_e2);
        if (u < 0.f || u > 1.f) return 0;

        vector origin_cross_e1 = p1_to_origin ^ e1;
        float v = f * (r.d * origin_cross_e1);
        if (v < 0.f || (u + v) > 1.f) return 0;

        float t = f * (e2 * origin_cross_e1);
        intersection_buffer[0] = intersection{t, object_index, u, v};
        return 1;
    }

    __host__ __device__ vec3 triangle::local_normal_at(const point& local_surface_point, float u, float v) const {
        return ((n2 * u) + (n3 * v) + (n1 * (1.f - u - v))).normalize();
    }

    __host__ __device__ vec3 triangle::world_normal_at(const point& world_surface_point, float u, float v) const {
        return ((n2 * u) + (n3 * v) + (n1 * (1.f - u - v))).normalize();
    }

    __host__ __device__ shape_data& shape_data::operator=(const struct sphere& other) {
        sphere = other;

        return *this;
    }

    __host__ __device__ phong::phong(const vec3& color, float ambient, float diffuse, float specular, float shininess): 
        color(color), ambient(ambient), diffuse(diffuse), specular(specular), shininess(shininess) {}

    __host__ __device__ light::light(const vec3& color): color(color) {};

    __host__ __device__ mat_data& mat_data::operator=(const struct light& other) {
        light = other;

        return *this;
    }

}