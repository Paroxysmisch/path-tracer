#pragma once
#include "ray.cuh"
#include "util.cuh"

namespace pathtracer {
    struct intersection {
        float t_value;
        int object_index;
    };

    __host__ __device__ intersection* get_closest_positive_intersection(intersection* intersection_buffer, int size);

    struct shape {
        vec3 lower;
        vec3 upper;
        mat4 transformation_to_world;
        mat4 transformation_to_object;
        mat4 inverse_transpose;

        __host__ __device__ shape(vec3 lower, vec3 upper, mat4 transformation_to_world);

        __host__ __device__ virtual int find_intersections(const ray& r, intersection* intersection_buffer, int object_index) = 0;

        __host__ __device__ virtual vec3 local_normal_at(const point& local_surface_point) const = 0;

        __host__ __device__ virtual vec3 world_normal_at(const point& world_surface_point) const = 0;
    };

    struct sphere : shape {
        __host__ __device__ sphere(const mat4& transformation_to_world = mat4::get_identity());

        // __host__ __device__ sphere& operator=(const sphere& other);

        __host__ __device__ virtual int find_intersections(const ray& r, intersection* intersection_buffer, int object_index) override;

        __host__ __device__ virtual vec3 local_normal_at(const point& local_surface_point) const override;
        
        __host__ __device__ virtual vec3 world_normal_at(const point& world_surface_point) const override;
    };

    enum shape_type {
        SPHERE
    };

    union shape_data {
        __host__ __device__ shape_data& operator=(const sphere& other);
        sphere sphere;
    };

    enum mat_type {
        PHONG,
        LIGHT,
        MICROFACET
    };

    struct phong {
        __host__ __device__ phong(const vec3& color, float ambient, float diffuse, float specular, float shininess);
        vec3 color;
        float ambient;
        float diffuse;
        float specular;
        float shininess;
    };

    struct microfacet {
        vec3 color;
        vec3 emission;
        float metalness;
        float roughness;
        float transmissiveness;
        float refractive_index;
    };

    struct light {
        __host__ __device__ light(const vec3& color);
        vec3 color;
    };

    union mat_data {
      phong phong;
      microfacet microfacet;
      __host__ __device__ mat_data& operator=(const light& other);
      light light;
    };

    struct object {
        shape_type shape_t;
        shape_data shape_d;
        mat_type mat_t;
        mat_data mat_d;
    };
}