#pragma once
#include "ray.cuh"
#include "util.cuh"

namespace pathtracer {
    struct intersection {
        float t_value;
        int object_index;
        float u;
        float v;
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

        __host__ __device__ virtual vec3 local_normal_at(const point& local_surface_point, float u = -1.f, float v = -1.f) const = 0;

        __host__ __device__ virtual vec3 world_normal_at(const point& world_surface_point, float u = -1.f, float v = -1.f) const = 0;
    };

    struct sphere : shape {
        __host__ __device__ sphere(const mat4& transformation_to_world = mat4::get_identity());

        // __host__ __device__ sphere& operator=(const sphere& other);

        __host__ __device__ virtual int find_intersections(const ray& r, intersection* intersection_buffer, int object_index) override;

        __host__ __device__ virtual vec3 local_normal_at(const point& local_surface_point, float u = -1.f, float v = -1.f) const override;
        
        __host__ __device__ virtual vec3 world_normal_at(const point& world_surface_point, float u = -1.f, float v = -1.f) const override;

        __host__ __device__ vec3 world_tangent_at(const point& world_surface_point) const;
    };

    __host__ __device__ point triangle_get_lower(const point& p1, const point& p2, const point& p3);

    __host__ __device__ point triangle_get_upper(const point& p1, const point& p2, const point& p3);

    struct triangle : shape {
        // Two edge vectors
        vector p1;
        vector p2;
        vector p3;
        vector e1;
        vector e2;
        vector n1;
        vector n2;
        vector n3;
        vector tex1;
        vector tex2;
        vector tex3;
        int texture_idx;
        vector tan1;
        vector tan2;

        __host__ __device__ triangle(const point& p1, const point& p2, const point& p3);

        __host__ __device__ triangle(const point& p1, const point& p2, const point& p3, const vector& n1, const vector& n2, const vector& n3, const vector& tex1 = {0.f, 0.f, -1.f}, const vector& tex2 = {0.f, 0.f, -1.f}, const vector& tex3 = {0.f, 0.f, -1.f}, int texture_idx = -1);

        __host__ __device__ virtual int find_intersections(const ray& r, intersection* intersection_buffer, int object_index) override;

        __host__ __device__ virtual vec3 local_normal_at(const point& local_surface_point, float u, float v) const override;

        __host__ __device__ virtual vec3 world_normal_at(const point& world_surface_point, float u, float v) const override;
    };

    enum shape_type {
        SPHERE,
        TRIANGLE
    };

    union shape_data {
        __host__ __device__ shape_data& operator=(const sphere& other);
        sphere sphere;
        triangle triangle;
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
        float transmissiveness = 0.f;
        float refractive_index = 1.f;
        float reflectance = 0.04f;
        float clear_coat_strength = 0.f;
        float clear_coat_roughness = 0.f;
        float anisotropy = 0.f;
        float transmissive_roughness = 0.f;
        float optical_density = 0.f;
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