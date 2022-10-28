#pragma once

#include "util.cuh"
#include "ray.cuh"

namespace pathtracer {

    struct camera {
        const int height;
        const int width;
        const float fov;
        float half_height;
        float half_width;
        float pixel_size;
        mat4 inverse_transform;


        camera(int height, 
               int width, 
               float fov, 
               const point& from = {0.f, 0.f, 0.f}, 
               const point& to = {0.f, 0.f, -1.f}, 
               const vector& up = {0.f, 1.f, 0.f}, 
               const mat4& additional_transformation = mat4::get_identity());

        mat4 gen_view_transform(const point& from, const point& to, const vector& up, const mat4& additional_transformation = mat4::get_identity());

        __host__ __device__ ray gen_ray_for_pixel(int i, int j);
    };

}