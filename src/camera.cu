#include <cmath>
#include "camera.cuh"
#include "constants.h"
#include "util.cuh"

namespace pathtracer {

    camera::camera(int height, 
                   int width, 
                   float fov, 
                   const point& from, 
                   const point& to, 
                   const vector& up, 
                   const mat4& additional_transformation): 
                       height(height), width(width), fov(fov) {
        const float half_view = tanf(fov / 2.f);
        const float aspect_ratio = static_cast<float>(width) / height;

        if (aspect_ratio >= 1.f) {
            half_width = half_view;
            half_height = half_view / aspect_ratio;
        } else {
            half_width = half_view * aspect_ratio;
            half_height = half_view;
        }

        pixel_size = (half_width * 2) / width;

        bool success_flag1;
        bool success_flag2;

        inverse_transform = gen_view_transform(from, to, up, additional_transformation.inverse(success_flag1)).inverse(success_flag2);
    }

    mat4 camera::gen_view_transform(const point& from, const point& to, const vector& up, const mat4& additional_transformation) {
        vector forward = (to - from).normalize();
        vector up_normalized{up};
        up_normalized.normalize();
        vector left = forward ^ up_normalized;
        vector true_up = left ^ forward;

        mat4 orientation = {
            left.x, left.y, left.z, 0.f,
            true_up.x, true_up.y, true_up.z, 0.f,
            -forward.x, -forward.y, -forward.z, 0.f,
            0.f, 0.f, 0.f, 1.f
        };

        return orientation * (additional_transformation * mat4::get_translation(-from.x, -from.y, -from.z));
    }

    __host__ __device__ ray camera::gen_ray_for_pixel(int i, int j) {
        float y_offset = (i + 0.5f) * pixel_size;
        float x_offset = (j + 0.5f) * pixel_size;

        float world_y = half_height - y_offset;
        float world_x = half_width - x_offset;

        point pixel = inverse_transform.transform_point({world_x, world_y, -1.f});
        point origin = inverse_transform.transform_point({0.f, 0.f, 0.f});
        vector direction = (pixel - origin).normalize();

        return ray(origin, direction);
    }

}