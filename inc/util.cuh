#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include "check_cuda_errors.h"
#include "constants.h"

#include "tinyexr.h"

namespace pathtracer {

    __host__ __device__ bool f_equal(float a, float b);

    struct vec3;

    using point = vec3;
    using vector = vec3;
    using color = vec3;

    struct vec3_d {
        double x;
        double y;
        double z;

        __host__ __device__ vec3_d();

        __host__ __device__ vec3_d(float x, float y, float z);

        __host__ __device__ vec3_d(double x, double y, double z);

        __host__ __device__ vec3_d& operator+=(const vec3& other);

        __host__ __device__ vec3_d operator&(const vec3_d& other) const;

        __host__ __device__ vec3 operator-(const float scalar);

        __host__ __device__ vec3 operator-(const vec3& other);

        __host__ __device__ vec3 operator/(const float scalar);

        __host__ __device__ vec3_d& operator/=(const float scalar);
    };

    struct vec3 {
        float x;
        float y;
        float z;

        __host__ __device__ vec3();

        __host__ __device__ vec3(float x, float y, float z);

        __host__ __device__ vec3(float x);

        __host__ __device__ bool operator==(const vec3& other) const;

        __host__ __device__ vec3 operator+(const vec3& other) const;

        __host__ __device__ vec3& operator+=(const vec3& other);

        __host__ __device__ vec3 operator-(const vec3& other) const;

        __host__ __device__ vec3& operator-=(const vec3& other);

        __host__ __device__ vec3 operator-() const;

        __host__ __device__ vec3 operator*(const float scalar) const;

        __host__ __device__ vec3& operator*=(const float scalar);

        __host__ __device__ vec3 operator&(const vec3& other) const;

        __host__ __device__ vec3& operator&=(const vec3& other);

        __host__ __device__ vec3 operator/(const float scalar) const;

        __host__ __device__ vec3& operator/=(const float scalar);

        __host__ __device__ float mag() const;

        __host__ __device__ float mag_2() const;

        __host__ __device__ vec3& normalize();

        __host__ __device__ float operator*(const vec3& other) const;

        __host__ __device__ vec3 operator^(const vec3& other) const;

        __host__ __device__ vec3& operator^=(const vec3& other);

        __host__ __device__ static vec3 gen_orthogonal(const vec3& v);

        __host__ __device__ inline float max_component() {
            return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
        }

        __host__ __device__ inline float min_component() {
            return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
        }

        __host__ __device__ vector reflect(vector normal);
    };

    __host__ std::ostream &operator<<(std::ostream &os, const vec3& vec_3);

    __host__ __device__ unsigned char to_byte(float n);

    struct canvas {
        vec3* m_data;
        // Only call cudaFree if you are the original owner of the canvas
        // You are not the owner if you were copy constructed canvas
        bool m_owner;

        int m_height;
        int m_width;

        __host__ canvas(const int height, const int width);

        __host__ canvas(canvas& other);

        __host__ ~canvas();

        __host__ __device__ canvas& write_pixel(int y, int x, const vec3& data);

        __host__ void export_as_PPM(const std::string& filename) const;

        __host__ bool export_as_EXR(const std::string& filename) const;
    };

    struct mat4 {
        float m_data[16];

        __host__ __device__ mat4();

        __host__ __device__ mat4(float _00, float _01, float _02, float _03,
                                 float _10, float _11, float _12, float _13,
                                 float _20, float _21, float _22, float _23,
                                 float _30, float _31, float _32, float _33);

        __host__ __device__ static mat4 get_identity();

        __host__ __device__ bool operator==(const mat4& other) const;

        __host__ __device__ mat4 operator*(const mat4& other) const;

        __host__ __device__ mat4& operator*=(const mat4& other);

        __host__ __device__ mat4 transpose();

        __host__ __device__ mat4& operator=(const mat4& other);

        __host__ __device__ mat4 inverse(bool& success_flag) const;

        __host__ __device__ static mat4 get_translation(float x, float y, float z);

        __host__ __device__ static mat4 get_scaling(float x, float y, float z);

        __host__ __device__ static mat4 get_rotation_x(float rad);

        __host__ __device__ static mat4 get_rotation_y(float rad);

        __host__ __device__ static mat4 get_rotation_z(float rad);

        __host__ __device__ static mat4 get_shearing(float x_y, float x_z,
                                                     float y_x, float y_z,
                                                     float z_x, float z_y);

        __host__ __device__ point transform_point(const point& p) const;

        __host__ __device__ vector transform_vector(const vector& v) const;
    };

    struct quaternion {
        float w;
        vec3 ijk;

        __host__ __device__ quaternion(float w, vec3 ijk);

        __host__ __device__ quaternion(float w, float i, float j, float k);

        __host__ __device__ quaternion conjugate();

        __host__ __device__ quaternion& normalize();

        __host__ __device__ static quaternion get_rotation_between(vec3 u, vec3 v);

        __host__ __device__ static vec3 rotate_vector_by_quaternion(const vec3& v, const quaternion& q);

        __host__ __device__ static quaternion get_rotation_to_z_axis(const vec3& normalized_v);

        __host__ __device__ static quaternion get_rotation_from_z_axis(const vec3& normalized_v);

        __host__ __device__ static quaternion get_inverse_rotation(const quaternion& q);
    };

    __host__ __device__ float maxf(float a, float b);

    __host__ __device__ float minf(float a, float b);

}
