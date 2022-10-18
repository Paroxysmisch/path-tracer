#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include "check_cuda_errors.h"
#include "constants.h"

namespace pathtracer {

    __host__ __device__ bool f_equal(float a, float b);

    struct vec3 {
        float x;
        float y;
        float z;

        __host__ __device__ vec3();

        __host__ __device__ vec3(float x, float y, float z);

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
    };

    using point = vec3;
    using vector = vec3;
    using color = vec3;

    __host__ __device__ unsigned char to_byte(float n);

    template <size_t height, size_t width>
    struct canvas {
        vec3* m_data;
        // Only call cudaFree if you are the original owner of the canvas
        // You are not the owner if you were copy constructed canvas
        bool m_owner;

        __host__ canvas();

        __host__ canvas(canvas<height, width>& other);

        __host__ ~canvas();

        __host__ __device__ canvas& write_pixel(size_t y, size_t x, const vec3& data);

        __host__ void export_as_PPM(const std::string& filename) const;
    };

    template <size_t height, size_t width>
    __host__ canvas<height, width>::canvas(): m_owner(true) {
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&m_data), 
                                           sizeof(vec3) * height * width) );

        for (size_t i{0}; i < height * width; ++i) {
            (m_data)[i] = {0, 0, 0};
        }
    }

    template <size_t height, size_t width>
    __host__ canvas<height, width>::canvas(canvas<height, width>& other): m_owner(false), m_data(other.m_data) {}

    template <size_t height, size_t width>
    __host__ canvas<height, width>::~canvas() {
        if (m_owner) {
            cudaFree(m_data);
        }
    }

    template <size_t height, size_t width>
    __host__ __device__ canvas<height, width>& canvas<height, width>::write_pixel(size_t y, size_t x, const vec3& data) {
            m_data[y * width + x] = data;
            return *this;
    }

    template <size_t height, size_t width>
    __host__ void canvas<height, width>::export_as_PPM(const std::string& filename) const {
        std::ofstream file{filename};

        file << "P3" << std::endl
             << width << " " << height << std::endl
             << 255 << std::endl;

        constexpr size_t max_per_line{64};

        size_t counter{0};

        for (size_t i{0}; i < height; ++i) {
            for (size_t j{0}; j < width; ++j) {
                if (counter >= max_per_line) {
                    file << std::endl;
                    counter = 0;
                }

                const vec3& color = m_data[i * width + j];

                file << static_cast<int>(to_byte(color.x)) << " "
                     << static_cast<int>(to_byte(color.y)) << " "
                     << static_cast<int>(to_byte(color.z)) << " ";

                ++counter;
            }
        }

        // PPMs end with a newline character
        file << std::endl;

        file.close();
    }

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

        __host__ __device__ mat4 inverse(bool& success_flag);

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

}
