#pragma once

#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include "check_cuda_errors.h"

namespace pathtracer {

    __host__ __device__ bool f_equal(float a, float b);

    struct vec3 {
        float x;
        float y;
        float z;

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

        __host__ __device__ vec3& normalize();

        __host__ __device__ float operator*(const vec3& other) const;

        __host__ __device__ vec3 operator^(const vec3& other) const;

        __host__ __device__ vec3& operator^=(const vec3& other);
    };

    using point = vec3;
    using vector = vec3;
    using color = vec3;

    __host__ __device__ unsigned char to_byte(float n);

    template <size_t height, size_t width>
    struct canvas {
        std::array<std::array<vec3, width>, height>* m_data;

        __host__ __device__ canvas();

        __host__ __device__ canvas& write_pixel(size_t y, size_t x, const vec3& data);

        __host__ __device__ void export_as_PPM(const std::string& filename) const;
    };

    template <size_t height, size_t width>
    __host__ __device__ canvas<height, width>::canvas() {
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&m_data), 
                                           sizeof(std::array<std::array<vec3, width>, height>)) );
    }

    template <size_t height, size_t width>
    __host__ __device__ canvas<height, width>& canvas<height, width>::write_pixel(size_t y, size_t x, const vec3& data) {
            m_data[y][x] = data;
            return *this;
    }

    template <size_t height, size_t width>
    __host__ __device__ void canvas<height, width>::export_as_PPM(const std::string& filename) const {
        std::ofstream file{filename};

        file << "P3" << std::endl
             << width << " " << height << std::endl
             << 255 << std::endl;
    }

}
